"""HTTP API (contract from the Patterno brief appendix) + dashboard, served on port 8080."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel

from . import timeutil
from .audit import verification_entry
from .config import Settings
from .http import DocumentStore, HostRateLimiter, parse_host_intervals
from .llm import LLMRouter
from .store import Store
from .worker import Manager

DASHBOARD = os.path.join(os.path.dirname(__file__), "dashboard.html")
HEAL_PAGE = os.path.join(os.path.dirname(__file__), "heal.html")
CROSSREF_PAGE = os.path.join(os.path.dirname(__file__), "crossref.html")


class AddPortal(BaseModel):
    url: str


def _portal_entry(store: Store, p: dict) -> dict:
    return {
        "id": p["id"],
        "url": p["url"],
        "state": p["state"],
        "reason": p["reason"],
        "tenderCount": store.tender_count(p["id"]),
        "lastSuccessAt": p["last_success_at"],
        # Patterno, 20 Nov: fields may be added. Integration health = state; data coverage is reported separately.
        "coverage": _coverage(store, p["id"]),
        # Continuous verification (audit.py): the latest sampled audit of served values against the portal itself.
        "verification": verification_entry(store.latest_audit(p["id"])),
    }


def _coverage(store: Store, pid: str, runs: list[dict] | None = None, documents: dict | None = None) -> dict:
    """Integration health is the portal state; coverage says how much of the portal's data we actually deliver:
    tenders complete/partial, details pending, documents available/pending/partial/unavailable/none (+ reasons).
    Built from the latest successful read; cheap enough for every GET /portals."""
    if runs is None:
        ok = store.latest_run(pid, "ok")
    else:
        ok = next((r for r in reversed(runs) if r["result"] == "ok"), None)
    if ok is None:
        return {"tenders": "unknown", "details": {"pending": 0}, "documents": "unknown", "reasons": []}
    st = _run_stats(ok)
    if documents is None:
        documents = {"downloaded": 1 if store.has_documents(pid) else 0, "pending": int(st.get("pendingDocs") or 0),
                     "pendingLists": int(st.get("pendingDocLists") or 0)}
    reasons = []
    if st.get("capped"):
        tenders = "partial"
        reasons.append(f"the portal shows only part of its listing ({st.get('stopReason') or 'capped'})")
    elif st.get("shortfall"):
        tenders = "partial"
        reasons.append(f"{st['shortfall']} listed tenders not seen (the listing changed while it was read)")
    elif st.get("droppedIncomplete"):
        tenders = "partial"
        reasons.append(f"{st['droppedIncomplete']} listed tenders could not be read completely and are not delivered")
    else:
        tenders = "complete"
    spec = store.current_spec(pid)
    docs_spec = (spec[1].get("documents") or {}) if spec else {}
    if not (docs_spec.get("from") or docs_spec.get("sel") or docs_spec.get("path")):
        docs = "none"
        reasons.append("no documents found for this portal's tenders")
    elif st.get("docBlocked") and not documents.get("downloaded"):
        docs = "unavailable"
        reasons.append(f"documents not available from this portal ({st.get('docBlockReason') or 'blocked'})")
    elif st.get("docBlocked"):
        docs = "partial"
        reasons.append(f"{st['docBlocked']} documents blocked ({st.get('docBlockReason') or 'blocked'})")
    elif documents.get("pending") or documents.get("pendingLists"):
        docs = "pending"
    else:
        docs = "available"
    return {"tenders": tenders, "details": {"pending": int(st.get("pendingDetails") or 0)}, "documents": docs,
            "reasons": reasons}


_STATS: dict[str, dict] = {}     # finished run id -> parsed stats (a finished run's detail never changes)


def _run_stats(run: dict) -> dict:
    hit = _STATS.get(run["id"])
    if hit is None:
        hit = json.loads(run["detail"] or "{}").get("stats") or {}
        if run.get("finished_at"):
            if len(_STATS) > 5000:
                _STATS.clear()
            _STATS[run["id"]] = hit
    return hit


def _run_entry(r: dict) -> dict:
    return {"id": r["id"], "kind": r["kind"], "startedAt": r["started_at"],
            "finishedAt": r["finished_at"], "result": r["result"]}


def create_app(settings: Settings | None = None, pipeline=None, llm=None, transport=None) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.data_dir)
    limiter = HostRateLimiter(settings.min_request_interval, parse_host_intervals(settings.host_intervals))
    docs = DocumentStore(store, settings.max_document_bytes, settings.max_zip_total_bytes)
    llm = llm(store) if callable(llm) else LLMRouter(settings, store)
    if pipeline is None:
        from .engine import Engine
        pipeline = Engine(settings)
    manager = Manager(settings, store, pipeline, llm, limiter, docs, transport=transport)

    # WATCHTOWER_WORKER=0: serve the API and pages over existing data without reading any portal (UI previews on a
    # copy of the database).
    run_worker = os.environ.get("WATCHTOWER_WORKER", "1") != "0"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_worker:
            await manager.start()
        yield
        if run_worker:
            await manager.stop()
        if hasattr(llm, "aclose"):
            await llm.aclose()

    app = FastAPI(title="Watchtower", lifespan=lifespan)
    app.state.store, app.state.manager, app.state.settings = store, manager, settings

    def _get(pid: str) -> dict:
        p = store.get_portal(pid)
        if p is None:
            raise HTTPException(404, "unknown portal id")
        return p

    # ---------------------------------------------------------------- contract
    @app.post("/portals", status_code=201)
    async def add_portal(body: AddPortal):
        if not body.url or not body.url.strip():
            raise HTTPException(400, "url is required")
        pid, _created = manager.add(body.url)
        return {"id": pid}

    # Pages poll these (the heal page every 1.5 s): the work runs in a thread so the event loop stays responsive.
    @app.get("/portals")
    async def list_portals():
        return await asyncio.to_thread(lambda: [_portal_entry(store, p) for p in store.list_portals()])

    @app.get("/portals/{pid}")
    async def get_portal(pid: str):
        p = _get(pid)
        return await asyncio.to_thread(_portal_entry, store, p)

    @app.delete("/portals/{pid}", status_code=204)
    async def delete_portal(pid: str):
        _get(pid)
        manager.delete(pid)
        return Response(status_code=204)

    @app.post("/portals/{pid}/refresh", status_code=202)
    async def refresh(pid: str):
        _get(pid)
        return {"runId": manager.refresh(pid)}

    @app.get("/portals/{pid}/tenders")
    async def tenders(pid: str, changedSince: str | None = Query(default=None)):
        _get(pid)
        since = None
        if changedSince:
            try:
                since = timeutil.parse(changedSince)
            except ValueError:
                raise HTTPException(400, "changedSince must be an ISO date-time with offset")
        return store.served_tenders(pid, since)

    @app.get("/portals/{pid}/runs")
    async def runs(pid: str):
        _get(pid)
        return [_run_entry(r) for r in store.list_runs(pid)]

    @app.get("/files/{sha256}")
    async def file(sha256: str):
        if not all(c in "0123456789abcdef" for c in sha256) or len(sha256) != 64 or not store.has_file(sha256):
            raise HTTPException(404, "unknown file")
        return FileResponse(store.file_path(sha256), media_type="application/octet-stream")

    # ---------------------------------------------------------------- dashboard
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        with open(DASHBOARD, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    card_counts: dict[str, tuple] = {}     # pid -> (fingerprint, computed at, tender counts, document counts)

    def _card_counts(pid: str, now) -> tuple[dict, dict]:
        """Open/withdrawn and document counts for one card. Parsing every tender is the expensive part, so it is
        redone only when the portal's tenders changed (fingerprint) or after 30 s (deadlines pass)."""
        from .store import derive_status
        fp = store.tender_fingerprint(pid)
        hit = card_counts.get(pid)
        if hit and hit[0] == fp and time.monotonic() - hit[1] < 30:
            return hit[2], hit[3]
        tender_counts = {"open": 0, "withdrawn": 0}
        documents = {"downloaded": 0, "pending": 0, "pendingLists": 0}
        for entry in store.get_tenders(pid).values():
            tender = entry["data"]
            status = derive_status(tender, now)
            if status not in tender_counts:
                continue
            tender_counts[status] += 1
            documents["downloaded"] += len(tender.get("documents") or [])
            documents["pending"] += len(set(tender.get("_docUrls") or []) - set(tender.get("_docDone") or []))
            documents["pendingLists"] += bool(tender.get("_docListPending"))
        card_counts[pid] = (fp, time.monotonic(), tender_counts, documents)
        return tender_counts, documents

    def _cards() -> list[dict]:
        now = timeutil.now()
        cards = []
        for p in store.list_portals():
            tender_counts, documents = _card_counts(p["id"], now)
            last = store.latest_run(p["id"])
            cards.append({
                **_portal_entry(store, p),
                "tenders": dict(tender_counts),
                "documents": dict(documents),
                "coverage": _coverage(store, p["id"], None, documents),
                "lastRun": _run_entry(last) if last else None,
            })
        return cards

    @app.get("/_ui/portals")
    async def ui_portals():
        """Return counts and the latest run for each card in a single request."""
        return await asyncio.to_thread(_cards)

    @app.get("/_ui/portals/{pid}")
    async def ui_detail(pid: str):
        """Extra detail for the dashboard (kept off the contract endpoints)."""
        p = _get(pid)
        spec = store.current_spec(pid)
        runs = []
        for r in reversed(store.list_runs(pid)[-30:]):
            entry = _run_entry(r)
            entry.update({"summary": r["summary"], "requests": r["requests"], "llmUsd": round(r["llm_usd"], 4),
                          "detail": json.loads(r["detail"] or "{}")})
            runs.append(entry)
        sample = store.served_tenders(pid)[:5]
        return {"portal": _portal_entry(store, p), "specVersion": spec[0] if spec else None,
                "specKind": (spec[1].get("list", {}) or {}).get("kind") if spec else None,
                "llm": store.llm_totals(pid), "runs": runs, "sampleTenders": sample}

    @app.get("/_ui/events")
    async def ui_events(since: str | None = Query(default=None), portal: str | None = Query(default=None),
                        limit: int = Query(default=500, ge=1, le=5000)):
        """Timeline per portal: state changes, runs (kind, result, duration, requests, model calls, $) and spec
        promotions (with the selectors/paths that changed). `since` = ISO date-time: only later events."""
        from . import events
        after = None
        if since:
            try:
                after = timeutil.parse(since)
            except ValueError:
                raise HTTPException(400, "since must be an ISO date-time with offset")
        if portal:
            _get(portal)
        return events.build(store, after, portal, limit)

    @app.get("/_ui/heal", response_class=HTMLResponse)
    async def heal_page():
        with open(HEAL_PAGE, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    from . import demo
    app.state.demo = demo.install(app, settings, store, manager, limiter, _portal_entry)

    @app.get("/_ui/audit/{pid}")
    async def ui_audit(pid: str, request: Request):
        """The latest audit in full (every disagreement with both values and the tender URL) and the history.
        A browser opening this address (Accept: text/html) is sent to the readable page; scripts get the JSON."""
        _get(pid)
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse(f"/_ui/audit/{pid}/view", status_code=303)
        audits = store.list_audits(pid, 20)
        latest = audits[0] if audits else None
        return {"verification": verification_entry(latest), "running": pid in manager.audits,
                "latest": ({"id": latest["id"], "startedAt": latest["started_at"], "finishedAt": latest["finished_at"],
                            "specVersion": latest["spec_version"], "status": latest["status"], **latest["result"]}
                           if latest else None),
                "history": [{"id": a["id"], "finishedAt": a["finished_at"], "status": a["status"],
                             "sampled": a["result"].get("sampled", 0),
                             "fieldsChecked": a["result"].get("fieldsChecked", 0),
                             "fieldsMatched": a["result"].get("fieldsMatched", 0),
                             "accuracy": a["result"].get("accuracy"), "costUsd": a["result"].get("costUsd")}
                            for a in audits]}

    @app.post("/_ui/audit/{pid}", status_code=202)
    async def ui_audit_now(pid: str):
        """Run one audit now (dashboard button). Coalesces with a running audit."""
        _get(pid)
        if not store.tender_count(pid):
            raise HTTPException(409, "nothing is served yet for this portal")
        return {"auditId": manager.audit(pid)}

    # ---------------------------------------------------------------- cross-portal view (dashboard only)
    from .crossref import Cache as CrossrefCache
    crossref_cache = CrossrefCache(store, ttl=float(os.environ.get("WATCHTOWER_CROSSREF_TTL", "300")))

    @app.get("/_ui/crossref")
    async def ui_crossref(refresh: bool = Query(default=False)):
        """Tenders that several portals (or one portal twice) serve, with evidence and disagreements."""
        return await crossref_cache.get(force=refresh)

    @app.get("/_ui/crossref/view", response_class=HTMLResponse)
    async def ui_crossref_view():
        with open(CROSSREF_PAGE, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    from . import ui_tenders; ui_tenders.register(app, store)  # noqa: E702  /_ui/tenders, /_ui/tenders/view
    from . import ui_pages; ui_pages.register(app, store, crossref_cache)   # audit view, verifier page, totals

    # ---------------------------------------------------------------- shared page assets (nav bar, styles, PDFs)
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    @app.get("/_ui/static/{name}")
    async def ui_static(name: str):
        path = os.path.join(static_dir, os.path.basename(name))
        if name != os.path.basename(name) or not os.path.isfile(path):
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app
# Run with: uvicorn watchtower.api:create_app --factory --host 0.0.0.0 --port 8080
