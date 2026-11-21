"""Continuous verification: re-measure, per portal, how many of the values we serve match the portal itself.

An audit samples K served tenders (stratified: newest, oldest, some with documents, the rest at random), opens each
one's source (the JSON detail when the spec reads one, else the tender page; pages that render only with JavaScript
go through the same bounded browser path as attach), lets the cheap model copy the values the source shows WITHOUT
seeing ours (blind_verify's prompt plus status and document count), and compares field by field:
title, buyer, publication date, deadline (to the minute, timezone-aware), status, and document count.

An audit only reports. It never changes served data, the spec or the portal state; the result is stored per audit
(store.audits) and summarised as the "verification" object of the contract portal entry.
"""
from __future__ import annotations

import json
import math
import random
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import timeutil
from .extract import is_no_value, notice_status, page_text
from .http import BudgetExceeded
from .llm import LLMUnavailable
from .onboard import (VERIFY_PROMPT, _date_problem, _needs_browser, _norm, _rendered_text, _shown_interpretations)
from .pipeline import AttachFailed
from .store import derive_status

FIELDS = ("title", "buyerName", "publicationDate", "submissionDeadline", "status", "documents")

AUDIT_PROMPT = VERIFY_PROMPT + (
    ", \"status\": \"withdrawn\" if the page says the tender was withdrawn, cancelled or ended early, \"closed\" if "
    "it says the tender is closed or awarded, or the notice reports a result (an award notice, an ex-post "
    "publication, a contract already awarded, prior information only), \"open\" if it still accepts offers, null if "
    "the page does not say, \"noticeType\": the kind of notice or procedure as the page names it (e.g. \"Contract "
    "notice\", \"Award notice\", \"Ex post\"), or null, "
    "\"documentCount\": how many tender documents the page lists for download (count the files; a single "
    "'download all' archive counts as 1), or null if the page lists none}")

_DOC_EXT = re.compile(r"\.(pdf|zip|docx?|xlsx?|odt|ods|rtf|txt|xml|dwg|7z|rar|gaeb|x8[0-9])(?:$|[?#])", re.I)
_DOC_WORD = re.compile(r"download|dokument|document|unterlagen|bijlage|telecharg|fichier|dce", re.I)


# ------------------------------------------------------------------ sampling
def _pub_key(t: dict) -> str:
    return str(t.get("publicationDate") or "")


def sample(tenders: list[dict], k: int, rng: random.Random | None = None) -> list[tuple[dict, str]]:
    """Up to k served tenders, stratified: (tender, why). A quarter newest and a quarter oldest by publication date,
    a quarter with documents, the rest at random; each tender at most once."""
    rng = rng or random.Random()
    pool = [t for t in tenders if t.get("id")]
    if k <= 0 or not pool:
        return []
    q = max(1, k // 4)
    dated = sorted((t for t in pool if t.get("publicationDate")), key=_pub_key)
    picked: list[tuple[dict, str]] = []
    seen: set[str] = set()

    def take(cands, why, n):
        for t in cands:
            if n <= 0 or len(picked) >= k:
                return
            if str(t["id"]) in seen:
                continue
            seen.add(str(t["id"]))
            picked.append((t, why))
            n -= 1

    take(reversed(dated), "newest", q)
    take(dated, "oldest", q)
    with_docs = [t for t in pool if t.get("documents")]
    rng.shuffle(with_docs)
    take(with_docs, "with documents", q)
    rest = list(pool)
    rng.shuffle(rest)
    take(rest, "random", k - len(picked))
    return picked


# ------------------------------------------------------------------ comparison
def _outcome(field: str, result: str, shown: Any = None, ours: Any = None, note: str = "") -> dict:
    out = {"field": field, "result": result, "page": shown, "ours": ours}
    if note:
        out["note"] = note
    return out


def _shown(v) -> bool:
    return v is not None and not isinstance(v, (dict, list)) and str(v).strip() != "" and not is_no_value(v)


def compare_text(field: str, shown: Any, ours: Any, others: list[str] | None = None) -> dict | None:
    """Title / buyer: equal after normalising case, spaces and punctuation; or one contains the other (a page may
    shorten a title or add the department to a buyer); or nearly equal (ratio >= 0.9). Any of `others` (other names
    the page gives for the buying body) also counts. None: the page shows no value."""
    if not _shown(shown):
        return None
    if not _shown(ours):
        return _outcome(field, "missing", shown, ours)
    g = _norm(ours)
    for s in [shown] + list(others or []):
        n = _norm(s)
        if not n or not g:
            continue
        short, long_ = sorted((n, g), key=len)
        if n == g or (len(short) >= 8 and short in long_) or SequenceMatcher(None, n, g).ratio() >= 0.9:
            return _outcome(field, "match", shown, ours)
    return _outcome(field, "mismatch", shown, ours)


_LOCAL_NOTE = re.compile(r"\((?=[^()]*\b(?:heure|hora|ora|time|zeit|tijd|czasu?)\b)[^()\d]*\)", re.I)


# abbreviated month names %b does not know ('13 mrt. 2017', '3 okt. 2026', '5 févr. 2027')
_ABBR = {"mrt": "Mar", "mär": "Mar", "mrz": "Mar", "okt": "Oct", "mei": "May", "dez": "Dec", "janv": "Jan",
         "févr": "Feb", "fevr": "Feb", "avr": "Apr", "juil": "Jul", "sept": "Sep", "déc": "Dec", "ene": "Jan",
         "abr": "Apr", "ago": "Aug", "dic": "Dec"}
_SLASH_MONTH_FIRST = ["%m/%d/%Y %H:%M", "%m/%d/%Y, %H:%M", "%m/%d/%Y %I:%M %p", "%m/%d/%Y, %I:%M %p", "%m/%d/%Y"]
_ABBR_RE = re.compile(r"\b(" + "|".join(map(re.escape, _ABBR)) + r")\b", re.I)


def _prep_date(shown: str) -> str:
    s = " ".join(str(shown).split())
    s = _LOCAL_NOTE.sub(" ", s).strip()       # '(heure de Paris)', '(Ortszeit Berlin)': the portal's own time zone
    s = re.sub(r"\b([^\W\d_]{3,5})\.(?=[\s,]|$)", r"\1", s)        # 'nov. 2026' -> 'nov 2026'
    s = _ABBR_RE.sub(lambda m: _ABBR[m.group(1).lower()], s)
    s = re.sub(r"(?<=\d{4})\s+-\s+(?=\d{1,2}[:.]\d{2})", " ", s)   # '24.11.2026 - 08:59'
    return re.sub(r"(?<=\d)\s*h\s*(?=\d{2}\b)", ":", s)   # '12h00' (French) -> '12:00'


def compare_date(field: str, shown: Any, ours: Any, tz: str, formats: list[str] | None = None) -> dict | None:
    """publicationDate as a calendar date; submissionDeadline as an instant to the minute (an explicit offset on the
    page counts, else the portal's timezone). A page that shows the deadline without a time is compared by date.
    A page value no format can read is 'unreadable' (not counted)."""
    if not _shown(shown):
        return None
    if not _shown(ours):
        return _outcome(field, "missing", shown, ours)
    s = _prep_date(shown)
    # a slash date may be month-first (English-language pages, e.g. '12/04/2026' = 4 December): both readings count
    formats = list(formats or []) + (_SLASH_MONTH_FIRST if "/" in s else [])
    interps = _shown_interpretations(s, formats)
    if not interps:
        return _outcome(field, "unreadable", shown, ours, "the page's date format could not be read")
    if field == "submissionDeadline" and not any(h for _, h in interps):
        try:
            zone = ZoneInfo(tz or "UTC")
        except (ZoneInfoNotFoundError, ValueError):
            zone = ZoneInfo("UTC")
        try:
            got = datetime.fromisoformat(str(ours).replace("Z", "+00:00"))
        except ValueError:
            return _outcome(field, "mismatch", shown, ours, "ours is not an ISO date-time")
        got_d = (got.astimezone(zone) if got.tzinfo else got).date()
        ok = any(dt.date() == got_d for dt, _ in interps)
        return _outcome(field, "match" if ok else "mismatch", shown, ours, "the page shows the date only")
    problem = _date_problem(field, s, ours, tz or "UTC", list(formats or []))
    if problem is None:
        return _outcome(field, "match", shown, ours)
    return _outcome(field, "mismatch", shown, ours, "wrong offset" if "offset mismatch" in problem else "")


def page_status(truth: dict, tz: str, formats: list[str] | None = None, now: datetime | None = None) -> str | None:
    """The status the source shows: the model's reading; else the notice type it copied (an award, result or
    cancellation notice is not open); else a deadline on the page that has passed (closed). None: not shown."""
    s = str(truth.get("status") or "").strip().lower()
    if s in ("open", "withdrawn", "closed"):
        nk = notice_status(truth.get("noticeType"))
        return nk if (s == "open" and nk) else s
    nk = notice_status(truth.get("noticeType"))
    if nk:
        return nk
    shown = truth.get("submissionDeadline")
    if not _shown(shown):
        return None
    p = _prep_date(shown)
    interps = _shown_interpretations(p, list(formats or []) + (_SLASH_MONTH_FIRST if "/" in p else []))
    if not interps:
        return None
    try:
        zone = ZoneInfo(tz or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    at = now or timeutil.now()
    # every reading of the page's deadline is in the past (a date-only deadline counts until the end of that day)
    past = all(((dt if dt.tzinfo else dt.replace(tzinfo=zone)) if has_time
                else (dt if dt.tzinfo else dt.replace(tzinfo=zone)).replace(hour=23, minute=59)) <= at
               for dt, has_time in interps)
    return "closed" if past else None


def compare_status(shown: Any, ours: str) -> dict | None:
    s = str(shown or "").strip().lower()
    if s not in ("open", "withdrawn", "closed"):
        return None
    return _outcome("status", "match" if s == ours else "mismatch", s, ours)


def compare_documents(shown: Any, ours: int, pending: bool) -> dict | None:
    """The page's document count against the files we serve. We may serve more (a ZIP is unpacked into its files),
    never fewer; while our downloads for that tender are still pending, a shortfall is not counted."""
    try:
        n = int(shown)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if ours >= n:
        return _outcome("documents", "match", n, ours)
    if pending:
        return _outcome("documents", "pending", n, ours, "our downloads for this tender are still in progress")
    return _outcome("documents", "missing" if ours == 0 else "mismatch", n, ours)


def compare_tender(truth: dict, t: dict, tz: str, spec: dict | None = None, now: datetime | None = None,
                   docs_pending: bool = False) -> list[dict]:
    """Every field the source shows, compared with the tender as we serve it."""
    fields = (spec or {}).get("fields") or {}
    others = [o for o in (truth.get("otherBuyerNames") or []) if isinstance(o, str) and o.strip()] \
        if isinstance(truth.get("otherBuyerNames"), list) else []
    out = [compare_text("title", truth.get("title"), t.get("title")),
           compare_text("buyerName", truth.get("buyerName"), t.get("buyerName"), others)]
    for f in ("publicationDate", "submissionDeadline"):
        fmts = list((fields.get(f) or {}).get("formats") or [])
        out.append(compare_date(f, truth.get(f), t.get(f), tz, fmts))
    dl_formats = list((fields.get("submissionDeadline") or {}).get("formats") or [])
    out.append(compare_status(page_status(truth, tz, dl_formats, now), derive_status(t, now or timeutil.now())))
    out.append(compare_documents(truth.get("documentCount"), len(t.get("documents") or []), docs_pending))
    return [o for o in out if o is not None]


COUNTED = ("match", "mismatch")          # delivered values the source confirmed or contradicted


def wilson_low(matched: int, n: int, z: float = 1.96) -> float | None:
    """Lower bound of the 95% Wilson interval: with few fields checked, 100% means 'at least this much'."""
    if n <= 0:
        return None
    p = matched / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - r) / d, 3)


def summarise(outcomes: list[tuple[dict, str, dict]], degraded_below: float = 0.9) -> dict:
    """outcomes: (tender, url, outcome). Accuracy = matched / checked, where checked are the values we deliver AND the
    source shows (match or mismatch). Values the source shows but we do not serve are counted as 'missing' (a
    coverage gap, not a wrong value); unreadable or pending ones are listed but not counted."""
    by: dict[str, dict] = {}
    mismatches = []
    for t, url, o in outcomes:
        b = by.setdefault(o["field"], {"checked": 0, "matched": 0, "missing": 0, "notCounted": 0})
        if o["result"] in COUNTED:
            b["checked"] += 1
            b["matched"] += o["result"] == "match"
        elif o["result"] == "missing":
            b["missing"] += 1
        else:
            b["notCounted"] += 1
        if o["result"] != "match":
            mismatches.append({"tenderId": t.get("id"), "url": url, **o})
    checked = sum(b["checked"] for b in by.values())
    matched = sum(b["matched"] for b in by.values())
    for b in by.values():
        b["accuracy"] = round(b["matched"] / b["checked"], 3) if b["checked"] else None
    acc = round(matched / checked, 3) if checked else None
    return {"fieldsChecked": checked, "fieldsMatched": matched, "accuracy": acc,
            "accuracyLow95": wilson_low(matched, checked), "byField": by, "mismatches": mismatches,
            "status": ("ok" if acc is not None and acc >= degraded_below else "degraded") if checked else None}


# ------------------------------------------------------------------ reading the source
def _doc_links(html: str, limit: int = 60) -> list[str]:
    """Names of the download links on a page (page_text drops links; the model needs them to count documents)."""
    out = []
    try:
        tree = HTMLParser(html or "")
    except Exception:
        return out
    for a in tree.css("a[href]"):
        href = a.attributes.get("href") or ""
        txt = " ".join((a.text() or "").split())
        if _DOC_EXT.search(href) or (_DOC_WORD.search(href) and txt):
            out.append((txt or href)[:120])
            if len(out) >= limit:
                break
    return out


def source_url(spec: dict, t: dict) -> str | None:
    det = (spec or {}).get("detail") or {}
    if det.get("url_from") == "template" and det.get("template") and t.get("id"):
        return t.get("_detailUrl") or det["template"].replace("{id}", str(t["id"]))
    return t.get("url")


async def read_source(ctx, spec: dict, t: dict, renders: dict) -> tuple[str | None, str, str | None]:
    """(text, source-or-why-not, url). JSON detail when the spec reads one, else the tender page (rendered in a
    browser when it is a JavaScript shell, at most renders['left'] times per audit)."""
    url = source_url(spec, t)
    if not url:
        return None, "the tender has no source address", None
    if renders.get("shellSeen") and renders.get("left", 0) <= 0 and url not in renders.get("pages", {}):
        # the portal's tender pages are JavaScript shells and no render is left: don't fetch more of them
        return None, "the page renders only with JavaScript (no browser render left)", url
    resp = await ctx.client.get(url)
    if resp.status_code >= 400:
        return None, f"HTTP {resp.status_code}", url
    if ((spec or {}).get("detail") or {}).get("kind") == "json":
        try:
            return json.dumps(resp.json(), ensure_ascii=False)[:8000], "json detail", url
        except ValueError:
            return None, "the JSON detail did not return JSON", url
    html = resp.text or ""
    if "json" in (resp.headers.get("content-type") or ""):
        try:
            return json.dumps(resp.json(), ensure_ascii=False)[:8000], "json detail", url
        except ValueError:
            pass
    if _needs_browser(html):
        renders["shellSeen"] = True
        text, why = await _rendered_text(ctx, url, t, renders)
        if text is None:
            return None, f"the page renders only with JavaScript ({why})", url
        return text, why, url
    text = page_text(html, 12000)
    links = _doc_links(html)
    if links:
        text += "\n\nDownload links on this page:\n" + "\n".join(f"- {x}" for x in links)
    if t.get("_followUrl"):
        # values the spec reads from the page this one links to (detail.follow) are shown there
        from .onboard import linked_page_text
        from . import runner
        if runner.follow_spec(spec or {}):
            text += await linked_page_text(ctx, spec, t["_followUrl"])
    return text, "page", url


def _shows_anything(truth: dict) -> bool:
    return any(_shown(truth.get(f)) for f in ("id", "title", "buyerName", "publicationDate", "submissionDeadline"))


def _not_possible(skipped: list[dict]) -> str:
    if not skipped:
        return "not possible: no served tenders to sample"
    whys = [s["why"] for s in skipped]
    top = max(set(whys), key=whys.count)
    return f"not possible: {top} ({whys.count(top)} of {len(skipped)} sampled tender sources)"


async def audit_portal(ctx, spec: dict, tenders: list[dict], k: int = 8, renders: int = 2,
                       rng: random.Random | None = None, degraded_below: float = 0.9,
                       pending_ids: set[str] | None = None) -> dict:
    """One audit of the served `tenders` (as served, plus `_` bookkeeping keys if available). Read-only: the result
    is returned, nothing is written. Stops early when the request or model budget runs out (what was compared
    still counts)."""
    started = timeutil.now()
    now = started
    tz = (spec or {}).get("tz") or "UTC"
    picked = sample([t for t in tenders if derive_status(t, now) in ("open", "withdrawn")], k, rng)
    budget = {"left": max(0, int(renders))}
    outcomes: list[tuple[dict, str, dict]] = []
    skipped: list[dict] = []
    checked_tenders = []
    stopped = None
    for t, why in picked:
        try:
            text, source, url = await read_source(ctx, spec, t, budget)
            if text is None:
                skipped.append({"tenderId": t.get("id"), "url": url, "why": source})
                continue
            truth = await ctx.llm.json_call("audit", AUDIT_PROMPT, text, portal_id=ctx.portal_id,
                                            run_id=ctx.run_id)
        except BudgetExceeded as e:
            stopped = f"request budget used up: {e}"
            break
        except (LLMUnavailable, AttachFailed) as e:     # incl. the job's model-call budget
            stopped = f"model unavailable: {e}"
            break
        except Exception as e:                          # a network error on one source: skip that tender
            skipped.append({"tenderId": t.get("id"), "url": source_url(spec, t), "why": f"{type(e).__name__}"})
            continue
        if not isinstance(truth, dict) or not _shows_anything(truth):
            skipped.append({"tenderId": t.get("id"), "url": url, "why": "the source showed no tender values"})
            continue
        pend = bool(t.get("_docListPending")) or bool(set(t.get("_docUrls") or []) - set(t.get("_docDone") or [])) \
            or str(t.get("id")) in (pending_ids or set())
        res = compare_tender(truth, t, tz, spec, now, docs_pending=pend)
        outcomes.extend((t, url, o) for o in res)
        checked_tenders.append({"tenderId": t.get("id"), "url": url, "why": why, "source": source,
                                "fields": {o["field"]: o["result"] for o in res}})
    s = summarise(outcomes, degraded_below)
    if not checked_tenders:
        s["status"] = _not_possible(skipped) if not stopped else f"not possible: {stopped}"
    elif s["status"] is None:
        s["status"] = "not possible: the sampled sources show none of the audited fields"
    return {"startedAt": timeutil.iso(started), "finishedAt": timeutil.iso(timeutil.now()), "requested": k,
            "sampled": len(checked_tenders), "tenders": checked_tenders, "skipped": skipped, "stopped": stopped,
            "requests": getattr(ctx.client, "requests", 0), "tz": tz, **s}


def verification_entry(audit: dict | None) -> dict:
    """The contract's per-portal 'verification' object (Patterno allows added fields), from the latest audit."""
    if not audit:
        return {"lastAuditAt": None, "sampled": 0, "fieldsChecked": 0, "fieldsMatched": 0, "accuracy": None,
                "byField": {}, "status": "pending"}
    r = audit.get("result") or {}
    return {"lastAuditAt": audit.get("finished_at") or r.get("finishedAt"), "sampled": r.get("sampled", 0),
            "fieldsChecked": r.get("fieldsChecked", 0), "fieldsMatched": r.get("fieldsMatched", 0),
            "accuracy": r.get("accuracy"), "accuracyLow95": r.get("accuracyLow95"),
            "byField": {f: {"checked": b.get("checked", 0), "matched": b.get("matched", 0),
                            "missing": b.get("missing", 0)} for f, b in (r.get("byField") or {}).items()},
            "mismatches": sum(1 for m in r.get("mismatches") or [] if m.get("result") == "mismatch"),
            "costUsd": r.get("costUsd"), "status": audit.get("status") or r.get("status") or "pending"}


# ------------------------------------------------------------------ live check against a running API (read-only)
_TLD_TZ = {"nl": "Europe/Amsterdam", "de": "Europe/Berlin", "fr": "Europe/Paris", "es": "Europe/Madrid",
           "pl": "Europe/Warsaw", "ch": "Europe/Zurich", "ie": "Europe/Dublin", "uk": "Europe/London",
           "at": "Europe/Vienna", "be": "Europe/Brussels", "it": "Europe/Rome"}


async def _live(api: str, pids: list[str], k: int, renders: int, seed: int | None) -> list[dict]:
    """Audit portals of a running Watchtower through its public API only (GET /portals, GET /portals/{id}/tenders),
    against the tender pages as served in `url`. No spec is needed: what a person would open is what is compared."""
    import tempfile
    from urllib.parse import urlsplit

    import httpx

    from .config import Settings
    from .http import HostRateLimiter, PoliteClient, parse_host_intervals
    from .llm import LLMRouter
    from .pipeline import JobContext
    from .store import Store, new_run_id

    s = Settings(data_dir=tempfile.mkdtemp(prefix="wt-audit-"))
    store = Store(s.data_dir)
    llm = LLMRouter(s, store)
    limiter = HostRateLimiter(s.min_request_interval, parse_host_intervals(s.host_intervals))
    out = []
    async with httpx.AsyncClient(timeout=120) as api_client:
        portals = {p["id"]: p for p in (await api_client.get(f"{api}/portals")).json()}
        for pid in pids:
            p = portals.get(pid)
            if p is None:
                out.append({"portal": pid, "status": "not possible: unknown portal"})
                continue
            tenders = (await api_client.get(f"{api}/portals/{pid}/tenders")).json()
            host = urlsplit(p["url"]).hostname or ""
            tz = _TLD_TZ.get(host.rsplit(".", 1)[-1], "UTC")
            rid = new_run_id().replace("r_", "a_")
            client = PoliteClient(limiter, s.user_agent, s.http_timeout, s.audit_request_budget,
                                  respect_robots=s.respect_robots)
            ctx = JobContext(pid, p["url"], rid, client, None, llm, store)
            try:
                res = await audit_portal(ctx, {"tz": tz}, tenders, k=k, renders=renders,
                                         rng=random.Random(seed), degraded_below=s.audit_degraded_below)
            finally:
                await client.aclose()
            usd = store._q("SELECT COALESCE(SUM(usd),0) AS s, COUNT(*) AS n FROM llm_calls WHERE run_id=?", (rid,))[0]
            res.update({"portal": pid, "url": p["url"], "costUsd": round(usd["s"], 5), "modelCalls": usd["n"],
                        "log": ctx.log})
            out.append(res)
    await llm.aclose()
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Audit served tenders of a running Watchtower (read-only).")
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--renders", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the full results as JSON to this file")
    ap.add_argument("portals", nargs="+")
    a = ap.parse_args(argv)
    res = asyncio.run(_live(a.api.rstrip("/"), a.portals, a.k, a.renders, a.seed))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    for r in res:
        print(json.dumps({k: r.get(k) for k in ("portal", "url", "status", "sampled", "fieldsChecked",
                                                "fieldsMatched", "accuracy", "accuracyLow95", "costUsd",
                                                "modelCalls", "requests")}, ensure_ascii=False))
        for f, b in (r.get("byField") or {}).items():
            print(f"   {f}: {b['matched']}/{b['checked']} matched, {b['missing']} missing, {b['notCounted']} not counted")
        for m in (r.get("mismatches") or [])[:12]:
            print(f"   - {m['result']} {m['field']} {m['tenderId']}: page {str(m.get('page'))[:70]!r} "
                  f"ours {str(m.get('ours'))[:70]!r} {m.get('note', '')} {m.get('url')}")
        for sk in (r.get("skipped") or [])[:5]:
            print(f"   skipped {sk['tenderId']}: {sk['why']}")


if __name__ == "__main__":
    main()
