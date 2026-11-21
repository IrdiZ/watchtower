"""Timeline of what happened per portal, for the dashboard and the heal page (GET /_ui/events).

Three kinds of event, merged and sorted by time:
- state: a change of portal state (with the tenders served at that moment)
- run:   a job (attach/read/repair): result, duration, requests, model calls and $
- spec:  a promoted spec version, with the selectors/paths that changed against the version before
"""
from __future__ import annotations

import json
from datetime import datetime

from . import timeutil

# Spec keys that are bookkeeping or evidence rather than "how to read the portal": left out of the diff.
_DIFF_SKIP = {"profile", "verification", "verifiedBlind", "totalReliable"}
MAX_CHANGES = 60


def _flatten(value, path: str, out: dict) -> None:
    if isinstance(value, dict):
        if not value and path:
            out[path] = "{}"
        for k, v in value.items():
            if not path and k in _DIFF_SKIP:
                continue
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        for i, v in enumerate(value):
            _flatten(v, f"{path}[{i}]", out)
    else:
        out[path] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def spec_diff(old: dict | None, new: dict) -> list[dict]:
    """Leaf-level differences: [{"path": "list.items", "old": "tr.tender-row", "new": "tr.ergebnis-zeile"}, ...]
    (old or new is None for a path that was added or removed)."""
    a: dict = {}
    b: dict = {}
    _flatten(old or {}, "", a)
    _flatten(new or {}, "", b)
    changes = []
    for path in sorted(set(a) | set(b)):
        if a.get(path) != b.get(path):
            changes.append({"path": path, "old": a.get(path), "new": b.get(path)})
    return changes[:MAX_CHANGES]


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return round((timeutil.parse(end) - timeutil.parse(start)).total_seconds(), 1)
    except ValueError:
        return None


def build(store, since: datetime | None = None, portal: str | None = None, limit: int = 500) -> dict:
    """Events after `since` (all when None), oldest first. A run shows up when it starts and again (same id) when it
    finishes, so a client polling with `since` can upsert by id."""
    pids = [portal] if portal else [p["id"] for p in store.list_portals()]
    events: list[dict] = []

    def after(at: str | None) -> bool:
        if since is None:
            return True
        if not at:
            return False
        try:
            return timeutil.parse(at) > since
        except ValueError:
            return False

    for row in store.state_changes(portal):
        if after(row["at"]):
            events.append({"id": f"state:{row['id']}", "type": "state", "at": row["at"], "portal": row["portal_id"],
                           "from": row["from_state"], "to": row["to_state"], "reason": row["reason"],
                           "tenders": row["tenders"]})

    specs = store.list_specs(portal)
    first_spec = {}
    for s in specs:
        first_spec.setdefault(s["portal_id"], s["created_at"])
    for pid in pids:
        llm = store.llm_by_run(pid)
        for r in store.list_runs(pid):
            # a read job that ran before the portal had any spec was the attach (the contract calls it "read")
            kind = "attach" if r["kind"] == "read" and (pid not in first_spec or r["started_at"] < first_spec[pid]) \
                else r["kind"]
            at = r["finished_at"] or r["started_at"]
            if not after(at):
                continue
            detail = json.loads(r["detail"] or "{}") if r["finished_at"] else {}
            stats = detail.get("stats") or {}
            calls = llm.get(r["id"], {"calls": 0, "usd": 0.0, "purposes": {}})
            events.append({
                "id": f"run:{r['id']}", "type": "run", "at": at, "portal": pid, "runId": r["id"], "kind": kind,
                "result": r["result"], "startedAt": r["started_at"], "finishedAt": r["finished_at"],
                "durationS": _seconds(r["started_at"], r["finished_at"]), "requests": r["requests"],
                "llmCalls": calls["calls"], "llmUsd": round(calls["usd"] or r["llm_usd"] or 0.0, 4),
                "llmPurposes": calls["purposes"], "summary": r["summary"],
                "failures": detail.get("failures") or [], "was": detail.get("was") or [],
                "attempts": detail.get("attempts") or [], "counts": detail.get("counts") or {},
                "tendersRead": stats.get("rows") if isinstance(stats.get("rows"), int) else None,
            })

    previous: dict[str, dict] = {}
    for s in specs:
        pid = s["portal_id"]
        before = previous.get(pid)
        previous[pid] = s
        if pid not in pids or not after(s["created_at"]):
            continue
        events.append({"id": f"spec:{pid}:{s['version']}", "type": "spec", "at": s["created_at"], "portal": pid,
                       "version": s["version"], "previous": before["version"] if before else None,
                       "sourceRun": s["source_run"],
                       "changes": spec_diff(before["spec"] if before else None, s["spec"]) if before else []})

    events.sort(key=lambda e: (e["at"], {"state": 1, "spec": 0, "run": 2}[e["type"]]))
    return {"now": timeutil.iso(timeutil.now()), "events": events[-limit:]}
