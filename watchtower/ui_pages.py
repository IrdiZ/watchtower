"""HTML pages and small read-only totals for the dashboard: the audit view, the verifier page, and two light
endpoints the dashboard's hero strip reads (model spend; how many tenders several portals carry).
Nothing here changes data or an existing endpoint."""
from __future__ import annotations

import os

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

_HERE = os.path.dirname(__file__)
AUDIT_PAGE = os.path.join(_HERE, "audit_view.html")
VERIFIER_PAGE = os.path.join(_HERE, "verifier.html")


def _page(path: str) -> HTMLResponse:
    with open(path, encoding="utf-8") as f:        # read per request: edits show without a restart
        return HTMLResponse(f.read())


def register(app, store, crossref_cache=None) -> None:
    @app.get("/_ui/audit/{pid}/view", response_class=HTMLResponse)
    async def ui_audit_view(pid: str):
        if store.get_portal(pid) is None:
            raise HTTPException(404, "unknown portal id")
        return _page(AUDIT_PAGE)

    @app.get("/_ui/verifier", response_class=HTMLResponse)
    async def ui_verifier():
        return _page(VERIFIER_PAGE)

    @app.get("/_ui/totals")
    async def ui_totals():
        """Model spend over every portal (all calls recorded)."""
        r = store._q("SELECT COUNT(*) AS calls, COALESCE(SUM(usd),0) AS usd,"
                     " SUM(CASE WHEN usd IS NULL THEN 1 ELSE 0 END) AS unpriced FROM llm_calls")[0]
        return {"llm": {"calls": r["calls"], "usd": round(r["usd"], 4), "unpricedCalls": r["unpriced"] or 0}}

    pending = {"task": None}

    @app.get("/_ui/totals/crossref")
    async def ui_totals_crossref():
        """The cross-portal summary without the groups (the full result is megabytes). Shares the crossref cache."""
        if crossref_cache is None:
            raise HTTPException(404, "not available")
        # Never wait here and never start a second computation: the comparison takes most of a minute of CPU.
        # Until the first result exists this answers {"computing": true} and the dashboard asks again later.
        res = getattr(crossref_cache, "_value", None)
        if res is None:
            import asyncio
            if pending["task"] is None or (pending["task"].done() and pending["task"].exception() is not None):
                pending["task"] = asyncio.create_task(crossref_cache.get())
            return {"computing": True, "computedAt": None, "crossPortalGroups": None}
        s = res.get("summary") or {}
        return {"computing": False, "computedAt": res.get("computedAt"),
                **{k: s.get(k) for k in ("records", "groups", "crossPortalGroups", "withinPortalGroups",
                                         "portalsInvolved", "crossPortalPortals", "conflicts")}}
