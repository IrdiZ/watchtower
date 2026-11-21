"""Demo layer: the mock portal served by Watchtower itself (under /_demo/portal/) plus the heal-page controls.

Only active with WATCHTOWER_DEMO=1 (the default) and when the `mock_portal` package is present. Nothing here is used
by the engine: the demo portal is attached through the normal contract like any other portal, at its real URL
(WATCHTOWER_DEMO_URL, default http://localhost:8080/_demo/portal/), and read over HTTP by the worker.

Controls (never on contract endpoints):
  GET  /_ui/heal/demo     the demo portal's address, id, state and current mutation
  POST /_ui/heal/attach   attach the demo portal (POST /portals with its URL)
  POST /_ui/heal/break    {"mutation": "...", "refresh": true}: change the portal, then read it
  POST /_ui/heal/restore  {"refresh": false}: the portal as it was
  POST /_ui/heal/refresh  {"portal": id?}: read now (default: the demo portal)
"""
from __future__ import annotations

import logging
import os
import re

from fastapi import Body, FastAPI, HTTPException

log = logging.getLogger("watchtower.demo")

PREFIX = "/_demo/portal"

# What the heal page offers, in the order shown: (mutation, button label, what changes on the portal)
CHOICES = [
    ("rename_classes", "Rename classes", "The list's CSS class names change, as in a redesign."),
    ("swap_columns", "Swap columns", "Title and buyer swap places: same shape, wrong data."),
    ("restructure", "New layout", "The results table becomes a list of cards."),
    ("pagination", "New pager", "?page=N becomes ?p=N, with 4 tenders per page."),
    ("overlay", "Pop-up in front", "A notice dialog with its own table sits in front of the list (cosmetic: the "
                                   "reader should not need a repair)."),
    # Not offered: extra_table (awarded contracts in the list's own row markup, before the list). It is accepted
    # today as wrong data (the read stops at the stated total counting foreign rows, closes real tenders and serves
    # the awarded ones as open): an engine gap to fix first.
    ("date_format", "Date format", "Dates are written out in words."),
    ("login_wall", "Login wall", "The list now asks for a login."),
]

# Root-relative addresses in the mock's pages, scripts and JSON: attributes (href='/...', also href='/'), and quoted
# strings starting with "/<letter>" (fetch('/api/...'), "href": "/download/..."). The prefix starts with "/_", so a
# rewritten address is never rewritten twice.
_ATTR = re.compile(r"""(\b(?:href|src|action)=)(['"])/(?!/)""")
_QUOTED = re.compile(r"""(?<![=\w])(['"`])/(?=[A-Za-z])""")
_TEXTUAL = ("text/", "application/javascript", "application/json", "application/xml")


def rewrite(text: str, prefix: str = PREFIX) -> str:
    text = _ATTR.sub(lambda mt: f"{mt.group(1)}{mt.group(2)}{prefix}/", text)
    return _QUOTED.sub(lambda mt: f"{mt.group(1)}{prefix}/", text)


class Prefixed:
    """ASGI wrapper: serves an app written for the site root under a path prefix (links, redirects, scripts)."""

    def __init__(self, app, prefix: str = PREFIX):
        self.app, self.prefix = app, prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start: dict = {}
        chunks: list[bytes] = []

        async def wrapped(message):
            if message["type"] == "http.response.start":
                start.update(message)
                return
            if message["type"] != "http.response.body":
                return await send(message)
            chunks.append(message.get("body", b""))
            if message.get("more_body"):
                return
            body = b"".join(chunks)
            headers = []
            ctype = ""
            for k, v in start.get("headers", []):
                name = k.decode("latin-1").lower()
                if name == "content-type":
                    ctype = v.decode("latin-1").lower()
                if name == "location" and v.startswith(b"/") and not v.startswith(b"//"):
                    v = self.prefix.encode() + v
                if name != "content-length":
                    headers.append((k, v))
            if ctype.startswith(_TEXTUAL):
                body = rewrite(body.decode("utf-8", "replace"), self.prefix).encode("utf-8")
            headers.append((b"content-length", str(len(body)).encode()))
            await send({"type": "http.response.start", "status": start.get("status", 200), "headers": headers})
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, wrapped)


def _mock():
    try:
        from mock_portal import app as mock
        return mock
    except ImportError:
        return None


def install(app: FastAPI, settings, store, manager, limiter, portal_entry) -> bool:
    """Mount the demo portal and its controls. Returns False (nothing mounted) when the demo is off or the mock
    portal package is not available."""
    enabled = os.environ.get("WATCHTOWER_DEMO", "1") == "1"
    mock = _mock() if enabled else None
    url = os.environ.get("WATCHTOWER_DEMO_URL", "http://localhost:8080" + PREFIX + "/")
    if mock is not None:
        app.mount(PREFIX, Prefixed(mock.app, PREFIX))
        # The demo portal lives on this machine (loopback): it may be read faster than a real portal.
        try:
            from urllib.parse import urlsplit
            host = (urlsplit(url).hostname or "").lower()
            limiter.host_intervals.setdefault(host, float(os.environ.get("WATCHTOWER_DEMO_INTERVAL", "0.25")))
        except ValueError:
            pass
    elif enabled:
        log.info("demo portal not mounted: the mock_portal package is not available")

    def demo_pid() -> str | None:
        from .urls import normalize_url
        norm = normalize_url(url)
        return next((p["id"] for p in store.list_portals() if p["norm_url"] == norm), None)

    def need_mock():
        if mock is None:
            raise HTTPException(404, "the demo portal is off (WATCHTOWER_DEMO=1 and the mock_portal package)")
        return mock

    def refresh(pid: str | None) -> str | None:
        if pid is None or store.get_portal(pid) is None:
            return None
        return manager.refresh(pid)

    @app.get("/_ui/heal/demo")
    async def heal_demo():
        pid = demo_pid() if mock is not None else None
        p = store.get_portal(pid) if pid else None
        return {"enabled": mock is not None, "url": url if mock is not None else None, "path": PREFIX + "/",
                "portalId": pid, "portal": portal_entry(store, p) if p else None,
                "mutations": sorted(mock.CONFIG["mutations"]) if mock is not None else [],
                "choices": [{"mutation": k, "label": lab, "what": w} for k, lab, w in CHOICES]}

    @app.post("/_ui/heal/attach")
    async def heal_attach():
        need_mock()
        pid, created = manager.add(url)
        return {"id": pid, "created": created}

    @app.post("/_ui/heal/break")
    async def heal_break(body: dict = Body(default={})):
        m = need_mock()
        mutation = str(body.get("mutation") or "")
        if mutation not in {k for k, _, _ in CHOICES}:
            raise HTTPException(400, f"unknown mutation; one of {[k for k, _, _ in CHOICES]}")
        m.CONFIG["skin"] = "html"
        m.CONFIG["mutations"] = {mutation}
        run = refresh(demo_pid()) if body.get("refresh", True) else None
        return {"mutations": sorted(m.CONFIG["mutations"]), "runId": run}

    @app.post("/_ui/heal/restore")
    async def heal_restore(body: dict = Body(default={})):
        m = need_mock()
        m.CONFIG["skin"] = "html"
        m.CONFIG["mutations"] = set()
        run = refresh(demo_pid()) if body.get("refresh", False) else None
        return {"mutations": [], "runId": run}

    @app.post("/_ui/heal/refresh", status_code=202)
    async def heal_refresh(body: dict = Body(default={})):
        pid = body.get("portal") or demo_pid()
        run = refresh(pid)
        if run is None:
            raise HTTPException(404, "unknown portal id")
        return {"runId": run, "portal": pid}

    return mock is not None
