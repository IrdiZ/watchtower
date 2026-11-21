"""Browser traffic is paced at the receiving server: navigation, redirect hops and popups all arrive >= interval apart."""
import asyncio
import socket
import threading
import time
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from tests.test_browser_live import _chromium_ok
from watchtower import browser
from watchtower.http import HostRateLimiter, PoliteClient

ARRIVALS: list[tuple[float, str]] = []
app = FastAPI()


@app.middleware("http")
async def stamp(request: Request, call_next):
    ARRIVALS.append((time.monotonic(), request.url.path))
    return await call_next(request)


@app.get("/start")
def start():
    return HTMLResponse("""<html><body><h1>app</h1><script>
      window.open('/popup');
      fetch('/redir1').then(r => r.json()).then(d => { document.title = 'ok'; });
      </script></body></html>""")


@app.get("/redir1")
def redir1():
    return RedirectResponse("/redir2")


@app.get("/redir2")
def redir2():
    return RedirectResponse("/api/tenders")


@app.get("/api/tenders")
def tenders():
    return JSONResponse({"items": [{"id": i, "title": f"t{i}", "deadline": "2030-01-01", "buyer": "b"}
                                   for i in range(3)]})


@app.get("/popup")
def popup():
    return HTMLResponse("<html><body>popup</body></html>")


@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_browser_requests_are_paced_across_redirects_and_popups():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    ARRIVALS.clear()
    interval = 0.5
    try:
        async def go():
            client = PoliteClient(HostRateLimiter(interval), "test-agent", 10, budget=50)
            ctx = SimpleNamespace(client=client, note=lambda m: None)
            cap = await browser.capture(ctx, f"http://127.0.0.1:{port}/start", max_wait_s=40, quiet_s=4)
            await client.aclose()
            return cap, client.requests
        cap, n = asyncio.run(go())
    finally:
        server.should_exit = True
        th.join(timeout=5)
    paths = [p for _, p in ARRIVALS]
    assert cap["error"] is None, cap
    assert {"/start", "/popup", "/redir1", "/redir2", "/api/tenders"} <= set(paths), paths
    assert any(d.get("items") for _, d in cap["json"])  # the page received the redirected data
    stamps = [t for t, _ in ARRIVALS]
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert min(gaps) >= interval - 0.02, (min(gaps), paths)
    assert n == len(ARRIVALS)  # every arrival (including each redirect hop) was counted by the client


@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_browser_respects_request_budget():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    ARRIVALS.clear()
    try:
        async def go():
            client = PoliteClient(HostRateLimiter(0.05), "test-agent", 10, budget=2)
            ctx = SimpleNamespace(client=client, note=lambda m: None)
            await browser.capture(ctx, f"http://127.0.0.1:{port}/start", max_wait_s=10, quiet_s=2)
            await client.aclose()
        asyncio.run(go())
    finally:
        server.should_exit = True
        th.join(timeout=5)
    assert len(ARRIVALS) <= 2
