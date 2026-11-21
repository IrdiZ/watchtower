"""Locate walk, harder ways in: a list kept in another application of the site (a central portal whose tenders
forward into the platform that runs them), JavaScript app frames, and search pages whose results appear only after
the search control is pressed (browser, skipped without Chromium)."""
import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from tests.fake_llm import FakeLLM
from watchtower import browser, locate
from watchtower.config import Settings
from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext
from watchtower.store import Store

HOST = "http://portal.test"


def _rows(prefix: str, n: int, cls: str = "") -> str:
    return "".join(f"<tr class='{cls}'><td><a href='{prefix}{i}'>Lieferung {i}</a></td><td>0{i % 9 + 1}.12.2026</td>"
                   f"<td>1{i % 9}.11.2026</td></tr>" for i in range(1, n + 1))


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(f"<html><head><script src='/static/app.js'></script></head><body>{body}</body></html>")


def _two_apps() -> FastAPI:
    """'/' -> /Center/company/welcome.do: the latest 5 tenders (an excerpt) whose links forward into /Sat/; 'all
    notices' there is an empty search page. The full list lives at /Sat/company/welcome.do, linked from nowhere."""
    app = FastAPI()
    head = "<thead><tr><th>Titel</th><th>Angebotsfrist</th><th>Veröffentlicht</th></tr></thead>"

    @app.get("/")
    def root():
        return RedirectResponse("/Center/company/welcome.do", 302)

    @app.get("/Center/company/welcome.do")
    def center():
        return _page(f"<h1>Neueste Bekanntmachungen</h1><table>{head}<tbody>{_rows('/Center/fwd.do?pid=', 5)}"
                     f"</tbody></table><a href='/Center/company/search.do'>Suche</a>")

    @app.get("/Center/company/search.do")
    def center_search():
        return _page("<h1>Suche</h1><p>Bitte Suchkriterien eingeben.</p>")

    @app.get("/Center/fwd.do")
    def forward(pid: int):
        return RedirectResponse(f"/Sat/public/project/{pid}/overview", 302)

    @app.get("/Sat/public/project/{pid}/overview")
    def tender(pid: int):
        return _page(f"<h1>Lieferung {pid}</h1><p>Angebotsfrist 01.12.2026</p>")

    @app.get("/Sat/company/welcome.do")
    def satellite():
        return _page(f"<h1>Bekanntmachungen</h1><table>{head}<tbody>{_rows('/Sat/public/project/', 20, 'tender-row')}"
                     f"</tbody></table><a href='/Sat/company/welcome.do?page=2'>2</a>")
    return app


def _ctx(tmp_path, app, start):
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    store = Store(str(tmp_path))
    client = PoliteClient(HostRateLimiter(0), "t", 10, budget=200, transport=httpx.ASGITransport(app=app))
    return JobContext("p1", start, "r1", client, DocumentStore(store, 10 ** 8, 10 ** 9), FakeLLM(store), store,
                      mode="attach")


def _walk(tmp_path, app, start):
    ctx = _ctx(tmp_path, app, start)

    async def go():
        try:
            return await locate.locate_list(ctx, start)
        finally:
            await ctx.client.aclose()
    return ctx, asyncio.run(go())


def test_context_helpers():
    assert locate._context("https://p.de/VMPCenter/company/welcome.do") == "VMPCenter"
    assert locate._context("https://p.de/") is None and locate._context("https://p.de/index.html") is None
    assert locate._in_context("https://p.de/A/company/welcome.do?x=1", "B") == "https://p.de/B/company/welcome.do?x=1"
    assert locate._in_context("https://p.de/A", "B") is None          # an application root alone is no page
    assert locate._app_sig("<script src='/a.js?v=1'></script><script src='/b.js'></script>") == ("/a.js", "/b.js")
    assert locate._app_sig("<p>no scripts</p>") is None


def test_list_in_the_application_the_tenders_forward_into(tmp_path):
    ctx, found = _walk(tmp_path, _two_apps(), HOST + "/")
    assert found["url"] == HOST + "/Sat/company/welcome.do" and found["tier"] == "A"
    vias = [r["via"] for r in found["walk"].trace]
    assert "item" in vias and "context" in vias
    assert any("forwards from the application /Center/ into /Sat/" in l for l in ctx.log)


def test_list_in_the_given_addresses_own_application(tmp_path):
    """The given address (/Sat) answers with a buyers' login; the list found under /Center is tried under /Sat."""
    app = _two_apps()

    @app.get("/Sat")
    def sat_root():
        return RedirectResponse("/Sat/mandator/login.do", 302)

    @app.get("/Sat/mandator/login.do")
    def login():
        return _page("<h1>Login Vergabestellen</h1><form method=post><input name=u>"
                     "<input type=password name=p></form>")
    _, found = _walk(tmp_path, app, HOST + "/Sat")
    assert found["url"] == HOST + "/Sat/company/welcome.do" and found["tier"] == "A"
    assert found["walk"].used["pages"] <= 5


# ------------------------------------------------------------------ browser: search press, booting apps
def _chromium_ok():
    try:
        from playwright.async_api import async_playwright

        async def m():
            async with async_playwright() as p:
                b = await p.chromium.launch(args=["--no-sandbox"])
                await b.close()
        asyncio.run(m())
        return True
    except Exception:
        return False


SEARCH_APP = """<html><body><div id=app></div><script>
setTimeout(function () {
  document.getElementById('app').innerHTML =
    '<h1>Erweiterte Suche</h1><p>Finden Sie öffentliche Ausschreibungen nach Stichwort, Ort und Frist.</p>' +
    '<form onsubmit="return false"><input name=q>' +
    '<button type=button id=login aria-label="Login">Login</button>' +
    '<button type=button id=reset aria-label="Reset selection">x</button>' +
    '<button type=button id=go aria-label="Apply filter"><i class="mdi mdi-magnify"></i></button></form>';
  document.getElementById('login').onclick = function () { fetch('/api/login'); };
  document.getElementById('reset').onclick = function () { fetch('/api/reset'); };
  document.getElementById('go').onclick = function () {
    fetch('/api/search?q=').then(function (r) { return r.json(); }).then(function (d) {
      history.pushState({}, '', '/search?q=');
      document.getElementById('app').innerHTML += d.items.map(function (t) { return '<p>' + t.title + '</p>'; }).join('');
    });
  };
}, BOOT_MS);
</script></body></html>"""


def _serve(boot_ms: int):
    app = FastAPI()
    hits: list[str] = []

    @app.get("/")
    def home():
        return HTMLResponse(SEARCH_APP.replace("BOOT_MS", str(boot_ms)))

    @app.get("/api/{what}")
    def api(what: str):
        hits.append(what)
        if what != "search":
            return JSONResponse({"ok": True})
        return JSONResponse({"items": [{"id": i, "title": f"Tender {i}", "deadline": "2026-12-01",
                                        "publicationDate": "2026-11-01"} for i in range(5)]})
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}", hits


def _capture(tmp_path, base, **kw):
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    store = Store(str(tmp_path))
    client = PoliteClient(HostRateLimiter(0.1), "t", 10, budget=100)
    ctx = JobContext("p1", base, "r1", client, DocumentStore(store, 10 ** 8, 10 ** 9), None, store, mode="attach")

    async def go():
        try:
            return await browser.capture(ctx, base + "/", max_wait_s=40, **kw)
        finally:
            await client.aclose()
    return asyncio.run(go())


@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_browser_presses_the_search_control_not_login_or_reset(tmp_path):
    server, base, hits = _serve(300)
    try:
        cap = _capture(tmp_path, base, submit_search=True)
    finally:
        server.should_exit = True
    assert cap["submitted"] == "Apply filter"
    assert hits == ["search"]                       # never the login or the reset control
    assert any("/api/search" in u for u, _ in cap["json"])
    assert cap["final_url"].endswith("/search?q=") and "Tender 3" in cap["html"]


@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_browser_waits_for_an_app_that_boots_slowly(tmp_path):
    """Nothing drawn and no requests for 10 s (a big bundle parsing): not 'quiet' yet - the app is still booting."""
    server, base, hits = _serve(10000)
    try:
        cap = _capture(tmp_path, base, quiet_s=1.0, submit_search=True)
    finally:
        server.should_exit = True
    assert "Erweiterte Suche" in cap["html"] and hits == ["search"]
