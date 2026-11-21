"""Blind verification on tender pages that render only with JavaScript: the page is rendered in a browser (at most
VERIFY_RENDERS per learn) and compared; when even that shows nothing, the spec is NOT marked verified and the
serving reason says so ("Verification: not possible ...")."""
import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_portal import app as mock
from tests.fake_llm import BASE, FakeLLM
from tests.test_browser_live import _chromium_ok
from watchtower import browser, onboard
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient
from watchtower.pipeline import AttachFailed, JobContext, ReadResult, Verdict
from watchtower.store import Store
from watchtower.validate import decide

BERLIN = "Europe/Berlin"
SHELL = "<!doctype html><html><body><div id='app'></div><script src='/main.js'></script></body></html>"
RENDERED = ("<html><body><div id='app'><article><h2>Dachsanierung Grundschule</h2><dl><dt>Vergabe-Nr.</dt>"
            "<dd>MH-1</dd><dt>Vergabestelle</dt><dd>Stadt Musterhausen</dd><dt>Veröffentlicht</dt><dd>20.11.2026</dd>"
            "<dt>Angebotsfrist</dt><dd>10.12.2026 10:00</dd></dl></article></div></body></html>")
SPEC = {"tz": BERLIN, "detail": {"kind": "html", "url_from": "url"},
        "fields": {"publicationDate": {"formats": ["%d.%m.%Y"]}, "submissionDeadline": {"formats": ["%d.%m.%Y %H:%M"]}}}
GOOD = {"id": "MH-1", "url": "https://x.test/app/tender/MH-1", "title": "Dachsanierung Grundschule",
        "buyerName": "Stadt Musterhausen", "publicationDate": "2026-11-20",
        "submissionDeadline": "2026-12-10T10:00:00+01:00"}


def _model(text: str) -> dict:
    """A careful reader: copies what the text shows, nothing else."""
    if "Dachsanierung" not in text:
        return {"id": None, "title": None, "buyerName": None, "publicationDate": None, "submissionDeadline": None}
    return {"id": "MH-1", "title": "Dachsanierung Grundschule", "buyerName": "Stadt Musterhausen",
            "publicationDate": "20.11.2026", "submissionDeadline": "10.12.2026 10:00"}


def _ctx(seen: list, status=200, html=SHELL):
    async def get(url, **kw):
        return SimpleNamespace(status_code=status, text=html, json=lambda: {})

    async def json_call(task, system, user, **kw):
        seen.append(user)
        return _model(user)
    return SimpleNamespace(client=SimpleNamespace(get=get), llm=SimpleNamespace(json_call=json_call),
                           portal_id="p", run_id="r", note=lambda m: None)


def _fake_capture(monkeypatch, html="", json_=None):
    calls = []

    async def capture(ctx, url, **kw):
        calls.append(url)
        return {"html": html, "json": list(json_ or []), "requests": 3, "seconds": 1.0, "error": None}
    monkeypatch.setattr(browser, "capture", capture)
    return calls


def test_js_tender_page_is_rendered_and_compared(monkeypatch):
    calls = _fake_capture(monkeypatch, RENDERED)
    seen: list = []
    spec = dict(SPEC)
    assert asyncio.run(onboard.blind_verify(_ctx(seen), spec, [dict(GOOD)], n=1)) == []
    assert calls == [GOOD["url"]] and "Dachsanierung" in seen[0] and "verification" not in spec
    wrong = dict(GOOD, buyerName="Landkreis Beispiel")        # a spec that reads the wrong buyer is caught
    probs = asyncio.run(onboard.blind_verify(_ctx([]), dict(SPEC), [wrong], n=1))
    assert len(probs) == 1 and "buyerName mismatch" in probs[0], probs


def test_server_rendered_pages_are_not_rendered(monkeypatch):
    calls = _fake_capture(monkeypatch, RENDERED)
    assert asyncio.run(onboard.blind_verify(_ctx([], html=RENDERED), dict(SPEC), [dict(GOOD)], n=1)) == []
    assert calls == []


def test_data_the_page_loads_is_used_when_the_render_stays_empty(monkeypatch):
    record = {"id": "MH-1", "title": "Dachsanierung Grundschule", "buyer": "Stadt Musterhausen"}
    _fake_capture(monkeypatch, SHELL, [("https://x.test/api/menu", {"items": []}), ("https://x.test/api/t", record)])
    seen: list = []
    spec = dict(SPEC)
    assert asyncio.run(onboard.blind_verify(_ctx(seen), spec, [dict(GOOD)], n=1)) == []
    assert '"buyer": "Stadt Musterhausen"' in seen[0] and "verification" not in spec


def test_nothing_rendered_is_not_a_pass_and_renders_are_bounded(monkeypatch):
    calls = _fake_capture(monkeypatch, "<html><body><div id='app'>Cookie-Einstellungen</div></body></html>")
    tenders = [dict(GOOD, id=f"MH-{i}", url=f"https://x.test/app/tender/MH-{i}") for i in range(6)]
    renders = {"left": onboard.VERIFY_RENDERS}
    spec = dict(SPEC)
    assert asyncio.run(onboard.blind_verify(_ctx([]), spec, tenders, renders=renders)) == []
    assert spec["verification"] == onboard.VERIFY_JS_ONLY
    assert len(calls) == onboard.VERIFY_RENDERS == 2
    # a later attempt of the same learn renders nothing new
    spec2 = dict(SPEC)
    assert asyncio.run(onboard.blind_verify(_ctx([]), spec2, tenders, renders=renders)) == []
    assert len(calls) == 2 and spec2["verification"] == onboard.VERIFY_JS_ONLY


def test_pages_that_cannot_be_opened_still_fail(monkeypatch):
    calls = _fake_capture(monkeypatch, RENDERED)
    spec = dict(SPEC)
    probs = asyncio.run(onboard.blind_verify(_ctx([], status=404), spec, [dict(GOOD)], n=1))
    assert probs == ["could not open any tender page to verify"] and "verification" not in spec and calls == []


def test_serving_reason_names_a_verification_that_was_not_possible():
    res = ReadResult([{"id": "1"}], True, {})
    d = decide(Verdict(ok=True), res, mode="attach", first=True, spec={"verification": onboard.VERIFY_JS_ONLY})
    assert d.commit and "Verification: not possible: tender pages render only with JavaScript." in d.reason(1, 1)
    assert "Verification" not in decide(Verdict(ok=True), res, mode="attach", first=True, spec={}).reason(1, 1)


# ------------------------------------------------------------------ on the mock portal (api skin)
class PageOnlyLLM(FakeLLM):
    """A model that reads everything from the list API and links the app's tender pages: no detail API, so blind
    verification has only the JavaScript tender page /app/tender/{id} to compare against."""

    def _list(self, d):
        spec = super()._list(d)
        if "jsonApi" in d:
            spec["detail"] = {"kind": "html", "url_from": "url"}
        return spec

    def _detail(self, d):
        return {"fields": {}, "documents": {}, "block_markers": []}


def _job_ctx(tmp_path):
    store = Store(str(tmp_path))
    client = PoliteClient(HostRateLimiter(0), "t", 10, budget=800, transport=httpx.ASGITransport(app=mock.app))
    return JobContext("p1", BASE + "/", "r1", client, DocumentStore(store, 10 ** 8, 10 ** 9), PageOnlyLLM(store),
                      store, mode="attach")


def _learn(ctx):
    async def go():
        try:
            return await onboard.learn(ctx, BASE + "/")
        finally:
            await ctx.client.aclose()
    return asyncio.run(go())


@pytest.fixture
def api_skin():
    mock.CONFIG.update(skin="api", mutations=set())
    yield
    mock.CONFIG.update(skin="html", mutations=set())


chromium = pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")


@chromium
def test_mock_js_tender_pages_are_rendered_and_verified(tmp_path, api_skin, monkeypatch):
    real, renders = browser.capture, []

    async def counting(ctx, url, **kw):
        renders.append(url)
        return await real(ctx, url, **kw)
    monkeypatch.setattr(browser, "capture", counting)
    ctx = _job_ctx(tmp_path)
    spec = _learn(ctx)
    assert spec["verifiedBlind"] is True and "verification" not in spec
    assert renders and all("/app/tender/" in u for u in renders) and len(renders) <= onboard.VERIFY_RENDERS
    assert any("renders only with JavaScript" in m for m in ctx.log), ctx.log


@chromium
def test_mock_js_tender_pages_catch_a_wrong_spec(tmp_path, api_skin):
    mock.CONFIG["mutations"] = {"swap_columns"}       # the list API swaps title and buyer, the tender page does not
    ctx = _job_ctx(tmp_path)
    with pytest.raises(AttachFailed) as e:
        _learn(ctx)
    assert "mismatch" in str(e.value), e.value


@chromium
def test_mock_blank_js_tender_pages_are_not_verified(tmp_path, api_skin):
    mock.CONFIG["mutations"] = {"detail_js_blank"}
    ctx = _job_ctx(tmp_path)
    spec = _learn(ctx)
    assert spec["verifiedBlind"] is False and spec["verification"] == onboard.VERIFY_JS_ONLY


def test_attach_serves_with_the_verification_caveat_in_the_reason(tmp_path, api_skin, monkeypatch):
    _fake_capture(monkeypatch, "")                     # no browser here: the tender pages stay empty shells
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    app = create_app(s, llm=PageOnlyLLM, transport=httpx.ASGITransport(app=mock.app))
    with TestClient(app) as client:
        pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
        end = time.time() + 30
        while time.time() < end:
            p = client.get(f"/portals/{pid}").json()
            if p["state"] in ("serving", "failed"):
                break
            time.sleep(0.05)
    assert p["state"] == "serving", p
    assert "Verification: not possible: tender pages render only with JavaScript." in p["reason"], p["reason"]
