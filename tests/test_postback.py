"""Postback pagination (PRADO / ASP.NET WebForms pagers that post the page's form): every page is read, a read is
complete only when the pager ends at the stated total, and a broken pager is never complete."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from mock_portal import app as mock
from tests.fake_llm import BASE
from watchtower import onboard, postback, runner
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext, Verdict
from watchtower.validate import decide


@pytest.fixture(autouse=True)
def postback_skin():
    mock.CONFIG.update(skin="postback", mutations=set(), postback_flavor="prado")
    mock.PB_POSTS.clear()
    yield
    mock.CONFIG.update(skin="html", mutations=set())
    for k in ("postback_flavor", "pb_total_extra"):
        mock.CONFIG.pop(k, None)


def spec_for(pagination: dict | None = None) -> dict:
    return {"tz": "Europe/Berlin",
            "list": {"kind": "html", "url": f"{BASE}/ausschreibungen", "items": "tr.tender-row",
                     "pagination": pagination or {"type": "postback", "next_sel": "a#ctl0_pager_next"},
                     "total": {"sel": "p.total", "regex": r"(\d+)"}},
            "fields": {"id": {"sel": "td.nr"}, "title": {"sel": "td.title a"},
                       "url": {"sel": "td.title a", "attr": "href"}, "buyerName": {"sel": "td.buyer"},
                       "publicationDate": {"sel": "td.published", "formats": ["%d.%m.%Y"]},
                       "submissionDeadline": {"sel": "td.deadline", "formats": ["%d.%m.%Y %H:%M"]}},
            "detail": {"kind": "html", "url_from": "url"}}


class Recording(httpx.AsyncBaseTransport):
    """The mock portal, plus a log of every request URL (and an optional hook before each request)."""

    def __init__(self, before=None):
        self.inner = httpx.ASGITransport(app=mock.app)
        self.urls: list[str] = []
        self.before = before

    async def handle_async_request(self, request):
        self.urls.append(f"{request.method} {request.url}")
        if self.before:
            self.before(request)
        return await self.inner.handle_async_request(request)


def read(spec, transport=None, max_pages=None):
    transport = transport or Recording()

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=transport)
        ctx = JobContext("p", BASE, "r", pc, None, None, None)
        try:
            return await runner.read(ctx, spec, {}, runner.Limits(max_pages=max_pages, max_details=0,
                                                                  fetch_docs=False)), ctx
        finally:
            await pc.aclose()
    res, ctx = asyncio.run(go())
    return res, transport, ctx


def get_html(path: str) -> str:
    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=mock.app), base_url=BASE) as c:
            return (await c.get(path)).text
    return asyncio.run(go())


@pytest.mark.parametrize("flavor,target_field", [("prado", "PRADO_POSTBACK_TARGET"), ("aspnet", "__EVENTTARGET")])
def test_postback_reads_every_page(flavor, target_field):
    mock.CONFIG["postback_flavor"] = flavor
    res, tr, _ = read(spec_for())
    assert sorted(t["id"] for t in res.tenders) == sorted(t["id"] for t in mock.TENDERS)
    assert res.complete and res.stats["stopReason"] == "reached stated total" and not res.stats["capped"]
    assert res.stats["pages"] == 4 and res.stats["shortfall"] == 0
    assert [u.split(" ")[0] for u in tr.urls] == ["GET", "POST", "POST", "POST"]
    # the browser's submission: the list form only (not the search form before it), the state, the clicked control
    assert all(p[target_field] == "ctl0$pager$next" for p in mock.PB_POSTS)
    assert all("q" not in p for p in mock.PB_POSTS)
    assert all(("PRADO_PAGESTATE" in p) == (flavor == "prado") for p in mock.PB_POSTS)
    assert all(p["ctl0$pager$size"] == "3" for p in mock.PB_POSTS)
    d = decide(Verdict(ok=True), res, mode="read", first=False)
    assert d.commit and d.close_missing


@pytest.mark.parametrize("flavor", ["prado", "aspnet"])
def test_postback_page_size_control(flavor):
    mock.CONFIG["postback_flavor"] = flavor
    spec = spec_for({"type": "postback", "next_sel": "a#ctl0_pager_next", "size_sel": "select#ctl0_pager_size"})
    res, tr, _ = read(spec)
    assert res.complete and len(res.tenders) == 12
    first = mock.PB_POSTS[0]
    assert first.get("PRADO_POSTBACK_TARGET", first.get("__EVENTTARGET")) == "ctl0$pager$size"
    assert first["ctl0$pager$size"] == "5"
    assert len(tr.urls) == 4          # GET, resize, 2 x next (5 + 5 + 2) instead of GET + 3 x next at 3 per page


def test_page_size_control_is_found_and_proven_without_a_model():
    def found(mutations=()):
        mock.CONFIG["mutations"] = set(mutations)
        spec = spec_for()
        dry, _, _ = read(spec_for(), max_pages=2)       # what learn's dry run saw: 2 pages of 3

        async def go():
            pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=Recording())
            ctx = JobContext("p", BASE, "r", pc, None, None, None)
            try:
                await onboard._postback_size_safely(ctx, spec, dry)
            finally:
                await pc.aclose()
        asyncio.run(go())
        return spec["list"]["pagination"].get("size_sel")
    assert found() == "select#ctl0_pager_size"
    assert found({"pager_ignores_target"}) is None      # the control is there but does nothing: not kept


def test_pager_that_ignores_the_postback_is_not_complete():
    mock.CONFIG["mutations"] = {"pager_ignores_target"}
    res, _, _ = read(spec_for())
    assert not res.complete and not res.stats["capped"]
    assert res.stats["stopReason"] == "page 2 repeated earlier results"
    d = decide(Verdict(ok=True), res, mode="read", first=False)
    assert not d.commit and not d.close_missing


def test_pager_that_ends_early_is_not_complete():
    mock.CONFIG["mutations"] = {"pager_stops"}
    res, _, _ = read(spec_for())
    assert not res.complete and not res.stats["capped"] and len(res.tenders) == 6
    assert res.stats["stopReason"] == "the pager ended at page 2 but the portal states 12 (read 6)"


def test_expired_session_is_not_the_end_of_the_list():
    """The pager needs the session cookie: kept across the walk it works; lost, the post shows no rows -> not complete."""
    def drop_sessions(request):
        if request.method == "POST":
            mock.PB_SESSIONS.clear()
    res, _, _ = read(spec_for(), Recording(before=drop_sessions))
    assert not res.complete and len(res.tenders) == 3
    assert res.stats["stopReason"].startswith("the postback for page 2 showed no rows")


def test_listing_that_changed_during_the_walk():
    """Every page walked but the stated total is 1 ahead (a notice published mid-read): served, nothing closed.
    Further off than the drift allowance, the walk is not complete."""
    mock.CONFIG["pb_total_extra"] = 1
    res, _, _ = read(spec_for())
    assert res.complete and res.stats["shortfall"] == 1 and "listing changed" in res.stats["stopReason"]
    d = decide(Verdict(ok=True), res, mode="read", first=False)
    assert d.commit and not d.close_missing and any("not closed" in n for n in d.notes)
    mock.CONFIG["pb_total_extra"] = 3
    res, _, _ = read(spec_for())
    assert not res.complete and res.stats["stopReason"].startswith("the pager ended at page 4 but the portal states 15")


def test_rows_the_spec_cannot_read_are_kept_as_evidence():
    mock.CONFIG["mutations"] = {"external_row"}
    spec = spec_for()
    spec["fields"]["url"]["sel"] = "td.title a.intern"
    res, _, _ = read(spec)
    # the learned selector misses the external row; the runner takes the row's own link, and still keeps the row
    # as evidence for a later repair
    assert res.complete and all(t.get("url") for t in res.tenders) and res.stats["urlFallbacks"] == 1
    assert {t["id"]: t["url"] for t in res.tenders}["MH-2026-1011"] == "https://extern.test/pub/MH-2026-1011"
    ev = res.stats["missingExamples"]["url"]
    assert "https://extern.test/pub/MH-2026-1011" in ev and "MH-2026-1011" in ev and len(ev) <= 2200
    assert "missingExamples" not in read(spec_for())[0].stats


def test_a_list_page_lost_in_transport_is_retried_once(monkeypatch):
    monkeypatch.setattr(runner, "LIST_RETRY_PAUSE", 0)
    fails = {"left": 1}

    def flaky(request):
        if request.method == "POST" and fails["left"]:
            fails["left"] -= 1
            raise httpx.ReadTimeout("slow page")
    res, tr, ctx = read(spec_for(), Recording(before=flaky))
    assert res.complete and len(res.tenders) == 12 and any("retrying once" in n for n in ctx.log)
    fails["left"] = 2
    with pytest.raises(httpx.ReadTimeout):         # twice in a row: the read fails (and says why)
        read(spec_for(), Recording(before=flaky))


def test_javascript_next_link_is_not_followed():
    """The model once declared the PRADO href as a next link: never fetched, the stop reason names postback."""
    res, tr, _ = read(spec_for({"type": "next_link", "next_sel": "a#ctl0_pager_next"}))
    assert not res.complete and not res.stats["capped"]
    assert "postback" in res.stats["stopReason"]
    assert not any("javascript" in u for u in tr.urls) and len(tr.urls) == 1


def test_pager_hint_names_the_next_control():
    for flavor in ("prado", "aspnet"):
        mock.CONFIG["postback_flavor"] = flavor
        hint = postback.pager_hint(mock._postback_page(1, 3))
        assert hint and '"postback"' in hint and "a#ctl0_pager_next" in hint, (flavor, hint)
    mock.CONFIG["skin"] = "html"
    assert postback.pager_hint(get_html("/ausschreibungen")) is None    # ordinary ?page= links: no hint


def test_validate_spec_postback():
    ok = spec_for()
    assert onboard.validate_spec(ok) == []
    no_sel = spec_for({"type": "postback"})
    assert any("next_sel" in p for p in onboard.validate_spec(no_sel))
    as_json = spec_for()
    as_json["list"]["kind"] = "json"
    assert any("postback" in p for p in onboard.validate_spec(as_json))
    bad = spec_for({"type": "postback", "next_sel": "a#x", "size_sel": "select[[["})
    assert any("size_sel" in p for p in onboard.validate_spec(bad))


def test_form_fields_follow_browser_rules():
    form = HTMLParser("""<form><input name=a value=1><input type=checkbox name=c1 value=x checked>
        <input type=checkbox name=c2 value=y><input type=radio name=r value=r1><input type=radio name=r value=r2 checked>
        <input type=submit name=go value=Go><input name=d value=no disabled><input type=hidden name=h>
        <select name=s><option value=1>1</option><option value=2 selected>2</option></select>
        <select name=s0><option>first</option><option>second</option></select>
        <textarea name=t>text</textarea></form>""").css_first("form")
    assert postback.form_fields(form) == [("a", "1"), ("c1", "x"), ("r", "r2"), ("h", ""), ("s", "2"),
                                          ("s0", "first"), ("t", "text")]


def test_submit_button_pager_and_get_forms():
    """A pager made of submit buttons posts the clicked button's name; a GET form becomes a query string."""
    html = ("<form method=get action='/list'><input type=hidden name=state value=7>"
            "<input type=submit name='pager$next' value='Next'></form>")
    tree = HTMLParser(html)
    req, why = postback.next_request(tree, html, "http://p.test/list?x=1", "utf-8",
                                     {"type": "postback", "next_sel": "input[name='pager$next']"})
    assert why is None and req.method == "GET" and req.url == "http://p.test/list?state=7&pager%24next=Next"


def test_unresolvable_control_is_a_reason_not_the_end():
    html = "<form method=post><input type=hidden name=__VIEWSTATE value=s><a id=n href='#' onclick='go()'>Next</a></form>"
    tree = HTMLParser(html)
    req, why = postback.next_request(tree, html, "http://p.test/", "utf-8", {"type": "postback", "next_sel": "a#n"})
    assert req is None and "no postback target" in why
