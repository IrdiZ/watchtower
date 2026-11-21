"""Pagers driven by JavaScript, followed without a browser: an address inside the next control's script, a form whose
page field a script sets, and rows a script request appends ("load more"). Every page is read, a read is complete
only at the stated total, and a pager that cannot be followed is never served as complete."""
from __future__ import annotations

import json

import pytest
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from mock_portal import app as mock
from tests.fake_llm import BASE
from tests.test_e2e_mock import attach, env, wait  # noqa: F401  (fixture)
from tests.test_postback import get_html, read
from watchtower import jspager, onboard

TABLE_FIELDS = {"id": {"sel": "td.nr"}, "title": {"sel": "td.title a"}, "url": {"sel": "td.title a", "attr": "href"},
                "buyerName": {"sel": "td.buyer"}, "publicationDate": {"sel": "td.published", "formats": ["%d.%m.%Y"]},
                "submissionDeadline": {"sel": "td.deadline", "formats": ["%d.%m.%Y %H:%M"]}}
STREAM_FIELDS = {"id": {"sel": "span.nr"}, "title": {"sel": "span.title a"},
                 "url": {"sel": "span.title a", "attr": "href"}, "buyerName": {"sel": "span.buyer"},
                 "publicationDate": {"sel": "span.published", "formats": ["%d.%m.%Y"]},
                 "submissionDeadline": {"sel": "span.deadline", "formats": ["%d.%m.%Y %H:%M"]}}
ALL_IDS = {t["id"] for t in mock.TENDERS}


@pytest.fixture(autouse=True)
def clean():
    mock.CONFIG.update(skin="html", mutations=set())
    mock.JS_HITS.clear()
    yield
    mock.CONFIG.update(skin="html", mutations=set())


def jsurl_spec(pagination: dict) -> dict:
    return {"tz": "Europe/Berlin",
            "list": {"kind": "html", "url": f"{BASE}/jsurl/liste", "items": "tr.tender-row", "pagination": pagination,
                     "total": {"sel": ".browsePagesText", "regex": r"Gesamteinträge:\s*(\d+)"}},
            "fields": TABLE_FIELDS, "detail": {"kind": "html", "url_from": "url"}}


def jsform_spec(pagination: dict) -> dict:
    return {"tz": "Europe/Berlin",
            "list": {"kind": "html", "url": f"{BASE}/jsform/liste", "items": "tr.tender-row", "pagination": pagination,
                     "total": {"sel": "h1.resultTitle span.count", "regex": r"(\d+)"}},
            "fields": TABLE_FIELDS, "detail": {"kind": "html", "url_from": "url"}}


def xhr_spec(**over) -> dict:
    pag = {"type": "load_more", "url": "/xhr/api/more", "method": "POST",
           "fields": {"offset": "{offset}", "filter": "0", "search": "", "form": {"orderBy": "created desc"},
                      "CSRF_TOKEN": "{token:CSRF_TOKEN}"},
           "html_path": "items", **over}
    return {"tz": "Europe/Berlin",
            "list": {"kind": "html", "url": f"{BASE}/xhr/liste", "items": "ul.stream > li[id^='tender-']",
                     "pagination": pag, "total": {"sel": "div.hits", "regex": r"hit:\s*([\d.]+)"}},
            "fields": STREAM_FIELDS, "detail": {"kind": "html", "url_from": "url"}}


# ------------------------------------------------------------------ an address inside the next control's script
def test_onclick_address_as_postback_submits_the_form_to_that_address():
    """setTargetAndSubmit('/l?seite=2'): the script points the list form at the address and submits it."""
    res, tr, _ = read(jsurl_spec({"type": "postback", "next_sel": "nav.browsePages a.browseForward"}))
    assert res.complete and res.stats["stopReason"] in ("no next-page control", "reached stated total"), res.stats
    assert {t["id"] for t in res.tenders} == ALL_IDS
    later = mock.JS_HITS[1:]
    assert [h["query"].get("seite") for h in later] == ["2", "3"]
    assert all(h["method"] == "POST" and h["form"].get("filterStatus") == "expanded" and "JSESSIONID" in h["cookie"]
               for h in later), later


def test_onclick_address_as_next_link_is_followed():
    res, _, _ = read(jsurl_spec({"type": "next_link", "next_sel": "nav.browsePages a.browseForward"}))
    assert res.complete and {t["id"] for t in res.tenders} == ALL_IDS, res.stats
    assert [h["method"] for h in mock.JS_HITS] == ["GET", "GET", "GET"]


@pytest.mark.parametrize("ptype", ["postback", "next_link"])
def test_script_without_an_address_is_never_complete(ptype):
    mock.CONFIG["mutations"] = {"jsurl_no_address"}
    res, _, _ = read(jsurl_spec({"type": ptype, "next_sel": "nav.browsePages a.browseForward"}))
    assert not res.complete and len(res.tenders) == 5, res.stats
    assert "JavaScript action" in res.stats["stopReason"] or "no address in its script" in res.stats["stopReason"]


def test_js_url_reads_one_address_only():
    def el(attrs):
        return HTMLParser(f"<a {attrs}>x</a>").css_first("a")
    assert jspager.js_url(el("href='javascript:void(0);' onclick=\"go('/a.do?p=2')\"")) == "/a.do?p=2"
    assert jspager.js_url(el("href=\"javascript:location.href='?page=3'\"")) == "?page=3"
    assert jspager.js_url(el("href='javascript:;//ctl0_pager_next'")) is None          # PRADO: bound by a script
    assert jspager.js_url(el("href=\"javascript:__doPostBack('ctl0$next','')\"")) is None
    assert jspager.js_url(el("onclick=\"go('/a?p=2', '/b?p=2')\"")) is None             # two addresses: no guess
    assert jspager.js_url(el("onclick=\"alert('Seite 2 laden')\"")) is None              # text, not an address


# ------------------------------------------------------------------ a form whose page field a script sets
def test_form_pager_reads_every_page():
    res, _, _ = read(jsform_spec({"type": "form", "form_sel": "form#hiddenForm", "field": "bean.page", "start": 0,
                                  "next_sel": "span[aria-label='Fiche suivante']"}))
    assert res.complete and {t["id"] for t in res.tenders} == ALL_IDS, res.stats
    posts = [h for h in mock.JS_HITS if h["method"] == "POST"]
    assert [h["form"]["bean.page"] for h in posts] == ["1", "2"]
    assert all(h["form"]["bean.dateFrom"] == "20/11/2026" and h["path"] == "/jsform/suche" for h in posts)


def test_form_pager_with_the_wrong_start_is_not_complete():
    """start 1 asks for the third page second: page 2 is skipped, the pager ends short of the stated total."""
    res, _, _ = read(jsform_spec({"type": "form", "form_sel": "form#hiddenForm", "field": "bean.page", "start": 1,
                                  "next_sel": "span[aria-label='Fiche suivante']"}))
    assert not res.complete and "the pager ended" in res.stats["stopReason"], res.stats


def test_form_pager_without_a_form_stops_honestly():
    res, _, _ = read(jsform_spec({"type": "form", "form_sel": "form#nope", "field": "bean.page", "start": 0}))
    assert not res.complete and "no form matches" in res.stats["stopReason"], res.stats


# ------------------------------------------------------------------ rows a script request appends
def test_load_more_reads_every_row_with_the_pages_token():
    res, _, _ = read(xhr_spec())
    assert res.complete and {t["id"] for t in res.tenders} == ALL_IDS, res.stats
    more = [h for h in mock.JS_HITS if h["path"] == "/xhr/api/more"]
    assert [h["form"]["offset"] for h in more] == ["5", "10"]          # rows shown so far, as the page counts
    assert all(h["xhr"] == "XMLHttpRequest" and h["form"]["form[orderBy]"] == "created desc" for h in more)
    # rows appended to the list page resolve their links against the list page, not the request's address
    assert {t["url"] for t in res.tenders} == {f"{BASE}/tender/{i}" for i in ALL_IDS}


def test_load_more_with_a_stale_token_is_never_complete():
    spec = xhr_spec()
    spec["list"]["pagination"]["fields"]["CSRF_TOKEN"] = "0123456789abcdef0123"    # copied from an old visit
    res, _, _ = read(spec)
    assert not res.complete and len(res.tenders) == 5, res.stats
    assert "load-more answer for page 2 is not JSON" in res.stats["stopReason"], res.stats


def test_load_more_token_missing_from_the_page_stops_honestly():
    spec = xhr_spec()
    spec["list"]["pagination"]["fields"]["CSRF_TOKEN"] = "{token:NO_SUCH_TOKEN}"
    res, _, _ = read(spec)
    assert not res.complete and "no value named 'NO_SUCH_TOKEN'" in res.stats["stopReason"], res.stats


def test_literal_token_from_the_page_becomes_a_per_visit_token():
    html = get_html("/xhr/liste")
    token = jspager.page_value(HTMLParser(html), html, "CSRF_TOKEN")
    spec = xhr_spec()
    spec["list"]["pagination"]["fields"]["CSRF_TOKEN"] = token
    onboard._per_visit_tokens(spec, html)
    assert spec["list"]["pagination"]["fields"]["CSRF_TOKEN"] == "{token:CSRF_TOKEN}"
    assert spec["list"]["pagination"]["fields"]["offset"] == "{offset}"


def test_attach_load_more_portal_learns_the_script_request_from_the_feedback(env):
    """The reduced page shows no pager, so the model first declares none; the dry run sees 5 of a stated 12 and
    quotes the page's script; the model copies its request (token included, as shown) and the portal serves all 12,
    on later refreshes too (a new visit, a new token)."""
    client, llm = env
    pid, p = attach(client, "/xhr/liste")
    assert p["state"] == "serving" and p["tenderCount"] == 12, p
    lst = client.app.state.store.current_spec(pid)[1]["list"]
    pag = lst["pagination"]
    assert pag["type"] == "load_more" and pag["fields"]["CSRF_TOKEN"] == "{token:CSRF_TOKEN}", pag
    assert lst["total"]["sel"] == "div.hits", lst       # the rewrite dropped it; the earlier attempt's is kept
    run = json.loads(client.app.state.store.list_runs(pid)[-1]["detail"])
    assert any("the portal states 12 results but the spec read 5" in line for line in run["log"])
    assert any("kept the earlier attempt's stated-total selector" in line for line in run["log"])
    assert run["stats"]["statedTotal"] == 12 and run["stats"]["stopReason"] == "reached stated total", run["stats"]
    n = len(llm.calls)
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"serving", "failed"})
    assert p["state"] == "serving" and p["tenderCount"] == 12 and len(llm.calls) == n, p


# ------------------------------------------------------------------ what the model is told, what is checked
def test_pager_notes_say_what_each_script_does():
    xhr = onboard.pager_notes(get_html("/xhr/liste"))
    assert "load_more" in xhr and "/xhr/api/more" in xhr and "CSRF_TOKEN" in xhr, xhr
    form = onboard.pager_notes(get_html("/jsform/liste"))
    assert '"type": "form"' in form and "form#hiddenForm" in form and "bean.page" in form, form
    assert '$("#nextPrevPage").val(1)' in form, form
    url = onboard.pager_notes(get_html("/jsurl/liste"))
    assert "/jsurl/liste?method=showTable&seite=2" in url and '"postback"' in url, url
    assert onboard.pager_notes(get_html("/ausschreibungen")) is None        # a plain ?page= pager: nothing to add


def test_validate_spec_checks_the_scripted_pagers():
    assert onboard.validate_spec(xhr_spec()) == []
    assert onboard.validate_spec(jsform_spec({"type": "form", "form_sel": "form#hiddenForm", "field": "bean.page",
                                              "start": 0})) == []
    bad = onboard.validate_spec(xhr_spec(fields={"filter": "0"}))
    assert any("{offset}" in p for p in bad), bad
    bad = onboard.validate_spec(xhr_spec(url="javascript:void(0)", html_path="items[", start="2"))
    assert len(bad) == 3, bad
    bad = onboard.validate_spec(jsform_spec({"type": "form", "field": ""}))
    assert any("form_sel" in p for p in bad) and any("pagination.field" in p for p in bad), bad


def test_today_token_in_the_portals_own_date_format(monkeypatch):
    from datetime import datetime, timezone
    from watchtower import runner, timeutil
    monkeypatch.setattr(timeutil, "now", lambda: datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc))
    spec = {"tz": "Europe/Paris"}
    assert runner.fill_tokens("from={today}&d={today:%d/%m/%Y}", spec) == "from=2026-11-21&d=21/11/2026"
    assert runner.fill_tokens("{today:%d.%m.%Y}", spec) == "21.11.2026"


def test_load_more_request_encodes_like_the_page_script():
    req = jspager.load_more_request({"url": "more?x=1", "method": "GET", "fields": {"o": "{offset}", "a": [1, 2],
                                     "n": {"k": None, "b": True}, "d": "{today}"}},
                                    "https://p.test/list/", 2, 40, {}, lambda v: v.replace("{today}", "2026-11-21"))
    assert req.method == "GET" and req.url == ("https://p.test/list/more?x=1&o=40&a%5B%5D=1&a%5B%5D=2&n%5Bk%5D=&"
                                               "n%5Bb%5D=true&d=2026-11-21"), req.url
    req = jspager.load_more_request({"url": "/api", "json": {"from": "{offset}", "p": "{page}", "t": "{token:T}"}},
                                    "https://p.test/list", 3, 20, {"T": "abc"}, lambda v: v)
    assert req.method == "POST" and req.content == b'{"from": 20, "p": 3, "t": "abc"}', req.content
