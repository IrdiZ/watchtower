"""Cursor ("load more" / keyset) paging, POST search APIs paged inside the JSON body, and replaying what the browser
saw with the same method and body.

Shapes taken from the live portals (20 Nov): simap.ch answers {"projects": [...], "pagination": {"lastItem":
"20261120|38951", "itemsPerPage": 20}} and wants ?lastItem=<that> for the next page (no total; an empty page with
lastItem "" at the end). oeffentlichevergabe.de answers POST /bkmk/searches {"SELECT": ..., "WHERE": [...], "PAGE":
{"number": 0, "size": 25}, ...} with {"totalElements", "offset", "elements"} and refuses page*size > 10,000."""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from tests.test_browser_live import _chromium_ok
from tests.test_contract import setup  # noqa: F401  (fixture)
from watchtower import onboard, runner
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext

HOST = "http://keyset.test"


def _rows(n: int):
    return [{"id": f"p{i:03d}", "projectNumber": str(1000 + i), "title": f"Project {i}", "office": "Kanton X",
             "publicationDate": "2026-11-20", "deadline": "2030-12-09T10:00:00+01:00"} for i in range(n)]


def _fields():
    return {"id": {"path": "id"}, "title": {"path": "title"}, "buyerName": {"path": "office"},
            "publicationDate": {"path": "publicationDate"}, "submissionDeadline": {"path": "deadline"},
            "url": {"template": f"{HOST}/project/{{id}}"}}


def _read(client, handler, spec, limits=None):
    store, docs = client.app.state.store, client.app.state.manager.docs

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
        ctx = JobContext("p", HOST, "r", pc, docs, None, store)
        try:
            return await runner.read(ctx, spec, {}, limits or runner.Limits(fetch_docs=False))
        finally:
            await pc.aclose()
    return asyncio.run(go())


# ----------------------------------------------------------------------------------- cursor from the response body
def keyset_handler(rows, hits, page=20, cursor_key="lastItem"):
    """simap-style keyset paging: the cursor is '<date>|<number>' of the last row; unknown cursors start over."""
    keys = [f"{r['publicationDate'].replace('-', '')}|{r['projectNumber']}" for r in rows]

    def handler(req: httpx.Request):
        after = req.url.params.get("lastItem")
        hits.append(after)
        start = keys.index(after) + 1 if after in keys else 0
        chunk = rows[start:start + page]
        last = keys[start + len(chunk) - 1] if chunk else ""
        return httpx.Response(200, json={"projects": chunk, "pagination": {cursor_key: last, "itemsPerPage": page}})
    return handler


def simap_spec(**pag):
    return {"tz": "Europe/Zurich",
            "list": {"kind": "json", "url": f"{HOST}/rest/project-search?lang=de", "items": "projects",
                     "pagination": {"type": "cursor", "param": "lastItem", "cursor_path": "pagination.lastItem",
                                    **pag}},
            "fields": _fields()}


def test_cursor_from_response_reads_every_page_and_ends_on_portal_evidence(setup):
    client, _ = setup
    hits = []
    res = _read(client, keyset_handler(_rows(47), hits), simap_spec())
    assert len(res.tenders) == 47 and res.complete, res.stats
    assert res.stats["stopReason"] == "empty page" and res.stats["pages"] == 4
    assert hits == [None, "20261120|1019", "20261120|1039", "20261120|1046"]   # first page: no cursor at all


def test_cursor_missing_from_response_is_not_complete(setup):
    client, _ = setup
    hits = []
    # the spec points at a key the response does not have: never a silent "complete" after one page
    res = _read(client, keyset_handler(_rows(47), hits), simap_spec(cursor_path="pagination.nextToken"))
    assert not res.complete and len(res.tenders) == 20
    assert "no next cursor at 'pagination.nextToken'" in res.stats["stopReason"]
    assert res.stats["capped"] is False      # breakage, not a portal cap: decide() rejects it


def test_cursor_missing_after_a_short_page_is_complete(setup):
    client, _ = setup

    def handler(req):
        return httpx.Response(200, json={"projects": _rows(7), "pagination": {"itemsPerPage": 20}})
    res = _read(client, handler, simap_spec(size=20))
    assert res.complete and res.stats["stopReason"] == "short page" and len(res.tenders) == 7


def test_cursor_explicitly_null_ends_the_list(setup):
    client, _ = setup
    rows = _rows(30)

    def handler(req):
        if req.url.params.get("cursor") == "t2":
            return httpx.Response(200, json={"data": rows[20:], "next": None})
        return httpx.Response(200, json={"data": rows[:20], "next": "t2"})
    spec = simap_spec()
    spec["list"].update(items="data", pagination={"type": "cursor", "param": "cursor", "cursor_path": "next"})
    res = _read(client, handler, spec)
    assert res.complete and res.stats["stopReason"] == "the portal gives no next cursor" and len(res.tenders) == 30


def test_cursor_has_more_flag_false_ends_the_list(setup):
    client, _ = setup

    def handler(req):
        return httpx.Response(200, json={"data": _rows(5), "next": "same", "hasMore": False})
    spec = simap_spec()
    spec["list"].update(items="data", pagination={"type": "cursor", "param": "c", "cursor_path": "next",
                                                  "more_path": "hasMore"})
    res = _read(client, handler, spec)
    assert res.complete and len(res.tenders) == 5


def test_cursor_repeated_is_not_complete(setup):
    client, _ = setup
    rows = _rows(60)
    calls = []

    def handler(req):
        calls.append(1)
        n = len(calls) - 1
        return httpx.Response(200, json={"data": rows[n * 20:(n + 1) * 20], "next": "stuck"})
    spec = simap_spec()
    spec["list"].update(items="data", pagination={"type": "cursor", "param": "c", "cursor_path": "next"})
    res = _read(client, handler, spec)
    assert not res.complete and "cursor that was already used" in res.stats["stopReason"]
    assert len(calls) == 2


def test_cursor_from_last_item(setup):
    client, _ = setup
    rows = _rows(25)
    seen = []

    def handler(req):
        after = req.url.params.get("after")
        seen.append(after)
        start = next((i + 1 for i, r in enumerate(rows) if r["id"] == after), 0)
        return httpx.Response(200, json={"items": rows[start:start + 10]})
    spec = simap_spec()
    spec["list"].update(items="items", pagination={"type": "cursor", "param": "after", "cursor_from_last": "id",
                                                   "size": 10})
    res = _read(client, handler, spec)
    # a short page that still yields a cursor is not taken as the end: one more request proves it (an empty page)
    assert res.complete and len(res.tenders) == 25 and res.stats["stopReason"] == "empty page"
    assert seen == [None, "p009", "p019", "p024"]


# ---------------------------------------------------------------------------------- POST search, paged in the body
def bkmk_handler(rows, seen, window=10000):
    """oeffentlichevergabe-style POST search: GET is not allowed; PAGE.number/PAGE.size in the body; a window."""
    def handler(req: httpx.Request):
        if req.method != "POST":
            return httpx.Response(405, json={"error": "Method Not Allowed"})
        body = json.loads(req.content)
        seen.append(body)
        n, size = body["PAGE"]["number"], body["PAGE"]["size"]
        if (n + 1) * size > window:
            return httpx.Response(400, text=f"Combination of Page and Size ({(n + 1) * size} results) exceeds limit "
                                            f"of {window} results.")
        ordered = rows[::-1] if (body.get("ORDER") or {}).get("direction") == "ASC" else rows
        return httpx.Response(200, json={"totalElements": len(rows), "offset": n * size,
                                         "elements": ordered[n * size:(n + 1) * size]})
    return handler


def bkmk_spec():
    return {"tz": "Europe/Berlin",
            "list": {"kind": "json", "url": f"{HOST}/bkmk/searches", "method": "POST",
                     "body": {"SELECT": "ALL", "WHERE": [{"fields": ["active"], "operator": "=", "operands": [True]},
                                                         {"fields": ["allDeadlines"], "operator": ">=",
                                                          "operands": ["{today}"]}],
                              "FROM": "lots", "PAGE": {"number": 0, "size": 25},
                              "ORDER": {"field": "publicationDate", "direction": "DESC"}},
                     "items": "elements", "total": {"path": "totalElements"},
                     "pagination": {"type": "page", "start": 0, "body_path": "PAGE.number",
                                    "size_body_path": "PAGE.size", "size": 10}},
            "fields": _fields()}


def test_post_search_is_paged_inside_the_body(setup):
    client, _ = setup
    seen = []
    res = _read(client, bkmk_handler(_rows(23), seen), bkmk_spec())
    assert res.complete and len(res.tenders) == 23 and res.stats["stopReason"] == "reached stated total"
    assert [b["PAGE"] for b in seen] == [{"number": 0, "size": 10}, {"number": 1, "size": 10},
                                         {"number": 2, "size": 10}]
    assert seen[0]["WHERE"][1]["operands"] == [runner.today_local(bkmk_spec())]    # {today} filled in the body
    assert seen[0]["ORDER"] == {"field": "publicationDate", "direction": "DESC"}  # the rest of the body unchanged
    assert bkmk_spec()["list"]["body"]["PAGE"]["number"] == 0                      # the spec itself is not mutated


def test_post_search_result_window_is_a_cap_not_complete(setup):
    client, _ = setup
    res = _read(client, bkmk_handler(_rows(45), [], window=20), bkmk_spec())
    assert not res.complete and res.stats["capped"] is True and len(res.tenders) == 20
    assert "HTTP 400" in res.stats["stopReason"]


def _reversible():
    spec = bkmk_spec()
    spec["list"]["pagination"]["reverse"] = {"body_path": "ORDER.direction", "value": "ASC"}
    assert onboard.validate_spec(spec) == []
    return spec


def test_windowed_list_is_read_from_both_ends_and_complete_only_at_the_stated_total(setup):
    client, _ = setup
    seen = []
    res = _read(client, bkmk_handler(_rows(38), seen, window=20), _reversible())
    assert res.complete and res.stats["stopReason"] == "reached stated total" and res.stats["bothEnds"] is True
    assert len(res.tenders) == 38 and res.stats["capped"] is False
    assert [(b["ORDER"]["direction"], b["PAGE"]["number"]) for b in seen] == [
        ("DESC", 0), ("DESC", 1), ("DESC", 2), ("ASC", 0), ("ASC", 1)]


def test_window_too_large_for_both_ends_stays_capped(setup):
    client, _ = setup
    res = _read(client, bkmk_handler(_rows(45), [], window=20), _reversible())
    assert not res.complete and res.stats["capped"] is True and len(res.tenders) == 40


def test_reverse_that_the_portal_ignores_is_not_complete(setup):
    client, _ = setup
    spec = _reversible()
    spec["list"]["pagination"]["reverse"] = {"body_path": "ORDER.field", "value": "ignored"}   # same order again
    res = _read(client, bkmk_handler(_rows(38), [], window=20), spec)
    assert not res.complete and "repeated earlier results" in res.stats["stopReason"]


def test_lots_with_identical_mapped_fields_count_as_rows(setup):
    client, _ = setup
    lots = [{**r, "id": f"N{i // 2}", "lot": f"LOT-{i % 2}"} for i, r in enumerate(_rows(4))]
    for x in lots:
        x["title"] = f"Notice {x['id']}"
    res = _read(client, bkmk_handler(lots, []), bkmk_spec())
    assert res.complete and res.stats["rows"] == 4 and res.stats["mergedNotices"] == 2 and len(res.tenders) == 2


def test_post_search_page_size_is_raised_by_evidence(setup):
    client, _ = setup
    spec = bkmk_spec()
    spec["list"]["pagination"] = {"type": "page", "start": 0, "body_path": "PAGE.number"}  # model gave no size field
    seen = []
    store, docs = client.app.state.store, client.app.state.manager.docs

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(bkmk_handler(_rows(150), seen)))
        ctx = JobContext("p", HOST, "r", pc, docs, None, store)
        try:
            await onboard.narrow_to_open(ctx, spec)
        finally:
            await pc.aclose()
    asyncio.run(go())
    assert spec["list"]["pagination"]["size_body_path"] == "PAGE.size" and spec["list"]["pagination"]["size"] == 100
    assert spec["list"]["body"]["WHERE"] == bkmk_spec()["list"]["body"]["WHERE"]     # the portal's filters untouched
    assert [b["PAGE"]["size"] for b in seen] == [25, 100]


def test_validate_spec_reverse():
    bad = bkmk_spec()
    bad["list"]["pagination"]["reverse"] = {"body_path": "ORDER.direction"}
    assert any("pagination.reverse must be" in p for p in onboard.validate_spec(bad))
    cur = simap_spec()
    cur["list"]["pagination"]["reverse"] = {"param": "order", "value": "asc"}
    assert any("page or offset pagination only" in p for p in onboard.validate_spec(cur))


def test_templates_may_use_any_list_field():
    spec = simap_spec()
    spec["fields"]["projectNumber"] = {"path": "projectNumber"}
    spec["fields"]["url"] = {"template": "https://portal/project/{projectNumber}"}
    assert onboard.validate_spec(spec) == []
    spec["fields"]["url"] = {"template": "https://portal/project/{nope}"}
    assert any("'nope' is not a field read from the list" in p for p in onboard.validate_spec(spec))
    spec["fields"]["url"] = {"template": "https://portal/project/"}
    assert any("needs {id}" in p for p in onboard.validate_spec(spec))
    from watchtower.extract import assemble, fill_field_template
    t = assemble({"id": "x1", "projectNumber": "4711"},
                 {"fields": {"id": {"path": "id"}, "projectNumber": {"path": "projectNumber"},
                             "url": {"template": "https://portal/p/{projectNumber}/{id}"}}}, "https://portal/")
    assert t["url"] == "https://portal/p/4711/x1"
    assert fill_field_template("https://portal/{a}/{id}", {"id": "1"}) is None


def test_post_search_with_ignored_page_field_is_not_complete(setup):
    client, _ = setup
    spec = bkmk_spec()
    spec["list"]["pagination"] = {"type": "page", "param": "page", "start": 0}   # query param: the API ignores it
    res = _read(client, bkmk_handler(_rows(45), []), spec)
    assert not res.complete and "repeated earlier results" in res.stats["stopReason"]
    assert "body_path" in onboard._paging_ignored_feedback(spec)


def test_cursor_inside_a_post_body(setup):
    client, _ = setup
    rows = _rows(15)
    bodies = []

    def handler(req):
        body = json.loads(req.content)
        bodies.append(body)
        after = body.get("searchAfter")
        start = next((i + 1 for i, r in enumerate(rows) if r["id"] == after), 0)
        chunk = rows[start:start + 6]
        return httpx.Response(200, json={"hits": chunk, "next": chunk[-1]["id"] if chunk else None})
    spec = simap_spec()
    spec["list"].update(method="POST", body={"q": "", "searchAfter": None}, items="hits",
                        pagination={"type": "cursor", "body_path": "searchAfter", "cursor_path": "next"})
    assert onboard.validate_spec(spec) == []
    res = _read(client, handler, spec)
    assert res.complete and len(res.tenders) == 15
    assert [b["searchAfter"] for b in bodies] == [None, "p005", "p011", "p014"]


# --------------------------------------------------------------------------------------------- spec checks + docs
def test_validate_spec_cursor_and_body_paging():
    ok = simap_spec()
    assert onboard.validate_spec(ok) == []
    no_src = simap_spec()
    del no_src["list"]["pagination"]["cursor_path"]
    assert any("cursor_path / cursor_from_last" in p for p in onboard.validate_spec(no_src))
    bad_path = simap_spec(cursor_path="pagination.[")
    assert any("pagination.cursor_path: invalid JMESPath" in p for p in onboard.validate_spec(bad_path))
    html = simap_spec()
    html["list"]["kind"] = "html"
    html["list"]["items"] = "tr.row"
    html["fields"] = {"id": {"sel": "td.id"}, "url": {"sel": "a", "attr": "href"}}
    assert any("needs a JSON list" in p for p in onboard.validate_spec(html))
    assert onboard.validate_spec(bkmk_spec()) == []
    get_body = bkmk_spec()
    get_body["list"]["method"] = "GET"
    assert any("needs list.method POST" in p for p in onboard.validate_spec(get_body))
    weird = bkmk_spec()
    weird["list"]["pagination"]["body_path"] = "PAGE[0].number"
    assert any("dotted path" in p for p in onboard.validate_spec(weird))
    assert "cursor" in onboard.SPEC_DOC and "body_path" in onboard.SPEC_DOC


def test_detail_template_may_use_other_list_fields(setup):
    client, _ = setup
    rows = [{"id": "proj-1", "pub": "pub-9", "title": "A", "office": "X", "publicationDate": "2026-11-20"},
            {"id": "proj-2", "pub": "pub-7", "title": "B", "office": "X", "publicationDate": "2026-11-20"}]
    deadlines = {("proj-1", "pub-9"): "2030-12-14T23:59:00+01:00", ("proj-2", "pub-7"): "2020-01-01T10:00:00+01:00"}
    opened = []

    def handler(req):
        p = req.url.path
        if p.startswith("/rest/project-search"):
            return httpx.Response(200, json={"projects": rows, "pagination": {"lastItem": ""}})
        _, _, _, proj, _, pub = p.split("/")          # /rest/project/<id>/details/<pub>
        opened.append((proj, pub))
        return httpx.Response(200, json={"dates": {"offerDeadline": deadlines[(proj, pub)]}})
    spec = simap_spec()
    spec["fields"] = {"id": {"path": "id"}, "publicationId": {"path": "pub"}, "title": {"path": "title"},
                      "buyerName": {"path": "office"}, "publicationDate": {"path": "publicationDate"},
                      "submissionDeadline": {"from": "detail", "path": "dates.offerDeadline"},
                      "url": {"template": f"{HOST}/project/{{id}}"}}
    spec["detail"] = {"kind": "json", "url_from": "template",
                      "template": f"{HOST}/rest/project/{{id}}/details/{{publicationId}}"}
    assert onboard.validate_spec(spec, full=True) == []
    res = _read(client, handler, spec)
    assert sorted(opened) == [("proj-1", "pub-9"), ("proj-2", "pub-7")]
    assert [t["id"] for t in res.tenders] == ["proj-1"]                       # the closed one is dropped
    assert res.tenders[0]["_detailUrl"] == f"{HOST}/rest/project/proj-1/details/pub-9"
    bad = json.loads(json.dumps(spec))
    bad["detail"]["template"] = f"{HOST}/rest/project/{{id}}/details/{{nope}}"
    assert any("{nope}" in p for p in onboard.validate_spec(bad, full=True))
    assert runner.fill_row_template("http://x/{id}/{pub}", {"pub": None}, "1") is None


# ------------------------------------------------------------------------- replaying the browser's POST, not a GET
def _capture_with_post_search():
    rows = [{"id": i, "title": f"t{i}", "buyer": "b", "deadline": "2030-01-01"} for i in range(3)]
    body = {"WHERE": [{"fields": ["active"], "operator": "=", "operands": [True]}], "PAGE": {"number": 0, "size": 3}}
    return {"html": "<html><body><div id=app></div></body></html>", "links": [], "error": None,
            "requests": 3, "seconds": 1.0, "types": {},
            "json": [(f"{HOST}/nuts", {"nuts": [{"code": "DE1", "name": "x"}, {"code": "DE2", "name": "y"}]}),
                     (f"{HOST}/bkmk/searches", {"totalElements": 3, "elements": rows})],
            "json_meta": [{"method": "GET", "body": None, "contentType": None},
                          {"method": "POST", "body": json.dumps(body), "contentType": "application/json"}],
            "data_requests": [f"{HOST}/nuts", f"{HOST}/bkmk/searches"], "data_calls": []}, body, rows


def test_locate_replays_a_post_search_with_the_same_body(setup, monkeypatch):
    client, _ = setup
    cap, body, rows = _capture_with_post_search()
    from watchtower import browser

    async def fake_capture(ctx, url, **kw):
        return cap
    monkeypatch.setattr(browser, "capture", fake_capture)
    sent = []

    def handler(req: httpx.Request):
        sent.append((req.method, req.url.path, req.content))
        if req.url.path == "/bkmk/searches":
            if req.method != "POST":
                return httpx.Response(405, json={"error": "Method Not Allowed"})
            return httpx.Response(200, json={"totalElements": 3, "elements": rows})
        return httpx.Response(200, text="<html><body><div id=app></div><script src='/app.js'></script></body></html>",
                              headers={"content-type": "text/html"})
    store, docs = client.app.state.store, client.app.state.manager.docs

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
        ctx = JobContext("p", HOST, "r", pc, docs, None, store)
        try:
            return await onboard.locate_list(ctx, f"{HOST}/ui/search/"), ctx
        finally:
            await pc.aclose()
    found, ctx = asyncio.run(go())
    assert found["api"][0] == f"{HOST}/bkmk/searches"
    assert found["apiRequest"] == {"method": "POST", "body": body}
    posts = [s for s in sent if s[0] == "POST"]
    assert len(posts) == 1 and json.loads(posts[0][2]) == body
    assert any("JSON POST" in n for n in ctx.log), ctx.log
    # the spec keeps the portal's own request even if the model leaves method/body out
    spec = {"list": {"kind": "json", "url": f"{HOST}/bkmk/searches", "items": "elements"}}
    onboard._apply_captured_request(spec, found)
    assert spec["list"]["method"] == "POST" and spec["list"]["body"] == body


def test_captured_endpoints_keep_post_and_get_apart():
    rows = [{"id": 1, "title": "a", "deadline": "x"}, {"id": 2, "title": "b", "deadline": "y"}]
    cap = {"json": [("http://x/s", {"items": rows}), ("http://x/s", {"items": rows})],
           "json_meta": [{"method": "GET", "body": None}, {"method": "POST", "body": "{\"page\": 0}"}]}
    eps = onboard.captured_endpoints(cap)
    assert [e["meta"]["method"] for e in eps] == ["GET", "POST"]
    assert onboard._untried_entry(eps[1]) == {"url": "http://x/s", "meta": {"method": "POST", "body": "{\"page\": 0}"}}
    assert onboard._untried_entry(eps[0]) == "http://x/s"
    # a form post is not replayed as JSON (another mechanism)
    form = {"method": "POST", "body": "a=1&b=2"}
    ctx = SimpleNamespace(client=None, note=lambda m: None)
    assert asyncio.run(onboard.replay_request(ctx, "http://x/s", form)) == (None, None)


# --------------------------------------------------------------- real browser: capture records method and body
@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_browser_capture_records_post_body_and_attach_serves(tmp_path, monkeypatch):
    """End to end on the mock portal (skin spa_post): the app loads its tenders with POST /api/search (GET -> 405).
    The attach must replay the POST, page inside the body, and serve all 12 tenders."""
    import uvicorn
    from fastapi.testclient import TestClient

    import tests.fake_llm as fake_llm
    from mock_portal import app as mock
    from tests.fake_llm import FakeLLM
    from watchtower import browser
    from watchtower.api import create_app
    from watchtower.config import Settings

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(mock.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(fake_llm, "BASE", base)
    mock.CONFIG.update(skin="spa_post", mutations=set())
    try:
        async def cap_only():
            pc = PoliteClient(HostRateLimiter(0.05), "tests", 10)
            try:
                return await browser.capture(SimpleNamespace(client=pc), base + "/")
            finally:
                await pc.aclose()
        cap = asyncio.run(cap_only())
        posts = [(u, m) for (u, _d), m in zip(cap["json"], cap["json_meta"]) if m["method"] == "POST"]
        assert posts and posts[0][0].endswith("/api/search"), cap["json_meta"]
        assert json.loads(posts[0][1]["body"])["page"] == {"number": 0, "size": 5}
        assert any(c["method"] == "POST" for c in cap["data_calls"])

        s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0.05)
        app = create_app(s, llm=FakeLLM)
        with TestClient(app) as client:
            pid = client.post("/portals", json={"url": base + "/"}).json()["id"]
            end = time.time() + 90
            while time.time() < end:
                p = client.get(f"/portals/{pid}").json()
                if p["state"] in ("serving", "failed"):
                    break
                time.sleep(0.2)
            runs = client.get(f"/portals/{pid}/runs").json()
            assert p["state"] == "serving", (p, runs)
            assert p["tenderCount"] == 12
    finally:
        server.should_exit = True
        th.join(timeout=5)
        mock.CONFIG.update(skin="html", mutations=set())
