"""W1 regressions: trustworthy pagination, document checkpoints, pending-detail priority, blocked downloads."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

import httpx
import pytest

from tests.test_contract import setup  # noqa: F401  (fixture)
from watchtower import runner
from watchtower.http import DocumentBlocked, HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext

HOST = "http://p.test"


def tenders_json(ids):
    return [{"id": i, "title": f"Title {i}", "buyer": "City", "pub": "2026-11-01", "dl": "2030-12-09T10:00"}
            for i in ids]


def base_spec(**over):
    spec = {"tz": "UTC",
            "list": {"kind": "json", "url": f"{HOST}/api/list", "items": "items",
                     "pagination": {"type": "page", "param": "page", "start": 1}},
            "fields": {"id": {"path": "id"}, "title": {"path": "title"}, "buyerName": {"path": "buyer"},
                       "publicationDate": {"path": "pub", "formats": ["%Y-%m-%d"]},
                       "submissionDeadline": {"path": "dl", "formats": ["%Y-%m-%dT%H:%M"]},
                       "url": {"template": f"{HOST}/t/{{id}}"}}}
    spec.update(over)
    return spec


def docs_spec():
    return base_spec(documents={"from": "template", "url": f"{HOST}/docs/{{id}}", "kind": "json", "path": "docs"})


def run_read(client, handler, spec, previous=None, limits=None):
    store, docs = client.app.state.store, client.app.state.manager.docs

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
        ctx = JobContext("p", HOST, "r", pc, docs, None, store)
        try:
            return await runner.read(ctx, spec, previous or {}, limits or runner.Limits())
        finally:
            await pc.aclose()
    return asyncio.run(go())


def prev_of(result):
    return {t["id"]: {"data": t, "last_verified_at": None} for t in result.tenders}


# ----------------------------------------------------------------------------------------------- pagination
def test_repeated_page_is_not_complete_and_keeps_live_tender_open(setup):
    client, _ = setup
    hits = []

    def handler(req):
        hits.append(req.url.params.get("page"))
        return httpx.Response(200, json={"items": tenders_json(["A", "B"])})   # page parameter ignored
    previous = {"LIVE": {"data": {"id": "LIVE"}, "last_verified_at": None}}
    res = run_read(client, handler, base_spec(), previous)
    assert not res.complete
    assert res.stats["stopReason"] == "page 2 repeated earlier results"
    assert res.stats["capped"] is False
    assert "LIVE" not in {t["id"] for t in res.tenders}
    assert hits == ["1", "2"]


def test_repeated_page_is_complete_when_stated_total_reached(setup):
    client, _ = setup
    spec = base_spec()
    spec["list"]["total"] = {"path": "total"}
    res = run_read(client, lambda r: httpx.Response(200, json={"items": tenders_json(["A", "B"]), "total": 2}), spec)
    assert res.complete and res.stats["stopReason"] == "reached stated total"


def test_empty_array_ends_but_missing_array_does_not(setup):
    client, _ = setup

    def handler_for(second):
        def handler(req):
            if req.url.params.get("page") == "1":
                return httpx.Response(200, json={"items": tenders_json(["A", "B"])})
            return httpx.Response(200, json=second)
        return handler
    ok = run_read(client, handler_for({"items": []}), base_spec())
    assert ok.complete and ok.stats["stopReason"] == "empty page"
    for bad in ({"error": "x"}, {"items": None}, {"items": "nope"}):
        res = run_read(client, handler_for(bad), base_spec())
        assert not res.complete and not res.stats["capped"]
        assert "no valid items array" in res.stats["stopReason"]
    with pytest.raises(RuntimeError):      # broken first page is breakage, not an empty portal
        run_read(client, lambda r: httpx.Response(200, json={"error": "x"}), base_spec())


def test_short_offset_page_below_stated_total_is_not_complete(setup):
    client, _ = setup
    spec = base_spec()
    spec["list"]["pagination"] = {"type": "offset", "param": "offset", "start": 0, "size": 5}
    spec["list"]["total"] = {"path": "total"}
    res = run_read(client, lambda r: httpx.Response(200, json={"items": tenders_json(["A", "B"]), "total": 40}), spec)
    assert not res.complete and not res.stats["capped"]


# ----------------------------------------------------------------------------------------------- documents
def doc_portal(state):
    files = {"/f/1": b"%PDF one", "/f/2": b"%PDF two"}

    def handler(req):
        path = req.url.path
        if path == "/api/list":
            return httpx.Response(200, json={"items": tenders_json(["A"])})
        if path == "/docs/A":
            state["list_calls"] = state.get("list_calls", 0) + 1
            return httpx.Response(200, json={"docs": [f"{HOST}/f/1", f"{HOST}/f/2"]})
        if path in files:
            state.setdefault("downloads", []).append(path)
            if state.get("fail") == path:
                return httpx.Response(500)
            return httpx.Response(200, content=files[path], headers={"content-type": "application/pdf"})
        return httpx.Response(404)
    return handler, files


def test_documents_are_checkpointed_per_url_under_max_docs(setup):
    client, _ = setup
    state = {}
    handler, files = doc_portal(state)
    lim = runner.Limits(max_docs=2)       # one list request + one download per read
    r1 = run_read(client, handler, docs_spec(), limits=lim)
    t1 = r1.tenders[0]
    assert len(t1["documents"]) == 1 and t1["_docDone"] == [f"{HOST}/f/1"]
    assert r1.stats["pendingDocs"] == 1 and r1.stats["docAttempts"] == 2
    r2 = run_read(client, handler, docs_spec(), prev_of(r1), lim)    # discovered list unchanged
    t2 = r2.tenders[0]
    assert state["list_calls"] == 1                                   # no re-listing
    assert sorted(d["sizeBytes"] for d in t2["documents"]) == sorted(len(v) for v in files.values())
    assert r2.stats["pendingDocs"] == 0
    for d in t2["documents"]:
        body = next(v for v in files.values() if hashlib.sha256(v).hexdigest() == d["sha256"])
        assert d["path"] == f"/files/{d['sha256']}"
        got = client.get(d["path"])
        assert got.status_code == 200 and got.content == body


def test_failed_document_is_retried_and_previous_entries_kept(setup):
    client, _ = setup
    state = {"fail": "/f/2"}
    handler, _files = doc_portal(state)
    r1 = run_read(client, handler, docs_spec())
    t1 = r1.tenders[0]
    assert len(t1["documents"]) == 1 and r1.stats["pendingDocs"] == 1
    assert any(e.startswith("document error") for e in r1.stats["errors"])
    state["fail"] = None
    r2 = run_read(client, handler, docs_spec(), prev_of(r1))
    assert len(r2.tenders[0]["documents"]) == 2 and r2.stats["pendingDocs"] == 0
    assert state["downloads"].count("/f/1") == 1                       # done URLs are not fetched again
    state["fail"] = "/f/1"                                             # nothing to fetch now: untouched
    r3 = run_read(client, handler, docs_spec(), prev_of(r2))
    assert r3.tenders[0]["documents"] == r2.tenders[0]["documents"]


def test_document_list_request_counts_and_is_retried(setup):
    client, _ = setup
    state = {}
    handler, _ = doc_portal(state)

    def two(req):
        if req.url.path == "/api/list":
            return httpx.Response(200, json={"items": tenders_json(["A", "B"])})
        if req.url.path == "/docs/B":
            return httpx.Response(200, json={"docs": []})
        return handler(req)
    r1 = run_read(client, two, docs_spec(), limits=runner.Limits(max_docs=1))
    assert r1.stats["docAttempts"] == 1 and r1.stats["pendingDocLists"] == 1
    b = next(t for t in r1.tenders if t["id"] == "B")
    assert b["_docListPending"]
    r2 = run_read(client, two, docs_spec(), prev_of(r1), runner.Limits(max_docs=1))
    assert r2.stats["pendingDocLists"] == 0


# ----------------------------------------------------------------------------------------------- pending details
def detail_spec():
    spec = base_spec()
    spec["fields"]["referenceNumber"] = {"path": "ref", "from": "detail"}
    spec["detail"] = {"kind": "json", "url_from": "template", "template": f"{HOST}/api/d/{{id}}"}
    return spec


def detail_portal(state):
    def handler(req):
        p = req.url.path
        if p == "/api/list":
            return httpx.Response(200, json={"items": [dict(t, **state.get("row", {}).get(t["id"], {}))
                                                       for t in tenders_json(state["ids"])]})
        if p.startswith("/api/d/"):
            state.setdefault("opened", []).append(p.rsplit("/", 1)[1])
            if p.rsplit("/", 1)[1] in state.get("broken", ()):
                return httpx.Response(500)
            return httpx.Response(200, json={"ref": "R-" + p.rsplit("/", 1)[1]})
        return httpx.Response(404)
    return handler


def test_pending_details_prioritised_cleared_and_sampled_from_listing(setup):
    client, _ = setup
    state = {"ids": ["A", "B", "C"]}
    handler = detail_portal(state)
    r1 = run_read(client, handler, detail_spec(), limits=runner.Limits(max_details=2, sample=1))
    assert state["opened"] == ["A", "B"] and r1.stats["pendingDetails"] == 1     # C deferred
    assert {t["id"]: t["_pendingDetail"] for t in r1.tenders} == {"A": False, "B": False, "C": True}
    prev = prev_of(r1)
    prev["A"]["last_verified_at"] = "2026-01-01"
    prev["B"]["last_verified_at"] = "2026-02-01"
    prev["GONE"] = {"data": {"id": "GONE"}, "last_verified_at": "2000-01-01"}    # not in the listing: never sampled
    state["opened"].clear()
    state["ids"] = ["A", "B", "C", "D"]                                           # D is new
    r2 = run_read(client, handler, detail_spec(), prev, runner.Limits(max_details=2, sample=1))
    assert state["opened"] == ["D", "C"]           # new, then pending: the sample (A) waits
    assert r2.stats["pendingDetails"] == 0
    assert {t["id"]: t["_pendingDetail"] for t in r2.tenders}["C"] is False
    assert sorted(r2.verified_ids) == ["C", "D"]
    state["opened"].clear()
    r3 = run_read(client, handler, detail_spec(), prev_of(r2) | {"GONE": prev["GONE"]},
                  runner.Limits(max_details=3, sample=1))
    assert "GONE" not in state["opened"] and len(state["opened"]) == 1             # nothing new: one sample only


def test_changed_row_deferred_by_cap_stays_pending(setup):
    client, _ = setup
    state = {"ids": ["A", "B"]}
    handler = detail_portal(state)
    r1 = run_read(client, handler, detail_spec(), limits=runner.Limits(sample=0))
    prev = prev_of(r1)
    state["row"] = {"A": {"title": "Changed A"}, "B": {"title": "Changed B"}}
    state["opened"].clear()
    r2 = run_read(client, handler, detail_spec(), prev, runner.Limits(max_details=1, sample=0))
    assert state["opened"] == ["A"]
    assert {t["id"]: t["_pendingDetail"] for t in r2.tenders} == {"A": False, "B": True}
    b = next(t for t in r2.tenders if t["id"] == "B")
    assert b["referenceNumber"] == "R-B"                                           # old detail kept meanwhile
    state["opened"].clear()
    r3 = run_read(client, handler, detail_spec(), prev_of(r2), runner.Limits(max_details=1, sample=0))
    assert state["opened"] == ["B"] and r3.stats["pendingDetails"] == 0


def test_failed_detail_fetch_stays_pending(setup):
    client, _ = setup
    state = {"ids": ["A"], "broken": {"A"}}
    r1 = run_read(client, detail_portal(state), detail_spec(), limits=runner.Limits(sample=0))
    assert r1.tenders[0]["_pendingDetail"] and r1.verified_ids == []
    state["broken"] = set()
    r2 = run_read(client, detail_portal(state), detail_spec(), prev_of(r1), runner.Limits(sample=0))
    assert not r2.tenders[0]["_pendingDetail"] and r2.verified_ids == ["A"]


# ----------------------------------------------------------------------------------------------- blocked downloads
def wall_portal(response_for):
    def handler(req):
        if req.url.path == "/api/list":
            return httpx.Response(200, json={"items": tenders_json(["A"])})
        if req.url.path == "/docs/A":
            return httpx.Response(200, json={"docs": [f"{HOST}/f/1"]})
        return response_for()
    return handler


LOGIN = b"<html><body><form><input type='password' name='p'><button>Log in</button></form></body></html>"
CAPTCHA = b"<html><body><div class='g-recaptcha'></div><p>Please verify you are human</p></body></html>"


@pytest.mark.parametrize("make", [
    lambda: httpx.Response(200, content=LOGIN, headers={"content-type": "text/html; charset=utf-8"}),
    lambda: httpx.Response(200, content=CAPTCHA, headers={"content-type": "text/html"}),
    lambda: httpx.Response(200, content=LOGIN),                      # no content type: sniffed
    lambda: httpx.Response(401, content=b"nope"),
    lambda: httpx.Response(403, content=b"<html>forbidden</html>", headers={"content-type": "text/html"}),
], ids=["login", "captcha", "sniffed", "401", "403"])
def test_wall_is_never_stored_and_is_reported(setup, make):
    client, _ = setup
    files = client.app.state.store.files_dir
    before = set(os.listdir(files))
    res = run_read(client, wall_portal(make), docs_spec())
    t = res.tenders[0]
    assert t["documents"] == [] and res.stats["docBlocked"] == 1 and res.stats["docBlockReason"]
    assert res.stats["pendingDocs"] == 1 and res.stats["docsFetched"] == 0
    assert set(os.listdir(files)) == before                           # nothing stored, no .dl-* left behind
    again = run_read(client, wall_portal(lambda: httpx.Response(200, content=b"%PDF real")), docs_spec(),
                     prev_of(res))
    assert len(again.tenders[0]["documents"]) == 1 and again.stats["docBlocked"] == 0   # retried and cleared


def test_transient_error_and_decorative_captcha_are_not_blocks(setup):
    client, _ = setup
    res = run_read(client, wall_portal(lambda: httpx.Response(503)), docs_spec())
    assert res.stats["docBlocked"] == 0 and res.stats["pendingDocs"] == 1
    assert any(e.startswith("document error") for e in res.stats["errors"])
    big = b"<html><body><script src='https://www.google.com/recaptcha/api.js'></script><p>" + b"word " * 400 + \
          b"</p><input type='password'></body></html>"
    res = run_read(client, wall_portal(lambda: httpx.Response(200, content=big, headers={"content-type": "text/html"})),
                   docs_spec())
    # not a captcha wall (lots of text), but a web page is still not a document file
    assert res.tenders[0]["documents"] == [] and res.stats["docBlocked"] == 1
    assert "web page, not a file" in res.stats["docBlockReason"] and "captcha" not in res.stats["docBlockReason"]


def test_a_web_page_is_not_a_document_and_old_ones_are_dropped(setup):
    """Live, 21 Nov: 3,503 stored 'documents' were the platforms' own HTML pages (e.g. PLACE's index.php)."""
    client, _ = setup
    page = b"<!DOCTYPE html>\n<html lang='fr'><head><title>PLACE - Plate-forme des achats</title></head><body>" + \
           b"<p>" + b"texte " * 300 + b"</p></body></html>"
    res = run_read(client, wall_portal(lambda: httpx.Response(200, content=page, headers={"content-type": "text/html"})),
                   docs_spec())
    t = res.tenders[0]
    assert t["documents"] == [] and "web page, not a file" in t["documentsUnavailable"]
    # a PDF served with a wrong text/html header is still a file
    pdf = run_read(client, wall_portal(lambda: httpx.Response(200, content=b"%PDF-1.7 real",
                                                              headers={"content-type": "text/html"})), docs_spec())
    assert len(pdf.tenders[0]["documents"]) == 1
    # an older version stored the web page as a document: the next read drops it and says why, without a download
    store = client.app.state.store
    sha = "cd" * 32
    with open(store.file_path(sha), "wb") as f:
        f.write(page)
    prev = prev_of(pdf)
    old = dict(prev[t["id"]]["data"])
    url = old["_docUrls"][0]
    old["_docEntries"] = {url: [{"fileName": "index.php", "sizeBytes": len(page), "sha256": sha, "path": f"/files/{sha}"}]}
    old["documents"] = old["_docEntries"][url]
    prev[t["id"]] = {**prev[t["id"]], "data": old}
    again = run_read(client, wall_portal(lambda: httpx.Response(200, content=page, headers={"content-type": "text/html"})),
                     docs_spec(), prev)
    assert again.tenders[0]["documents"] == [] and "web page, not a file" in again.tenders[0]["documentsUnavailable"]


def test_docstore_raises_document_blocked_and_cleans_temp(setup):
    client, _ = setup
    docs = client.app.state.manager.docs

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "t", 5,
                          transport=httpx.MockTransport(lambda r: httpx.Response(200, content=LOGIN)))
        try:
            await docs.fetch(pc, "http://x.test/doc")
        finally:
            await pc.aclose()
    with pytest.raises(DocumentBlocked):
        asyncio.run(go())
    assert not [n for n in os.listdir(client.app.state.store.files_dir) if n.startswith(".dl-")]


def test_tender_with_only_blocked_documents_says_so(setup):
    """Patterno, 20 Nov: deliver the tender without documents and make that recognisable on the tender itself."""
    client, _ = setup
    res = run_read(client, wall_portal(lambda: httpx.Response(403)), docs_spec())
    t = res.tenders[0]
    assert t["documents"] == [] and res.stats["docBlocked"] >= 1
    assert "403" in t["documentsUnavailable"]
    ok = run_read(client, wall_portal(lambda: httpx.Response(200, content=b"%PDF-1.7 x",
                                                             headers={"content-type": "application/pdf"})), docs_spec())
    assert "documentsUnavailable" not in ok.tenders[0]
