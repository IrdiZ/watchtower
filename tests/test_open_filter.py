"""Open tenders only, on an archive-style JSON API (the TenderNed failure of 20 Nov):
the feed mixes open calls with awards (no deadline) and corrections of closed calls (deadline before publication),
and refuses pages beyond a window. Closed rows must never reach validate(), and onboarding must find an open-only
filter and a bigger page size by evidence, not by trusting the model."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from watchtower import onboard, runner, timeutil, validate
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext, ReadResult

HOST = "http://archive.test"
TODAY = timeutil.now().date()


def _iso(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat() + "T12:00:00"


def archive(n: int = 60):
    """n notices, newest first: every 3rd an open call, every 3rd an award, every 3rd a late correction."""
    rows = []
    for i in range(n):
        kind = ("AAO", "AGO", "REC")[i % 3]
        dl = {"AAO": _iso(10 + i), "AGO": None, "REC": _iso(-40)}[kind]
        rows.append({"publicatieId": str(1000 + i), "naam": f"Tender {i}", "koper": "Gemeente X",
                     "publicatieDatum": TODAY.isoformat(), "sluitingsDatum": dl, "type": kind})
    return rows


def make_handler(hits: list, window_pages: int = 100, max_size: int = 100, n: int = 60):
    rows = archive(n)

    def handler(req: httpx.Request):
        q = req.url.params
        hits.append(dict(q))
        page, size = int(q.get("page", "0")), int(q.get("size", "20"))
        if page > window_pages - 1 or size > max_size:
            return httpx.Response(400, text="[PARAMETER] must be less than or equal")
        sel = rows
        if q.get("sluitingsDatumVanaf"):          # the real filter; unknown params are ignored, like TenderNed
            sel = [r for r in sel if r["sluitingsDatum"] and r["sluitingsDatum"][:10] >= q["sluitingsDatumVanaf"]]
        return httpx.Response(200, json={"content": sel[page * size:(page + 1) * size],
                                         "totalElements": len(sel)})
    return handler


def spec_for():
    return {"tz": "UTC",
            "list": {"kind": "json", "url": f"{HOST}/papi/publicaties?page=0&size=20", "items": "content",
                     "pagination": {"type": "page", "param": "page", "start": 0},
                     "total": {"path": "totalElements"}},
            "fields": {"id": {"path": "publicatieId"}, "title": {"path": "naam"}, "buyerName": {"path": "koper"},
                       "publicationDate": {"path": "publicatieDatum"},
                       "submissionDeadline": {"path": "sluitingsDatum"},
                       "url": {"template": f"{HOST}/aankondigingen/{{id}}"}}}


class ProposingLLM:
    """Proposes one wrong guess (ignored by the API) and the real filter."""
    def __init__(self):
        self.calls = 0

    async def json_call(self, task, system, user, **kw):
        assert task == "narrow"
        self.calls += 1
        return {"candidates": [{"params": {"typePublicatie": "AAO"}, "why": "wrong name"},
                               {"params": {"sluitingsDatumVanaf": "{today}"}, "why": "deadline from today"}]}


def _ctx(client, handler, llm=None):
    store, docs = client.app.state.store, client.app.state.manager.docs
    pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
    return JobContext("p", HOST, "r", pc, docs, llm, store), pc


def test_closed_rows_never_reach_validate(setup):
    client, _ = setup
    hits = []

    async def go():
        ctx, pc = _ctx(client, make_handler(hits))
        try:
            return await runner.read(ctx, spec_for(), {}, runner.Limits(fetch_docs=False))
        finally:
            await pc.aclose()
    res = asyncio.run(go())
    assert res.complete
    ids = {t["id"] for t in res.tenders}
    assert len(res.tenders) == 40                       # 20 open calls + 20 undated awards; 20 late corrections gone
    assert all((int(i) - 1000) % 3 != 2 for i in ids)
    assert res.stats["closedDropped"] == 20 and res.stats["rows"] == 60
    verdict = validate.validate(spec_for(), res, {}, mode="attach")
    assert not any("implausible" in f for f in verdict.failures), verdict.failures
    assert not any("stated" in f for f in verdict.failures)       # 60 listed == 60 stated, not 40
    d = validate.decide(verdict, res, mode="attach", first=True)
    assert d.commit and any("past their deadline" in n for n in d.notes)


def test_implausible_dates_compare_parsed_values():
    spec = spec_for()

    def verdict_for(pub, dl):
        t = {"id": "1", "url": "u", "title": "t", "publicationDate": pub, "submissionDeadline": dl}
        return validate.validate(spec, ReadResult([t], True, {"rows": 1}), {}, mode="attach")
    ok = verdict_for(TODAY.isoformat(), _iso(5) + "+01:00")
    assert not any("implausible" in f for f in ok.failures)
    future_pub = verdict_for((TODAY + timedelta(days=30)).isoformat(), _iso(40) + "+00:00")
    assert any("published in the future" in f for f in future_pub.failures)
    bad_year = verdict_for(TODAY.isoformat(), "2207-01-01T10:00:00+00:00")
    assert any("deadline year 2207" in f for f in bad_year.failures)


def test_today_token_is_filled_on_every_read():
    spec = {"tz": "Europe/Amsterdam", "list": {"url": "http://x/api?from={today}", "params": {"d": "{today}"}}}
    assert runner.fill_tokens(spec["list"]["url"], spec) == f"http://x/api?from={runner.today_local(spec)}"
    assert runner.fill_tokens(5, spec) == 5


def test_onboarding_finds_open_filter_and_page_size_by_evidence(setup):
    client, _ = setup
    hits = []
    llm = ProposingLLM()

    async def go():
        ctx, pc = _ctx(client, make_handler(hits, window_pages=2, n=300), llm)  # 40 rows reachable at size 20
        spec = spec_for()
        try:
            await onboard._narrow_safely(ctx, spec)
            res = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            return spec, res, ctx.log
        finally:
            await pc.aclose()
    spec, res, log = asyncio.run(go())
    assert spec["list"]["params"]["sluitingsDatumVanaf"] == "{today}"          # stored as a token, not a date
    assert "typePublicatie" not in spec["list"]["params"]                      # ignored by the API -> rejected
    assert spec["list"]["params"]["size"] == 100
    assert "300 -> 100" in spec["list"]["openFilter"]["evidence"]
    assert any(h.get("sluitingsDatumVanaf") == TODAY.isoformat() for h in hits)  # filled at request time
    assert res.complete and len(res.tenders) == 100 and res.stats["closedDropped"] == 0
    assert llm.calls == 1


def test_narrowing_leaves_an_open_only_api_alone(setup):
    client, _ = setup
    hits = []
    open_rows = [r for r in archive() if r["type"] == "AAO"]

    def handler(req):
        hits.append(dict(req.url.params))
        return httpx.Response(200, json={"content": open_rows[:20], "totalElements": len(open_rows)})

    class NoCalls:
        async def json_call(self, *a, **k):
            raise AssertionError("no model call needed when page 1 is all open")

    async def go():
        ctx, pc = _ctx(client, handler, NoCalls())
        spec = spec_for()
        spec["list"]["pagination"] = {"type": "none"}
        try:
            await onboard._narrow_safely(ctx, spec)
            return spec
        finally:
            await pc.aclose()
    spec = asyncio.run(go())
    assert "openFilter" not in spec["list"] and not spec["list"].get("params")
    assert len(hits) == 1


def test_open_filter_found_when_spec_maps_no_total(setup):
    """The 20 Nov TenderNed spec had no list.total: the filter must still be found (total guessed from the body)."""
    client, _ = setup
    hits = []

    async def go():
        ctx, pc = _ctx(client, make_handler(hits, window_pages=2, n=300), ProposingLLM())
        spec = spec_for()
        spec["list"].pop("total")
        try:
            await onboard._narrow_safely(ctx, spec)
            res = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            return spec, res
        finally:
            await pc.aclose()
    spec, res = asyncio.run(go())
    assert spec["list"]["params"]["sluitingsDatumVanaf"] == "{today}"
    assert "total" not in spec["list"]          # a guessed total is never stored (it may be a per-page count)
    assert res.complete and len(res.tenders) == 100


def test_publication_filter_loses_without_a_total(setup):
    """Review round 2: with no total anywhere, 'published from today' also makes page 1 all open. It must lose,
    because open rows from page 1 (published earlier) are gone from its page 1."""
    client, _ = setup
    rows = [{"publicatieId": str(500 - i), "naam": f"T{i}", "koper": "K",
             "publicatieDatum": (TODAY - timedelta(days=i // 5)).isoformat(),
             "sluitingsDatum": (_iso(20) if i % 4 else _iso(-30)), "type": "AAO"} for i in range(200)]

    def handler(req):
        q = req.url.params
        sel = rows
        if q.get("sluitingsDatumVanaf"):
            sel = [r for r in sel if r["sluitingsDatum"][:10] >= q["sluitingsDatumVanaf"]]
        if q.get("publicatieDatumVanaf"):
            sel = [r for r in sel if r["publicatieDatum"] >= q["publicatieDatumVanaf"]]
        size, page = int(q.get("size", "20")), int(q.get("page", "0"))
        return httpx.Response(200, json={"content": sel[page * size:(page + 1) * size]})   # no total at all

    class PubFirst:
        async def json_call(self, task, system, user, **kw):
            return {"candidates": [{"params": {"publicatieDatumVanaf": "{today}"}},
                                   {"params": {"sluitingsDatumVanaf": "{today}"}}]}

    async def go():
        ctx, pc = _ctx(client, handler, PubFirst())
        spec = spec_for()
        spec["list"].pop("total")
        try:
            await onboard._narrow_safely(ctx, spec)
            return spec
        finally:
            await pc.aclose()
    spec = asyncio.run(go())
    assert spec["list"]["openFilter"]["params"] == {"sluitingsDatumVanaf": "{today}"}


def test_filter_keeping_most_rows_wins_over_a_subset(setup):
    """Review finding 1: a type+deadline filter also passes 'all rows open' but drops open calls of another type
    (a prior-information notice used as a call). The candidate keeping the most rows must win."""
    client, _ = setup
    rows = [{"publicatieId": str(i), "naam": f"T{i}", "koper": "K", "publicatieDatum": TODAY.isoformat(),
             "sluitingsDatum": (_iso(5 + i) if i % 3 != 2 else None), "type": ("AAO", "VAK", "AGO")[i % 3]}
            for i in range(90)]

    def handler(req):
        q = req.url.params
        sel = rows
        if q.get("type"):
            sel = [r for r in sel if r["type"] == q["type"]]
        if q.get("sluitingsDatumVanaf"):
            sel = [r for r in sel if r["sluitingsDatum"] and r["sluitingsDatum"][:10] >= q["sluitingsDatumVanaf"]]
        size, page = int(q.get("size", "20")), int(q.get("page", "0"))
        return httpx.Response(200, json={"content": sel[page * size:(page + 1) * size], "totalElements": len(sel)})

    class SubsetFirst:
        async def json_call(self, task, system, user, **kw):
            return {"candidates": [{"params": {"type": "AAO", "sluitingsDatumVanaf": "{today}"}},
                                   {"params": {"sluitingsDatumVanaf": "{today}"}}]}

    async def go():
        ctx, pc = _ctx(client, handler, SubsetFirst())
        spec = spec_for()
        try:
            await onboard._narrow_safely(ctx, spec)
            return spec
        finally:
            await pc.aclose()
    spec = asyncio.run(go())
    assert spec["list"]["openFilter"]["params"] == {"sluitingsDatumVanaf": "{today}"}
    assert "type" not in spec["list"]["params"]
    assert spec["list"]["openFilter"]["undatedOnPage1"] > 0          # honest about what the filter leaves out


def test_all_closed_listing_with_stated_total_is_not_a_break():
    """Review finding 4: every listed row is past its deadline -> a warning, not 'portal states N' breakage."""
    past = (timeutil.now() - timedelta(days=3)).isoformat()
    previous = {"X": {"data": {"id": "X", "submissionDeadline": past, "status": "open"}, "last_verified_at": None}}
    res = ReadResult([], True, {"rows": 5, "closedDropped": 5, "statedTotal": 5})
    v = validate.validate(spec_for(), res, previous, mode="read")
    assert v.ok, v.failures
    assert any("past their deadline" in w for w in v.warnings)


# ------------------------------------------------------------------------- documents of a JSON API (TenderNed shape)
def _docs_handler(hits, docs_html=False):
    base = make_handler(hits, n=60)

    def handler(req: httpx.Request):
        path = req.url.path
        m = __import__("re").match(r"^/papi/publicaties/(\d+)/(\w+)$", path)
        if m:
            hits.append({"path": path})
            if m.group(2) != "documenten":
                return httpx.Response(404, text="not found")
            if docs_html:
                return httpx.Response(200, text="<html>app</html>", headers={"content-type": "text/html"})
            tid = m.group(1)
            return httpx.Response(200, json={"documenten": [
                {"documentNaam": f"Bestek {tid}", "grootte": 12,
                 "links": {"download": {"href": f"/papi/publicaties/{tid}/documenten/1/content", "title": "download"}}},
                {"documentNaam": f"Bijlage {tid}", "grootte": 9,
                 "links": {"download": {"href": f"/papi/publicaties/{tid}/documenten/2/content", "title": "download"}}}]})
        if path.endswith("/content"):
            hits.append({"path": path})
            return httpx.Response(200, content=b"%PDF-1.7 " + path.encode(), headers={"content-type": "application/pdf"})
        return base(req)
    return handler


def test_doc_link_path_finds_download_links():
    data = {"documenten": [{"documentNaam": "a", "links": {"download": {"href": "/x/1/content"}, "self": {"href": "/x/1"}}},
                           {"documentNaam": "b", "links": {"download": {"href": "/x/2/content"}, "self": {"href": "/x/2"}}}]}
    path, links = onboard.doc_link_path(data)
    assert path == "documenten[].links.download.href" and links == ["/x/1/content", "/x/2/content"]
    assert onboard.doc_link_path([{"name": "a.pdf", "url": "https://h/a.pdf"}]) == ("[].url", ["https://h/a.pdf"])
    assert onboard.doc_link_path({"documenten": []}) == ("", [])
    assert onboard.doc_link_path({"menu": [{"label": "Home"}]}) is None


def test_documents_endpoint_discovered_and_read(setup):
    client, _ = setup
    hits = []

    async def go():
        ctx, pc = _ctx(client, _docs_handler(hits))
        spec = spec_for()
        try:
            first = await runner.read(ctx, spec, {}, runner.Limits(max_pages=1, fetch_docs=False))
            await onboard._discover_docs_safely(ctx, spec, first.tenders)
            res = await runner.read(ctx, spec, {}, runner.Limits(max_pages=1))   # list requests count as doc attempts
            return spec, res, ctx.log
        finally:
            await pc.aclose()
    spec, res, log = asyncio.run(go())
    d = spec["documents"]
    assert d["from"] == "template" and d["url"] == f"{HOST}/papi/publicaties/{{id}}/documenten"
    assert d["path"] == "documenten[].links.download.href"
    with_docs = [t for t in res.tenders if t["documents"]]
    assert with_docs and with_docs[0]["documents"][0]["path"].startswith("/files/")
    assert res.stats["docsFetched"] >= 2 and not [e for e in res.stats["errors"] if e.startswith("document")]


def test_documents_discovery_rejects_html_and_leaves_spec_alone(setup):
    client, _ = setup
    hits = []

    async def go():
        ctx, pc = _ctx(client, _docs_handler(hits, docs_html=True))
        spec = spec_for()
        try:
            first = await runner.read(ctx, spec, {}, runner.Limits(max_pages=1, fetch_docs=False))
            await onboard._discover_docs_safely(ctx, spec, first.tenders)
            return spec, ctx.log
        finally:
            await pc.aclose()
    spec, log = asyncio.run(go())
    assert not spec.get("documents")
    assert any("no per-tender documents address" in l for l in log)
    assert len([h for h in hits if "path" in h]) <= onboard.DOC_MAX_PROBES


def test_notices_of_one_procedure_become_one_tender(setup):
    """Audit 20 Nov: TenderNed lists the original, its corrections and the termination as separate notices sharing
    one procedure number. Keyed on that number they must become ONE tender: newest notice's values, and a
    termination sticks even if an older notice comes later in the listing."""
    client, _ = setup
    rows = [  # newest first, like the portal
        {"proc": "P1", "naam": "Bestek v3 (rectificatie)", "pub": TODAY.isoformat(), "dl": _iso(30), "type": "REC"},
        {"proc": "P2", "naam": "Other tender", "pub": TODAY.isoformat(), "dl": _iso(12), "type": "AAO"},
        {"proc": "P1", "naam": "Bestek v1", "pub": (TODAY - timedelta(days=9)).isoformat(), "dl": _iso(20), "type": "AAO"},
        {"proc": "P2", "naam": "Other tender", "pub": (TODAY - timedelta(days=3)).isoformat(), "dl": _iso(12),
         "type": "VBE"},
    ]

    def handler(req):
        page = int(req.url.params.get("page", "0"))
        return httpx.Response(200, json={"content": rows if page == 0 else [], "totalElements": len(rows)})
    spec = spec_for()
    spec["fields"].update({"id": {"path": "proc"}, "title": {"path": "naam"}, "buyerName": {"path": "naam"},
                           "publicationDate": {"path": "pub"}, "submissionDeadline": {"path": "dl"},
                           "withdrawn": {"path": "type", "regex": "(VBE)"}})
    spec["totalReliable"] = True

    async def go():
        ctx, pc = _ctx(client, handler)
        try:
            return await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
        finally:
            await pc.aclose()
    res = asyncio.run(go())
    by = {t["id"]: t for t in res.tenders}
    assert sorted(by) == ["P1", "P2"]
    assert by["P1"]["title"] == "Bestek v3 (rectificatie)" and by["P1"]["status"] == "open"
    assert by["P2"]["status"] == "withdrawn"                    # older termination notice still wins the status
    assert res.complete and res.stats["rows"] == 4 and res.stats["mergedNotices"] == 2
    assert res.stats["closedDropped"] == 0                     # merged notices are not "closed"
    v = validate.validate(spec, res, {}, mode="attach")
    assert not any("states" in f for f in v.failures), v.failures   # 4 notices listed == 4 stated


def test_documents_per_notice_when_tender_is_a_procedure(setup):
    """TenderNed after the procedure merge: the tender id is the procedure number, documents hang off each notice
    id. Discovery must find {noticeId}; the tender's documents are the union over its merged notices."""
    client, _ = setup
    hits = []
    rows = [{"publicatieId": "9002", "kenmerk": "P1", "naam": "Bestek (rectificatie)", "koper": "K",
             "publicatieDatum": TODAY.isoformat(), "sluitingsDatum": _iso(30)},
            {"publicatieId": "9001", "kenmerk": "P1", "naam": "Bestek", "koper": "K",
             "publicatieDatum": (TODAY - timedelta(days=5)).isoformat(), "sluitingsDatum": _iso(30)}]

    def handler(req):
        path = req.url.path
        hits.append(path)
        m = __import__("re").match(r"^/papi/publicaties/(\w+)/documenten$", path)
        if m:
            if m.group(1) not in ("9001", "9002"):
                return httpx.Response(404, text="nope")
            n = m.group(1)
            return httpx.Response(200, json={"documenten": [
                {"links": {"download": {"href": f"/papi/publicaties/{n}/documenten/1/content"}}}]})
        if path.endswith("/content"):
            return httpx.Response(200, content=b"%PDF " + path.encode(), headers={"content-type": "application/pdf"})
        page = int(req.url.params.get("page", "0"))
        return httpx.Response(200, json={"content": rows if page == 0 else []})
    spec = spec_for()
    spec["list"].pop("total")
    spec["fields"]["id"] = {"path": "kenmerk"}

    async def go():
        ctx, pc = _ctx(client, handler)
        try:
            first = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            await onboard._discover_docs_safely(ctx, spec, first.tenders)
            res = await runner.read(ctx, spec, {}, runner.Limits())
            return res
        finally:
            await pc.aclose()
    res = asyncio.run(go())
    assert spec["documents"]["url"] == f"{HOST}/papi/publicaties/{{noticeId}}/documenten"
    assert spec["fields"]["noticeId"] == {"from": "list", "path": "publicatieId"}
    [t] = res.tenders
    assert t["id"] == "P1" and t["title"] == "Bestek (rectificatie)"
    assert len(t["documents"]) == 2                              # files of both notices


def test_flag_parameters_survive_in_list_urls():
    """Atexo (marches-publics, maximilien): '&AllCons' without a value IS the list; dropping it shows an empty form."""
    u = "https://x.fr/?page=Entreprise.EntrepriseAdvancedSearch&AllCons"
    assert runner._with_params(u, {}) == u
    assert "AllCons=" in runner._with_params(u, {"PageNumber": 2})


def test_documents_budget_leaves_room_for_downloads(setup):
    """Listing files may use at most half of max_docs: a read must also download."""
    client, _ = setup
    hits = []

    async def go():
        ctx, pc = _ctx(client, _docs_handler(hits))
        spec = spec_for()
        try:
            first = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            await onboard._discover_docs_safely(ctx, spec, first.tenders)
            return await runner.read(ctx, spec, {}, runner.Limits(max_docs=10))
        finally:
            await pc.aclose()
    res = asyncio.run(go())
    assert res.stats["docAttempts"] == 10 and res.stats["docsFetched"] == 5      # 5 lists, 5 downloads
    assert res.stats["pendingDocLists"] > 0


def test_complete_looking_read_far_below_stated_total_fails_on_attach():
    """Atexo, 20 Nov: 10 rows read, 'single page', portal states 1,640 -> must not become 'serving'."""
    ts = [{"id": str(i), "url": f"u{i}", "title": "t", "publicationDate": TODAY.isoformat(),
           "submissionDeadline": _iso(9) + "+00:00"} for i in range(10)]
    res = ReadResult(ts, True, {"rows": 10, "statedTotal": 1640, "stopReason": "single page"})
    v = validate.validate(spec_for(), res, {}, mode="attach")
    assert not v.ok and any("portal states 1640" in f for f in v.failures)
    v_read = validate.validate(spec_for(), res, {}, mode="read")       # routine reads keep the old warning
    assert not any("portal states" in f for f in v_read.failures)
