"""A JSON list without the deadline (simap-shaped, 20 Nov): the deadline is only in a per-tender JSON record at an
address the list does not give (another API version, a second id). Onboarding must find that record by evidence;
reads must never serve a notice as open before its record shows an open deadline when the listing also carries
closed notices, and must not reopen records of notices already known to be closed."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from watchtower import onboard, runner, timeutil, validate
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext

HOST = "http://cantons.test"
NOW = timeutil.now()


def _dt(days: int) -> str:
    return (NOW + timedelta(days=days)).replace(microsecond=0).isoformat()


def projects(n: int = 30):
    """Newest first; a third open calls, a third awards (no deadline), a third calls past their deadline."""
    rows = []
    for i in range(n):
        kind = ("award", "tender", "tender")[i % 3]
        dl = None if kind == "award" else (_dt(12 + i) if i % 3 == 1 else _dt(-20 - i))
        rows.append({"id": f"p{i:03d}", "publicationId": f"n{i:03d}", "pubType": kind, "title": f"Project {i}",
                     "publicationDate": NOW.date().isoformat(), "_deadline": dl})
    return rows


def make_handler(hits: list, n: int = 30):
    rows = projects(n)
    by_id = {r["id"]: r for r in rows}

    def handler(req: httpx.Request):
        hits.append(str(req.url))
        path = req.url.path
        if path == "/rest/pub/v2/project/project-search":
            last = req.url.params.get("lastItem")
            start = 0 if not last else next(i for i, r in enumerate(rows) if r["id"] == last) + 1
            page = rows[start:start + 10]
            out = [{k: v for k, v in r.items() if k != "_deadline"} for r in page]
            return httpx.Response(200, json={"projects": out, "pagination": {
                "lastItem": page[-1]["id"] if len(page) == 10 and start + 10 < len(rows) else None}})
        parts = path.strip("/").split("/")
        # /rest/pub/v1/project/{id}/publication-details/{publicationId}
        if len(parts) == 7 and parts[:4] == ["rest", "pub", "v1", "project"] and parts[5] == "publication-details":
            r = by_id.get(parts[4])
            if r and r["publicationId"] == parts[6]:
                body = {"id": r["publicationId"], "projectId": r["id"], "base": {"title": r["title"]},
                        "dates": {"publicationDate": r["publicationDate"] + "T08:00:00+01:00"}}
                if r["_deadline"]:
                    body["dates"]["offerDeadline"] = r["_deadline"]
                    body["dates"]["questionsUntil"] = _dt(-60)
                return httpx.Response(200, json=body)
        return httpx.Response(404, json={"error": "not found"})
    return handler


def list_spec():
    return {"tz": "Europe/Zurich",
            "list": {"kind": "json", "url": f"{HOST}/rest/pub/v2/project/project-search?lang=de", "items": "projects",
                     "pagination": {"type": "cursor", "param": "lastItem", "cursor_path": "pagination.lastItem"}},
            "fields": {"id": {"path": "id"}, "title": {"path": "title"},
                       "publicationDate": {"path": "publicationDate"},
                       "url": {"template": f"{HOST}/project/{{id}}"}},
            "detail": {"kind": "html", "url_from": "url"}}


class DetailLLM:
    """Proposes a wrong address first, then labels the offer deadline among the record's dates."""
    def __init__(self, propose=True):
        self.calls = []
        self.propose = propose

    async def json_call(self, task, system, user, **kw):
        self.calls.append(task)
        assert task == "detail_api"
        if "URL templates" in system:
            return {"templates": [f"{HOST}/rest/pub/v2/project/{{id}}/details"] if self.propose else [],
                    "rows": [1, 4]}
        return {"path": "dates.offerDeadline" if "dates.offerDeadline" in user else None, "why": "offers"}


def _ctx(client, handler, llm=None):
    store, docs = client.app.state.store, client.app.state.manager.docs
    pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
    return JobContext("p", HOST, "r", pc, docs, llm, store), pc


def test_static_candidates_include_other_version_and_second_id():
    c = onboard._detail_candidates_static(f"{HOST}/rest/pub/v2/project/project-search?lang=de",
                                          {"id": "x", "publicationId": "y", "projectNumber": "1"})
    assert f"{HOST}/rest/pub/v1/project/{{id}}/publication-details/{{publicationId}}" in c
    assert f"{HOST}/rest/pub/v2/project/{{id}}" in c
    # the spec reads the tender id from another field: the row's own 'id' becomes a list field of its own
    fields = {"id": {"path": "projectNumber"}}
    t = onboard._spec_template(f"{HOST}/p/{{id}}/n/{{publicationId}}/{{projectNumber}}", fields)
    assert t == f"{HOST}/p/{{recordId}}/n/{{publicationId}}/{{id}}"
    assert fields["recordId"] == {"from": "list", "path": "id"}
    assert fields["publicationId"] == {"from": "list", "path": "publicationId"}


def test_api_sample_keeps_pagination_block_visible():
    data = {"projects": [{"t": "x" * 900, "i": i} for i in range(20)], "pagination": {"lastItem": "20261120|1"}}
    s = onboard._api_sample(data, 12000)
    assert len(s) < 12200 and '"lastItem": "20261120|1"' in s and "has 20 rows" in s


def test_repeated_query_params_survive_paging():
    u = runner._with_params("http://h.test/s?lang=de&lang=fr&x=1", {"lastItem": "a|1", "x": "2"})
    assert u == "http://h.test/s?lang=de&lang=fr&x=2&lastItem=a%7C1"


def test_discovers_json_record_by_evidence(setup):  # noqa: F811
    client = setup[0]
    hits: list = []
    llm = DetailLLM()
    ctx, pc = _ctx(client, make_handler(hits), llm)
    spec = list_spec()
    got = asyncio.run(onboard.discover_detail_json(ctx, spec))
    asyncio.run(pc.aclose())
    assert got and got["prefer"] == ["p001"]
    assert spec["detail"]["kind"] == "json" and spec["detail"]["url_from"] == "template"
    assert spec["detail"]["template"] == f"{HOST}/rest/pub/v1/project/{{id}}/publication-details/{{publicationId}}"
    assert spec["fields"]["submissionDeadline"] == {"from": "detail", "path": "dates.offerDeadline"}
    assert spec["fields"]["publicationId"] == {"from": "list", "path": "publicationId"}
    assert "mixesClosed" in spec["list"]          # rows 0..3 include an award (no deadline) and a closed call
    assert not onboard.validate_spec(spec)
    assert len(hits) < 45


def test_no_endpoint_no_change(setup):  # noqa: F811
    client = setup[0]
    hits: list = []

    def handler(req):
        if req.url.path.endswith("project-search"):
            return make_handler(hits)(req)
        hits.append(str(req.url))
        return httpx.Response(404)
    ctx, pc = _ctx(client, handler, DetailLLM(propose=False))
    spec = list_spec()
    before = dict(spec["fields"])
    assert asyncio.run(onboard.discover_detail_json(ctx, spec)) is None
    asyncio.run(pc.aclose())
    assert spec["fields"] == before and spec["detail"]["kind"] == "html"


def discovered_spec():
    s = list_spec()
    s["fields"]["publicationId"] = {"from": "list", "path": "publicationId"}
    s["fields"]["submissionDeadline"] = {"from": "detail", "path": "dates.offerDeadline"}
    s["detail"] = {"kind": "json", "url_from": "template",
                   "template": f"{HOST}/rest/pub/v1/project/{{id}}/publication-details/{{publicationId}}"}
    s["list"]["mixesClosed"] = "evidence"
    return s


def test_mixed_listing_serves_only_confirmed_open_and_remembers_closed(setup):  # noqa: F811
    client = setup[0]
    hits: list = []
    ctx, pc = _ctx(client, make_handler(hits))
    spec = discovered_spec()
    lim = runner.Limits(max_details=12, max_docs=0, fetch_docs=False)
    r1 = asyncio.run(runner.read(ctx, spec, {}, lim))
    st = r1.stats
    # 12 records read: 4 open calls, 4 awards, 4 closed calls; 18 not read yet are NOT served
    assert sorted(t["id"] for t in r1.tenders) == ["p001", "p004", "p007", "p010"]
    assert all(t["submissionDeadline"] for t in r1.tenders)
    assert st["heldBack"] == 18 and st["noDeadline"] == 4 and st["closedDropped"] == 4
    assert st["pendingDetails"] == 18
    assert len(r1.closed) == 8
    d = validate.decide(validate.Verdict(ok=True), r1, mode="read", first=False)
    text = " ".join(d.notes)
    assert "Coverage: partial" in text and "18 listed notices are not served yet" in text
    assert "Details for 18 tenders still pending" in text and "Coverage: complete" not in text

    # next read: the 8 known-closed records are not opened again; the next 12 unseen ones are
    prev = {t["id"]: {"data": t, "last_verified_at": None} for t in r1.tenders}
    hits.clear()
    r2 = asyncio.run(runner.read(ctx, spec, prev, lim, known_closed=r1.closed))
    opened = [h for h in hits if "publication-details" in h]
    assert not any(f"/{tid}/" in h for h in opened for tid in r1.closed)
    assert r2.stats["knownClosedSkipped"] == 8
    assert r2.stats["heldBack"] == 6
    assert all(t["id"] not in r1.closed for t in r2.tenders)

    # a changed list row (a new notice of a closed project) opens its record again
    changed = dict(r1.closed)
    some = next(iter(changed))
    changed[some] = {**changed[some], "sig": "different"}
    r3 = asyncio.run(runner.read(ctx, spec, prev, lim, known_closed=changed))
    asyncio.run(pc.aclose())
    assert some in r3.forget_closed


def test_open_only_listing_keeps_serving_pending(setup):  # noqa: F811
    """Without evidence of closed notices in the listing (the portal's own current view), a tender whose page is
    not read yet is served as before."""
    client = setup[0]
    hits: list = []
    ctx, pc = _ctx(client, make_handler(hits))
    spec = discovered_spec()
    del spec["list"]["mixesClosed"]
    r = asyncio.run(runner.read(ctx, spec, {}, runner.Limits(max_details=3, max_docs=0, fetch_docs=False)))
    asyncio.run(pc.aclose())
    assert r.stats["heldBack"] == 0 and len(r.tenders) > 20


def test_known_closed_persisted_after_accept(setup):  # noqa: F811
    client = setup[0]
    store = client.app.state.store
    store.create_portal("p_kc", "http://x.test", "x.test")
    store.remember_closed("p_kc", {"a": {"sig": "s", "deadline": None, "why": "no deadline on its tender page"}})
    assert store.known_closed("p_kc")["a"]["sig"] == "s"
    store.remember_closed("p_kc", {}, forget=["a"])
    assert store.known_closed("p_kc") == {}
    store.remember_closed("p_kc", {"b": {"sig": "s", "deadline": _dt(-1), "why": "deadline passed"}})
    store.delete_portal("p_kc")
    assert store.known_closed("p_kc") == {}
