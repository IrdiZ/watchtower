"""Change feeds (list kind "feed"): an official ATOM syndication read as a log - bounded backfill on the first load,
only the delta afterwards (high-water mark + conditional GET), newest entry per procedure wins, tenders close only on
an explicit status, a tombstone or their deadline - never because a delta did not mention them. And onboarding:
the locate walk follows a page's <link rel=alternate type=application/atom+xml> when the HTML has no list."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from watchtower import feed, onboard, runner, timeutil, validate
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext
from watchtower.store import Store

HOST = "http://feed.test"
NOW = timeutil.now()
NS = 'xmlns="http://www.w3.org/2005/Atom" xmlns:at="http://purl.org/atompub/tombstones/1.0" ' \
     'xmlns:x="urn:example:tender"'


def ts(hours_ago: float) -> str:
    return timeutil.iso(NOW - timedelta(hours=hours_ago))


def entry(n: int, hours_ago: float, status: str = "PUB", days_left: int = 20, title: str | None = None) -> str:
    dl = (NOW + timedelta(days=days_left)).date().isoformat()
    return (f"<entry><id>{HOST}/syndication/{n}</id><link href=\"{HOST}/tender/{n}\"/>"
            f"<title>{title or f'Tender number {n} for road works'}</title><updated>{ts(hours_ago)}</updated>"
            f"<x:Folder><x:FileID>F-{n}</x:FileID><x:StatusCode>{status}</x:StatusCode>"
            f"<x:Party><x:Name>City {n % 3}</x:Name></x:Party>"
            f"<x:Notice><x:IssueDate>{(NOW - timedelta(days=2)).date().isoformat()}</x:IssueDate></x:Notice>"
            f"<x:Deadline><x:EndDate>{dl}</x:EndDate><x:EndTime>14:00:00</x:EndTime></x:Deadline></x:Folder></entry>")


def feed_xml(entries: list[str], nxt: str | None = None, deleted: list[tuple[str, float]] = ()) -> str:
    links = f'<link rel="self" href="{HOST}/f.atom"/>' + (f'<link rel="next" href="{nxt}"/>' if nxt else "")
    tomb = "".join(f'<at:deleted-entry ref="{HOST}/syndication/{n}" when="{ts(h)}"><at:comment type="ANNULLED"/>'
                   f'</at:deleted-entry>' for n, h in deleted)
    return f'<?xml version="1.0" encoding="UTF-8"?><feed {NS}><title>t</title><updated>{ts(0)}</updated>{links}' \
           f'{tomb}{"".join(entries)}</feed>'


SPEC = {"tz": "Europe/Madrid",
        "list": {"kind": "feed", "url": f"{HOST}/f.atom", "namespaces": {"x": "urn:example:tender"},
                 "pagination": {"type": "feed", "older": "next"}},
        "fields": {"id": {"path": "atom:id", "regex": r"(\d+)$"},
                   "url": {"path": "atom:link/@href"},
                   "title": {"path": "atom:title"},
                   "buyerName": {"path": "x:Folder/x:Party/x:Name"},
                   "publicationDate": {"path": "x:Folder/x:Notice/x:IssueDate"},
                   "submissionDeadline": {"path": ["x:Folder/x:Deadline/x:EndDate", "x:Folder/x:Deadline/x:EndTime"]},
                   "referenceNumber": {"path": "Folder.FileID"},          # dotted local names work too
                   "status": {"path": "x:Folder/x:StatusCode",
                              "map": {"PUB": "open", "EV": "closed", "ADJ": "closed", "ANUL": "withdrawn"}}},
        "detail": {}, "documents": {}}


class Site:
    """Files by path; counts requests; answers 304 to a matching If-None-Match."""
    def __init__(self, files: dict[str, str], etag: str = '"v1"'):
        self.files, self.etag, self.hits = files, etag, []

    def handler(self, req: httpx.Request):
        self.hits.append((req.url.path, req.headers.get("if-none-match")))
        body = self.files.get(req.url.path)
        if body is None:
            return httpx.Response(404, text="gone")
        if req.url.path == "/f.atom" and req.headers.get("if-none-match") == self.etag:
            return httpx.Response(304)
        ctype = "text/html" if req.url.path.endswith((".html", "/")) else "application/atom+xml"
        return httpx.Response(200, text=body, headers={"content-type": ctype, "etag": self.etag})


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path))


def _read(store, site, spec=SPEC, state=None, previous=None, max_pages=None):
    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(site.handler))
        ctx = JobContext("p1", HOST, "r", pc, None, None, store)
        try:
            return await runner.read(ctx, spec, store.get_tenders("p1") if previous is None else previous,
                                     runner.Limits(fetch_docs=False, max_pages=max_pages), state=state)
        finally:
            await pc.aclose()
    return asyncio.run(go())


def _accept(store, res, first=False):
    """What worker._accept does: validate -> decide -> commit snapshot + read state."""
    prev = store.get_tenders("p1")
    v = validate.validate(SPEC, res, prev, mode="attach" if first else "read")
    d = validate.decide(v, res, mode="attach" if first else "read", first=first, serve_capped=True)
    assert d.commit, (v.failures, d.failures)
    counts = store.commit_snapshot("p1", res.tenders, d.close_missing, res.verified_ids)
    store.set_read_state("p1", res.state)
    return d, counts


def chain():
    """Three files, newest first: f.atom (0-2 h), f2 (3-5 h), f3 (old: 40 days)."""
    return {
        "/f.atom": feed_xml([entry(1, 0.5), entry(2, 1, "EV"), entry(3, 2)], nxt=f"{HOST}/f2.atom"),
        "/f2.atom": feed_xml([entry(1, 3, title="Tender number 1 for road works (old title)"), entry(4, 4),
                              entry(5, 5, "ADJ")], nxt=f"{HOST}/f3.atom"),
        "/f3.atom": feed_xml([entry(6, 24 * 40, days_left=60)]),
    }


def test_parse_fields_namespaces_and_dotted_names():
    doc = feed.parse(feed_xml([entry(7, 1)]).encode())
    assert doc.format == "atom" and doc.links["self"].endswith("/f.atom")
    e = doc.entries[0]
    ns = {**doc.ns, **SPEC["list"]["namespaces"]}
    assert feed.entry_value(e, SPEC["fields"]["submissionDeadline"], ns).endswith(" 14:00:00")
    assert feed.entry_value(e, {"path": "Folder.FileID"}, ns) == "F-7"
    assert feed.entry_value(e, {"path": "link@href"}, ns) == f"{HOST}/tender/7"
    assert feed.entry_value(e, SPEC["fields"]["id"], ns) == "7"
    assert feed.status_of(SPEC["fields"]["status"], "pub") == "open" and feed.status_of({"map": {}}, "X") is None
    assert feed.path_problems("x:Nope/[") and not feed.path_problems("atom:title", {})
    assert feed.looks_like_feed("text/xml", b'<?xml version="1.0"?><feed xmlns="x">') and \
        not feed.looks_like_feed("text/html", b"<html>")


def test_backfill_newest_entry_wins_and_closed_by_status(store):
    site = Site(chain())
    res = _read(store, site)
    ids = {t["id"]: t for t in res.tenders}
    # 1 (newest title), 3, 4 open; 2 (EV) and 5 (ADJ) closed and never served; 6 is older than the backfill window
    assert set(ids) == {"1", "3", "4"}
    assert ids["1"]["title"] == "Tender number 1 for road works"
    assert ids["1"]["submissionDeadline"].endswith("14:00:00+01:00") or "T14:00:00" in ids["1"]["submissionDeadline"]
    assert res.complete and res.stats["changeFeed"] and res.stats["feed"]["mode"] == "backfill"
    assert res.stats["feed"]["files"] == 3 and res.stats["closedDropped"] == 2
    assert res.state["hwm"] and res.state["reach"] and not res.state["pending"]
    d, _ = _accept(store, res, first=True)
    assert not d.close_missing
    assert "official change feed" in d.reason(3) and "Coverage: complete" in d.reason(3)


def test_file_cap_continues_backfill_on_next_read(store):
    site = Site(chain())
    spec = json.loads(json.dumps(SPEC))
    spec["list"]["pagination"]["max_files"] = 1
    res = _read(store, site, spec)
    assert not res.complete and res.stats["capped"] and res.state["pending"][0]["next"].endswith("/f2.atom")
    d, _ = _accept(store, res, first=True)
    assert "Coverage: partial" in d.reason(2) and "continues on the next reads" in d.reason(2)
    site.hits.clear()
    res2 = _read(store, site, spec, state=store.get_read_state("p1"))
    # the head is unchanged (304), the pending part is read from where it stopped
    assert site.hits[0] == ("/f.atom", '"v1"') and site.hits[1][0] == "/f2.atom"
    assert {t["id"] for t in res2.tenders} == {"4"}            # 1 is known with a newer entry: the old one is stale
    assert res2.stats["feed"]["stale"] == 1
    _accept(store, res2)
    assert store.get_tenders("p1")["1"]["data"]["title"] == "Tender number 1 for road works"


def test_delta_reads_only_new_changes_and_never_closes_absent_tenders(store):
    site = Site(chain())
    _accept(store, _read(store, site), first=True)
    state = store.get_read_state("p1")
    # nothing changed: one conditional request, an empty delta is accepted, nothing is closed
    site.hits.clear()
    res = _read(store, site, state=state)
    assert site.hits == [("/f.atom", '"v1"')] and res.tenders == [] and res.complete
    _d, counts = _accept(store, res)
    assert counts["closed_missing"] == 0
    assert all(p["data"]["status"] == "open" for p in store.get_tenders("p1").values())
    # new head generation: tender 3 awarded, 7 new, 4 annulled (tombstone); 1 is not mentioned
    files = chain()
    files["/f.atom"] = feed_xml([entry(3, -0.5, "ADJ"), entry(7, -0.4)], nxt=f"{HOST}/fx.atom",
                                deleted=[(4, -0.3)])
    files["/fx.atom"] = feed_xml([entry(1, 0.5), entry(2, 1, "EV"), entry(3, 2)], nxt=f"{HOST}/f2.atom")
    site2 = Site(files, etag='"v2"')
    res = _read(store, site2, state=store.get_read_state("p1"))
    assert [h[0] for h in site2.hits] == ["/f.atom", "/fx.atom"]   # stops at the high-water mark: f2 never read
    by = {t["id"]: t for t in res.tenders}
    assert by["3"]["status"] == "closed" and by["4"]["status"] == "withdrawn" and by["7"]["status"] == "open"
    _d, counts = _accept(store, res)
    served = {t["id"]: t["status"] for t in store.served_tenders("p1")}
    assert served == {"1": "open", "3": "closed", "4": "withdrawn", "7": "open"}
    assert counts["closed_missing"] == 0
    assert store.get_read_state("p1")["hwm"] > state["hwm"]


def test_rejected_read_keeps_the_high_water_mark(store):
    site = Site(chain())
    _accept(store, _read(store, site), first=True)
    before = store.get_read_state("p1")
    files = chain()
    files["/f.atom"] = feed_xml([entry(8, -1)], nxt=f"{HOST}/f2.atom")
    res = _read(store, Site(files, etag='"v3"'), state=before)
    assert res.state["hwm"] > before["hwm"]
    # not accepted -> not stored (the worker writes the state only in _accept after commit)
    assert store.get_read_state("p1") == before


def test_feed_spec_static_checks():
    assert onboard.validate_spec(SPEC) == []
    bad = json.loads(json.dumps(SPEC))
    bad["fields"]["title"] = {"path": "x:Folder/["}
    bad["fields"]["status"]["map"]["PRE"] = "maybe"
    probs = onboard.validate_spec(bad)
    assert any("title" in p and "XPath" in p for p in probs) and any("status.map" in p for p in probs)


class FeedLLM:
    """Classifies the landing page as 'landing' (no list); maps the sample entry like a careful model."""
    def __init__(self):
        self.calls = []

    async def json_call(self, task, system, user, **kw):
        self.calls.append(task)
        if task == "classify":
            return {"pageType": "landing", "links": [], "reason": "a home page"}
        assert task == "synthesize"
        data = json.loads(user)
        assert data["sampleEntries"] and data["namespaces"]["x"] == "urn:example:tender"
        f = {k: v for k, v in SPEC["fields"].items() if k != "status"}
        f["status"] = {"path": "x:Folder/x:StatusCode"}
        return {"tz": "Europe/Madrid", "older": "next", "fields": f,
                "status_map": {"PUB": "open", "EV": "closed", "ADJ": "closed"}}


def test_onboarding_follows_the_linked_feed(store):
    files = chain()
    files["/"] = (f'<html><head><link rel="alternate" type="application/atom+xml" href="{HOST}/f.atom"></head>'
                  f'<body><h1>Portal</h1><p>{"Welcome. " * 50}</p><a href="/about">About</a>'
                  f'<a href="/help">Help</a><a href="/contact">Contact</a></body></html>')
    site = Site(files)
    llm = FeedLLM()

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(site.handler))
        ctx = JobContext("p1", f"{HOST}/", "r", pc, None, llm, store)
        try:
            return await onboard.learn(ctx, f"{HOST}/"), ctx
        finally:
            await pc.aclose()
    spec, ctx = asyncio.run(go())
    assert spec["list"]["kind"] == "feed" and spec["list"]["url"] == f"{HOST}/f.atom"
    assert spec["fields"]["status"]["map"]["PUB"] == "open"
    assert spec["source"]["kind"] == "official feed"
    assert any("offers a feed" in m for m in ctx.log)


def test_onboarding_rejects_unknown_status_codes(store):
    files = chain()
    files["/f.atom"] = feed_xml([entry(1, 0.5), entry(2, 1, "ZZZ")], nxt=f"{HOST}/f2.atom")
    site = Site(files)

    class Forgetful(FeedLLM):
        async def json_call(self, task, system, user, **kw):
            out = await super().json_call(task, system, user, **kw)
            if task == "synthesize" and len([c for c in self.calls if c == "synthesize"]) >= 2:
                assert any("ZZZ" in p for p in json.loads(user)["problemsToFix"])
                out["status_map"]["ZZZ"] = "closed"
            return out
    llm = Forgetful()

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(site.handler))
        ctx = JobContext("p1", f"{HOST}/f.atom", "r", pc, None, llm, store)
        try:
            return await onboard.learn(ctx, f"{HOST}/f.atom")
        finally:
            await pc.aclose()
    spec = asyncio.run(go())
    assert spec["fields"]["status"]["map"]["ZZZ"] == "closed" and llm.calls.count("synthesize") == 2
