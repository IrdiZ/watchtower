"""Second hop (Find a Tender shaped, 21 Nov): the list shows the deadline's date only, the tender page shows no
deadline at all, and the time is on the full NOTICE page the tender page links to (list row -> tender page -> notice
page). detail.follow declares the link; fields "from": "detail2" are read there, only when needed and within the
detail budget (pages, not tenders). When the notice page refuses automated requests, the portal's open-data record of
the notice (an address built from the notice number) is the second hop."""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from watchtower import onboard, runner
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext

HOST = "http://hop.test"
N = 4


def _notice_no(i: int) -> str:
    return f"11037{i}-2026"


def make_handler(hits: list, notice_status: int = 200):
    def handler(req: httpx.Request):
        hits.append(str(req.url))
        p = req.url.path
        if p == "/search":
            rows = "".join(
                f"<div class='item'><h2 class='t'><a href='{HOST}/procurement/ocds-{i}'>Tender {i} services</a></h2>"
                f"<dl><dt>Contracting authority name:</dt><dd>Buyer {i}</dd>"
                f"<dt>Procurement identifier:</dt><dd>ocds-{i}</dd>"
                f"<dt>Submission deadline date:</dt><dd>2{i} December 2026</dd>"
                f"<dt>Publication date:</dt><dd>20 November 2026</dd></dl></div>" for i in range(N))
            return httpx.Response(200, html=f"<html><body><h1>Opportunities</h1><div class='results'>{rows}</div>"
                                            f"<p>We've found {N} result(s)</p></body></html>")
        m = re.fullmatch(r"/procurement/ocds-(\d)", p)
        if m:
            i = int(m.group(1))
            return httpx.Response(200, html=(
                f"<html><body><main><h1>Tender {i} services</h1><p>Buyer {i}</p>"
                f"<p>Procurement identifier (OCID): ocds-{i}</p><h2>Notices</h2>"
                f"<div class='entry'><h3><a href='{HOST}/Notice/{_notice_no(i)}'>UK4: Tender notice</a></h3>"
                f"<dl><dt>Published</dt><dd>20 November 2026, 9:42pm</dd></dl></div>"
                f"<div class='entry'><h3><a href='{HOST}/Notice/0178{i}-2025'>UK2: Preliminary market engagement "
                f"notice</a></h3></div></main><footer><a href='{HOST}/privacy-notice'>Privacy notice</a>"
                f"<a href='{HOST}/Developer/Documentation'>Data and API</a></footer></body></html>"))
        m = re.fullmatch(r"/Notice/(\d+-\d+)", p)
        if m:
            if notice_status != 200:
                return httpx.Response(notice_status, html="<html><body><h1>403 Forbidden</h1></body></html>")
            i = int(m.group(1)[5]) if m.group(1).startswith("11037") else 9
            return httpx.Response(200, html=(
                f"<html><body><h1>Tender {i} services</h1><h2>Submission</h2><dl>"
                f"<dt>Enquiry deadline</dt><dd>1{i} December 2026, 5:00pm</dd>"
                f"<dt>Tender submission deadline</dt><dd>2{i} December 2026, 3:00pm</dd></dl></body></html>"))
        m = re.fullmatch(r"/api/notices/(\d+-\d+)", p)
        if m:
            i = int(m.group(1)[5])
            return httpx.Response(200, json={"releases": [{"id": m.group(1), "date": "2026-11-20T21:42:17Z",
                                                           "parties": [{"name": f"Buyer {i}"}],
                                                           "tender": {"title": f"Tender {i} services",
                                                                      "enquiryPeriod": {"endDate": f"2026-12-1{i}T17:00:00+00:00"},
                                                                      "tenderPeriod": {"endDate": f"2026-12-2{i}T15:00:00+00:00"}}}]})
        if p == "/Developer/Documentation":
            return httpx.Response(200, html=(
                f"<html><body><h1>Data and API documentation</h1>"
                f"<a href='{HOST}/apidocumentation/GET-notices'>OCDS release package API</a>"
                f"<a href='{HOST}/Home/Cookies'>Cookies</a></body></html>"))
        return httpx.Response(404, html="<p>not found</p>")
    return handler


def _ctx(client, handler, llm=None):
    store, docs = client.app.state.store, client.app.state.manager.docs
    pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
    return JobContext("p", HOST, "r", pc, docs, llm, store), pc


def list_spec():
    return {"tz": "Europe/London",
            "list": {"kind": "html", "url": f"{HOST}/search", "items": "div.item"},
            "fields": {"id": {"from": "list", "label": "Procurement identifier:"},
                       "url": {"from": "list", "sel": "h2.t a", "attr": "href"},
                       "title": {"from": "list", "sel": "h2.t a"},
                       "buyerName": {"from": "list", "label": "Contracting authority name:"},
                       "publicationDate": {"from": "list", "label": "Publication date:", "formats": ["%d %B %Y"]}},
            "detail": {"kind": "html", "url_from": "url"}}


def hop_spec():
    s = list_spec()
    s["detail"]["follow"] = {"sel": "a[href*='/Notice/']", "attr": "href", "kind": "html"}
    s["fields"]["submissionDeadline"] = {"from": "detail2", "label": "Tender submission deadline",
                                         "formats": ["%d %B %Y, %I:%M%p"]}
    return s


def test_runner_reads_deadline_from_linked_page(setup):  # noqa: F811
    hits: list = []
    ctx, pc = _ctx(setup[0], make_handler(hits))
    r = asyncio.run(runner.read(ctx, hop_spec(), {}, runner.Limits(max_details=20, fetch_docs=False)))
    asyncio.run(pc.aclose())
    by = {t["id"]: t for t in r.tenders}
    assert by["ocds-1"]["submissionDeadline"] == "2026-12-21T15:00:00+00:00"     # 3:00pm London (GMT)
    assert all(t["submissionDeadline"] for t in r.tenders) and len(r.tenders) == N
    assert r.stats["followFetched"] == N and r.stats["detailsFetched"] == 2 * N
    # the newest notice (first link), never the preliminary one
    assert sum("/Notice/0178" in h for h in hits) == 0
    assert by["ocds-2"]["_followUrl"] == f"{HOST}/Notice/{_notice_no(2)}"


def test_linked_pages_count_against_the_detail_budget(setup):  # noqa: F811
    hits: list = []
    ctx, pc = _ctx(setup[0], make_handler(hits))
    r1 = asyncio.run(runner.read(ctx, hop_spec(), {}, runner.Limits(max_details=4, fetch_docs=False)))
    assert r1.stats["detailsFetched"] == 4 and r1.stats["pendingDetails"] == N - 2
    assert sum(1 for t in r1.tenders if t["submissionDeadline"]) == 2
    # the next read fills in the rest and keeps what was known (no second hop for tenders already read)
    prev = {t["id"]: {"data": t, "last_verified_at": "x"} for t in r1.tenders}
    hits.clear()
    r2 = asyncio.run(runner.read(ctx, hop_spec(), prev, runner.Limits(max_details=4, sample=0, fetch_docs=False)))
    asyncio.run(pc.aclose())
    assert all(t["submissionDeadline"] for t in r2.tenders), [t["submissionDeadline"] for t in r2.tenders]
    assert r2.stats["pendingDetails"] == 0


def test_no_second_hop_without_detail2_fields(setup):  # noqa: F811
    hits: list = []
    ctx, pc = _ctx(setup[0], make_handler(hits))
    s = hop_spec()
    s["fields"]["submissionDeadline"] = {"from": "detail", "label": "Published"}
    asyncio.run(runner.read(ctx, s, {}, runner.Limits(max_details=20, fetch_docs=False)))
    asyncio.run(pc.aclose())
    assert not any("/Notice/" in h for h in hits)


def test_refused_linked_page_stays_pending(setup):  # noqa: F811
    hits: list = []
    ctx, pc = _ctx(setup[0], make_handler(hits, notice_status=403))
    r = asyncio.run(runner.read(ctx, hop_spec(), {}, runner.Limits(max_details=20, fetch_docs=False)))
    asyncio.run(pc.aclose())
    assert r.stats["followFetched"] == N and not r.verified_ids
    assert all(t["_pendingDetail"] for t in r.tenders) and "linked page HTTP 403" in r.stats["errors"]


def test_open_data_template_hop(setup):  # noqa: F811
    hits: list = []
    ctx, pc = _ctx(setup[0], make_handler(hits, notice_status=403))
    s = list_spec()
    s["detail"]["follow"] = {"sel": "a[href*='/Notice/']", "attr": "href", "regex": r"/Notice/([^/?#]+)",
                             "template": f"{HOST}/api/notices/{{value}}", "kind": "json"}
    s["fields"]["submissionDeadline"] = {"from": "detail2", "path": "releases[0].tender.tenderPeriod.endDate"}
    assert not onboard.validate_spec(s, full=True)
    r = asyncio.run(runner.read(ctx, s, {}, runner.Limits(max_details=20, fetch_docs=False)))
    asyncio.run(pc.aclose())
    assert {t["id"]: t["submissionDeadline"] for t in r.tenders}["ocds-3"] == "2026-12-23T15:00:00+00:00"
    assert not any("/Notice/" in h for h in hits)


def test_validate_spec_checks_follow():
    s = list_spec()
    s["fields"]["submissionDeadline"] = {"from": "detail2", "label": "Deadline"}
    assert any("detail.follow" in p for p in onboard.validate_spec(s))
    s["detail"]["follow"] = {"sel": "a[href*='/Notice/']", "template": "https://x.test/api/notice"}
    assert any("{value}" in p for p in onboard.validate_spec(s))
    s["detail"]["follow"] = {"sel": "a[[bad"}
    assert any("detail.follow" in p for p in onboard.validate_spec(s))
    assert any("detail.follow.sel" in p for p in onboard.selector_problems(s))


def test_notice_links_and_selector():
    html = make_handler([])(httpx.Request("GET", f"{HOST}/procurement/ocds-1")).text
    links = onboard.notice_links(html, f"{HOST}/procurement/ocds-1")
    assert links[0] == (f"{HOST}/Notice/{_notice_no(1)}", "UK4: Tender notice", "a[href*='/Notice/']")
    assert all("privacy" not in u for u, _, _ in links)


def test_json_digest_puts_dates_first():
    d = {"releases": [{"parties": [{"name": "x" * 50, "n": i} for i in range(200)],
                       "tender": {"tenderPeriod": {"endDate": "2026-12-22T15:00:00+00:00"}}}]}
    out = onboard.json_digest(d, 2000)
    assert out.startswith("releases[0].tender.tenderPeriod.endDate = 2026-12-22T15:00:00+00:00")


class HopLLM:
    """A model that, like the live one on Find a Tender, maps everything but the deadline (the list shows only its
    date and the tender page none); verification copies what the given text shows."""
    def __init__(self, store=None, follow_json=False):
        self.calls, self.verify_texts, self.feedback = [], [], []
        self.follow_json = follow_json
        self.store = store

    async def json_call(self, task, system, user, **kw):
        self.calls.append(task)
        if task == "classify":
            return {"pageType": "list", "links": [], "reason": "rows"}
        if task == "synthesize":
            d = json.loads(user)
            self.feedback.append(d.get("problemsToFix") or [])
            if "detailPage" not in d:
                s = list_spec()
                return {k: s[k] for k in ("tz", "list", "fields", "detail")}
            fb = " ".join(d.get("problemsToFix") or [])
            if self.follow_json and "open data" in fb:
                return {"fields": {"submissionDeadline": {"from": "detail2",
                                                          "path": "releases[0].tender.tenderPeriod.endDate"}},
                        "follow": {"sel": "a[href*='/Notice/']", "regex": r"/Notice/([^/?#]+)",
                                   "template": f"{HOST}/api/notices/{{value}}", "kind": "json"},
                        "documents": {}, "block_markers": []}
            return {"fields": {"description": {"from": "detail", "sel": "main p"}}, "documents": {},
                    "block_markers": []}
        if task == "detail_api":
            return {"path": next((p for p in json.loads(user)["dates"] if "tenderPeriod" in p), None), "why": "x"}
        if task == "verify":
            self.verify_texts.append(user)
            m = re.search(r"Tender submission deadline (\d+ December 2026, [\d:]+pm)", user) or \
                re.search(r"tenderPeriod\.endDate = (\S+)", user)
            t = re.search(r"(Tender \d services)", user)
            return {"id": None, "title": t.group(1) if t else None, "buyerName": None, "publicationDate": None,
                    "submissionDeadline": m.group(1) if m else None, "labels": {}}
        if task == "narrow":
            return {"candidates": []}
        raise AssertionError(task)


def test_learn_finds_deadline_on_linked_notice_page(setup):  # noqa: F811
    """End to end on the mock: no attempt maps the deadline, the second hop finds it on the notice page by evidence,
    the dry run reads it there and blind verification sees the notice page too."""
    hits: list = []
    llm = HopLLM()
    ctx, pc = _ctx(setup[0], make_handler(hits), llm)
    spec = asyncio.run(onboard.learn(ctx, f"{HOST}/search"))
    asyncio.run(pc.aclose())
    assert spec["detail"]["follow"]["sel"] == "a[href*='/Notice/']" and spec["detail"]["follow"]["kind"] == "html"
    assert spec["fields"]["submissionDeadline"]["from"] == "detail2"
    assert spec["fields"]["submissionDeadline"]["label"] == "Tender submission deadline"
    assert spec["verifiedBlind"] is True
    assert llm.verify_texts and all("THE FULL NOTICE THIS PAGE LINKS TO" in t for t in llm.verify_texts)


def test_learn_refused_notice_page_feedback_leads_to_open_data(setup):  # noqa: F811
    """The notice page answers 403: the feedback names the refusal and the portal's open-data pointers; the next
    attempt declares the open-data record of the notice as the second hop, and it is verified there."""
    hits: list = []
    llm = HopLLM(follow_json=True)
    ctx, pc = _ctx(setup[0], make_handler(hits, notice_status=403), llm)
    spec = asyncio.run(onboard.learn(ctx, f"{HOST}/search"))
    asyncio.run(pc.aclose())
    fb = " ".join(" ".join(f) for f in llm.feedback)
    assert "answered HTTP 403" in fb and "OCDS release package API" in fb and "{value}" in fb
    assert spec["detail"]["follow"]["template"] == f"{HOST}/api/notices/{{value}}"
    assert spec["fields"]["submissionDeadline"] == {"from": "detail2",
                                                    "path": "releases[0].tender.tenderPeriod.endDate"}
    assert any("tenderPeriod.endDate = 2026-12-2" in t for t in llm.verify_texts)


class InsistLLM(HopLLM):
    """Always maps the deadline on the notice page (which refuses us): such a spec must never be accepted."""
    async def json_call(self, task, system, user, **kw):
        if task == "synthesize" and "detailPage" in json.loads(user):
            self.calls.append(task)
            return {"fields": {"submissionDeadline": {"from": "detail2", "label": "Tender submission deadline",
                                                      "formats": ["%d %B %Y, %I:%M%p"]}},
                    "follow": {"sel": "a[href*='/Notice/']", "attr": "href"}, "documents": {}, "block_markers": []}
        return await super().json_call(task, system, user, **kw)


def test_refused_linked_page_is_never_accepted(setup):  # noqa: F811
    """Live, 21 Nov: a spec whose deadline sits on a notice page that answers 403 was accepted (no tender page was
    'opened', so nothing was compared). It must fail with the refusal named."""
    import pytest
    from watchtower.pipeline import AttachFailed
    ctx, pc = _ctx(setup[0], make_handler([], notice_status=403), InsistLLM())
    with pytest.raises(AttachFailed) as e:
        asyncio.run(onboard.learn(ctx, f"{HOST}/search"))
    asyncio.run(pc.aclose())
    assert "linked page" in str(e.value) and "403" in str(e.value)
