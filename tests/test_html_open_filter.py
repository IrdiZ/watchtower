"""Open tenders only, on an HTML list (the eTenders failure of 20 Nov): the first list page shows only closed
tenders. The list page's own search form (and its 'advanced search' page) offers the portal's open/current settings;
onboarding must find them by evidence - the cheap model only proposes - and store them with {today} tokens.
Also: a CSS selector the parser rejects is a named, fixable spec problem (Find a Tender), and a JSON list the model
wrote blind (an open-data API it knows of, BOAMP) is sampled so its deadline field can be mapped."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl

import httpx

from watchtower import onboard, runner, timeutil
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.llm import LLMUnavailable
from watchtower.pipeline import JobContext

HOST = "http://forms.test"
NOW = timeutil.now()


def tenders(n: int = 30):
    """n tenders, oldest deadline first (the portal's default order): the first 12 are closed."""
    out = []
    for i in range(n):
        days = -30 + i if i < 12 else i - 7
        dl = (NOW + timedelta(days=days)).astimezone(timezone.utc).replace(hour=12, minute=0, second=0)
        out.append({"id": f"T{100 + i}", "title": f"Tender {i}", "deadline": dl,
                    "status": "ev" if days < 0 else "ts"})
    return out


def page_html(rows, total, form_method="post", page=1, size=10):
    body = "".join(f"<tr class='row'><td><a href='/t/{r['id']}'>{r['title']}</a></td><td>{r['id']}</td>"
                   f"<td>{r['deadline']:%d/%m/%Y %H:%M}</td></tr>" for r in rows[(page - 1) * size:page * size])
    return (f"<html><body><a href='/advanced'>Advanced search</a>"
            f"<form method='{form_method}' action='/search'><input type='hidden' name='status' value=''>"
            f"<input type='hidden' name='submissionFromDate' value=''><input type='hidden' name='type' value='cft'>"
            f"</form><p class='total'>{total} results</p>"
            f"<table><tr><th>Title</th><th>ID</th><th>Deadline</th></tr>{body}</table></body></html>")


ADVANCED = ("<html><body><form method='{m}' action='/search'><select name='status'><option value=''>All</option>"
            "<option value='ts'>Tender Submission</option><option value='ev'>Evaluation</option></select>"
            "<input type='text' name='submissionFromDate' placeholder='dd/mm/yyyy'>"
            "<input type='hidden' name='type' value='cft'><input type='submit' name='go' value='Search'>"
            "</form></body></html>")


def make_handler(hits: list, post_only: bool = False, newest_first: bool = False):
    rows = tenders()[::-1] if newest_first else tenders()
    method = "post" if post_only else "get"

    def handler(req: httpx.Request):
        q = dict(req.url.params)
        if req.method == "POST":
            q.update(dict(parse_qsl(req.content.decode(), keep_blank_values=True)))
        hits.append((req.method, req.url.path, q))
        if req.url.path == "/advanced":
            return httpx.Response(200, text=ADVANCED.format(m=method))
        if req.url.path == "/list":
            return httpx.Response(200, text=page_html(rows, len(rows), method, int(q.get("p", "1"))))
        if req.url.path == "/search":
            if post_only and req.method != "POST":
                return httpx.Response(405, text="method not allowed")
            sel = rows
            if q.get("status"):
                sel = [r for r in sel if r["status"] == q["status"]]
            if q.get("submissionFromDate"):
                d = datetime.strptime(q["submissionFromDate"], "%d/%m/%Y").date()
                sel = [r for r in sel if r["deadline"].date() >= d]
            return httpx.Response(200, text=page_html(sel, len(sel), method, int(q.get("p", "1"))))
        return httpx.Response(404, text="no")
    return handler


def spec_for():
    return {"tz": "UTC",
            "list": {"kind": "html", "url": f"{HOST}/list", "items": "tr.row",
                     "pagination": {"type": "page", "param": "p", "start": 1},
                     "total": {"sel": "p.total", "regex": "(\\d+)"}},
            "fields": {"id": {"column": "ID"}, "url": {"sel": "a", "attr": "href"}, "title": {"sel": "a"},
                       "submissionDeadline": {"column": "Deadline", "formats": ["%d/%m/%Y %H:%M"]}}}


class FormLLM:
    """Proposes a status that is not open (must be rejected) and the portal's own open setting."""
    def __init__(self):
        self.calls = 0

    async def json_call(self, task, system, user, **kw):
        assert task == "narrow" and "forms" in user
        self.calls += 1
        return {"candidates": [{"form": 1, "values": {"status": "ev"}, "why": "wrong"},
                               {"form": 1, "values": {"status": "ts", "submissionFromDate": "{today:%d/%m/%Y}"},
                                "why": "Tender Submission, deadline from today"}]}


class NoModel:
    async def json_call(self, *a, **k):
        raise LLMUnavailable("no model")


def _ctx(handler, llm):
    pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
    return JobContext("p", HOST, "r", pc, None, llm, None), pc


def _narrow_and_read(handler, llm):
    async def go():
        ctx, pc = _ctx(handler, llm)
        spec = spec_for()
        try:
            changed = await onboard.narrow_html_to_open(ctx, spec)
            res = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            return changed, spec, res, ctx.log
        finally:
            await pc.aclose()
    return asyncio.run(go())


def test_search_form_open_setting_found_by_evidence():
    hits = []
    llm = FormLLM()
    changed, spec, res, log = _narrow_and_read(make_handler(hits), llm)
    assert changed and llm.calls == 1
    lst = spec["list"]
    assert lst["url"] == f"{HOST}/search" and lst["method"] == "GET"
    assert lst["params"]["submissionFromDate"] == "{today:%d/%m/%Y}"          # a token, filled on every read
    assert lst["params"].get("status") != "ev"                                 # not open -> rejected
    assert "30 -> 18" in lst["openFilter"]["evidence"]
    today = runner.today_local(spec)
    assert any(q.get("submissionFromDate") == f"{today[8:10]}/{today[5:7]}/{today[:4]}" for _m, _p, q in hits)
    assert res.complete and len(res.tenders) == 18 and res.stats["closedDropped"] == 0
    assert any("advanced" in p for _m, p, _q in hits)       # the list page's form had no choices: its search page


def test_post_only_search_form_is_submitted_as_a_form():
    hits = []
    changed, spec, res, _log = _narrow_and_read(make_handler(hits, post_only=True), FormLLM())
    assert changed
    lst = spec["list"]
    assert lst["method"] == "POST" and lst["form"]["submissionFromDate"] == "{today:%d/%m/%Y}"
    assert not onboard.validate_spec(spec)
    assert res.complete and len(res.tenders) == 18 and res.stats["pages"] >= 2
    assert all(m == "POST" for m, p, _q in hits[-res.stats["pages"]:])


def test_closed_tenders_at_the_end_of_an_open_first_page_are_found():
    """eTenders 21 Nov: page 1 (newest first) is all open, the closed tenders sit on the last pages, so a read walks
    into the archive. The last page is the evidence; the filtered list's last page must then be open."""
    hits = []
    changed, spec, res, log = _narrow_and_read(make_handler(hits, newest_first=True), FormLLM())
    assert changed and any("last list page shows" in m for m in log)
    assert "{today" in str(spec["list"]["params"]) and len(res.tenders) == 18
    assert res.complete and res.stats["closedDropped"] == 0


def test_static_deadline_from_candidate_without_a_model():
    changed, spec, res, _log = _narrow_and_read(make_handler([]), NoModel())
    assert changed and "{today" in str(spec["list"]["params"])
    assert len(res.tenders) == 18 and res.stats["closedDropped"] == 0


def test_open_page_one_needs_no_narrowing():
    rows = [r for r in tenders() if r["status"] == "ts"]

    def handler(req):
        return httpx.Response(200, text=page_html(rows, len(rows)))

    class NoCalls:
        async def json_call(self, *a, **k):
            raise AssertionError("no model call when page 1 is all open")
    changed, spec, _res, _log = _narrow_and_read(handler, NoCalls())
    assert not changed and spec["list"]["url"] == f"{HOST}/list"


def test_today_token_with_format():
    spec = {"tz": "UTC"}
    today = runner.today_local(spec)
    assert runner.fill_tokens("{today:%d/%m/%Y}", spec) == f"{today[8:10]}/{today[5:7]}/{today[:4]}"
    assert runner.fill_tokens(["{today}", "x"], spec) == [today, "x"]
    assert runner._with_params("http://x/s", {"st": ["a", "b"], "q": "1"}) == "http://x/s?st=a&st=b&q=1"


# ------------------------------------------------------------------ unparseable selectors are named problems
def test_unparseable_total_selector_is_a_named_problem():
    spec = spec_for()
    spec["list"]["total"] = {"sel": ".app-search__summary .govuk-!-font-weight-bold"}
    probs = " | ".join(onboard.validate_spec(spec))
    assert "list.total.sel: invalid CSS selector '.app-search__summary .govuk-!-font-weight-bold'" in probs
    assert "cannot be parsed" in probs and "[class~=" in probs
    spec["list"]["total"] = {"sel": ".app-search__summary .govuk-\\!-font-weight-bold"}
    assert not onboard.validate_spec(spec)
    # harmless sloppiness is repaired instead of costing an attempt: unescaped or over-escaped class names
    for bad in (".app-search__summary span.govuk-!-font-weight-bold", ".app-search__summary .govuk-\\\\!-font-weight-bold"):
        spec["list"]["total"] = {"sel": bad}
        onboard.coerce_spec(spec)
        assert spec["list"]["total"]["sel"].endswith('[class~="govuk-!-font-weight-bold"]'), spec["list"]["total"]
        assert not onboard.validate_spec(spec)
    spec["list"]["total"] = {"sel": "div > p.total"}
    onboard.coerce_spec(spec)
    assert spec["list"]["total"]["sel"] == "div > p.total"


def test_run_time_parse_error_becomes_named_feedback():
    spec = spec_for()
    spec["fields"]["title"] = {"sel": "a:contains('x')"}
    msg = onboard._run_error_problem(Exception("Can't parse CSS selector."), spec)
    assert msg.startswith("spec problem: field title: invalid CSS selector") and "could not be run" not in msg
    generic = onboard._run_error_problem(Exception("Can't parse CSS selector."), None)
    assert "cannot be parsed" in generic
    assert onboard._run_error_problem(ValueError("boom"), spec).startswith("spec could not be run")
    bad = {"block_markers": ["form#login", "div[[x"]}
    onboard.coerce_spec(bad)
    assert bad["block_markers"] == ["form#login"]             # a marker that can never match is dropped


# ------------------------------------------------------------------ open-data APIs: sampled, deadline mappable
def test_open_data_records_candidates_from_seen_requests():
    seen = ["https://data.example.org/api/datasets/1.0/notices/?extrametas=true",
            "https://www.portal.test/api/datasets/1.0//?extrametas=true",
            "https://data.example.org/api/records/1.0/search/?dataset=notices&rows=10",
            "https://data.example.org/api/datasets/1.0/departements/?x=1",
            "https://other.test/api/v1/tenders"]
    assert onboard.open_data_records_candidates(seen) == [
        "https://data.example.org/api/explore/v2.1/catalog/datasets/notices/records?limit=20",
        "https://data.example.org/api/explore/v2.1/catalog/datasets/departements/records?limit=20"]


def test_blind_json_list_is_sampled_and_unmapped_deadline_named():
    data = {"total_count": 2, "results": [
        {"idweb": "26-1", "objet": "A", "dateparution": "2026-11-20", "datelimitereponse": "2026-12-21T11:00:00+01:00",
         "donnees": "x" * 5000},
        {"idweb": "26-2", "objet": "B", "dateparution": "2026-11-19", "datelimitereponse": "2026-12-03T12:00:00+01:00"}]}

    def handler(req):
        return httpx.Response(200, json=data)
    spec = {"tz": "Europe/Paris",
            "list": {"kind": "json", "url": "http://ods.test/api/explore/v2.1/catalog/datasets/x/records",
                     "items": "results", "pagination": {"type": "none"}},
            "fields": {"id": {"path": "idweb"}, "url": {"template": "http://ods.test/{id}"},
                       "title": {"path": "objet"}}}
    found = {"url": "http://ods.test/search", "html": "<html></html>", "api": None}

    async def go():
        ctx, pc = _ctx(handler, None)
        try:
            return await onboard._see_blind_json_list(ctx, found, spec), \
                await onboard._see_blind_json_list(ctx, found, spec)
        finally:
            await pc.aclose()
    first, again = asyncio.run(go())
    assert first and not again and found["api"][1] == data
    msg = onboard._never_found(spec, "submissionDeadline", [{"id": "26-1"}], found)[0]
    assert "submissionDeadline is not mapped" in msg and "datelimitereponse='2026-12-21T11:00:00+01:00'" in msg
    assert len(onboard._trim_long_strings(data)["results"][0]["donnees"]) < 500


def test_unmapped_deadline_points_at_what_the_list_shows():
    spec = {"list": {"kind": "html"}, "fields": {"id": {"sel": "a"}}}
    found = {"html": "<div class='item'><h2><a href='/n/1'>Lab system</a></h2><dl><dt>Procurement identifier:</dt>"
                     "<dd>ocds-1</dd><dt>Submission deadline date:</dt><dd>18 December 2026</dd></dl></div>"}
    msg = onboard._never_found(spec, "submissionDeadline", [{"id": "1"}], found)[0]
    assert "Submission deadline date: 18 December 2026" in msg and "WITHOUT a time" in msg
    found["html"] = found["html"].replace("18 December 2026", "18 December 2026, 1:00pm")
    assert "map it from the list" in onboard._never_found(spec, "submissionDeadline", [{"id": "1"}], found)[0]
