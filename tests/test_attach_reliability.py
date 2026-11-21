"""First-try attach reliability: deterministic evidence over model choice (pager, deadline label), Patterno rule A
for reads that cannot be completed (a two-ended read without a mirror, an early stop the total does not confirm,
a verified spec whose pager is not followed), and the learn loop keeping its best attempt."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from tests.test_cursor_post import _read, _reversible, _rows, bkmk_handler, bkmk_spec
from watchtower import onboard
from watchtower.pipeline import ReadResult, Verdict
from watchtower.validate import decide, validate

HOST = "http://reliable.test"


# ------------------------------------------------------------------ rule A: capped instead of failed
def test_two_ended_read_without_mirror_serves_the_first_pass_capped(setup):
    client, _ = setup
    spec = _reversible()
    spec["list"]["pagination"]["reverse"] = {"body_path": "ORDER.field", "value": "ignored"}   # same order again
    res = _read(client, bkmk_handler(_rows(38), [], window=20), spec)
    assert not res.complete and res.stats["capped"] is True and len(res.tenders) == 20
    assert "repeated earlier results" in res.stats["stopReason"] and "capped" in res.stats["stopReason"]
    d = decide(validate(spec, res, {}, mode="attach"), res, mode="attach", first=True, serve_capped=True)
    assert d.commit and d.state == "serving" and not d.close_missing
    assert any(n.startswith("Coverage: partial") for n in d.notes)


def _sorted_html(n_open: int, n_closed: int, per: int = 10):
    """A publication-sorted HTML list whose stated total counts many more (closed, old) rows than are read."""
    now = datetime.now(timezone.utc)
    rows = [((now + timedelta(days=30)).strftime("%d.%m.%Y %H:%M"), (now - timedelta(days=5)).strftime("%d.%m.%Y"))
            for _ in range(n_open)]
    rows += [((now - timedelta(days=400)).strftime("%d.%m.%Y %H:%M"), (now - timedelta(days=500)).strftime("%d.%m.%Y"))
             for _ in range(n_closed)]
    total = 10 * len(rows)

    def handler(req: httpx.Request):
        p = int(req.url.params.get("p") or 1)
        chunk = rows[(p - 1) * per:p * per]
        trs = "".join(f"<tr><td><a href='/t/{(p - 1) * per + i}'>T{(p - 1) * per + i}</a></td><td>Buyer</td>"
                      f"<td>{pub}</td><td>{dl}</td></tr>" for i, (dl, pub) in enumerate(chunk))
        return httpx.Response(200, text=f"<html><body><p class=total>{total} results</p><table><thead><tr><th>Title"
                                        f"</th><th>Buyer</th><th>Published</th><th>Deadline</th></tr></thead>"
                                        f"<tbody>{trs}</tbody></table></body></html>")
    spec = {"tz": "UTC",
            "list": {"kind": "html", "url": f"{HOST}/list", "order": "published_desc", "items": "tbody > tr",
                     "pagination": {"type": "page", "param": "p", "start": 1},
                     "total": {"sel": "p.total", "regex": r"(\d+)"}},
            "fields": {"id": {"sel": "a", "attr": "href", "regex": r"/t/(\d+)"}, "url": {"sel": "a", "attr": "href"},
                       "title": {"sel": "a"}, "buyerName": {"column": "Buyer"},
                       "publicationDate": {"column": "Published", "formats": ["%d.%m.%Y"]},
                       "submissionDeadline": {"column": "Deadline", "formats": ["%d.%m.%Y %H:%M"]}}}
    return spec, handler


def test_sorted_early_stop_the_total_does_not_confirm_is_capped_not_complete(setup):
    client, _ = setup
    spec, handler = _sorted_html(15, 10)
    res = _read(client, handler, spec)
    assert not res.complete and res.stats["capped"] is True and "not confirmed" in res.stats["stopReason"]
    assert len(res.tenders) == 15
    v = validate(spec, res, {}, mode="attach")
    assert v.ok, v.failures              # no "read N as if complete but the portal states M"
    d = decide(v, res, mode="attach", first=True, serve_capped=True)
    assert d.commit and not d.close_missing


def test_partial_listing_spec_is_always_capped(setup):
    client, _ = setup
    spec = bkmk_spec()
    spec["list"]["pagination"] = {"type": "none"}
    spec["list"]["partial"] = "the pager is not followed"
    res = _read(client, bkmk_handler(_rows(23), []), spec)
    assert not res.complete and res.stats["capped"] is True and "partial listing" in res.stats["stopReason"]


def test_coverage_only_problems():
    assert onboard._coverage_only(["the portal states 1091 results but the spec read 20 and stopped (single page): x",
                                   "hint: the list page loads more rows"])
    assert not onboard._coverage_only(["the portal states 9 results but the spec read 2 and stopped (single page)",
                                       "submissionDeadline is never found"])
    assert not onboard._coverage_only(["the portal states 12 results but the spec read 10 and stopped (the pager "
                                       "ended at page 2 but the portal states 12 (read 10))"])   # breakage
    assert not onboard._coverage_only([])


# ------------------------------------------------------------------ deterministic pager evidence
_PB_PAGE = ("<form id=f method=post action='/l'><input type=hidden name=__VIEWSTATE value=s1>"
            "<input type=hidden name=__EVENTTARGET><input type=hidden name=__EVENTARGUMENT>"
            "<table><tbody><tr><td>row</td></tr></tbody></table>"
            "<div class=pager><a href=\"javascript:__doPostBack('grid','Page$2')\" class=next>Next</a></div></form>")


def test_static_pager_sets_postback_when_the_spec_cannot_follow_the_pager():
    for pag in ({"type": "none"}, {"type": "next_link", "next_sel": "a.next"}, {"type": "postback", "next_sel": "a.x"}):
        spec = {"list": {"kind": "html", "url": f"{HOST}/l", "items": "tbody > tr", "pagination": dict(pag)}}
        note = onboard.static_pager(spec, _PB_PAGE, f"{HOST}/l")
        assert note and spec["list"]["pagination"] == {"type": "postback", "next_sel": "a.next"}, pag


def test_static_pager_keeps_a_working_or_parameter_pager():
    for pag in ({"type": "postback", "next_sel": "a.next"}, {"type": "page", "param": "p"},
                {"type": "load_more", "url": "/more"}):
        spec = {"list": {"kind": "html", "url": f"{HOST}/l", "items": "tbody > tr", "pagination": dict(pag)}}
        assert onboard.static_pager(spec, _PB_PAGE, f"{HOST}/l") is None and spec["list"]["pagination"] == pag


def test_static_pager_real_next_link_and_no_stray_next():
    page = ("<table><tbody><tr><td>r</td></tr></tbody></table>"
            "<nav class=pagination><a class=nxt href='/l?page=2' rel=next>Next</a></nav>")
    spec = {"list": {"kind": "html", "url": f"{HOST}/l", "items": "tbody > tr", "pagination": {"type": "none"}}}
    assert onboard.static_pager(spec, page) and spec["list"]["pagination"]["type"] == "next_link"
    stray = "<table><tbody><tr><td>r</td></tr></tbody></table><div class=carousel><button>Next</button></div>"
    spec = {"list": {"kind": "html", "url": f"{HOST}/l", "items": "tbody > tr", "pagination": {"type": "none"}}}
    assert onboard.static_pager(spec, stray) is None and spec["list"]["pagination"] == {"type": "none"}


# ------------------------------------------------------------------ deterministic deadline evidence
def test_deadline_label_found_on_the_tender_page():
    page = ("<dl><dt>Enquiry deadline</dt><dd>1 December 2026, 5:00pm</dd>"
            "<dt>Submission deadline</dt><dd>22 December 2026, 12:00pm</dd></dl>")
    got = onboard.deadline_from_page(page, "Europe/London")
    assert got["label"] == "Submission deadline" and got["formats"] == ["%d %B %Y, %I:%M%p"]
    de = ("<table><tr><td>Fragefrist</td><td>01.12.2026 10:00 Ortszeit</td></tr>"
          "<tr><td>Abgabefrist</td><td>10.12.2026 09:00 Ortszeit</td></tr></table>")
    assert onboard.deadline_from_page(de, "Europe/Berlin")["label"] == "Abgabefrist"
    assert onboard.deadline_from_page("<p>Deadline: 10 December 2026</p>", "UTC") is None    # no time: no deadline


class _Ctx:
    def __init__(self):
        self.log = []

    def note(self, m):
        self.log.append(m)


def test_deadline_evidence_replaces_only_a_mapping_that_reads_nothing():
    page = "<dl><dt>Closing date</dt><dd>10.12.2026 09:00</dd></dl>"
    spec = {"tz": "UTC", "detail": {"kind": "html"}, "fields": {}}
    onboard._apply_deadline_evidence(_Ctx(), spec, page, [{"id": "1"}])
    assert spec["fields"]["submissionDeadline"] == {"from": "detail", "label": "Closing date"}
    ok = {"from": "detail", "label": "Other", "formats": ["%d.%m.%Y %H:%M"]}
    spec = {"tz": "UTC", "detail": {"kind": "html"}, "fields": {"submissionDeadline": dict(ok)}}
    onboard._apply_deadline_evidence(_Ctx(), spec, "<dl><dt>Other</dt><dd>11.12.2026 10:00</dd></dl>" + page, [])
    assert spec["fields"]["submissionDeadline"] == ok          # the model's mapping reads a value: kept
    listed = {"from": "list", "column": "Deadline", "formats": ["%d.%m.%Y"]}
    spec = {"tz": "UTC", "detail": {"kind": "html"}, "fields": {"submissionDeadline": dict(listed)}}
    onboard._apply_deadline_evidence(_Ctx(), spec, page, [{"id": "1", "submissionDeadline": None}])
    assert spec["fields"]["submissionDeadline"]["label"] == "Closing date"   # the list gave no deadline (no time)


def test_best_attempt_payload():
    best = {"spec": {"tz": "UTC", "list": {"url": "u"}, "fields": {"id": {}}, "profile": {"x": 1}}, "rows": 20,
            "problems": ["p"]}
    got = onboard._best_payload(best)
    assert got["spec"] == {"tz": "UTC", "list": {"url": "u"}, "fields": {"id": {}}} and got["rowsReadInDryRun"] == 20
    assert onboard._best_payload(None) is None


# ------------------------------------------------------------------ transient transport errors
def test_a_dropped_keepalive_connection_is_retried_once_for_a_read():
    import asyncio

    from watchtower.http import HostRateLimiter, PoliteClient
    fail = {"n": 1}

    def handler(req):
        if fail["n"] > 0:
            fail["n"] -= 1
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.", request=req)
        return httpx.Response(200, text="ok")

    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(handler))
        try:
            r = await c.get(f"{HOST}/tender/1")
            assert r.status_code == 200 and c.requests == 2
            fail["n"] = 1                   # a form post is never sent twice
            try:
                await c.request("POST", f"{HOST}/form", content=b"a=1")
            except httpx.RemoteProtocolError:
                pass
            else:
                raise AssertionError("POST must not be retried")
            fail["n"] = 2                   # a read is retried once, not forever
            try:
                await c.get(f"{HOST}/tender/2")
            except httpx.RemoteProtocolError:
                pass
            else:
                raise AssertionError("only one retry")
        finally:
            await c.aclose()
    asyncio.run(go())


def test_list_deadline_column_with_the_time_in_a_title_is_found():
    rows = "".join(
        f"<tr><td><abbr title='20.11.2026 um 19:15 Uhr'>20.11.2026</abbr></td><td>{dl}</td><td><a href='/p?pid={i}'>"
        f"T{i}</a></td></tr>" for i, dl in enumerate([
            "<abbr title='Die Frist ist bei diesem Verfahren nicht vorhanden'>nv</abbr>",
            "<abbr title='27.11.2026 um 12:00 Uhr'>27.11.2026</abbr>",
            "<abbr title='11.12.2026 um 09:00 Uhr'>11.12.2026</abbr>"]))
    html = (f"<table class=t><thead><tr><th>Veröffentlicht</th><th>Angebots- / Teilnahmefrist</th><th>Aktion</th></tr>"
            f"</thead><tbody>{rows}</tbody></table>")
    spec = {"tz": "Europe/Berlin", "list": {"kind": "html", "url": f"{HOST}/l", "items": "table.t > tbody > tr"},
            "fields": {"submissionDeadline": {"from": "list", "column": "Angebots- / Teilnahmefrist",
                                              "formats": ["%d.%m.%Y"]}}}
    assert onboard._apply_list_deadline_evidence(_Ctx(), spec, html)
    assert spec["fields"]["submissionDeadline"] == {"from": "list", "column": "Angebots- / Teilnahmefrist",
                                                    "sel": "abbr", "attr": "title", "formats": ["%d.%m.%Y um %H:%M"]}
    assert not onboard._apply_list_deadline_evidence(_Ctx(), spec, html)     # now it reads: left alone
