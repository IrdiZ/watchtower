"""W3: retryable malformed specs, blind date verification, verifiedBlind, endpoint fallback."""
import asyncio
from types import SimpleNamespace

import httpx

from mock_portal import app as mock
from tests.fake_llm import BASE, FakeLLM
from watchtower import onboard
from watchtower.config import Settings
from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient
from watchtower.onboard import _date_problem, validate_spec
from watchtower.pipeline import JobContext
from watchtower.store import Store

BERLIN = "Europe/Berlin"
F = ["%d.%m.%Y %H:%M"]


# ------------------------------------------------------------------ dates
def test_deadline_correct_forms():
    assert _date_problem("submissionDeadline", "15.03.2027 14:30", "2027-03-15T14:30:00+01:00", BERLIN, F) is None
    assert _date_problem("submissionDeadline", "15.07.2027 14:30", "2027-07-15T14:30:00+02:00", BERLIN, F) is None  # DST
    assert _date_problem("submissionDeadline", "15 March 2027, 14:30", "2027-03-15T14:30:00+01:00", BERLIN, []) is None
    assert _date_problem("submissionDeadline", "2027-03-15T14:30:00+02:00", "2027-03-15T12:30:00+00:00", BERLIN, []) is None
    assert _date_problem("submissionDeadline", "15.03.2027 14:30 UTC+2", "2027-03-15T14:30:00+02:00", BERLIN, F) is None


def test_deadline_wrong_month():
    msg = _date_problem("submissionDeadline", "15.03.2027 14:30", "2027-04-15T14:30:00+02:00", BERLIN, F)
    assert msg and "mismatch" in msg
    # the day number alone matching must not be enough
    assert _date_problem("publicationDate", "15.03.2027", "2027-04-15", BERLIN, ["%d.%m.%Y"])
    assert _date_problem("publicationDate", "15.03.2027", "2026-03-15", BERLIN, ["%d.%m.%Y"])
    assert _date_problem("publicationDate", "15.03.2027", "2027-03-15", BERLIN, ["%d.%m.%Y"]) is None


def test_deadline_wrong_offset():
    # page states +02:00 explicitly, the spec attached the portal's +01:00 instead
    msg = _date_problem("submissionDeadline", "2027-03-15T14:30:00+02:00", "2027-03-15T14:30:00+01:00", BERLIN, [])
    assert msg and "offset" in msg
    msg = _date_problem("submissionDeadline", "15.03.2027 14:30", "2027-03-15T14:30:00+00:00", BERLIN, F)  # wrong tz
    assert msg and "offset" in msg


def test_deadline_missing_time_must_be_null():
    assert _date_problem("submissionDeadline", "15.03.2027", None, BERLIN, F) is None
    msg = _date_problem("submissionDeadline", "15.03.2027", "2027-03-15T00:00:00+01:00", BERLIN, F)
    assert msg and "no time" in msg


def test_missing_field():
    assert _date_problem("submissionDeadline", None, None, BERLIN, F) is None  # page shows nothing, spec nothing
    assert "missing" in _date_problem("submissionDeadline", "15.03.2027 14:30", None, BERLIN, F)
    assert "missing" in _date_problem("publicationDate", "15.03.2027", None, BERLIN, F)


def _stub_ctx(page_text_value, truth):
    html = f"<html><body><p>{page_text_value}</p></body></html>"

    async def get(url, **kw):
        return SimpleNamespace(status_code=200, text=html, json=lambda: {})

    async def json_call(task, system, user, **kw):
        return truth
    return SimpleNamespace(client=SimpleNamespace(get=get), llm=SimpleNamespace(json_call=json_call),
                           portal_id="p", run_id="r")


def test_blind_verify_flags_wrong_month_and_passes_correct():
    spec = {"tz": BERLIN, "detail": {"url_from": "url"},
            "fields": {"submissionDeadline": {"formats": F}, "publicationDate": {"formats": ["%d.%m.%Y"]}}}
    truth = {"id": "A1", "title": "T", "buyerName": "B", "publicationDate": "01.03.2027",
             "submissionDeadline": "15.03.2027 14:30"}
    good = {"id": "A1", "url": "http://x/1", "title": "T", "buyerName": "B", "publicationDate": "2027-03-01",
            "submissionDeadline": "2027-03-15T14:30:00+01:00"}
    bad = dict(good, submissionDeadline="2027-04-15T14:30:00+02:00")
    run = lambda t: asyncio.run(onboard.blind_verify(_stub_ctx("x", truth), spec, [t], n=1))
    assert run(good) == []
    probs = run(bad)
    assert len(probs) == 1 and "submissionDeadline" in probs[0]


# ------------------------------------------------------------------ spec validation + retry
def test_validate_spec_reports_concise_problems():
    spec = {"tz": "Mars/Base", "list": {"kind": "html", "url": "/relative", "items": "div[[bad",
                                        "pagination": {"type": "page"}},
            "fields": {"id": {"sel": "td.nr"}, "title": {"sel": "a", "formats": ["%Q"]}}}
    probs = " | ".join(validate_spec(spec))
    for needle in ("tz", "absolute", "CSS selector", "param", "'url' is not mapped", "date format"):
        assert needle in probs, probs


def _job_ctx(tmp_path, llm_factory):
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    store = Store(str(tmp_path))
    client = PoliteClient(HostRateLimiter(0), "t", 10, budget=500, transport=httpx.ASGITransport(app=mock.app))
    return JobContext("p1", BASE + "/", "r1", client, DocumentStore(store, 10 ** 8, 10 ** 9), llm_factory(store),
                      store, mode="attach")


class FlakyLLM(FakeLLM):
    """First list synthesis is malformed (invalid selector, bad date format); later ones are fine."""

    def __init__(self, store):
        super().__init__(store)
        self.list_calls = 0
        self.feedback_seen: list[list[str]] = []

    def _list(self, d):
        self.list_calls += 1
        self.feedback_seen.append(d.get("problemsToFix", []))
        spec = super()._list(d)
        if self.list_calls == 1:
            spec["list"]["items"] = "div[[bad"
            spec["fields"]["submissionDeadline"]["formats"] = ["%Q"]
        return spec


def test_malformed_spec_is_retried_with_feedback_and_marked_verified(tmp_path):
    mock.CONFIG.update(skin="html", mutations=set())
    ctx = _job_ctx(tmp_path, FlakyLLM)

    async def go():
        try:
            return await onboard.learn(ctx, BASE + "/")
        finally:
            await ctx.client.aclose()
    spec = asyncio.run(go())
    assert ctx.llm.list_calls == 2
    assert any("CSS selector" in p for p in ctx.llm.feedback_seen[1]), ctx.llm.feedback_seen
    assert spec["verifiedBlind"] is True
    assert spec["list"]["items"].startswith(("tr.", "div."))


# ------------------------------------------------------------------ endpoint fallback
def test_next_endpoint_skips_unreplayable_and_non_tender_endpoints():
    tenders = {"items": [{"id": 1, "title": "a", "deadline": "x"}, {"id": 2, "title": "b", "deadline": "y"}]}
    menu = {"menu": [{"label": "a"}, {"label": "b"}]}

    async def get(url, **kw):
        if url.endswith("/dead"):
            return SimpleNamespace(status_code=401, json=lambda: {})
        data = menu if url.endswith("/menu") else tenders
        return SimpleNamespace(status_code=200, json=lambda: data)
    ctx = SimpleNamespace(client=SimpleNamespace(get=get), note=lambda m: None)
    found = {"api": ("http://x/first", {}), "untried": ["http://x/dead", "http://x/menu", "http://x/good", "http://x/z"]}
    assert asyncio.run(onboard._next_endpoint(ctx, found)) is True
    assert found["api"][0] == "http://x/good" and found["untried"] == ["http://x/z"]
    assert asyncio.run(onboard._next_endpoint(ctx, {"api": None, "untried": ["http://x/dead"]})) is False
