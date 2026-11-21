"""Lead's policy: partial (capped) listings don't go green unless WATCHTOWER_SERVE_CAPPED=1; sorted early stop."""
from datetime import datetime, timedelta, timezone

from watchtower.pipeline import ReadResult, Verdict
from watchtower.runner import _sorted_list_finished
from watchtower.validate import decide


def _res(**stats):
    return ReadResult(tenders=[{"id": "1"}], complete=False, stats={"capped": True,
                      "stopReason": "portal refuses pages beyond 100 (HTTP 400)", **stats})


def test_capped_first_attach_fails_by_default_with_reason():
    d = decide(Verdict(ok=True), _res(), mode="attach", first=True)
    assert not d.commit and d.state == "failed"
    assert "only part of its listing" in d.failure_text and "WATCHTOWER_SERVE_CAPPED" in d.failure_text


def test_capped_served_when_allowed():
    d = decide(Verdict(ok=True), _res(), mode="attach", first=True, serve_capped=True)
    assert d.commit and d.state == "serving" and not d.close_missing


def _iso(days):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


SPEC = {"tz": "UTC", "list": {"order": "deadline_desc"},
        "fields": {"submissionDeadline": {"path": "d"}, "publicationDate": {"path": "p"}}}


def test_deadline_sorted_list_stops_on_all_closed_page():
    closed = [{"submissionDeadline": _iso(-1), "publicationDate": "2026-01-01"} for _ in range(3)]
    assert "closed" in _sorted_list_finished(SPEC, closed)
    mixed = closed + [{"submissionDeadline": _iso(5), "publicationDate": "2026-01-01"}]
    assert _sorted_list_finished(SPEC, mixed) is None
    missing = closed + [{"submissionDeadline": None, "publicationDate": "2026-01-01"}]
    assert _sorted_list_finished(SPEC, missing) is None   # no evidence -> keep reading


def test_published_sorted_list_needs_closed_and_old():
    spec = {**SPEC, "list": {"order": "published_desc"}}
    old = (datetime.now(timezone.utc) - timedelta(days=500)).date().isoformat()
    recent = (datetime.now(timezone.utc) - timedelta(days=20)).date().isoformat()
    assert _sorted_list_finished(spec, [{"submissionDeadline": _iso(-1), "publicationDate": old}] * 3)
    assert _sorted_list_finished(spec, [{"submissionDeadline": _iso(-1), "publicationDate": recent}] * 3) is None
    assert _sorted_list_finished({**SPEC, "list": {}}, [{"submissionDeadline": _iso(-1)}]) is None
