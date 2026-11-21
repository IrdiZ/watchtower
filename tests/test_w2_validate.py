"""W2 P1: legitimately empty vs broken, same-shape drift on very small listings, repair evidence, and the
commit/reject decision (`validate.decide`) on the runner->worker contract fields."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from watchtower.pipeline import ReadResult, Verdict
from watchtower.validate import decide, validate

FUTURE = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
PAST = "2020-01-01T10:00:00+01:00"


def t(i, **kw):
    d = {"id": f"T-{i}", "url": f"https://p.example/t/{i}", "title": f"Dachsanierung Schule {i}",
         "buyerName": f"Stadt Musterhausen {i}", "publicationDate": "2026-11-01",
         "submissionDeadline": FUTURE, "status": "open"}
    d.update(kw)
    return d


def prev(*tenders):
    return {x["id"]: {"data": x} for x in tenders}


def res(tenders, complete=True, **stats):
    return ReadResult(tenders=tenders, complete=complete, stats=stats)


SPEC = {"list": {"kind": "html"}}


# ------------------------------------------------------------------ empty vs broken
def test_empty_after_every_deadline_passed_is_valid_even_if_stored_status_is_stale():
    # stored status still says "open" (committed before the deadline), but effective status at read time is closed
    previous = prev(t(1, submissionDeadline=PAST), t(2, submissionDeadline=PAST, status="closed"))
    v = validate({**SPEC, "list": {"kind": "html", "total": {"sel": "p.total"}}}, res([], statedTotal=0), previous)
    assert v.ok, v.failures
    assert any("past their deadline" in w for w in v.warnings)
    # and without a stated-total element in the spec
    assert validate(SPEC, res([]), previous).ok


def test_empty_while_open_tenders_are_still_due_is_rejected():
    previous = prev(t(1), t(2, submissionDeadline=PAST))
    v = validate(SPEC, res([], statedTotal=0), previous)
    assert not v.ok and any("no tenders found" in f for f in v.failures)


def test_open_tender_without_deadline_is_never_assumed_expired():
    previous = prev(t(1, submissionDeadline=None))
    assert not validate(SPEC, res([]), previous).ok


def test_empty_but_selector_or_total_element_missing_is_rejected():
    previous = prev(t(1, submissionDeadline=PAST))
    with_total = {"list": {"kind": "html", "total": {"sel": "p.total"}}}
    v = validate(with_total, res([], statedTotal=None), previous)  # the "N tenders" element is gone too
    assert not v.ok and any("structure" in f for f in v.failures)
    assert not validate(SPEC, res([], selectorMissing=True), previous).ok
    assert not validate(SPEC, res([], errors=["list selector 'tr.row' matched nothing"]), previous).ok
    assert not validate(with_total, res([], statedTotal=7), previous).ok  # the portal says it has some
    assert not validate(SPEC, res([], complete=False), previous).ok       # read did not even finish


def test_empty_first_attach_and_empty_repair_are_rejected():
    assert not validate(SPEC, res([]), {}, mode="attach").ok
    previous = prev(t(1, submissionDeadline=PAST))
    assert not validate(SPEC, res([]), previous, mode="repair").ok


# ------------------------------------------------------------------ same-shape drift on tiny listings
def swapped(x):
    return {**x, "title": x["buyerName"], "buyerName": x["title"]}


def test_title_buyer_swap_on_three_tenders_is_caught():
    old = [t(i) for i in range(3)]
    v = validate(SPEC, res([swapped(x) for x in old]), prev(*old))
    assert not v.ok and any("title/buyer" in f or "swapped" in f for f in v.failures)


def test_title_buyer_swap_on_a_single_tender_is_caught():
    old = [t(1)]
    assert not validate(SPEC, res([swapped(old[0])]), prev(*old)).ok


def test_one_amended_title_among_three_is_not_drift():
    old = [t(i) for i in range(3)]
    new = [{**old[0], "title": "Dachsanierung Schule 0 (Berichtigung)"}, old[1], old[2]]
    assert validate(SPEC, res(new), prev(*old)).ok


def test_change_feed_amendments_are_not_drift_but_a_changed_same_entry_is():
    # A change feed lists a known tender again because it changed: newer entry versions may carry a new title,
    # buyer spelling or publication date. The same entry version with different values means the reader broke.
    old = [t(i, _feedUpdated="2026-11-17T10:00:00+00:00") for i in range(3)]
    amended = [{**x, "title": x["title"] + " (Corrección)", "publicationDate": "2026-11-20",
                "_feedUpdated": "2026-11-20T08:00:00+00:00"} for x in old]
    assert validate(SPEC, res(amended, changeFeed=True), prev(*old)).ok
    same_entry = [{**x, "title": "Etwas ganz anderes", "publicationDate": "2026-10-01"} for x in old]
    v = validate(SPEC, res(same_entry, changeFeed=True), prev(*old))
    assert not v.ok and any("title/buyer" in f for f in v.failures)
    # a non-feed read keeps the strict check
    assert not validate(SPEC, res(amended), prev(*old)).ok


# ------------------------------------------------------------------ repair evidence with < 5 shared ids
def test_repair_with_few_shared_ids_warns_without_verified_blind_but_does_not_fail():
    old = [t(i) for i in range(3)]
    v = validate(dict(SPEC), res([dict(x) for x in old] + [t(9)]), prev(*old), mode="repair")
    assert v.ok and any("blind verification" in w for w in v.warnings)


def test_repair_with_few_shared_ids_and_verified_blind_is_quiet():
    old = [t(i) for i in range(3)]
    v = validate({**SPEC, "verifiedBlind": True}, res([dict(x) for x in old]), prev(*old), mode="repair")
    assert v.ok and not any("blind" in w for w in v.warnings)


def test_repair_that_contradicts_the_last_good_data_fails_unless_blind_verified():
    old = [t(i) for i in range(4)]
    new = [{**x, "title": f"Etwas ganz anderes {x['id']}", "buyerName": "Anderer Auftraggeber",
            "publicationDate": "2026-10-01"} for x in old]
    assert not validate(dict(SPEC), res(new), prev(*old), mode="repair").ok
    # (the generic drift check fires first for >= 2 shared tenders; one shared tender hits the agreement check)
    one = validate(dict(SPEC), res(new[:1]), prev(*old[:1]), mode="repair")
    assert not one.ok
    blind = validate({**SPEC, "verifiedBlind": True}, res(new[:1]), prev(*old[:1]), mode="repair")
    assert blind.ok and any("agrees" in w for w in blind.warnings)  # evidence from the source wins, but it is noted


# ------------------------------------------------------------------ the gate's decision
def ok():
    return Verdict(ok=True)


def test_gate_rejects_incomplete_uncapped_read():
    first = decide(ok(), res([t(1)], complete=False, capped=False, stopReason="repeated page"), mode="attach", first=True)
    assert not first.commit and first.state == "failed" and "incomplete" in first.failure_text
    later = decide(ok(), res([t(1)], complete=False, capped=False), mode="read", first=False)
    assert not later.commit and later.state == "healing"
    in_repair = decide(ok(), res([t(1)], complete=False), mode="repair", first=False)
    assert not in_repair.commit and in_repair.state == "healing"


def test_gate_publishes_capped_first_read_with_honest_reason_and_counts():
    d = decide(ok(), res([t(1)], complete=False, capped=True, stopReason="portal refuses pages beyond 200",
                         pendingDetails=7, pendingDocs=3, docBlocked=2, docBlockReason="login wall"),
               mode="attach", first=True, serve_capped=True)
    assert d.commit and d.state == "serving" and not d.close_missing
    r = d.reason(10, 8)
    assert "unseen tenders kept, not closed" in r and "10 tenders" in r and "8 open" in r
    assert "7 tenders" in r and "3 documents" in r and "Documents: not available from this portal (login wall)" in r
    assert "Coverage: partial" in r


def test_gate_complete_read_may_close_and_missing_contract_fields_default_quietly():
    d = decide(ok(), ReadResult(tenders=[t(1)], complete=True), mode="read", first=False)  # no stats at all
    assert d.commit and d.close_missing and d.notes == ["Coverage: complete."]


def test_gate_rejects_a_failed_verdict_whatever_the_flags():
    d = decide(Verdict(ok=False, failures=["buyerName broken"]), res([t(1)], complete=True), mode="read", first=False)
    assert not d.commit and d.failures == ["buyerName broken"] and not d.close_missing
