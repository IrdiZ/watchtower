"""Award, result and cancellation notices are not open tenders (vergabe-westfalen, 21 Nov: ex-post award notices
"Auftrag bereits vergeben" were served as open; PLACE served a closed call without a deadline as open).
1. a notice type / phase the portal shows (fields.noticeType) is classified statically, in several languages;
2. a new open tender without a deadline, in a read where most tenders show one, is held back;
3. the audit's status comparison also reads the page's notice type and a deadline that has passed."""
from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from tests.test_contract import setup  # noqa: F401  (fixture)
from tests.test_e2e_mock import attach, env, wait  # noqa: F401  (fixture)
from mock_portal import app as mock
from watchtower import audit, runner, timeutil, validate
from watchtower.extract import assemble, notice_status
from watchtower.http import HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext

HOST = "http://notices.test"
TODAY = timeutil.now().date()


@pytest.mark.parametrize("value", [
    "Bekanntmachung über vergebene Aufträge", "Bekanntmachung vergebener Aufträge", "Ex post Veröffentlichung",
    "Ex-post-Transparenz", "Vergabebekanntmachung", "Zuschlagsbekanntmachung", "Contract award notice", "Awarded",
    "Result", "Avis d'attribution", "Attribué", "Aankondiging van een gegunde opdracht", "Gunning",
    "Anuncio de adjudicación", "Adjudicado", "Formalización", "Avviso di aggiudicazione", "Esito di gara",
    "Ogłoszenie o udzieleniu zamówienia", "Vorinformation", "Prior information notice", "Avis de préinformation",
    "Vooraankondiging", "Closed", "Clôturée", "abgelaufen"])
def test_result_notices_are_closed(value):
    assert notice_status(value) == "closed", value


@pytest.mark.parametrize("value", ["Aufhebung", "Verfahren aufgehoben", "Annulation", "Cancelled", "Unieważnienie",
                                   "Anulación", "Desierto", "Vroegtijdige beëindiging", "Intrekking"])
def test_cancellations_are_withdrawn(value):
    assert notice_status(value) == "withdrawn", value


@pytest.mark.parametrize("value", [None, "", "Ausschreibung", "Contract notice", "Auftragsbekanntmachung",
                                   "Avis d'appel public à la concurrence", "Anuncio de licitación",
                                   "Aankondiging van een opdracht", "Bando di gara", "Ogłoszenie o zamówieniu",
                                   "Vorinformation mit Aufruf zum Wettbewerb", "Berichtigung", "Avis rectificatif",
                                   "Offenes Verfahren", "Freihändige Vergabe", "Veröffentlicht", "Open"])
def test_calls_and_corrections_give_no_signal(value):
    assert notice_status(value) is None, value


def test_explicit_map_comes_first():
    fs = {"map": {"ADJ": "closed", "PUB": "open", "ANUL": "withdrawn"}}
    assert notice_status("ADJ", fs) == "closed" and notice_status("anul", fs) == "withdrawn"
    assert notice_status("PUB", fs) is None
    assert notice_status("Contract award notice", fs) == "closed"     # an unmapped value: the keywords


def test_assemble_status_from_notice_type():
    spec = {"tz": "UTC", "fields": {"noticeType": {"sel": "td.kind"}}}
    base = {"id": "1", "url": "https://x/1", "title": "T"}
    assert assemble({**base, "noticeType": "Vergabebekanntmachung"}, spec, "https://x")["status"] == "closed"
    assert assemble({**base, "noticeType": "Aufhebung"}, spec, "https://x")["status"] == "withdrawn"
    assert assemble({**base, "noticeType": "Ausschreibung"}, spec, "https://x")["status"] == "open"
    # a cancellation sticks: an award type does not turn a withdrawn tender into a closed one
    assert assemble({**base, "withdrawn": True, "noticeType": "Award"}, spec, "https://x")["status"] == "withdrawn"
    assert "noticeType" not in assemble({**base, "noticeType": "Award"}, spec, "https://x")


# ------------------------------------------------------------------ runner
def _iso(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat() + "T12:00:00"


def _spec(notice_on: str | None = "list", deadline_on: str = "list"):
    fields = {"id": {"path": "id"}, "title": {"path": "name"}, "buyerName": {"path": "buyer"},
              "publicationDate": {"path": "pub"}, "url": {"template": f"{HOST}/t/{{id}}"}}
    if deadline_on == "list":
        fields["submissionDeadline"] = {"path": "dl"}
    else:
        fields["submissionDeadline"] = {"from": "detail", "path": "deadline"}
    if notice_on == "list":
        fields["noticeType"] = {"path": "type"}
    elif notice_on == "detail":
        fields["noticeType"] = {"from": "detail", "path": "noticeType"}
    spec = {"tz": "UTC", "list": {"kind": "json", "url": f"{HOST}/api?page=0", "items": "rows",
                                  "pagination": {"type": "page", "param": "page", "start": 0},
                                  "total": {"path": "total"}},
            "fields": fields}
    if deadline_on == "detail" or notice_on == "detail":
        spec["detail"] = {"kind": "json", "url_from": "template", "template": f"{HOST}/api/t/{{id}}"}
    return spec


def _rows():
    """8 open calls with a deadline, 3 ex-post award notices (no deadline), 1 call without any deadline,
    1 cancellation."""
    rows = [{"id": f"C{i}", "name": f"Call {i}", "buyer": "Stadt X", "pub": TODAY.isoformat(), "dl": _iso(10 + i),
             "type": "Ausschreibung"} for i in range(8)]
    rows += [{"id": f"A{i}", "name": f"Award {i}", "buyer": "Stadt X", "pub": TODAY.isoformat(), "dl": None,
              "type": t} for i, t in enumerate(["Bekanntmachung vergebener Aufträge", "Ex post Veröffentlichung",
                                                "Contract award notice"])]
    rows.append({"id": "N1", "name": "No deadline", "buyer": "Stadt X", "pub": TODAY.isoformat(), "dl": None,
                 "type": "Ausschreibung"})
    rows.append({"id": "W1", "name": "Cancelled", "buyer": "Stadt X", "pub": TODAY.isoformat(), "dl": None,
                 "type": "Aufhebung"})
    return rows


def _handler(rows, hits):
    by = {r["id"]: r for r in rows}

    def handler(req: httpx.Request):
        hits.append(str(req.url))
        if req.url.path.startswith("/api/t/"):
            r = by[req.url.path.rsplit("/", 1)[-1]]
            return httpx.Response(200, json={"id": r["id"], "deadline": r["dl"], "noticeType": r["type"]})
        page = int(req.url.params.get("page", "0"))
        return httpx.Response(200, json={"rows": rows if page == 0 else [], "total": len(rows)})
    return handler


def _read(client, spec, rows, previous=None, known_closed=None, limits=None, hits=None):
    store, docs = client.app.state.store, client.app.state.manager.docs
    hits = hits if hits is not None else []
    pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(_handler(rows, hits)))
    ctx = JobContext("p", HOST, "r", pc, docs, None, store)

    async def go():
        try:
            return await runner.read(ctx, spec, previous or {}, limits or runner.Limits(fetch_docs=False),
                                     known_closed=known_closed)
        finally:
            await pc.aclose()
    return asyncio.run(go())


def test_award_rows_on_the_list_are_not_served(setup):
    client, _ = setup
    hits = []
    res = _read(client, _spec("list"), _rows(), hits=hits)
    by = {t["id"]: t for t in res.tenders}
    assert sorted(by) == sorted([f"C{i}" for i in range(8)] + ["W1"]), sorted(by)
    assert all(by[f"C{i}"]["status"] == "open" for i in range(8)) and by["W1"]["status"] == "withdrawn"
    assert res.stats["notOpenNotices"] == 3
    assert res.stats["noDeadline"] == 1 and "N1" in res.closed        # most tenders show a deadline: held back
    assert res.closed["N1"]["why"] == "no deadline in its list row"
    v = validate.validate(_spec("list"), res, {}, mode="attach")
    assert v.ok, v.failures
    d = validate.decide(v, res, mode="attach", first=True)
    assert d.commit and any("award, result or prior-information" in n for n in d.notes), d.notes


def test_award_notice_of_a_served_tender_closes_it(setup):
    """A tender served before whose newest notice is now an award stays in the read with status closed."""
    client, _ = setup
    rows = _rows()
    prev = {"A0": {"data": {"id": "A0", "title": "Award 0", "status": "open", "submissionDeadline": None}}}
    res = _read(client, _spec("list"), rows, previous=prev)
    by = {t["id"]: t for t in res.tenders}
    assert by["A0"]["status"] == "closed"
    assert res.stats["notOpenNotices"] == 2


def test_notice_type_on_the_tender_page(setup):
    """noticeType and deadline only on the tender page: awards are recognised once the page is read; a notice
    whose page is not read yet is held back while most read pages show a deadline; known ones are not re-read."""
    client, _ = setup
    spec = _spec("detail", deadline_on="detail")
    rows = _rows()
    hits = []
    res = _read(client, spec, rows, limits=runner.Limits(fetch_docs=False, max_details=10), hits=hits)
    by = {t["id"]: t for t in res.tenders}
    assert all(by[i]["status"] == "open" and by[i]["submissionDeadline"] for i in by if i.startswith("C")), by
    assert not any(i.startswith("A") or i == "N1" for i in by)
    # pages read: C0..C7 (deadlines), A0, A1 (award notices); not read yet: A2, N1, W1 (held back)
    assert sorted(by) == [f"C{i}" for i in range(8)], sorted(by)
    assert res.stats["detailsFetched"] == 10 and res.stats["heldBack"] == 3
    assert res.stats["notOpenNotices"] == 2 and set(res.closed) == {"A0", "A1"}
    assert res.stats["deadlineShare"] == [8, 8]
    # next read: the held-back notices are opened, the known award notices are not opened again (row unchanged)
    previous = {tid: {"data": t} for tid, t in by.items()}
    hits2 = []
    res2 = _read(client, spec, rows, previous=previous, known_closed=res.closed,
                 limits=runner.Limits(fetch_docs=False, max_details=10), hits=hits2)
    by2 = {t["id"]: t for t in res2.tenders}
    assert set(by2) == {f"C{i}" for i in range(8)} | {"W1"}, sorted(by2)
    assert by2["W1"]["status"] == "withdrawn"
    assert res2.stats["heldBack"] == 0 and res2.stats["noDeadline"] == 1 and res2.stats["notOpenNotices"] == 3
    assert not any(h.endswith(("/api/t/A0", "/api/t/A1")) for h in hits2), hits2


def test_unread_pages_stay_served_without_award_evidence(setup):
    """Deadline only on the tender page, no award notice seen: tenders whose page is not read yet are served
    (pending details), as before - a large open-only portal keeps its count on the first read."""
    client, _ = setup
    spec = _spec(None, deadline_on="detail")
    rows = [r for r in _rows() if r["id"].startswith("C")]
    res = _read(client, spec, rows, limits=runner.Limits(fetch_docs=False, max_details=3))
    assert len(res.tenders) == 8 and res.stats["heldBack"] == 0 and res.stats["pendingDetails"] == 5


def test_procedure_type_naming_a_result_publication():
    """No noticeType mapped: a procedure type that says 'Ex post' (where the portal prints the procedure type) is a
    result publication; ordinary procedure names are not."""
    spec = {"tz": "UTC", "fields": {"procedureType": {"from": "detail", "label": "Vergabeart:"}}}
    base = {"id": "1", "url": "https://x/1", "title": "T"}
    assert assemble({**base, "procedureType": "Ex post Veröffentlichung"}, spec, "https://x")["status"] == "closed"
    for p in ("Offenes Verfahren", "VgV TNW", "Adjudicación directa", "Freihändige Vergabe", None):
        assert assemble({**base, "procedureType": p}, spec, "https://x")["status"] == "open", p


def test_no_deadline_rule_needs_a_majority(setup):
    """Half or fewer of the tenders with a deadline: nothing is held back for lacking one (an undated portal)."""
    client, _ = setup
    rows = [{"id": f"U{i}", "name": f"T{i}", "buyer": "B", "pub": TODAY.isoformat(), "dl": _iso(5) if i < 4 else None,
             "type": "Ausschreibung"} for i in range(8)]
    res = _read(client, _spec(None), rows)
    assert len(res.tenders) == 8 and res.stats["noDeadline"] == 0 and res.stats["heldBack"] == 0


# ------------------------------------------------------------------ audit
def test_audit_page_status_reads_notice_type_and_past_deadline():
    now = timeutil.now()
    assert audit.page_status({"status": None, "noticeType": "Ex post Veröffentlichung"}, "Europe/Berlin") == "closed"
    assert audit.page_status({"status": "open", "noticeType": "Bekanntmachung vergebener Aufträge"},
                             "Europe/Berlin") == "closed"       # the page's notice type wins over a model's 'open'
    assert audit.page_status({"status": None, "submissionDeadline": "11/11/2025"}, "Europe/Paris", now=now) == "closed"
    assert audit.page_status({"status": None, "submissionDeadline": (now + timedelta(days=9)).strftime("%d.%m.%Y")},
                             "Europe/Berlin", now=now) is None
    assert audit.page_status({"status": "withdrawn"}, "UTC") == "withdrawn"
    assert audit.page_status({}, "UTC") is None
    # compared against what we serve: an award served as open is a status mismatch
    truth = {"title": "X", "status": None, "noticeType": "Ex post"}
    out = audit.compare_tender(truth, {"title": "X", "status": "open", "submissionDeadline": None}, "Europe/Berlin")
    st = [o for o in out if o["field"] == "status"]
    assert st and st[0]["result"] == "mismatch" and st[0]["page"] == "closed"
    assert '"noticeType"' in audit.AUDIT_PROMPT


# ------------------------------------------------------------------ end to end on the mock portal
def test_e2e_award_rows_are_not_served(env):
    client, llm = env
    mock.CONFIG["mutations"] = {"award_rows"}
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    ids = {t["id"] for t in client.get(f"/portals/{pid}/tenders").json()}
    assert ids == {t["id"] for t in mock.TENDERS}, ids          # the 12 calls; no award notice, no undated call
    assert "award, result or prior-information" in p["reason"], p["reason"]
    # a regular read: no model call, the same tenders, the known undated call's page is not opened again
    n = len(llm.calls)
    mock.REQUESTS.clear()
    client.post(f"/portals/{pid}/refresh")
    assert wait(client, pid, {"serving"})["state"] == "serving"
    assert len(llm.calls) == n
    assert {t["id"] for t in client.get(f"/portals/{pid}/tenders").json()} == ids
    assert not any("/tender/MH-2026-09" in path for _, path in mock.REQUESTS)
