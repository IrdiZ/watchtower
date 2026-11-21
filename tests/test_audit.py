"""Continuous verification (watchtower/audit.py): comparison rules, stratified sampling, the audit loop on a fake
portal, and the contract's 'verification' object + /_ui/audit endpoints."""
from __future__ import annotations

import asyncio
import random
import re
import time

import httpx
from fastapi.testclient import TestClient

from watchtower import audit
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.onboard import VERIFY_PROMPT
from watchtower.pipeline import JobContext, ReadResult, Verdict


# ------------------------------------------------------------------ comparison
def test_text_compare_normalises_and_tolerates_shortening():
    assert audit.compare_text("title", "Lieferung  von Büromöbeln.", "lieferung von büromöbeln")["result"] == "match"
    assert audit.compare_text("buyerName", "Stadt Delbrück - Der Bürgermeister", "Stadt Delbrück")["result"] == "match"
    assert audit.compare_text("buyerName", "Gemeente Utrecht", "Provincie Utrecht")["result"] == "mismatch"
    assert audit.compare_text("title", "Bau einer Schule", None)["result"] == "missing"
    assert audit.compare_text("title", None, "anything") is None            # page shows nothing: not checked
    assert audit.compare_text("title", "-", "anything") is None
    # another name the page gives for the buyer counts
    assert audit.compare_text("buyerName", "Vergabestelle Nord", "Landkreis Süd",
                              ["Landkreis Süd"])["result"] == "match"
    # very short containment is not a match ("Stadt" in every German buyer name)
    assert audit.compare_text("buyerName", "Stadt", "Stadt Köln")["result"] == "mismatch"


def test_deadline_compare_to_the_minute_timezone_aware():
    tz = "Europe/Berlin"
    ours = "2026-12-10T09:00:00+01:00"
    assert audit.compare_date("submissionDeadline", "10.12.2026 09:00", ours, tz)["result"] == "match"
    assert audit.compare_date("submissionDeadline", "10.12.2026, 09:00 Uhr", ours, tz)["result"] == "match"
    assert audit.compare_date("submissionDeadline", "10.12.2026 09:01", ours, tz)["result"] == "mismatch"
    # same instant written in UTC on the page
    assert audit.compare_date("submissionDeadline", "2026-12-10T08:00:00Z", ours, tz)["result"] == "match"
    # same wall time, wrong offset: a mismatch, named as such
    o = audit.compare_date("submissionDeadline", "10.12.2026 09:00", "2026-12-10T09:00:00+00:00", tz)
    assert o["result"] == "mismatch" and o["note"] == "wrong offset"
    # French '12h00'
    assert audit.compare_date("submissionDeadline", "10/12/2026 12h00", "2026-12-10T12:00:00+01:00",
                              "Europe/Paris")["result"] == "match"
    assert audit.compare_date("submissionDeadline", "30/09/2035 16:00 (heure de Paris)", "2035-09-30T16:00:00+02:00",
                              "Europe/Paris")["result"] == "match"
    assert audit.compare_date("submissionDeadline", "01.12.2026 07:30 Ortszeit", "2026-12-01T07:30:00+01:00",
                              tz)["result"] == "match"
    assert audit.compare_date("submissionDeadline", "24.11.2026 - 08:59 Uhr", "2026-11-24T08:59:00+01:00",
                              tz)["result"] == "match"
    nl = "Europe/Amsterdam"     # TenderNed's rendered pages
    assert audit.compare_date("submissionDeadline", "31 mrt. 2027, 9:00", "2027-03-31T09:00:00+02:00",
                              nl)["result"] == "match"
    assert audit.compare_date("submissionDeadline", "10 dec. 2026", "2026-12-10T23:59:59+01:00", nl)["result"] == "match"
    assert audit.compare_date("publicationDate", "20 nov. 2026", "2026-11-20", nl)["result"] == "match"
    assert audit.compare_date("publicationDate", "3 okt. 2026", "2026-10-03", nl)["result"] == "match"
    # an English page with a US date: 12/04/2026 is 4 December
    assert audit.compare_date("submissionDeadline", "12/04/2026", "2026-12-04T12:00:00+01:00", tz)["result"] == "match"
    assert audit.compare_date("publicationDate", "05/16/2025", "2025-05-13", tz)["result"] == "mismatch"
    # the page shows only the date: compared by date in the portal's zone
    d = audit.compare_date("submissionDeadline", "10.12.2026", "2026-12-10T23:30:00+00:00", tz)
    assert d["result"] == "mismatch"                     # 23:30 UTC is already 11 Dec in Berlin
    assert audit.compare_date("submissionDeadline", "11.12.2026", "2026-12-10T23:30:00+00:00", tz)["result"] == "match"
    assert audit.compare_date("submissionDeadline", "10.12.2026 09:00", None, tz)["result"] == "missing"
    assert audit.compare_date("submissionDeadline", "sometime soon", ours, tz)["result"] == "unreadable"
    assert audit.compare_date("publicationDate", "1 November 2026", "2026-11-01", tz)["result"] == "match"
    assert audit.compare_date("publicationDate", "02.11.2026", "2026-11-01", tz)["result"] == "mismatch"


def test_status_and_documents():
    assert audit.compare_status("open", "open")["result"] == "match"
    assert audit.compare_status("Withdrawn", "open")["result"] == "mismatch"
    assert audit.compare_status(None, "open") is None and audit.compare_status("maybe", "open") is None
    assert audit.compare_documents(3, 5, False)["result"] == "match"       # a ZIP unpacked into more files
    assert audit.compare_documents(3, 2, False)["result"] == "mismatch"
    assert audit.compare_documents(3, 0, False)["result"] == "missing"
    assert audit.compare_documents(3, 0, True)["result"] == "pending"      # downloads still running: not counted
    assert audit.compare_documents(0, 0, False) is None and audit.compare_documents(None, 2, False) is None


def test_summary_counts_only_delivered_values_and_lists_disagreements():
    t = {"id": "A"}
    outs = [(t, "u", {"field": "title", "result": "match", "page": "x", "ours": "x"}),
            (t, "u", {"field": "title", "result": "mismatch", "page": "x", "ours": "y"}),
            (t, "u", {"field": "buyerName", "result": "missing", "page": "x", "ours": None}),
            (t, "u", {"field": "documents", "result": "pending", "page": 2, "ours": 0})]
    s = audit.summarise(outs)
    assert (s["fieldsChecked"], s["fieldsMatched"], s["accuracy"]) == (2, 1, 0.5)
    assert s["byField"]["buyerName"] == {"checked": 0, "matched": 0, "missing": 1, "notCounted": 0, "accuracy": None}
    assert s["status"] == "degraded" and len(s["mismatches"]) == 3
    assert s["mismatches"][0] == {"tenderId": "A", "url": "u", "field": "title", "result": "mismatch",
                                  "page": "x", "ours": "y"}
    assert audit.wilson_low(20, 20) < 1.0 < 1.01 and audit.wilson_low(0, 0) is None


def test_prompt_reuses_blind_verify_and_asks_for_status_and_documents():
    assert audit.AUDIT_PROMPT.startswith(VERIFY_PROMPT)
    assert '"status"' in audit.AUDIT_PROMPT and '"documentCount"' in audit.AUDIT_PROMPT
    assert audit.AUDIT_PROMPT.count("{") == audit.AUDIT_PROMPT.count("}")


# ------------------------------------------------------------------ sampling
def _tenders(n=40):
    out = []
    for i in range(n):
        out.append({"id": f"T{i}", "url": f"http://p.test/t/{i}", "title": f"Tender {i}",
                    "publicationDate": f"2026-{1 + i % 10:02d}-{1 + i % 28:02d}",
                    "documents": [{"fileName": "a.pdf"}] if i % 7 == 0 else []})
    return out


def test_sampling_is_stratified_and_unique():
    ts = _tenders()
    picked = audit.sample(ts, 8, random.Random(1))
    ids = [t["id"] for t, _ in picked]
    assert len(ids) == 8 == len(set(ids))
    whys = [w for _, w in picked]
    assert whys.count("newest") == 2 and whys.count("oldest") == 2 and whys.count("with documents") == 2
    newest = max(t["publicationDate"] for t in ts)
    oldest = min(t["publicationDate"] for t in ts)
    assert picked[0][0]["publicationDate"] == newest and picked[2][0]["publicationDate"] == oldest
    assert all(t["documents"] for t, w in picked if w == "with documents")
    assert [t["id"] for t, _ in audit.sample(ts, 8, random.Random(1))] == ids     # reproducible with a seed
    assert len(audit.sample(ts[:3], 8, random.Random(1))) == 3
    assert audit.sample([], 8) == [] and audit.sample(ts, 0) == []


# ------------------------------------------------------------------ the audit on a fake portal
PAGES = {
    "T0": "<h1>Neubau Feuerwache</h1><p>Auftraggeber: Stadt Musterstadt</p><p>Veröffentlicht: 01.11.2026</p>"
          "<p>Angebotsfrist: 10.12.2026 09:00</p><a href='/f/lv.pdf'>Leistungsverzeichnis</a>",
    "T1": "<h1>Reinigung Rathaus</h1><p>Auftraggeber: Gemeinde Beispiel</p><p>Veröffentlicht: 02.11.2026</p>"
          "<p>Angebotsfrist: 11.12.2026 12:00</p>",
}


class _PageLLM:
    """Reads the fake pages like the model would (regexes over the page text) and logs a cost."""

    def __init__(self, store):
        self.store = store
        self.calls = []
        self.available = True

    async def aclose(self):
        pass

    async def json_call(self, task, system, user, schema=None, portal_id=None, run_id=None, schema_name="r"):
        self.calls.append(task)
        self.store.log_llm_call(portal_id, run_id, task, "fake", 900, 120, 0.0002)
        assert task == "audit" and "documentCount" in system

        def grab(rx):
            m = re.search(rx, user)
            return m.group(1).strip() if m else None
        return {"id": None, "title": grab(r"^(.+?)\s*Auftraggeber"), "buyerName": grab(r"Auftraggeber: (.+?)\s*Ver"),
                "publicationDate": grab(r"Veröffentlicht: ([\d.]+)"),
                "submissionDeadline": grab(r"Angebotsfrist: ([\d.]+ [\d:]+)"),
                "status": "open", "documentCount": user.count("\n- ") or None}


def _served():
    return [{"id": "T0", "url": "http://p.test/t/T0", "title": "Neubau Feuerwache", "buyerName": "Stadt Musterstadt",
             "publicationDate": "2026-11-01", "submissionDeadline": "2030-12-10T09:00:00+01:00",
             "status": "open", "documents": [{"fileName": "lv.pdf"}]},
            {"id": "T1", "url": "http://p.test/t/T1", "title": "Reinigung Rathaus", "buyerName": "Gemeinde Beispiel",
             "publicationDate": "2026-11-02", "submissionDeadline": "2030-12-11T13:00:00+01:00",   # wrong hour
             "status": "open", "documents": []}]


def _pages_transport(html_by_id):
    def handler(req: httpx.Request):
        tid = req.url.path.rsplit("/", 1)[-1]
        if tid in html_by_id:
            body = html_by_id[tid].replace("2026 09:00", "2030 09:00").replace("2026 12:00", "2030 12:00")
            return httpx.Response(200, text=f"<html><body>{body}</body></html>",
                                  headers={"content-type": "text/html"})
        return httpx.Response(404, text="gone")
    return httpx.MockTransport(handler)


class _FakeStore:
    def log_llm_call(self, *a):
        pass


def test_audit_portal_finds_the_wrong_deadline_and_reports_both_values():
    from watchtower.http import HostRateLimiter, PoliteClient

    async def go():
        client = PoliteClient(HostRateLimiter(0), "test", 5, 50, transport=_pages_transport(PAGES))
        llm = _PageLLM(_FakeStore())
        ctx = JobContext("p", "http://p.test/", "a_test", client, None, llm, None)
        try:
            return await audit.audit_portal(ctx, {"tz": "Europe/Berlin"}, _served() + [
                {"id": "T9", "url": "http://p.test/t/T9", "title": "Gone", "status": "open"}],
                k=8, rng=random.Random(0))
        finally:
            await client.aclose()
    r = asyncio.run(go())
    assert r["sampled"] == 2 and r["skipped"] == [{"tenderId": "T9", "url": "http://p.test/t/T9", "why": "HTTP 404"}]
    assert r["byField"]["submissionDeadline"]["checked"] == 2 and r["byField"]["submissionDeadline"]["matched"] == 1
    assert r["byField"]["documents"] == {"checked": 1, "matched": 1, "missing": 0, "notCounted": 0, "accuracy": 1.0}
    bad = [m for m in r["mismatches"] if m["result"] == "mismatch"]
    assert bad == [{"tenderId": "T1", "url": "http://p.test/t/T1", "field": "submissionDeadline",
                    "result": "mismatch", "page": "11.12.2030 12:00", "ours": "2030-12-11T13:00:00+01:00"}]
    assert r["fieldsChecked"] == 11 and r["fieldsMatched"] == 10 and r["status"] == "ok"
    assert r["requests"] == 3


def test_audit_says_not_possible_when_no_source_can_be_read():
    from watchtower.http import HostRateLimiter, PoliteClient

    async def go():
        client = PoliteClient(HostRateLimiter(0), "test", 5, 50, transport=_pages_transport({}))
        ctx = JobContext("p", "http://p.test/", "a_t", client, None, _PageLLM(_FakeStore()), None)
        try:
            return await audit.audit_portal(ctx, {"tz": "UTC"}, _served(), k=4)
        finally:
            await client.aclose()
    r = asyncio.run(go())
    assert r["sampled"] == 0 and r["status"].startswith("not possible: HTTP 404")


def test_javascript_only_pages_without_renders_are_not_possible_and_not_refetched():
    from watchtower.http import HostRateLimiter, PoliteClient
    shell = httpx.MockTransport(lambda req: httpx.Response(
        200, text="<html><body><div id='app'></div><script src='/app.js'></script></body></html>",
        headers={"content-type": "text/html"}))

    async def go():
        client = PoliteClient(HostRateLimiter(0), "test", 5, 50, transport=shell)
        ctx = JobContext("p", "http://p.test/", "a_js", client, None, _PageLLM(_FakeStore()), None)
        try:
            return await audit.audit_portal(ctx, {"tz": "UTC"}, _served(), k=4, renders=0)
        finally:
            await client.aclose()
    r = asyncio.run(go())
    assert r["sampled"] == 0 and r["requests"] == 1          # one shell seen; the next page is not fetched
    assert r["status"].startswith("not possible: the page renders only with JavaScript")


# ------------------------------------------------------------------ contract + endpoints
class _ServingPipeline:
    async def attach(self, ctx):
        return {"list": {"kind": "html"}, "tz": "Europe/Berlin", "fields": {}, "detail": {}}

    async def read(self, ctx, spec):
        return ReadResult(tenders=_served(), complete=True)

    async def validate(self, ctx, spec, result, previous):
        return Verdict(ok=True)

    async def repair(self, ctx, old_spec, failures):
        return old_spec


def _wait(fn, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_contract_verification_field_and_audit_endpoints(tmp_path):
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0, audit_interval=0)
    app = create_app(s, pipeline=_ServingPipeline(), llm=_PageLLM, transport=_pages_transport(PAGES))
    with TestClient(app) as client:
        pid = client.post("/portals", json={"url": "http://p.test/"}).json()["id"]
        _wait(lambda: client.get(f"/portals/{pid}").json()["state"] == "serving")
        v = client.get(f"/portals/{pid}").json()["verification"]
        assert v["status"] == "pending" and v["lastAuditAt"] is None
        assert client.post("/_ui/audit/p_unknown").status_code == 404
        r = client.post(f"/_ui/audit/{pid}")
        assert r.status_code == 202 and r.json()["auditId"].startswith("a_")
        _wait(lambda: client.get(f"/portals/{pid}").json()["verification"]["lastAuditAt"])
        v = client.get(f"/portals/{pid}").json()["verification"]
        for key in ("lastAuditAt", "sampled", "fieldsChecked", "fieldsMatched", "accuracy", "byField", "status"):
            assert key in v
        assert v["sampled"] == 2 and v["fieldsChecked"] == 11 and v["fieldsMatched"] == 10
        assert v["accuracy"] == round(10 / 11, 3) and v["status"] == "ok" and 0 <= v["accuracy"] <= 1
        assert v["byField"]["submissionDeadline"] == {"checked": 2, "matched": 1, "missing": 0}
        assert v["costUsd"] == 0.0004 and v["mismatches"] == 1
        d = client.get(f"/_ui/audit/{pid}").json()
        assert d["latest"]["mismatches"][0]["url"] == "http://p.test/t/T1" and len(d["history"]) == 1
        # a browser opening the data address is sent to the readable page; scripts keep the JSON
        page = client.get(f"/_ui/audit/{pid}", headers={"accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
                          follow_redirects=False)
        assert page.status_code == 303 and page.headers["location"] == f"/_ui/audit/{pid}/view"
        assert client.get(f"/_ui/audit/{pid}", headers={"accept": "*/*"}).json()["latest"]
        # an audit never changes what is served
        assert client.get(f"/portals/{pid}/tenders").json()[1]["submissionDeadline"] == "2030-12-11T13:00:00+01:00"
        assert [r["kind"] for r in client.get(f"/portals/{pid}/runs").json()] == ["read"]
