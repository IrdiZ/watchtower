"""H1 gate: API contract + state machine, using a fake pipeline (no network, no LLM)."""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.pipeline import AttachFailed, PortalBlocked, ReadResult, Verdict


def tender(i: int, **kw) -> dict:
    t = {"id": f"T-{i}", "url": f"https://portal.example/t/{i}", "title": f"Tender {i}",
         "buyerName": "City of Example", "publicationDate": "2026-11-01",
         "submissionDeadline": "2030-12-09T10:00:00+01:00", "status": "open", "documents": []}
    t.update(kw)
    return t


class FakePipeline:
    def __init__(self):
        self.tenders = [tender(i) for i in range(5)]
        self.complete = True
        self.read_ok = True
        self.block_on_attach = False
        self.attach_fails = False
        self.repair_ok = True
        self.read_delay = 0.0
        self.reads = 0
        self.repairs = 0

    async def attach(self, ctx):
        if self.block_on_attach:
            raise PortalBlocked("login required to view tender list")
        if self.attach_fails:
            raise AttachFailed("no tender list found")
        return {"version": 1, "list": {"kind": "html"}}

    async def read(self, ctx, spec):
        self.reads += 1
        if self.read_delay:
            await asyncio.sleep(self.read_delay)
        return ReadResult(tenders=[dict(t) for t in self.tenders], complete=self.complete)

    async def validate(self, ctx, spec, result, previous):
        if spec.get("repaired"):
            return Verdict(ok=True)
        return Verdict(ok=self.read_ok, failures=[] if self.read_ok else ["field profile broken: buyerName"])

    async def repair(self, ctx, old_spec, failures):
        self.repairs += 1
        if not self.repair_ok:
            raise AttachFailed("could not re-learn the portal")
        return {**old_spec, "repaired": True}


@pytest.fixture
def setup(tmp_path):
    fake = FakePipeline()
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    app = create_app(s, pipeline=fake)
    with TestClient(app) as client:
        yield client, fake


def wait_state(client, pid, states, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        p = client.get(f"/portals/{pid}").json()
        if p["state"] in states:
            return p
        time.sleep(0.05)
    raise AssertionError(f"state never reached {states}: {client.get(f'/portals/{pid}').json()}")


def wait_runs_done(client, pid, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        runs = client.get(f"/portals/{pid}/runs").json()
        if runs and all(r["finishedAt"] for r in runs):
            return runs
        time.sleep(0.05)
    raise AssertionError("runs never finished")


def test_add_is_idempotent_and_attaches(setup):
    client, _ = setup
    r = client.post("/portals", json={"url": "https://Portal.example/tenders/"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert client.post("/portals", json={"url": "https://portal.example/tenders"}).json()["id"] == pid
    p = wait_state(client, pid, {"serving"})
    assert p["tenderCount"] == 5 and p["lastSuccessAt"] is not None
    assert set(p) == {"id", "url", "state", "reason", "tenderCount", "lastSuccessAt", "coverage", "verification"}
    assert p["verification"]["status"] == "pending" and p["verification"]["accuracy"] is None
    assert p["coverage"]["tenders"] in ("complete", "partial", "unknown") and "documents" in p["coverage"]
    runs = wait_runs_done(client, pid)
    assert runs[0]["kind"] == "read" and runs[0]["result"] == "ok"
    tenders = client.get(f"/portals/{pid}/tenders").json()
    assert len(tenders) == 5 and all(t["status"] == "open" and t["lastChangedAt"] for t in tenders)


def test_unknown_id_404(setup):
    client, _ = setup
    for method, path in [("get", "/portals/nope"), ("delete", "/portals/nope"), ("post", "/portals/nope/refresh"),
                         ("get", "/portals/nope/tenders"), ("get", "/portals/nope/runs")]:
        assert getattr(client, method)(path).status_code == 404


def test_refresh_coalesces(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://a.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    fake.read_delay = 0.5
    r1 = client.post(f"/portals/{pid}/refresh")
    r2 = client.post(f"/portals/{pid}/refresh")
    assert r1.status_code == r2.status_code == 202
    assert r1.json()["runId"] == r2.json()["runId"]


def test_delete_stops_and_404s(setup):
    client, fake = setup
    fake.read_delay = 1.0
    pid = client.post("/portals", json={"url": "https://b.example"}).json()["id"]
    assert client.delete(f"/portals/{pid}").status_code == 204
    assert client.get(f"/portals/{pid}").status_code == 404
    reads_after_delete = fake.reads
    time.sleep(1.2)
    assert fake.reads == reads_after_delete


def test_blocked_portal_fails_with_reason(setup):
    client, fake = setup
    fake.block_on_attach = True
    pid = client.post("/portals", json={"url": "https://login.example"}).json()["id"]
    p = wait_state(client, pid, {"failed"})
    assert "login" in p["reason"].lower()


def test_partial_first_read_never_serves(setup):
    client, fake = setup
    fake.complete = False
    pid = client.post("/portals", json={"url": "https://partial.example"}).json()["id"]
    p = wait_state(client, pid, {"failed"})
    assert "incomplete" in p["reason"]
    assert client.get(f"/portals/{pid}/tenders").json() == []


def test_break_heal_serving_keeps_last_snapshot(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://heal.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    before = client.get(f"/portals/{pid}/tenders").json()
    # portal changes: reads now fail validation; repair is slow so we can observe `healing`
    fake.read_ok = False
    fake.tenders = [tender(i, buyerName=f"WRONG {i}") for i in range(5)]
    orig_repair = fake.repair

    async def slow_repair(ctx, old, failures):
        await asyncio.sleep(0.5)
        fake.tenders = [tender(i) for i in range(6)]
        return await orig_repair(ctx, old, failures)
    fake.repair = slow_repair
    client.post(f"/portals/{pid}/refresh")
    wait_state(client, pid, {"healing"})
    assert client.get(f"/portals/{pid}/tenders").json() == before  # still serving last correct
    p = wait_state(client, pid, {"serving"})
    assert p["tenderCount"] == 6
    kinds = [(r["kind"], r["result"]) for r in wait_runs_done(client, pid)]
    assert kinds[-2:] == [("read", "failed"), ("repair", "ok")]


def test_unrepairable_goes_failed_but_keeps_serving_data(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://dead.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    fake.read_ok, fake.repair_ok = False, False
    client.post(f"/portals/{pid}/refresh")
    p = wait_state(client, pid, {"failed"})
    assert "person is needed" in p["reason"]
    assert len(client.get(f"/portals/{pid}/tenders").json()) == 5


def test_changed_since_and_closing(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://cs.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    wait_runs_done(client, pid)
    t0 = client.get(f"/portals/{pid}/tenders").json()[0]["lastChangedAt"]
    time.sleep(1.1)
    fake.tenders = [tender(0, title="Tender 0 (amended)")] + [tender(i) for i in range(1, 4)]  # T-4 delisted
    client.post(f"/portals/{pid}/refresh")
    time.sleep(0.3)
    wait_runs_done(client, pid)
    # missing from ONE complete read: remembered, still open (one odd read never closes a tender)
    assert {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}["T-4"]["status"] == "open"
    client.post(f"/portals/{pid}/refresh")          # missing again: now it closes
    time.sleep(0.3)
    wait_runs_done(client, pid)
    changed = client.get(f"/portals/{pid}/tenders", params={"changedSince": t0}).json()
    ids = {t["id"]: t for t in changed}
    assert set(ids) == {"T-0", "T-4"}
    assert ids["T-4"]["status"] == "closed"
    assert client.get(f"/portals/{pid}/tenders", params={"changedSince": "garbage"}).status_code == 400


def test_past_deadline_is_closed(setup):
    client, fake = setup
    fake.tenders = [tender(0, submissionDeadline="2020-01-01T10:00:00+01:00"), tender(1)]
    pid = client.post("/portals", json={"url": "https://old.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    st = {t["id"]: t["status"] for t in client.get(f"/portals/{pid}/tenders").json()}
    assert st == {"T-0": "closed", "T-1": "open"}


def test_dashboard_served(setup):
    client, _ = setup
    r = client.get("/")
    assert r.status_code == 200 and "Watchtower" in r.text
