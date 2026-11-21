"""W2 P0/P2: one acceptance gate for attach, refresh and repair; terminal failures stamp the attempt;
refresh coalescing; ids; fractional-second change stamps."""
from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from tests.test_contract import FakePipeline, tender, wait_runs_done, wait_state
from watchtower import timeutil
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.http import BudgetExceeded
from watchtower.pipeline import PortalBlocked, ReadResult
from watchtower.store import Store
from watchtower.urls import portal_id_for


class StatsPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.stats = {}
        self.cap = False

    async def read(self, ctx, spec):
        self.reads += 1
        if self.read_delay:
            await asyncio.sleep(self.read_delay)
        return ReadResult(tenders=[dict(t) for t in self.tenders], complete=self.complete, stats=dict(self.stats))


@pytest.fixture
def env(tmp_path):
    fake = StatsPipeline()
    app = create_app(Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0, serve_capped=True),
                     pipeline=fake)
    with TestClient(app) as client:
        yield client, fake, app.state.store


def add(client, url="https://w2.example"):
    return client.post("/portals", json={"url": url}).json()["id"]


# ------------------------------------------------------------------ P0 the gate
def test_capped_first_attach_serves_with_an_honest_reason_naming_the_counts(env):
    client, fake, _ = env
    fake.complete = False
    fake.stats = {"capped": True, "stopReason": "portal refuses pages beyond 200 (HTTP 400)",
                  "pendingDetails": 4, "pendingDocs": 9, "docBlocked": 2, "docBlockReason": "login required"}
    p = wait_state(client, add(client), {"serving", "failed"})
    assert p["state"] == "serving" and p["tenderCount"] == 5
    r = p["reason"]
    assert "unseen tenders kept, not closed" in r and "4 tenders" in r and "9 documents" in r
    assert "Documents: not available from this portal (login required)" in r


def test_incomplete_uncapped_first_attach_fails_and_publishes_nothing(env):
    client, fake, store = env
    fake.complete = False
    fake.stats = {"capped": False, "stopReason": "page 2 repeated page 1"}
    pid = add(client)
    p = wait_state(client, pid, {"failed"})
    assert "incomplete" in p["reason"] and "page 2 repeated page 1" in p["reason"]
    assert client.get(f"/portals/{pid}/tenders").json() == [] and store.current_spec(pid) is None
    wait_runs_done(client, pid)
    row = store.get_portal(pid)
    assert row["failed_streak"] == 1 and row["last_attempt_at"]  # terminal failure stamped for the backoff


def test_incomplete_uncapped_refresh_heals_then_fails_without_closing_anything(env):
    client, fake, store = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    before = client.get(f"/portals/{pid}/tenders").json()
    fake.tenders = [tender(i) for i in range(3)]       # two tenders look delisted ...
    fake.complete, fake.stats = False, {"capped": False, "stopReason": "stopped early"}  # ... but the read is unfinished
    client.post(f"/portals/{pid}/refresh")
    p = wait_state(client, pid, {"failed"})
    kinds = [(r["kind"], r["result"]) for r in wait_runs_done(client, pid)]
    assert kinds[-2:] == [("read", "failed"), ("repair", "failed")], kinds
    assert "person is needed" in p["reason"] and "incomplete" in p["reason"]
    assert client.get(f"/portals/{pid}/tenders").json() == before          # nothing closed, nothing changed
    assert store.get_portal(pid)["failed_streak"] == 1                     # read+repair is ONE terminal failure


def test_capped_refresh_publishes_but_closes_nothing(env):
    client, fake, _ = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    fake.tenders = [tender(i, title=f"Tender {i} v2") for i in range(3)]
    fake.complete, fake.stats = False, {"capped": True, "stopReason": "page cap", "pendingDocs": 1}
    client.post(f"/portals/{pid}/refresh")
    wait_runs_done(client, pid)
    p = wait_state(client, pid, {"serving"})
    ts = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert ts["T-0"]["title"] == "Tender 0 v2" and ts["T-4"]["status"] == "open"
    assert "1 documents" in p["reason"] and "kept, not closed" in p["reason"]


def test_healing_is_reported_while_repair_runs_and_budget_exhaustion_is_explained(env):
    client, fake, store = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    before = client.get(f"/portals/{pid}/tenders").json()
    fake.read_ok = False
    gate = asyncio.Event()

    async def repair(ctx, old, failures):
        await gate.wait()
        raise BudgetExceeded("request budget of 1500 exhausted")
    fake.repair = repair
    client.post(f"/portals/{pid}/refresh")
    p = wait_state(client, pid, {"healing"})
    assert "serving the last correct 5 tenders" in p["reason"]
    assert client.get(f"/portals/{pid}/tenders").json() == before
    client.portal.call(gate.set)
    p = wait_state(client, pid, {"failed"})
    assert "budget" in p["reason"] and "incomplete" in p["reason"] and "last correct" in p["reason"]
    assert client.get(f"/portals/{pid}/tenders").json() == before
    assert store.get_portal(pid)["failed_streak"] == 1


def test_blocked_during_repair_is_a_terminal_failure_too(env):
    client, fake, store = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    fake.read_ok = False

    async def repair(ctx, old, failures):
        raise PortalBlocked("login required to view the tender list")
    fake.repair = repair
    client.post(f"/portals/{pid}/refresh")
    p = wait_state(client, pid, {"failed"})
    assert "login" in p["reason"].lower()
    wait_runs_done(client, pid)
    assert store.get_portal(pid)["failed_streak"] == 1


def test_successful_repair_resets_the_streak(env):
    client, fake, store = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    fake.read_ok = False
    client.post(f"/portals/{pid}/refresh")
    wait_runs_done(client, pid)
    wait_state(client, pid, {"serving"})
    assert store.get_portal(pid)["failed_streak"] == 0


# ------------------------------------------------------------------ P2 lifecycle and API edges
def test_same_url_same_id_across_spellings(env):
    client, _, _ = env
    a = add(client, "https://Same.example/tenders/")
    b = add(client, "https://same.example/tenders#frag")
    c = add(client, "same.example/tenders")
    assert a == b == c == portal_id_for("https://same.example/tenders")
    assert len(client.get("/portals").json()) == 1


def test_refresh_coalesces_during_read_and_during_repair(env):
    client, fake, _ = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    fake.read_delay = 0.6
    ids = {client.post(f"/portals/{pid}/refresh").json()["runId"] for _ in range(3)}
    assert len(ids) == 1
    wait_runs_done(client, pid)
    fake.read_delay, fake.read_ok = 0.0, False
    gate = asyncio.Event()
    orig = fake.repair

    async def repair(ctx, old, failures):
        await gate.wait()
        return await orig(ctx, old, failures)
    fake.repair = repair
    client.post(f"/portals/{pid}/refresh")
    wait_state(client, pid, {"healing"})
    active = [r for r in client.get(f"/portals/{pid}/runs").json() if r["finishedAt"] is None]
    assert [r["kind"] for r in active] == ["repair"]
    assert {client.post(f"/portals/{pid}/refresh").json()["runId"] for _ in range(3)} == {active[0]["id"]}
    client.portal.call(gate.set)
    wait_state(client, pid, {"serving"})
    assert fake.repairs == 1


def test_known_and_unknown_ids(env):
    client, _, _ = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    assert client.get(f"/portals/{pid}").status_code == 200
    assert client.get(f"/portals/{pid}/tenders").status_code == 200
    assert client.get(f"/portals/{pid}/runs").status_code == 200
    assert client.post(f"/portals/{pid}/refresh").status_code == 202
    for method, path in [("get", "/portals/p_nope"), ("get", "/portals/p_nope/tenders"),
                         ("get", "/portals/p_nope/runs"), ("post", "/portals/p_nope/refresh"),
                         ("delete", "/portals/p_nope")]:
        assert getattr(client, method)(path).status_code == 404, (method, path)
    assert client.get("/files/" + "0" * 64).status_code == 404 and client.get("/files/zz").status_code == 404
    wait_runs_done(client, pid)
    assert client.delete(f"/portals/{pid}").status_code == 204
    assert client.get(f"/portals/{pid}").status_code == 404 and client.delete(f"/portals/{pid}").status_code == 404


def test_delete_during_repair_stops_the_job(env):
    client, fake, store = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    fake.read_ok = False
    calls = {"n": 0}

    async def repair(ctx, old, failures):
        calls["n"] += 1
        await asyncio.sleep(5)
    fake.repair = repair
    client.post(f"/portals/{pid}/refresh")
    wait_state(client, pid, {"healing"})
    assert client.delete(f"/portals/{pid}").status_code == 204
    time.sleep(0.3)
    reads = fake.reads
    time.sleep(0.5)
    assert fake.reads == reads and calls["n"] == 1 and store.get_portal(pid) is None
    assert store.list_runs(pid) == [] and store.tender_count(pid) == 0


def test_fractional_seconds_in_last_changed_and_strict_changed_since(tmp_path, monkeypatch):
    store = Store(str(tmp_path))
    store.create_portal("p_x", "https://x.example", "https://x.example/")
    base = timeutil.now().replace(microsecond=250_000)
    clock = {"t": base}
    monkeypatch.setattr(timeutil, "now", lambda: clock["t"])
    store.commit_snapshot("p_x", [tender(1), tender(2)], True)
    clock["t"] = base + timedelta(microseconds=1500)               # same second, a moment later
    store.commit_snapshot("p_x", [tender(1), tender(2, title="amended")], True)
    ts = {t["id"]: t for t in store.served_tenders("p_x")}
    assert ts["T-1"]["lastChangedAt"] == base.isoformat() and "." in ts["T-1"]["lastChangedAt"]
    assert ts["T-2"]["lastChangedAt"] == (base + timedelta(microseconds=1500)).isoformat()
    cursor = timeutil.parse(ts["T-1"]["lastChangedAt"])
    assert [t["id"] for t in store.served_tenders("p_x", cursor)] == ["T-2"]               # strict: T-1 excluded
    assert store.served_tenders("p_x", timeutil.parse(ts["T-2"]["lastChangedAt"])) == []   # strict at the newest
    assert timeutil.parse("2026-11-20T10:00:00.123456 01:00").utcoffset() == timedelta(hours=1)  # '+' decoded as ' '


def test_changed_since_roundtrip_over_http_keeps_fractions(env):
    client, fake, _ = env
    pid = add(client)
    wait_state(client, pid, {"serving"})
    wait_runs_done(client, pid)
    stamps = [t["lastChangedAt"] for t in client.get(f"/portals/{pid}/tenders").json()]
    assert all("." in s or s.endswith("+00:00") for s in stamps)
    newest = max(stamps, key=timeutil.parse)
    assert client.get(f"/portals/{pid}/tenders", params={"changedSince": newest}).json() == []

