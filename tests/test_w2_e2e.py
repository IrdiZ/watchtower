"""W2 acceptance on the mock portal: green -> swap_columns -> failed read -> healing -> last snapshot served during a
slowed repair -> repaired output and document hashes match the fixture. Plus: DELETE stops requests mid-attach."""
from __future__ import annotations

import asyncio
import hashlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_portal import app as mock
from tests.fake_llm import BASE, FakeLLM
from watchtower.api import create_app
from watchtower.config import Settings

FIELDS = ("title", "buyerName", "publicationDate", "submissionDeadline", "referenceNumber", "status")


class SlowLLM(FakeLLM):
    """FakeLLM whose model calls can be slowed down (to observe `healing` from outside)."""
    delay = 0.0

    async def json_call(self, *a, **kw):
        if self.delay:
            await asyncio.sleep(self.delay)
        return await super().json_call(*a, **kw)


def make(tmp_path, interval=0):
    mock.CONFIG.update(skin="html", mutations=set())
    mock.REQUESTS.clear()
    holder = {}

    def make_llm(store):
        holder["llm"] = SlowLLM(store)
        return holder["llm"]
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=interval)
    return create_app(s, llm=make_llm, transport=httpx.ASGITransport(app=mock.app)), holder


@pytest.fixture
def env(tmp_path):
    app, holder = make(tmp_path)
    with TestClient(app) as client:
        yield client, holder["llm"], app.state.store
    mock.CONFIG.update(skin="html", mutations=set())


def poll(client, pid, states, timeout=30.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        p = client.get(f"/portals/{pid}").json()
        if p["state"] in states:
            return p
        time.sleep(step)
    raise AssertionError(client.get(f"/portals/{pid}").json())


def settle(client, pid, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        runs = client.get(f"/portals/{pid}/runs").json()
        if runs and all(r["finishedAt"] for r in runs):
            return runs
        time.sleep(0.02)
    raise AssertionError("runs never finished")


def doc_hashes(client, tender):
    out = {}
    for d in tender["documents"]:
        body = client.get(d["path"]).content
        assert hashlib.sha256(body).hexdigest() == d["sha256"] and len(body) == d["sizeBytes"]
        out[d["fileName"]] = d["sha256"]
    return out


def test_acceptance_green_swap_fail_heal_serve_during_repair_match_fixture(env):
    client, llm, store = env
    pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
    p = poll(client, pid, {"serving", "failed"})
    assert p["state"] == "serving", p
    settle(client, pid)
    before = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert len(before) == 12
    hashes_before = {tid: doc_hashes(client, t) for tid, t in before.items()}
    pdf = next(d for d in before["MH-2026-1000"]["documents"] if d["fileName"].endswith("Leistungsbeschreibung.pdf"))
    assert pdf["sha256"] == hashlib.sha256(mock.doc_bytes(pdf["fileName"])).hexdigest()

    # the portal silently swaps buyer and title (same shape), and the repair is slow
    mock.CONFIG["mutations"] = {"swap_columns"}
    llm.delay = 0.4
    client.post(f"/portals/{pid}/refresh")
    p = poll(client, pid, {"healing", "failed"})
    assert p["state"] == "healing", p                      # the bad read was caught, not published
    assert "serving the last correct 12 tenders" in p["reason"]
    # while the repair runs, the last correct snapshot is what is delivered (and nothing is swapped)
    served = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert served == before
    runs = client.get(f"/portals/{pid}/runs").json()
    assert [(r["kind"], r["result"]) for r in runs][-2:] == [("read", "failed"), ("repair", None)]
    assert client.get(pdf["path"]).content == mock.doc_bytes(pdf["fileName"])  # documents stay downloadable too

    p = poll(client, pid, {"serving", "failed"})
    assert p["state"] == "serving" and p["tenderCount"] == 12, p
    runs = settle(client, pid)
    assert [(r["kind"], r["result"]) for r in runs][-2:] == [("read", "failed"), ("repair", "ok")]
    after = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert set(after) == set(before)
    for tid, t in before.items():
        for f in FIELDS:
            assert after[tid].get(f) == t.get(f), (tid, f, after[tid].get(f), t.get(f))
        assert after[tid]["buyerName"] != after[tid]["title"]
        assert doc_hashes(client, after[tid]) == hashes_before[tid]  # byte-identical documents, same hashes
    assert store.get_portal(pid)["failed_streak"] == 0


def test_delete_during_attach_stops_further_requests(tmp_path):
    app, _ = make(tmp_path, interval=0.05)
    with TestClient(app) as client:
        pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
        end = time.time() + 20
        while len(mock.REQUESTS) < 8 and time.time() < end:
            time.sleep(0.02)
        assert len(mock.REQUESTS) >= 8 and client.get(f"/portals/{pid}").json()["state"] == "integrating"
        assert client.delete(f"/portals/{pid}").status_code == 204
        time.sleep(0.3)                       # let an in-flight request land
        n = len(mock.REQUESTS)
        time.sleep(1.0)
        assert len(mock.REQUESTS) == n, "requests continued after DELETE"
        assert client.get(f"/portals/{pid}").status_code == 404
        assert app.state.store.tender_count(pid) == 0 and app.state.store.list_runs(pid) == []
    mock.CONFIG.update(skin="html", mutations=set())


def test_legitimately_empty_listing_after_all_deadlines_is_served_as_zero_open(env, monkeypatch):
    """Attach green; time passes beyond every deadline; the portal's list is now empty -> still `serving`."""
    client, _, store = env
    pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
    assert poll(client, pid, {"serving", "failed"})["state"] == "serving"
    settle(client, pid)
    from datetime import datetime, timedelta, timezone
    from watchtower import timeutil
    future = datetime.now(timezone.utc) + timedelta(days=3650)
    monkeypatch.setattr(timeutil, "now", lambda: future)
    monkeypatch.setattr(mock, "TENDERS", [])
    client.post(f"/portals/{pid}/refresh")
    time.sleep(0.2)
    settle(client, pid)
    p = client.get(f"/portals/{pid}").json()
    assert p["state"] == "serving", p
    assert "(0 open)" in p["reason"]
    assert all(t["status"] in ("closed", "withdrawn") for t in client.get(f"/portals/{pid}/tenders").json())


def test_open_tenders_vanishing_is_rejected_and_kept(env, monkeypatch):
    client, _, _ = env
    pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
    assert poll(client, pid, {"serving", "failed"})["state"] == "serving"
    settle(client, pid)
    before = client.get(f"/portals/{pid}/tenders").json()
    monkeypatch.setattr(mock, "TENDERS", [])
    client.post(f"/portals/{pid}/refresh")
    time.sleep(0.2)
    settle(client, pid)
    assert client.get(f"/portals/{pid}/tenders").json() == before
    runs = client.get(f"/portals/{pid}/runs").json()
    read_runs = [r for r in runs if r["kind"] == "read"]
    assert read_runs[-1]["result"] == "failed"
