"""Demo layer: the mock portal mounted inside Watchtower (/_demo/portal/), the heal-page controls (/_ui/heal/...),
and the events timeline (/_ui/events). The worker reads the demo portal over HTTP through Watchtower's own app,
exactly as it would in the container."""
from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_portal import app as mock
from tests.fake_llm import FakeLLM
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.events import spec_diff
from watchtower.http import HostRateLimiter


@pytest.fixture
def demo(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHTOWER_DEMO", "1")
    mock.CONFIG.update(skin="html", mutations=set())
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    holder = {}

    def make_llm(store):
        holder["llm"] = FakeLLM(store)
        return holder["llm"]
    app = create_app(s, llm=make_llm)
    app.state.manager.transport = httpx.ASGITransport(app=app)   # the worker reaches the demo portal via this app
    with TestClient(app) as client:
        yield client, holder["llm"]
    mock.CONFIG.update(skin="html", mutations=set())


def settle(client, pid, states, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        p = client.get(f"/portals/{pid}").json()
        if p["state"] in states and not any(r["finishedAt"] is None for r in client.get(f"/portals/{pid}/runs").json()):
            return p
        time.sleep(0.05)
    raise AssertionError(client.get(f"/portals/{pid}").json())


def test_mounted_portal_links_stay_under_the_prefix(demo):
    client, _ = demo
    page = client.get("/_demo/portal/ausschreibungen?page=2").text
    assert "href='/_demo/portal/tender/MH-2026-1005'" in page and "href='/_demo/portal/ausschreibungen?page=1'" in page
    assert "href='/tender/" not in page and "href='/ausschreibungen" not in page
    assert client.get("/_demo/portal/download/MH-2026-1000-Vergabeunterlagen.zip").content == \
        mock.doc_bytes("MH-2026-1000-Vergabeunterlagen.zip")
    # the mock itself, at its own root, is unchanged
    with TestClient(mock.app) as raw:
        assert "href='/tender/MH-2026-1005'" in raw.get("/ausschreibungen?page=2").text


def test_break_heal_cached_read_and_events(demo):
    client, llm = demo
    info = client.get("/_ui/heal/demo").json()
    assert info["enabled"] and info["portalId"] is None and info["url"].endswith("/_demo/portal/")
    pid = client.post("/_ui/heal/attach").json()["id"]
    assert settle(client, pid, {"serving", "failed"})["state"] == "serving"
    assert client.get("/_ui/heal/demo").json()["portalId"] == pid
    mark = client.get("/_ui/events").json()["now"]

    r = client.post("/_ui/heal/break", json={"mutation": "rename_classes"}).json()
    assert r["mutations"] == ["rename_classes"] and r["runId"]
    p = settle(client, pid, {"serving", "failed"})
    assert p["state"] == "serving" and p["tenderCount"] == 12
    calls_after_heal = len(llm.calls)
    run = client.post("/_ui/heal/refresh", json={}).json()["runId"]
    settle(client, pid, {"serving"})

    ev = client.get("/_ui/events", params={"since": mark, "portal": pid}).json()["events"]
    assert all(e["at"] > mark for e in ev)
    states = [(e["from"], e["to"], e["tenders"]) for e in ev if e["type"] == "state"]
    assert states == [("serving", "healing", 12), ("healing", "serving", 12)], states
    runs = [e for e in ev if e["type"] == "run"]
    assert [(e["kind"], e["result"]) for e in runs] == [("read", "failed"), ("repair", "ok"), ("read", "ok")]
    noticed, repair, cached = runs
    assert noticed["failures"] and noticed["llmCalls"] == 0
    assert repair["llmCalls"] > 0 and repair["was"] == noticed["failures"] and repair["attempts"][-1]["ok"]
    assert cached["runId"] == run and cached["llmCalls"] == 0 and cached["durationS"] is not None
    assert len(llm.calls) == calls_after_heal
    spec = next(e for e in ev if e["type"] == "spec")
    assert (spec["previous"], spec["version"], spec["sourceRun"]) == (1, 2, repair["runId"])
    assert {"path": "list.items", "old": "tr.tender-row", "new": "tr.ergebnis-zeile"} in spec["changes"]

    assert client.post("/_ui/heal/restore").json()["mutations"] == []
    assert mock.CONFIG["mutations"] == set()
    first = client.get("/_ui/events", params={"portal": pid}).json()["events"]
    assert next(e for e in first if e["type"] == "run")["kind"] == "attach"


def test_cosmetic_popup_needs_no_repair(demo):
    client, llm = demo
    pid = client.post("/_ui/heal/attach").json()["id"]
    settle(client, pid, {"serving"})
    n = len(llm.calls)
    client.post("/_ui/heal/break", json={"mutation": "overlay"})
    assert "Wichtiger Hinweis" in client.get("/_demo/portal/ausschreibungen").text
    p = settle(client, pid, {"serving", "failed"})
    runs = client.get(f"/portals/{pid}/runs").json()
    assert p["state"] == "serving" and runs[-1]["kind"] == "read" and runs[-1]["result"] == "ok"
    assert len(llm.calls) == n


def test_controls_validate_input(demo):
    client, _ = demo
    assert client.post("/_ui/heal/break", json={"mutation": "nope"}).status_code == 400
    assert client.post("/_ui/heal/refresh", json={}).status_code == 404       # demo portal not attached yet
    assert client.post("/_ui/heal/refresh", json={"portal": "p_x"}).status_code == 404
    assert client.get("/_ui/events", params={"since": "yesterday"}).status_code == 400
    assert client.get("/_ui/heal").status_code == 200


def test_demo_off(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHTOWER_DEMO", "0")
    app = create_app(Settings(data_dir=str(tmp_path), refresh_interval=0), llm=FakeLLM)
    with TestClient(app) as client:
        assert client.get("/_demo/portal/").status_code == 404
        assert client.post("/_ui/heal/break", json={"mutation": "rename_classes"}).status_code == 404
        assert client.get("/_ui/heal/demo").json()["enabled"] is False
        assert client.get("/_ui/heal").status_code == 200 and client.get("/_ui/events").json()["events"] == []


def test_only_loopback_may_be_faster():
    lim = HostRateLimiter(1.05, {"localhost": 0.25, "example.org": 0.1, "slow.example": 2.0})
    assert lim.interval("localhost") == 0.25
    assert lim.interval("example.org") == 1.05 and lim.interval("slow.example") == 2.0
    assert lim.interval("127.0.0.1") == 1.05


def test_spec_diff_paths():
    old = {"list": {"items": "tr.a", "pagination": {"param": "page"}}, "fields": {"title": {"sel": "td.t"}},
           "profile": {"x": 1}}
    new = {"list": {"items": "tr.b", "pagination": {"param": "page"}}, "fields": {"title": {"sel": "td.t"},
           "buyerName": {"sel": "td.b"}}, "profile": {"x": 2}}
    assert spec_diff(old, new) == [{"path": "fields.buyerName.sel", "old": None, "new": "td.b"},
                                   {"path": "list.items", "old": "tr.a", "new": "tr.b"}]
