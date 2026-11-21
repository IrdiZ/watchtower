"""H2/H3 gates on the mock portal: attach from a bare URL, documents byte-exact, every mutation detected,
repaired (or honestly failed), last snapshot served while healing. Uses FakeLLM (no network)."""
from __future__ import annotations

import hashlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_portal import app as mock
from tests.fake_llm import BASE, FakeLLM
from watchtower.api import create_app
from watchtower.config import Settings


@pytest.fixture
def env(tmp_path):
    mock.CONFIG.update(skin="html", mutations=set())
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    holder = {}

    def make_llm(store):
        holder["llm"] = FakeLLM(store)
        return holder["llm"]
    app = create_app(s, llm=make_llm, transport=httpx.ASGITransport(app=mock.app))
    with TestClient(app) as client:
        yield client, holder["llm"]
    mock.CONFIG.update(skin="html", mutations=set())


def wait(client, pid, states, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        p = client.get(f"/portals/{pid}").json()
        if p["state"] in states and not any(r["finishedAt"] is None for r in client.get(f"/portals/{pid}/runs").json()):
            return p
        time.sleep(0.05)
    raise AssertionError(client.get(f"/portals/{pid}").json())


def attach(client, path="/"):
    pid = client.post("/portals", json={"url": BASE + path}).json()["id"]
    return pid, wait(client, pid, {"serving", "failed"})


def test_attach_html_from_homepage(env):
    client, llm = env
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    ts = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert len(ts) == 12
    t = ts["MH-2026-1000"]
    assert t["title"] == "Dachsanierung Grundschule" and t["buyerName"] == "Stadt Musterhausen"
    assert t["publicationDate"] == "2026-11-20"
    assert t["submissionDeadline"] == "2026-12-10T10:00:00+01:00"
    assert t["cpvCodes"] == ["45261000"] and t["referenceNumber"] == "VG 300/26"
    assert t["placeOfPerformance"] == {"postalCode": "12345", "city": "Musterhausen"}
    assert t["procedureType"] == "open"
    assert ts["MH-2026-1007"]["status"] == "withdrawn"
    names = sorted(d["fileName"] for d in t["documents"])
    assert names == ["Formblatt_Angebot.pdf", "MH-2026-1000-Leistungsbeschreibung.pdf", "Vertragsentwurf.docx"]
    pdf = next(d for d in t["documents"] if d["fileName"].endswith("Leistungsbeschreibung.pdf"))
    body = client.get(pdf["path"]).content
    expected = mock.doc_bytes("MH-2026-1000-Leistungsbeschreibung.pdf")
    assert body == expected and pdf["sha256"] == hashlib.sha256(expected).hexdigest()
    assert not any(k.startswith("_") for k in t)
    # a regular read makes no model call
    n = len(llm.calls)
    client.post(f"/portals/{pid}/refresh")
    assert wait(client, pid, {"serving"})["state"] == "serving"
    assert len(llm.calls) == n


def test_attach_json_api(env):
    client, _ = env
    mock.CONFIG["skin"] = "api"
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    ts = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert len(ts) == 12 and ts["MH-2026-1001"]["url"] == f"{BASE}/app/tender/MH-2026-1001"
    assert ts["MH-2026-1001"]["placeOfPerformance"]["countryCode"] == "DE"
    assert len(ts["MH-2026-1001"]["documents"]) == 3


@pytest.mark.parametrize("mutation", ["rename_classes", "restructure", "pagination", "date_format", "swap_columns"])
def test_break_and_heal_html(env, mutation):
    client, _ = env
    pid, p = attach(client, "/")
    assert p["state"] == "serving"
    before = client.get(f"/portals/{pid}/tenders").json()
    mock.CONFIG["mutations"] = {mutation}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"serving", "failed"})
    runs = client.get(f"/portals/{pid}/runs").json()
    assert [r["kind"] for r in runs][-2:] == ["read", "repair"], (mutation, runs)
    assert runs[-2]["result"] == "failed", f"{mutation} was not noticed"
    assert p["state"] == "serving" and runs[-1]["result"] == "ok", (mutation, p)
    after = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    for t in before:  # repaired output equals the truth
        for f in ("title", "buyerName", "publicationDate", "submissionDeadline"):
            assert after[t["id"]][f] == t[f], (mutation, f, after[t["id"]][f], t[f])


def test_move_urls_is_not_a_break(env):
    client, _ = env
    pid, _ = attach(client, "/")
    mock.CONFIG["mutations"] = {"move_urls"}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"serving", "failed"})
    assert p["state"] == "serving"
    assert all("/notice/" in t["url"] for t in client.get(f"/portals/{pid}/tenders").json())


def test_rename_keys_api_heals(env):
    client, _ = env
    mock.CONFIG["skin"] = "api"
    pid, _ = attach(client, "/")
    mock.CONFIG["mutations"] = {"rename_keys"}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"serving", "failed"})
    assert p["state"] == "serving", p
    assert client.get(f"/portals/{pid}/runs").json()[-1]["kind"] == "repair"


def test_login_wall_fails_with_reason_and_keeps_data(env):
    client, _ = env
    pid, _ = attach(client, "/")
    mock.CONFIG["mutations"] = {"login_wall"}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"failed"})
    assert "login" in p["reason"].lower()
    assert len(client.get(f"/portals/{pid}/tenders").json()) == 12


def test_login_wall_at_attach(env):
    client, _ = env
    mock.CONFIG["mutations"] = {"login_wall"}
    pid, p = attach(client, "/ausschreibungen")
    assert p["state"] == "failed" and "login" in p["reason"].lower(), p


def test_decorative_captcha_still_serves(env):
    client, _ = env
    mock.CONFIG["mutations"] = {"captcha_decor"}
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p


def test_error500_then_recovers(env):
    client, _ = env
    pid, _ = attach(client, "/")
    mock.CONFIG["mutations"] = {"error500"}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"failed"})
    assert len(client.get(f"/portals/{pid}/tenders").json()) == 12
    mock.CONFIG["mutations"] = set()
    client.post(f"/portals/{pid}/refresh")
    assert wait(client, pid, {"serving"})["state"] == "serving"


def test_rate_limit_holds_across_attach(tmp_path):
    """Every request (discovery, list, details, documents) respects the per-host spacing."""
    mock.CONFIG.update(skin="html", mutations=set())
    mock.REQUESTS.clear()
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0.1)
    app = create_app(s, llm=FakeLLM, transport=httpx.ASGITransport(app=mock.app))
    with TestClient(app) as client:
        pid, p = attach(client, "/")
        assert p["state"] == "serving"
    stamps = sorted(t for t, _ in mock.REQUESTS)
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) > 20 and min(gaps) >= 0.095, min(gaps)


def test_attach_directly_from_json_api_url(env):
    client, _ = env
    mock.CONFIG["skin"] = "api"
    pid, p = attach(client, "/api/tenders?offset=0&limit=5")
    assert p["state"] == "serving", p
    assert p["tenderCount"] == 12


def test_deep_paging_refusal_is_not_a_break(env):
    """A portal that refuses pages beyond a window (HTTP 400) still serves what it shows, closing nothing.
    The page size is capped too, so a bigger page cannot fit the listing into the window."""
    client, _ = env
    mock.CONFIG.update(skin="api", max_offset=10, max_limit=5)
    try:
        # opt-out policy (WATCHTOWER_SERVE_CAPPED=0): a capped listing is partial -> failed with the reason
        client.app.state.settings.serve_capped = False
        pid0, p0 = attach(client, "/")
        assert p0["state"] == "failed" and "only part of its listing" in p0["reason"], p0
        client.delete(f"/portals/{pid0}")
        client.app.state.settings.serve_capped = True   # Patterno accepts capped listings -> serve honestly
        pid, p = attach(client, "/")
        assert p["state"] == "serving", p
        assert p["tenderCount"] == 10 and "kept, not closed" in p["reason"] and "Coverage: partial" in p["reason"]
        client.post(f"/portals/{pid}/refresh")
        p = wait(client, pid, {"serving", "failed"})
        assert p["state"] == "serving" and p["tenderCount"] == 10
        assert all(t["status"] != "closed" or t["id"] == "MH-2026-1007"
                   for t in client.get(f"/portals/{pid}/tenders").json())
    finally:
        mock.CONFIG.pop("max_offset", None)
        mock.CONFIG.pop("max_limit", None)


def test_bigger_pages_fit_a_windowed_listing(env):
    """Same window, but the API takes bigger pages: onboarding raises the page size and the read is complete."""
    client, _ = env
    mock.CONFIG.update(skin="api", max_offset=10)
    try:
        pid, p = attach(client, "/")
        assert p["state"] == "serving" and "kept, not closed" not in p["reason"], p
        assert p["tenderCount"] >= 11
    finally:
        mock.CONFIG.pop("max_offset", None)
