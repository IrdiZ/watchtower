"""Attach a postback-paged portal from its bare address (FakeLLM first makes the real model's mistake: the pager's
javascript: href as a next link), serve every tender, and never serve a broken pager as a complete read."""
from __future__ import annotations

import json

import pytest

from mock_portal import app as mock
from tests.test_e2e_mock import attach, env, wait  # noqa: F401  (fixture)


@pytest.fixture
def postback_env(env):
    mock.CONFIG.update(skin="postback", postback_flavor="prado")
    yield env
    mock.CONFIG.pop("postback_flavor", None)


def last_spec(client, pid):
    return client.app.state.store.current_spec(pid)[1]


@pytest.mark.parametrize("flavor", ["prado", "aspnet"])
def test_attach_postback_portal_serves_every_tender(postback_env, flavor):
    client, llm = postback_env
    mock.CONFIG["postback_flavor"] = flavor
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    assert p["tenderCount"] == 12 and "not closed" not in p["reason"], p
    pag = last_spec(client, pid)["list"]["pagination"]
    assert pag["type"] == "postback" and pag["size_sel"] == "select#ctl0_pager_size", pag
    assert llm.calls.count("synthesize") == 2          # the javascript: next link is corrected from the page itself
    ts = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert set(ts) == {t["id"] for t in mock.TENDERS}
    assert ts["MH-2026-1011"]["title"] == "Glasfaserausbau Gewerbegebiet"      # a row only on the last page
    log = json.loads(client.app.state.store.list_runs(pid)[-1]["detail"])["log"]
    assert any("pager set from the list page: postback" in line for line in log)
    assert any("results per page raised" in line for line in log)
    # a regular refresh walks the pager again, no model call
    n = len(llm.calls)
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"serving", "failed"})
    assert p["state"] == "serving" and p["tenderCount"] == 12 and len(llm.calls) == n


def test_row_unlike_the_learned_pages_is_learned_from_the_full_read(postback_env):
    """Atexo lists some tenders hosted on another platform; none were on the pages the model saw. The learned link
    selector misses that row; the runner takes the row's own link (it matches the title), so it serves at once - no
    extra model calls. The row is still kept as evidence for a later repair."""
    client, llm = postback_env
    mock.CONFIG["mutations"] = {"external_row"}
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    ts = {t["id"]: t for t in client.get(f"/portals/{pid}/tenders").json()}
    assert len(ts) == 12 and ts["MH-2026-1011"]["url"] == "https://extern.test/pub/MH-2026-1011"
    assert ts["MH-2026-1000"]["url"] == "http://mock.test/tender/MH-2026-1000"
    stats = json.loads(client.app.state.store.list_runs(pid)[-1]["detail"])["stats"]
    assert stats["urlFallbacks"] >= 1 and "url" in stats.get("missingExamples", {})
    assert llm.calls.count("synthesize") < 6        # no second learning pass was needed


def test_full_read_without_evidence_is_not_relearned(postback_env):
    """A pager that ends early fails the attach with the reason; no row explains it, so nothing is re-learned."""
    client, llm = postback_env
    mock.CONFIG["mutations"] = {"pager_stops"}
    pid, p = attach(client, "/")
    assert p["state"] == "failed" and "the pager ended at page 2 but the portal states 12" in p["reason"], p
    log = json.loads(client.app.state.store.list_runs(pid)[-1]["detail"])["log"]
    assert not any("learning once more" in line for line in log)


@pytest.mark.parametrize("mutation", ["pager_ignores_target", "pager_stops"])
def test_broken_pager_is_never_served_as_complete(postback_env, mutation):
    client, _ = postback_env
    pid, p = attach(client, "/")
    assert p["state"] == "serving", p
    mock.CONFIG["mutations"] = {mutation}
    client.post(f"/portals/{pid}/refresh")
    p = wait(client, pid, {"failed", "serving"}, timeout=60)
    runs = client.app.state.store.list_runs(pid)
    read_run = [r for r in runs if r["kind"] == "read"][-1]
    assert read_run["result"] == "failed", read_run
    stop = json.loads(read_run["detail"])["stats"]["stopReason"]
    assert stop in ("page 2 repeated earlier results",
                    "the pager ended at page 2 but the portal states 12 (read 10)"), stop
    assert p["state"] == "failed", p                    # the repair cannot read it either: a person is needed
    served = client.get(f"/portals/{pid}/tenders").json()
    assert len(served) == 12 and all(t["status"] != "closed" for t in served)   # last good snapshot, nothing closed
