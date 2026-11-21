"""Locate walk: best-first search with backtracking, evidence-confirmed lists, sitemap candidates, ranked fallbacks.
Mock portal skins: login_root, dead_end, sitemap_only, decoy_first (see mock_portal/app.py)."""
import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from mock_portal import app as mock
from tests.fake_llm import BASE, FakeLLM
from watchtower import locate, onboard
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient
from watchtower.pipeline import JobContext, PortalBlocked
from watchtower.store import Store


@pytest.fixture(autouse=True)
def reset_mock():
    mock.CONFIG.update(skin="html", mutations=set())
    yield
    mock.CONFIG.update(skin="html", mutations=set())


def _ctx(tmp_path, llm_cls=FakeLLM):
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    store = Store(str(tmp_path))
    client = PoliteClient(HostRateLimiter(0), "t", 10, budget=500, transport=httpx.ASGITransport(app=mock.app))
    return JobContext("p1", BASE + "/", "r1", client, DocumentStore(store, 10 ** 8, 10 ** 9), llm_cls(store), store,
                      mode="attach")


def _run(ctx, coro_fn):
    async def go():
        try:
            return await coro_fn()
        finally:
            await ctx.client.aclose()
    return asyncio.run(go())


def _locate(tmp_path, skin, path="/", **kw):
    mock.CONFIG["skin"] = skin
    ctx = _ctx(tmp_path)

    async def go():
        found = await onboard.locate_list(ctx, BASE + path)
        second = await locate.next_candidate(ctx, found) if kw.get("second") else None
        return found, second
    found, second = _run(ctx, go)
    return ctx, found, second


def _visited(found):
    return [r["url"].removeprefix(BASE) for r in found["walk"].trace if r.get("via") != "sitemap"]


# ------------------------------------------------------------------ static signals
def test_keyword_score_multilingual_and_vetoes():
    s = locate.keyword_score
    for text in ("Alle Bekanntmachungen", "Aktuelle Ausschreibungen", "Find tenders", "Consultations en cours",
                 "Aankondigingen", "Licitaciones", "Ogłoszenia", "Appels d'offres"):
        assert s(text, "https://p.example/x") >= 0.7, text
    for text in ("Anmelden", "Registrierung", "Impressum", "Datenschutz", "Kontakt", "Login"):
        assert s(text, "https://p.example/x") < 0, text
    assert s("Öffentliche Bekanntmachungen ohne Anmeldung", "https://p.example/x") >= 0.7  # a public way in
    assert s("", "https://p.example/aankondigingen/overzicht") > 0                          # URL words count too
    assert s("Leistungsbeschreibung", "https://p.example/files/a.pdf") == -1.0


def test_row_evidence_needs_repeated_rows_with_link_and_date():
    rows = "".join(f"<tr><td><a href='/t/{i}'>Lieferung {i}</a></td><td>0{i}.12.2026 10:00</td></tr>" for i in range(1, 7))
    ev = locate.row_evidence(f"<body><table><thead><tr><th>Titel</th><th>Angebotsfrist</th></tr></thead>"
                             f"<tbody>{rows}</tbody></table></body>")
    assert ev["strong"] and ev["rows"] == 6 and ev["deadline"] and len(ev["hrefs"]) == 6
    nav = "".join(f"<li><a href='/s{i}'>Seite {i}</a></li>" for i in range(8))     # links, no dates
    news = "".join(f"<li><a href='/n{i}'>News {i}</a> 0{i}.11.2026</li>" for i in range(1, 6))
    assert not locate.row_evidence(f"<body><ul>{nav}</ul></body>")["weak"]
    ev = locate.row_evidence(f"<body><ul>{news}</ul></body>")
    assert ev["weak"] and not ev["strong"]   # dated links alone: weak (could be news); the model must agree
    # big wrappers holding the whole page are no rows: their links stay navigation
    page = "<body>" + "".join(f"<div><a href='/m{i}'>Menu {i}</a></div>" for i in range(3)) + \
        "<div><a href='/x'>Ausschreibungen</a> Stand 01.12.2026</div></body>"
    assert locate.row_evidence(page)["hrefs"] == []


def test_search_forms_and_soft_redirects():
    html = ("<form action='/suche' method='get'><input name='q'><input type='hidden' name='t' value='open'>"
            "<select name='s'><option value='a'>A</option><option value='b' selected>B</option></select>"
            "<button>Suchen</button></form><form method='post' action='/x?m=s'><input name='q'></form>"
            "<form method='post' name='langForm' action='/'><input name='lang'></form>"
            "<form method='post' action='/registrierung'><input name='firma'></form>"
            "<form action='/login'><input name='u'><input type='password' name='p'></form>")
    forms = locate.search_forms(html, "https://p.example/start")
    # GET forms first; a POST search form is offered as a GET; language, registration and sign-in forms never
    assert [f["url"] for f in forms] == ["https://p.example/suche?q=&t=open&s=b", "https://p.example/x?m=s&q="]
    assert forms[1]["post"] and not forms[0].get("post")
    # an empty action submits to the page itself; the form data REPLACES its query (no growing query on re-submit)
    again = locate.search_forms("<form><input name='q'></form>", "https://p.example/s?q=&lang=de")
    assert [f["url"] for f in again] == ["https://p.example/s?q="]
    assert locate.soft_redirects("<meta http-equiv='refresh' content='0; URL=/v2/'>", "https://p.example/", False) \
        == ["https://p.example/v2/"]
    assert locate.soft_redirects("<frameset><frame src='/main.html'></frameset>", "https://p.example/", False) \
        == ["https://p.example/main.html"]
    js = "<script>window.location.href = '/app/home';</script>"
    assert locate.soft_redirects(js, "https://p.example/", True) == ["https://p.example/app/home"]
    assert locate.soft_redirects(js, "https://p.example/", False) == []   # on a full page it is a click handler


def test_parse_verdict_top3_and_legacy_next_link():
    ptype, picks, _ = locate.parse_verdict({"pageType": "Landing", "links": [{"i": 2, "confidence": 85}, {"i": 9},
                                                                             {"i": "x"}, 3, 4], "nextLink": 1}, 5)
    assert ptype == "landing" and picks == [(1, 0.8), (2, 0.85), (3, 0.6)]


# ------------------------------------------------------------------ walks on the mock portal
def test_plain_portal_two_pages(tmp_path):
    _, found, _ = _locate(tmp_path, "html")
    assert found["url"] == BASE + "/ausschreibungen" and found["tier"] == "A" and found["staticRows"]
    assert found["walk"].used["pages"] == 2 and found["walk"].used["classify"] == 2


def test_login_page_at_root_is_a_branch_not_the_end(tmp_path):
    ctx, found, _ = _locate(tmp_path, "login_root")
    assert found["url"] == BASE + "/ausschreibungen" and found["tier"] == "A"
    walk = found["walk"]
    assert walk.trace[0]["final"] == BASE + "/login" and walk.trace[0]["verdict"] == "blocked"
    assert walk.blocks and "login" in walk.blocks[0]["reason"]
    assert any(l.startswith("locate: ") for l in ctx.log)


def test_dead_end_first_then_backtrack_and_ranked_fallback(tmp_path):
    _, found, second = _locate(tmp_path, "dead_end", second=True)
    assert _visited(found)[:3] == ["/", "/vergabe", "/ausschreibungen"]
    assert found["walk"].trace[1]["verdict"] == "dead end"
    assert found["url"] == BASE + "/ausschreibungen" and found["tier"] == "A"
    # the home page's 3-row teaser is a weaker candidate, handed out second
    assert [c["tier"] for c in found["candidates"]] == ["A", "B"]
    assert second is not None and second["url"] == BASE + "/" and second["tier"] == "B"


def test_list_reachable_only_via_sitemap(tmp_path):
    _, found, _ = _locate(tmp_path, "sitemap_only")
    assert found["url"] == BASE + "/ausschreibungen" and found["tier"] == "A"
    assert any(r.get("via") == "sitemap" and BASE + "/ausschreibungen" in r["pushed"] for r in found["walk"].trace)
    assert found["path"] == [BASE + "/ausschreibungen"]


def test_login_wall_everywhere_is_blocked(tmp_path):
    mock.CONFIG["mutations"] = {"login_wall"}
    with pytest.raises(PortalBlocked) as e:
        _locate(tmp_path, "html", "/ausschreibungen")
    assert "login" in str(e.value).lower()


def test_json_api_in_page_source(tmp_path):
    _, found, _ = _locate(tmp_path, "api")
    assert found["api"] and "/api/tenders" in found["api"][0] and found["tier"] == "A"


class NoNetworkLLM(FakeLLM):
    """The model's endpoint cannot be reached during the walk (DNS hiccup): classify raises a network error."""

    async def json_call(self, task, *a, **kw):
        if task == "classify":
            raise httpx.ConnectError("Name or service not known")
        return await super().json_call(task, *a, **kw)


def test_model_network_error_falls_back_to_static_signals(tmp_path):
    ctx = _ctx(tmp_path, NoNetworkLLM)
    found = _run(ctx, lambda: onboard.locate_list(ctx, BASE + "/"))
    # anchor words lead to the list; strong rows (link + date + deadline words) confirm it without the model
    assert found["url"] == BASE + "/ausschreibungen" and found["tier"] == "A"
    assert any("static signals only" in l for l in ctx.log)


def test_files_are_refused_unread_and_never_queued(tmp_path):
    ctx = _ctx(tmp_path)
    walk = locate.Walk(ctx, BASE + "/")

    async def go():
        return await walk._fetch(BASE + "/downloads/setup"), await walk._fetch(BASE + "/ausschreibungen")
    (big, why), (page, ok) = _run(ctx, go)
    assert why and "file" in why and ok is None and b"tender-row" in page.content
    assert not walk._push(locate.Node(BASE + "/files/setup.msi", 1, "model", 0.9))   # the model cannot override


def test_page_budget_is_respected(tmp_path, monkeypatch):
    monkeypatch.setitem(locate.BUDGET, "pages", 2)   # dead end + list need 3 pages: only the teaser is left
    _, found, _ = _locate(tmp_path, "dead_end")
    assert found["walk"].used["pages"] == 2 and found["tier"] == "B" and found["url"] == BASE + "/"


def test_learn_moves_to_the_next_candidate_when_a_page_has_no_readable_rows(tmp_path):
    mock.CONFIG["skin"] = "decoy_first"
    ctx = _ctx(tmp_path)
    spec = _run(ctx, lambda: onboard.learn(ctx, BASE + "/"))
    assert spec["list"]["url"] == BASE + "/ausschreibungen" and spec["learnedFrom"] == BASE + "/ausschreibungen"
    assert any("trying the next located candidate" in l for l in ctx.log), ctx.log


# ------------------------------------------------------------------ end to end through the API
@pytest.mark.parametrize("skin", ["login_root", "dead_end", "sitemap_only", "decoy_first"])
def test_attach_serves_through_harder_ways(tmp_path, skin):
    mock.CONFIG["skin"] = skin
    s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0)
    app = create_app(s, llm=FakeLLM, transport=httpx.ASGITransport(app=mock.app))
    with TestClient(app) as client:
        pid = client.post("/portals", json={"url": BASE + "/"}).json()["id"]
        end = time.time() + 30
        while time.time() < end:
            p = client.get(f"/portals/{pid}").json()
            if p["state"] in ("serving", "failed"):
                break
            time.sleep(0.05)
        assert p["state"] == "serving", p
        assert p["tenderCount"] == 12
