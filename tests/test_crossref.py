"""Cross-portal "same tender, several sources": matching precision, evidence, conflicts, API (synthetic data)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from watchtower import crossref
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.store import Store

NOW = datetime(2026, 11, 21, 2, 0, tzinfo=timezone.utc)


def t(tid, url, title, buyer, deadline, **kw):
    d = {"id": tid, "url": url, "title": title, "buyerName": buyer, "publicationDate": "2026-11-10",
         "submissionDeadline": deadline, "status": "open", "documents": []}
    d.update(kw)
    return d


def filler(host, n=40):
    """Unrelated tenders so buyer-word weights look like a real portal (many "Stadt", "Stichting" buyers)."""
    buyers = ["Stadt Köln", "Stadt Bonn", "Stichting Openbaar Onderwijs Noord", "Stichting Katholiek Onderwijs Zuid",
              "Gemeente Utrecht", "Stadt Essen", "Stichting Onderwijs Midden", "Ville de Lyon"]
    tag = host.replace("www.", "").replace(".", "")
    return [t(f"f{i}", f"https://{host}/t/f{i}", f"{tag} works package {i}", buyers[i % len(buyers)],
              f"2027-0{1 + i % 8}-1{i % 9}T10:00:00+01:00") for i in range(n)]


def run(portal_tenders: dict[str, list[dict]], urls: dict[str, str] | None = None) -> dict:
    portals = [{"id": pid, "url": (urls or {}).get(pid, f"https://{pid}.example/")} for pid in portal_tenders]
    return crossref.compute(portals, portal_tenders, now=NOW)


def groups_with(out, *tids):
    return [g for g in out["groups"] if set(tids) <= {r["tenderId"] for r in g["records"]}]


# ------------------------------------------------------------------------------------------ evidence
def test_link_from_one_portal_to_another_matches_and_shows_deadline_conflict():
    tn = [t("619152", "https://www.tenderned.nl/aankondigingen/overzicht/445010", "Open Raadsinformatie",
            "Vereniging van Nederlandse Gemeenten", "2026-11-24T12:00:00+01:00",
            _docUrls=["https://s2c.mercell.com/today/239061"])] + filler("www.tenderned.nl")
    me = [t("T239061", "https://s2c.mercell.com/today/239061", "Open Raadsinformatie (VNG)", "VNG",
            "2026-11-25T12:00:00+01:00")] + filler("s2c.mercell.com")
    out = run({"tn": tn, "me": me})
    [g] = groups_with(out, "619152", "T239061")
    assert g["kind"] == "cross-portal" and g["strongestEvidence"] == "link" and g["confidence"] >= 0.95
    [c] = [c for c in out["conflicts"] if c["field"] == "deadline"]
    shown = {v["host"]: v["display"] for v in c["values"]}
    assert shown == {"tenderned.nl": "24 Nov 2026 12:00 CET", "s2c.mercell.com": "25 Nov 2026 12:00 CET"}
    assert c["note"] == "1 day apart"
    assert all(v["url"].startswith("https://") for v in c["values"])
    assert out["summary"]["crossPortalGroups"] == 1 and out["summary"]["crossPortalConflicts"] >= 1


def test_boamp_style_link_to_place_consultation_id_matches_despite_different_titles():
    boamp = [t("26-112349", "https://www.boamp.fr/pages/avis/?q=idweb:26-112349", "Maîtrise d'oeuvre RN57",
               "DREAL Bourgogne-Franche-Comté", "2026-11-24T12:00:00+01:00",
               description="Dossier: https://www.marches-publics.gouv.fr/entreprise/consultation/3087506?orgAcronyme=d4t")]
    place = [t("3087506", "https://www.marches-publics.gouv.fr/app.php/entreprise/consultation/3087506?orgAcronyme=d4t",
               "RN57-La Main-La Vrine - Marché de procédures", None, None,
               referenceNumber="DrealBFC-26-RN57-LMLV-Procedures"),
             t("3087507", "https://www.marches-publics.gouv.fr/app.php/entreprise/consultation/3087507?orgAcronyme=d4t",
               "Autre consultation", None, None)]
    out = run({"boamp": boamp + filler("www.boamp.fr"), "place": place + filler("www.marches-publics.gouv.fr")})
    [g] = groups_with(out, "26-112349", "3087506")
    assert g["strongestEvidence"] == "link"
    assert not groups_with(out, "26-112349", "3087507")
    assert "title" in g["conflicts"]
    gap_fields = {x["field"] for x in g["gaps"]}
    assert {"deadline", "buyer"} <= gap_fields  # BOAMP gives what PLACE lacks


def test_same_reference_and_buyer_across_portals():
    a = [t("908830", "https://www.evergabe-online.de/tenderdetails.html?id=908830",
           "Konzeption und Grafik für Kampagne, Katalog und Web-App - JMB 241 / 2026",
           "Stiftung Jüdisches Museum Berlin", "2026-11-30T09:00:00+01:00", referenceNumber="JMB 241 / 2026")]
    b = [t("961b5353", "https://oeffentlichevergabe.de/ui/en/search/details?noticeId=961b5353",
           "Konzeption und Grafik für Kampagne, Katalog und Web -App zur Sonderausstellung",
           "Stiftung Jüdisches Museum Berlin", "2026-11-30T08:00:00Z", referenceNumber="JMB-241/2026")]
    out = run({"evo": a + filler("www.evergabe-online.de"), "oev": b + filler("oeffentlichevergabe.de")})
    [g] = groups_with(out, "908830", "961b5353")
    assert g["strongestEvidence"] == "reference" and g["confidence"] >= 0.85
    # 09:00 CET and 08:00 UTC are the same minute: no deadline conflict
    assert "deadline" not in g["conflicts"]


def test_title_buyer_and_same_deadline_minute_match_across_timezones():
    a = [t("57183032", "https://www.vergabe-westfalen.de/p?pid=57183032", "Umweltplanung K29 Riesenbecker Straße, Saerbeck",
           "Kreis Steinfurt", "2026-12-21T10:30:00+01:00", referenceNumber="CXPWYY2LKJC")]
    b = [t("26066910", "https://oeffentlichevergabe.de/d?noticeId=26066910", "Umweltplanung K29 Riesenbecker Strasse Saerbeck",
           "Kreis Steinfurt", "2026-12-21T09:30:00+00:00")]
    out = run({"w": a + filler("www.vergabe-westfalen.de"), "o": b + filler("oeffentlichevergabe.de")})
    [g] = groups_with(out, "57183032", "26066910")
    assert g["strongestEvidence"] == "title" and 0.8 <= g["confidence"] < 0.95
    assert g["conflicts"] == []


# ------------------------------------------------------------------------------------------ precision
def test_near_identical_titles_from_different_buyers_do_not_match():
    a = [t("1", "https://a.example/t/1", "Reinigung von Büroflächen im Rathaus", "Stadt Köln", "2026-12-01T10:00:00+01:00")]
    b = [t("2", "https://b.example/t/2", "Reinigung von Büroflächen im Rathaus", "Stadt Bonn", "2026-12-01T10:00:00+01:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    assert not groups_with(out, "1", "2")


def test_buyers_sharing_only_generic_words_do_not_match():
    a = [t("319569", "https://a.example/t/319569", "DAS Schoolmeubilair Primair Onderwijs",
           "Stichting Katholiek Onderwijs Veluwe-Vallei", "2029-07-31T10:00:00+02:00")]
    b = [t("319911", "https://b.example/t/319911", "DAS Schoolmeubilair Primair Onderwijs",
           "Stichting Onderwijs Pr1mair", "2029-07-31T10:00:00+02:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    assert not groups_with(out, "319569", "319911")


def test_different_lot_numbers_do_not_match():
    a = [t("1", "https://a.example/t/1", "Neubau Grundschule Los 1 Rohbau", "Stadt Ohrdruf", "2026-12-04T10:00:00+01:00")]
    b = [t("2", "https://b.example/t/2", "Neubau Grundschule Los 2 Rohbau", "Stadt Ohrdruf", "2026-12-04T10:00:00+01:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    assert not groups_with(out, "1", "2")


def test_same_deadline_but_different_title_does_not_match():
    a = [t("1", "https://a.example/t/1", "Winterdienst Kreisstraßen", "Kreis Steinfurt", "2026-12-04T10:00:00+01:00")]
    b = [t("2", "https://b.example/t/2", "Grünpflege Schulhöfe", "Kreis Steinfurt", "2026-12-04T10:00:00+01:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    assert out["groups"] == []


def test_lots_sharing_one_reference_inside_a_portal_are_not_merged():
    title = "Erweiterung und energetische Sanierung der Ludwig-Reinhard-Schule mit Neubau einer Sporthalle"
    lots = [t(f"k{i}", f"https://oev.example/d?id=k{i}", title, "Stadt Kaufbeuren", dl, referenceNumber="2020_3")
            for i, dl in enumerate(["2026-11-25T09:30:00+01:00", "2026-12-01T09:30:00+01:00",
                                    "2026-12-02T09:30:00+01:00", "2026-12-03T09:30:00+01:00"])]
    out = run({"oev": lots + filler("oev.example")})
    assert out["groups"] == []


def test_one_portal_publishing_the_same_procedure_twice_is_a_within_portal_duplicate():
    pair = [t(i, f"https://oev.example/d?id={i}", "Planung Regenwasserleitung", "Trinkwasserversorgung Magdeburg GmbH",
              "2026-12-16T14:00:00+01:00", referenceNumber="TWM_RWL_Plan_2026") for i in ("01a163e3", "c7f36bd0")]
    out = run({"oev": pair + filler("oev.example")})
    [g] = groups_with(out, "01a163e3", "c7f36bd0")
    assert g["kind"] == "within-portal" and g["confidence"] == 0.9
    assert out["summary"]["withinPortalGroups"] == 1 and out["summary"]["crossPortalGroups"] == 0


def test_portal_attached_twice_is_one_source_not_a_duplicate():
    ts = [t(f"{600000 + i}", f"https://www.tenderned.nl/aankondigingen/overzicht/{450000 + i}", f"Opdracht nummer {i}",
            "Gemeente Utrecht", "2026-12-17T13:00:00+01:00") for i in range(30)]
    out = run({"api": ts, "web": [dict(x) for x in ts]})
    assert out["groups"] == []
    [alias] = out["summary"]["excludedSelfDuplicates"]
    assert alias["portal"] == "web" and alias["sameAs"] == "api" and alias["sharedIds"] == 30


def test_status_conflict_is_listed():
    a = [t("1", "https://a.example/t/1", "Levering laboratoria chemicaliën", "Radboudumc",
           "2026-12-31T00:00:00+01:00", referenceNumber="RUMC-2026-118")]
    b = [t("2", "https://b.example/t/2", "Levering laboratoria chemicaliën", "Radboud Universitair Medisch Centrum",
           "2026-12-31T00:00:00+01:00", referenceNumber="RUMC-2026-118", status="withdrawn")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    [g] = groups_with(out, "1", "2")
    assert "status" in g["conflicts"]
    [c] = [c for c in out["conflicts"] if c["field"] == "status"]
    assert {v["value"] for v in c["values"]} == {"open", "withdrawn"}


def test_normalisation_helpers():
    assert crossref.ref_key("JMB 241 / 2026") == crossref.ref_key("jmb-241/2026") == "jmb2412026"
    assert crossref.ref_key("A-1") is None and crossref.ref_key("ABCDEF") is None
    assert crossref.norm_text("Münster  Straße – Łódź") == "munster strasse lodz"
    assert crossref.host_of("https://www.Example.org:8443/x") == "example.org"
    assert "3087506" in crossref.url_tokens("https://x.fr/entreprise/consultation/3087506?orgAcronyme=d4t")
    assert crossref.fmt_deadline("2026-07-31T10:00:00+02:00") == "31 Jul 2026 10:00 UTC+02:00"


# ------------------------------------------------------------------------------------------ API
class _Idle:
    async def attach(self, ctx):
        raise AssertionError("not used")


def test_ui_crossref_endpoint_and_page(tmp_path):
    store = Store(str(tmp_path))
    for pid, host, tid, dl in (("p_a", "a.example", "A1", "2026-11-24T12:00:00+01:00"),
                               ("p_b", "b.example", "B1", "2026-11-25T12:00:00+01:00")):
        store.create_portal(pid, f"https://{host}/", f"{host}/")
        store.set_state(pid, "serving", "ok")
        store.commit_snapshot(pid, [t(tid, f"https://{host}/t/{tid}", "Accountantsdiensten 2027-2030",
                                      "Vereniging van Nederlandse Gemeenten", dl, referenceNumber="VNG-2026-045")]
                              + filler(host, 10), complete=True)
    app = create_app(Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0), pipeline=_Idle())
    with TestClient(app) as client:
        body = client.get("/_ui/crossref").json()
        assert body["summary"]["crossPortalGroups"] == 1
        [c] = [c for c in body["conflicts"] if c["field"] == "deadline"]
        assert sorted(v["display"] for v in c["values"]) == ["24 Nov 2026 12:00 CET", "25 Nov 2026 12:00 CET"]
        assert client.get("/_ui/crossref").json()["computedAt"] == body["computedAt"]  # cached
        assert client.get("/_ui/crossref?refresh=true").json()["computedAt"] != body["computedAt"]
        page = client.get("/_ui/crossref/view")
        assert page.status_code == 200 and "/_ui/crossref" in page.text
        assert "/_ui/crossref/view" in client.get("/").text
        # the contract is unchanged: the portal list has no crossref fields
        assert all(set(p) == {"id", "url", "state", "reason", "tenderCount", "lastSuccessAt", "coverage", "verification"}
                   for p in client.get("/portals").json())


def test_one_trade_inside_another_title_is_not_the_same_tender():
    a = [t("1", "https://a.example/t/1", "Lüftung für den Neubau des Berufskollegs Meschede", "Hochsauerlandkreis",
           "2026-11-26T09:00:00+01:00")]
    b = [t("2", "https://b.example/t/2",
           "Schlosserarbeiten (Unterkonstruktion Lüftung) für den Neubau des Berufskollegs Meschede",
           "Hochsauerlandkreis", "2026-11-26T09:00:00+01:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    assert out["groups"] == []


def test_html_entities_do_not_make_a_title_conflict():
    a = [t("1", "https://a.example/t/1", 'Sanierung "Auf dem Düsen" - Straßenbau', "Stadt Werl",
           "2026-12-04T09:30:00+01:00")]
    b = [t("2", "https://b.example/t/2", "Sanierung &quot;Auf dem Düsen&quot; - Straßenbau", "Stadt Werl",
           "2026-12-04T09:30:00+01:00")]
    out = run({"a": a + filler("a.example"), "b": b + filler("b.example")})
    [g] = groups_with(out, "1", "2")
    assert g["conflicts"] == []


def test_identical_title_and_buyer_with_different_deadlines_is_a_flagged_conflict():
    a = [t("57211711", "https://www.vergabe-westfalen.de/p?pid=57211711", "Neubau Schlammentwässerung Kläranlage - Gewerk Schlammpresse",
           "Technische Betriebe Rheine", "2026-12-15T10:00:00+01:00")]
    b = [t("26022304", "https://oeffentlichevergabe.de/d?noticeId=26022304", "Neubau Schlammentwässerung Kläranlage - Gewerk Schlammpresse",
           "Technische Betriebe Rheine", "2026-12-01T10:00:00+01:00")]
    out = run({"w": a + filler("www.vergabe-westfalen.de"), "o": b + filler("oeffentlichevergabe.de")})
    [g] = groups_with(out, "57211711", "26022304")
    assert g["confidence"] == 0.6 and g["conflicts"] == ["deadline"]
    assert any(e["type"] == "caution" for x in g["evidence"] for e in x["evidence"])
    [c] = out["conflicts"]
    assert c["note"] == "14 days apart"
    # the same title twice in one portal (two lots): which one would it be? no match at all
    lot2 = dict(b[0], id="26022305", url="https://oeffentlichevergabe.de/d?noticeId=26022305",
                submissionDeadline="2026-12-02T10:00:00+01:00")
    out = run({"w": a + filler("www.vergabe-westfalen.de"), "o": b + [lot2] + filler("oeffentlichevergabe.de")})
    assert out["groups"] == []
