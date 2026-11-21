"""All-tenders explorer (/_ui/tenders): filters, search, paging, the one-registration-per-host rule, stable order."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from watchtower import ui_tenders
from watchtower.api import create_app
from watchtower.config import Settings
from watchtower.store import Store

# Far in the future, so nothing here closes by deadline while the suite runs.
NOW = datetime(2030, 1, 1, 2, 0, tzinfo=timezone.utc)


def t(tid, host, title, buyer="Stadt Köln", deadline="2030-03-01T10:00:00+01:00", **kw):
    d = {"id": tid, "url": f"https://{host}/t/{tid}", "title": title, "buyerName": buyer,
         "publicationDate": "2029-12-10", "submissionDeadline": deadline, "status": "open", "documents": []}
    d.update(kw)
    return d


def seed(tmp_path) -> Store:
    store = Store(str(tmp_path))
    portals = {
        # Two registrations of one host: the larger one is served, the other is left out.
        "p_tn_small": ("https://www.tenderned.nl/aankondigingen/overzicht",
                       [t("N1", "www.tenderned.nl", "Schoonmaak gemeentehuis", "Gemeente Utrecht")]),
        "p_tn_big": ("https://tenderned.nl/papi/v2/publicaties",
                     [t("N1", "www.tenderned.nl", "Schoonmaak gemeentehuis", "Gemeente Utrecht",
                        "2030-01-05T12:00:00+01:00", referenceNumber="UTR-2030-7"),
                      t("N2", "www.tenderned.nl", "Accountantsdiensten", "VNG", "2030-01-25T12:00:00+01:00"),
                      t("N3", "www.tenderned.nl", "Zonder termijn", "VNG", None)]),
        "p_de": ("https://www.evergabe-online.de",
                 [t(f"D{i:02d}", "www.evergabe-online.de", f"Bauleistung Los {i:02d}", "Stadt Bonn",
                    "2030-03-10T10:00:00+01:00", publicationDate=f"2029-12-{1 + i % 3:02d}") for i in range(30)]),
        "p_fr": ("https://www.achatpublic.com", [t("F1", "www.achatpublic.com", "Travaux de voirie", "Ville de Lyon",
                                                   "2030-01-02T12:00:00+01:00",
                                                   documents=[{"fileName": "DCE.zip", "sizeBytes": 2048,
                                                               "sha256": "ab" * 32, "path": "/files/" + "ab" * 32}])]),
        "p_uk": ("https://www.find-tender.service.gov.uk", []),  # attached, nothing served: not counted
    }
    for pid, (url, tenders) in portals.items():
        store.create_portal(pid, url, url)
        store.set_state(pid, "serving", "ok")
        if tenders:
            store.commit_snapshot(pid, tenders, complete=True)
    return store


def index(tmp_path) -> ui_tenders.TenderIndex:
    return ui_tenders.TenderIndex(seed(tmp_path))


def ids(out) -> set[str]:
    return {i["tenderId"] for i in out["items"]}


def test_country_and_host_rules():
    assert ui_tenders.host_of("https://www.tenderned.nl/x") == "tenderned.nl"
    hosts = ("evergabe-online.de", "boamp.fr", "tenderned.nl", "ezamowienia.gov.pl", "contrataciondelestado.es",
             "etenders.gov.ie", "simap.ch", "find-tender.service.gov.uk", "achatpublic.com", "x.example")
    assert [ui_tenders.country_of(h) for h in hosts] == ["DE", "FR", "NL", "PL", "ES", "IE", "CH", "UK", "FR", ""]
    more = ("hankintailmoitukset.fi", "doffin.no", "pmp.b2g.etat.lu", "marches.megalis.bretagne.bzh",
            "alsacemarchespublics.eu", "sell2wales.gov.wales")
    assert [ui_tenders.country_of(h) for h in more] == ["FI", "NO", "LU", "FR", "FR", "UK"]


def test_registrations_sharing_a_host_serve_only_the_largest(tmp_path):
    ix = index(tmp_path)
    out = ix.query(size=100, now=NOW)
    assert out["served"]["portals"] == 3 and out["served"]["all"] == 34  # 3 + 30 + 1; p_tn_small's row is left out
    assert {p["id"] for p in out["facets"]["portals"]} == {"p_tn_big", "p_de", "p_fr"}
    assert [i["tenderId"] for i in out["items"]].count("N1") == 1
    assert out["total"] == 34 and sum(p["count"] for p in out["facets"]["portals"]) == 34
    # A link that names the left-out registration lands on the one that is served for that host.
    assert ids(ix.query(portals=["p_tn_small"], now=NOW)) == {"N1", "N2", "N3"}


def test_filters_by_portal_country_status_and_closing(tmp_path):
    ix = index(tmp_path)
    assert ix.query(portals=["p_de"], now=NOW)["total"] == 30
    assert ix.query(portals=["p_de", "p_fr"], now=NOW)["total"] == 31
    nl = ix.query(country=["nl"], now=NOW)
    assert nl["total"] == 3 and all(i["country"] == "NL" for i in nl["items"])
    # Facets are counted without their own filter, so the other chips keep their counts.
    assert {c["country"]: c["count"] for c in nl["facets"]["countries"]} == {"DE": 30, "NL": 3, "FR": 1}
    assert {p["host"]: p["count"] for p in nl["facets"]["portals"]} == \
        {"tenderned.nl": 3, "evergabe-online.de": 0, "achatpublic.com": 0}
    assert ids(ix.query(closing=7, now=NOW)) == {"F1", "N1"}
    assert ids(ix.query(closing=30, now=NOW)) == {"F1", "N1", "N2"}
    # A deadline that has passed since the last read closes on the fly.
    later = datetime(2030, 1, 3, 0, 0, tzinfo=timezone.utc)
    assert "F1" not in ids(ix.query(size=100, now=later))
    assert ids(ix.query(status="closed", now=later)) == {"F1"}
    assert ix.query(status="all", now=later)["total"] == 34


def test_search_covers_title_buyer_and_reference(tmp_path):
    ix = index(tmp_path)
    assert ids(ix.query(q="accountants", now=NOW)) == {"N2"}
    assert ids(ix.query(q="LYON voirie", now=NOW)) == {"F1"}
    assert ids(ix.query(q="utr-2030-7", now=NOW)) == {"N1"}
    assert ix.query(q="nothing matches this", now=NOW)["total"] == 0


def test_paging_and_stable_sort(tmp_path):
    ix = index(tmp_path)
    pages = [ix.query(portals=["p_de"], sort="published", page=n, size=7, now=NOW) for n in range(1, 6)]
    assert [len(p["items"]) for p in pages] == [7, 7, 7, 7, 2] and all(p["total"] == 30 for p in pages)
    got = [i["tenderId"] for p in pages for i in p["items"]]
    assert len(set(got)) == 30  # no row is repeated or lost across pages although many share a date
    dates = [i["publicationDate"] for p in pages for i in p["items"]]
    assert dates == sorted(dates, reverse=True)
    for day in set(dates):  # equal keys keep one fixed order: by tender id
        same = [i for i, d in zip(got, dates) if d == day]
        assert same == sorted(same)
    assert got == [i["tenderId"] for i in ix.query(portals=["p_de"], sort="published", size=100, now=NOW)["items"]]
    by_deadline = [i["tenderId"] for i in ix.query(sort="deadline", size=100, now=NOW)["items"]]
    assert by_deadline[:3] == ["F1", "N1", "N2"] and by_deadline[-1] == "N3"  # no deadline sorts last
    titles = [i["title"] for i in ix.query(sort="title", size=100, now=NOW)["items"]]
    assert titles == sorted(titles, key=str.casefold)


def test_index_is_reused_until_counts_change(tmp_path, monkeypatch):
    store = seed(tmp_path)
    ix = ui_tenders.TenderIndex(store)
    monkeypatch.setattr(ui_tenders, "CHECK_SECONDS", 0.0)
    monkeypatch.setattr(ui_tenders, "BACKGROUND", False)
    ix.query(now=NOW)
    ix.query(q="bau", now=NOW)
    assert ix.builds == 1
    store.commit_snapshot("p_fr", [t("F1", "www.achatpublic.com", "Travaux de voirie", "Ville de Lyon"),
                                   t("F2", "www.achatpublic.com", "Fourniture de mobilier", "Ville de Lyon")],
                          complete=True)
    assert ix.query(portals=["p_fr"], now=NOW)["total"] == 2 and ix.builds == 2


def test_index_rebuilds_when_data_changes_without_a_count_change(tmp_path, monkeypatch):
    """Live, 21 Nov: web pages dropped from tenders' documents changed no count, so the page kept showing them."""
    store = seed(tmp_path)
    ix = ui_tenders.TenderIndex(store)
    monkeypatch.setattr(ui_tenders, "CHECK_SECONDS", 0.0)
    monkeypatch.setattr(ui_tenders, "BACKGROUND", False)
    assert ix.query(docs=True, now=NOW)["withDocuments"] == 1
    store.commit_snapshot("p_fr", [t("F1", "www.achatpublic.com", "Travaux de voirie", "Ville de Lyon",
                                     "2030-01-02T12:00:00+01:00")], complete=True)     # same count, no documents
    assert ix.query(docs=True, now=NOW)["withDocuments"] == 0 and ix.builds == 2


def test_api_and_page(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHTOWER_WORKER", "0")
    seed(tmp_path)
    app = create_app(Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0), pipeline=object())
    with TestClient(app) as client:
        body = client.get("/_ui/tenders", params={"portal": ["p_de", "p_fr"], "size": 5, "page": 2,
                                                  "sort": "title"}).json()
        assert body["total"] == 31 and body["page"] == 2 and body["size"] == 5 and len(body["items"]) == 5
        item = body["items"][0]
        assert set(item) >= {"title", "buyerName", "submissionDeadline", "publicationDate", "status",
                             "referenceNumber", "url", "portal", "host", "country"}
        assert item["url"].startswith("https://www.evergabe-online.de/t/")
        assert client.get("/_ui/tenders", params={"size": 101}).status_code == 422
        page = client.get("/_ui/tenders/view")
        assert page.status_code == 200 and "/_ui/static/ui.css" in page.text


def test_documents_filter_and_download_links(tmp_path):
    out = index(tmp_path).query(docs=True, now=NOW)
    assert ids(out) == {"F1"} and out["withDocuments"] == 1 and out["docs"] is True
    item = out["items"][0]
    assert item["documentCount"] == 1
    assert item["documents"] == [{"name": "DCE.zip", "size": 2048, "url": "/files/" + "ab" * 32}]
    everything = index(tmp_path).query(now=NOW)
    assert everything["withDocuments"] == 1 and len(everything["items"]) > 1
    assert all(i["documentCount"] == 0 for i in everything["items"] if i["tenderId"] != "F1")
