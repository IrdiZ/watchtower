"""Fields found by what the page says, not by position: a table column by its header text, a value by its label.
Plus 'nv'-style "no value" cells and the onboarding feedback for a deadline the list shows without its time.
Shapes modelled on cosinex portals (list: date-only deadline column, 'nv' when none; tender page: heading + block)."""
import asyncio
from types import SimpleNamespace

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from tests.test_contract import setup  # noqa: F401  (fixture)
from tests.test_w1_runner import run_read
from watchtower import onboard
from watchtower.extract import assemble, convert, html_value, is_no_value, label_value, table_cell
from watchtower.onboard import _date_problem, _never_found, validate_spec

BERLIN = "Europe/Berlin"
HEADERS = ["Veröffentlicht", "Angebots- / Teilnahmefrist", "Bezeichnung", "Typ", "Ausschreibende Stelle", "Aktion"]
ROWS = [  # pid, published, deadline (list: date only), title, type, buyer, deadline on the tender page
    ("57363830", "20.11.2026", "10.12.2030", "Gehölzbeseitigung bei Meschede", "Sonstige Ausschreibung",
     "Hochsauerlandkreis", "10.12.2030 09:00 Ortszeit"),
    ("56883668", "20.11.2026", "04.12.2030", "Brandmeldeanlage Berufskolleg", "VOB/A Ausschreibung",
     "Hochsauerlandkreis", "04.12.2030 10:30 Ortszeit"),
    ("57362887", "19.11.2026", "nv", "Wachalarmsteuerung Rettungswache", "Vergebener Auftrag",
     "Stadt Iserlohn", None),
]


def list_html(order=None) -> str:
    order = order or list(range(len(HEADERS)))
    heads = "".join(f"<th class='w-120'><a href='?sort={i}'><span>{HEADERS[i]}</span></a></th>" for i in order)
    body = ""
    for pid, pub, dl, title, typ, buyer, _ in ROWS:
        cells = [f"<abbr>{pub}</abbr>", f"<abbr>{dl}</abbr>", title, typ.replace(" ", "<br/>"), buyer,
                 f"<a class='lnk' href='/VMP/public/projectForwarding.do?pid={pid}'><img/></a>"]
        body += "<tr>" + "".join(f"<td class='text-center'>{cells[i]}</td>" for i in order) + "</tr>"
    return (f"<html><body><table class='csx-new-table'><thead><tr>{heads}</tr></thead><tbody>{body}</tbody>"
            f"</table><div class='browsePagesText'>Gesamteinträge: {len(ROWS)}</div></body></html>")


def detail_html(pid: str) -> str:
    row = next(r for r in ROWS if r[0] == pid)

    def block(label, value):
        return (f"<div><div class='sub-headline-container'><h4 class='sub-headline'><span>{label}</span>"
                f"<span class='mandatoryField'></span></h4></div><div class='control-group'>{value}</div></div>")
    deadline = block("Abgabefrist", row[6]) if row[6] else ""
    return (f"<html><body><nav><a href='#'>Übersicht</a></nav><div id='content'>"
            f"<table><thead><tr><th>Dateiname</th><th>Hinzugefügt am</th></tr></thead><tbody><tr><td>a.pdf</td>"
            f"<td>20.11.2026 16:07 Uhr</td></tr></tbody></table>"
            f"{block('Auftraggeber / Ausschreibende Stelle', row[5])}{deadline}"
            f"{block('Frist zur Einreichung von Aufklärungsfragen', '07.12.2026 23:59 Ortszeit')}"
            f"{block('Ausschreibungs-ID', 'CXS' + pid)}</div></body></html>")


def rows_of(html):
    return HTMLParser(html).css("table.csx-new-table > tbody > tr")


# ------------------------------------------------------------------ column by header text
def test_column_by_header_survives_reordering():
    for order in (None, [5, 4, 3, 2, 1, 0], [2, 0, 4, 1, 5, 3]):
        rows = rows_of(list_html(order))
        assert html_value(rows[0], {"column": "Angebots- / Teilnahmefrist"}) == "10.12.2030"
        assert html_value(rows[0], {"column": "angebots-/teilnahmefrist:"}) == "10.12.2030"   # loose match
        assert html_value(rows[1], {"column": "Ausschreibende Stelle"}) == "Hochsauerlandkreis"
        assert html_value(rows[2], {"column": "Aktion", "sel": "a", "attr": "href",
                                    "regex": r"pid=(\d+)"}) == "57362887"
        assert html_value(rows[0], {"column": "Veröffentlicht", "sel": "abbr"}) == "20.11.2026"
        assert html_value(rows[0], {"column": "Typ"}) == "Sonstige Ausschreibung"


def test_column_unknown_ambiguous_or_outside_a_table_is_none():
    row = rows_of(list_html())[0]
    assert html_value(row, {"column": "Losnummer"}) is None
    assert html_value(row, {"column": "Teilnahmefrist"}) == "10.12.2030"   # contained in exactly one header
    amb = HTMLParser("<table><tr><th>Frist A</th><th>Frist B</th></tr><tr><td>1</td><td>2</td></tr></table>")
    assert table_cell(amb.css("tr")[1], "Frist") is None      # contained in two headers: no guess
    assert table_cell(amb.css("tr")[1], "Frist B").text() == "2"
    div = HTMLParser("<div class='row'><span>x</span></div>").css_first("div")
    assert html_value(div, {"column": "Frist"}) is None


def test_column_counts_colspans_and_header_row_without_thead():
    t = HTMLParser("<table><tr><th colspan='2'>Nr.</th><th>Frist</th></tr>"
                   "<tr><td>A</td><td>1</td><td>10.12.2026 09:00</td></tr></table>")
    assert html_value(t.css("tr")[1], {"column": "Frist"}) == "10.12.2026 09:00"
    assert html_value(t.css("tr")[0], {"column": "Frist"}) is None     # the header row itself has no value


# ------------------------------------------------------------------ value by label
def test_label_heading_then_block():
    page = HTMLParser(detail_html("57363830"))
    assert html_value(page, {"label": "Abgabefrist"}) == "10.12.2030 09:00 Ortszeit"
    assert html_value(page, {"label": "Auftraggeber / Ausschreibende Stelle"}) == "Hochsauerlandkreis"
    assert html_value(page, {"label": "Ausschreibungs-ID", "regex": r"(CXS\d+)"}) == "CXS57363830"
    assert html_value(page, {"label": "Losnummer"}) is None
    v = convert("submissionDeadline", html_value(page, {"label": "Abgabefrist"}),
                {"formats": ["%d.%m.%Y %H:%M"]}, BERLIN, "")
    assert v == "2030-12-10T09:00:00+01:00"


def test_label_common_layouts():
    cases = {
        "<dl><dt>Angebotsfrist</dt><dd>10.12.2026 09:00</dd></dl>": "10.12.2026 09:00",
        "<table><tr><th>Angebotsfrist:</th><td><b>10.12.2026</b> 09:00</td></tr></table>": "10.12.2026 09:00",
        "<p><b>Angebotsfrist:</b> 10.12.2026 09:00 Uhr</p>": "10.12.2026 09:00 Uhr",
        "<p>Angebotsfrist: 10.12.2026 09:00</p>": "10.12.2026 09:00",
        "<p>Angebotsfrist: <b>10.12.2026</b> 09:00</p>": "10.12.2026 09:00",
        "<select><option>Angebotsfrist</option><option>Titel</option></select>"
        "<div><label>Angebotsfrist</label><span>10.12.2026 09:00</span></div>": "10.12.2026 09:00",
        "<p>Die Angebotsfrist endet bald.</p><p>Angebotsfristverlängerung: nein</p>": None,
        "<section><span class='label'>Angebotsfrist</span><span class='text'>: </span>"
        "<span class='value'>10.12.2026 09:00</span></section>": "10.12.2026 09:00",       # separator element
        "<div class='form-group'><label class='control-label'><span>Angebotsfrist:</span></label>"
        "<div class='col'><p class='form-control-static'>10.12.2026 09:00</p></div></div>": "10.12.2026 09:00",
        "<dl><dt>Angebotsfrist</dt><dd>-</dd><dt>Ort</dt><dd>Berlin</dd></dl>": "-",           # 'not given'
    }
    for html, want in cases.items():
        assert html_value(HTMLParser(f"<body>{html}</body>"), {"label": "Angebotsfrist"}) == want, html
    assert label_value(HTMLParser("<body><h4>Angebotsfrist</h4></body>"), "Angebotsfrist") is None


# ------------------------------------------------------------------ 'nv' = not given
def test_no_value_markers():
    for s in ("nv", "n.v.", "N.V.", "-", "–", "k.A.", "n/a", "keine Angabe", " entfällt "):
        assert is_no_value(s), s
        assert convert("submissionDeadline", s, {"formats": ["%d.%m.%Y %H:%M"]}, BERLIN, "") is None
    for s in ("10.12.2026", "Navigation", "nvidia", None, "", 0):
        assert not is_no_value(s), s


def test_german_national_procedure_names():
    p = lambda s: convert("procedureType", s, {}, BERLIN, "")
    assert p("Öffentliche Ausschreibung") == "open"
    assert p("Beschränkte Ausschreibung mit öffentlichem Teilnahmewettbewerb") == "restricted"
    assert p("Verhandlungsvergabe mit öffentlichem Teilnahmewettbewerb") == "negotiated"


# ------------------------------------------------------------------ the runner with such a spec
SPEC = {"tz": BERLIN,
        "list": {"kind": "html", "url": "https://vmp.test/VMP/list", "items": "table.csx-new-table > tbody > tr",
                 "pagination": {"type": "none"},
                 "total": {"label": "Gesamteinträge"}},
        "fields": {"id": {"column": "Aktion", "sel": "a", "attr": "href", "regex": r"pid=(\d+)"},
                   "url": {"column": "Aktion", "sel": "a", "attr": "href"},
                   "title": {"column": "Bezeichnung"},
                   "buyerName": {"column": "Ausschreibende Stelle"},
                   "publicationDate": {"column": "Veröffentlicht", "formats": ["%d.%m.%Y"]},
                   "submissionDeadline": {"from": "detail", "label": "Abgabefrist", "formats": ["%d.%m.%Y %H:%M"]}},
        "detail": {"kind": "html", "url_from": "url"}}


def _portal(order):
    def handler(req: httpx.Request):
        if req.url.path == "/VMP/list":
            return httpx.Response(200, html=list_html(order))
        if req.url.path.endswith("projectForwarding.do"):
            return httpx.Response(200, html=detail_html(req.url.params["pid"]))
        return httpx.Response(404)
    return handler


def test_runner_reads_columns_and_labels(setup):
    client, _ = setup
    assert validate_spec(SPEC, full=True) == []
    for order in (None, [4, 2, 5, 0, 1, 3]):
        res = run_read(client, _portal(order), SPEC)
        by_id = {t["id"]: t for t in res.tenders}
        assert set(by_id) == {"57363830", "56883668", "57362887"}
        t = by_id["57363830"]
        assert t["title"] == "Gehölzbeseitigung bei Meschede" and t["buyerName"] == "Hochsauerlandkreis"
        assert t["publicationDate"] == "2026-11-20"
        assert t["submissionDeadline"] == "2030-12-10T09:00:00+01:00"
        assert t["url"] == "https://vmp.test/VMP/public/projectForwarding.do?pid=57363830"
        assert by_id["56883668"]["submissionDeadline"] == "2030-12-04T10:30:00+01:00"
        assert by_id["57362887"]["submissionDeadline"] is None          # 'nv': no deadline, no error
        assert res.stats["statedTotal"] == 3 and res.complete


def test_validate_spec_column_and_label():
    spec = {"tz": BERLIN, "list": {"kind": "html", "url": "https://x/l", "items": "tr"},
            "fields": {"id": {"column": "Nr."}, "url": {"column": "Aktion", "sel": "a", "attr": "href"},
                       "title": {"column": ""}, "buyerName": {"from": "detail", "column": "Stelle"},
                       "submissionDeadline": {"from": "detail", "label": "Abgabefrist", "sel": "div[[bad"}}}
    probs = " | ".join(validate_spec(spec))
    assert "title: column must be the text" in probs
    assert "buyerName: column (a table header) works on list rows only" in probs
    assert "submissionDeadline: invalid CSS selector" in probs
    assert "field id" not in probs and "field url" not in probs


# ------------------------------------------------------------------ onboarding feedback
def _assembled(spec, raw):
    t = assemble(dict(raw), spec, "https://x/")
    t["_listSig"] = __import__("json").dumps({k: v for k, v in raw.items() if not k.startswith("_")})
    return t


def test_never_found_explains_a_date_only_list_deadline():
    spec = {"tz": BERLIN, "fields": {"submissionDeadline": {"from": "list", "column": "Angebots- / Teilnahmefrist",
                                                            "formats": ["%d.%m.%Y"]}}}
    ts = [_assembled(spec, {"id": "1", "submissionDeadline": "10.12.2026"}),
          _assembled(spec, {"id": "2", "submissionDeadline": "nv"})]
    assert all(t["submissionDeadline"] is None for t in ts)
    msg = _never_found(spec, "submissionDeadline", ts)[0]
    assert "only dates (e.g. '10.12.2026')" in msg and '"label"' in msg
    only_nv = [_assembled(spec, {"id": "3", "submissionDeadline": "nv"})]
    assert _never_found(spec, "submissionDeadline", only_nv) == []    # the portal says: no deadline


def test_deadline_with_ortszeit_is_verified_not_skipped():
    assert _date_problem("submissionDeadline", "10.12.2026 09:00 Ortszeit", "2026-12-10T09:00:00+01:00",
                         BERLIN, []) is None
    msg = _date_problem("submissionDeadline", "10.12.2026 09:00 Ortszeit", "2026-12-10T10:00:00+01:00", BERLIN, [])
    assert msg and "mismatch" in msg
    assert _date_problem("submissionDeadline", "10.12.2026 09:00 MEZ", "2026-12-10T08:00:00+00:00", BERLIN, []) is None


def _verify_ctx(truth, seen):
    async def get(url, **kw):
        seen.append(url)
        return SimpleNamespace(status_code=200, text="<html><body><p>Abgabefrist 10.12.2026 09:00 Ortszeit</p>"
                                                     "</body></html>", json=lambda: {})

    async def json_call(task, system, user, **kw):
        return dict(truth)
    return SimpleNamespace(client=SimpleNamespace(get=get), llm=SimpleNamespace(json_call=json_call),
                           portal_id="p", run_id="r")


def test_blind_verify_names_the_label_and_the_date_only_list_value():
    spec = {"tz": BERLIN, "detail": {"url_from": "url"},
            "fields": {"submissionDeadline": {"from": "list", "column": "Angebots- / Teilnahmefrist",
                                              "formats": ["%d.%m.%Y"]}}}
    truth = {"id": "57363830", "title": "T", "buyerName": "B", "publicationDate": None,
             "submissionDeadline": "10.12.2026 09:00 Ortszeit", "labels": {"submissionDeadline": "Abgabefrist"}}
    t = _assembled(spec, {"id": "57363830", "url": "https://x/t/57363830", "title": "T", "buyerName": "B",
                          "submissionDeadline": "10.12.2026"})
    probs = asyncio.run(onboard.blind_verify(_verify_ctx(truth, []), spec, [t], n=1))
    assert len(probs) == 1
    p = probs[0]
    assert "submissionDeadline missing although the page shows '10.12.2026 09:00 Ortszeit'" in p
    assert "the list shows only the date ('10.12.2026')" in p
    assert '{"from": "detail", "label": "Abgabefrist"}' in p


def test_blind_verify_only_compares_tenders_whose_page_was_opened():
    spec = {"tz": BERLIN, "detail": {"url_from": "url"},
            "fields": {"submissionDeadline": {"from": "detail", "label": "Abgabefrist",
                                              "formats": ["%d.%m.%Y %H:%M"]}}}
    truth = {"id": None, "title": None, "buyerName": None, "publicationDate": None,
             "submissionDeadline": "10.12.2026 09:00 Ortszeit"}
    unopened = {"id": "A", "url": "https://x/t/A", "submissionDeadline": None}
    opened = {"id": "B", "url": "https://x/t/B", "submissionDeadline": "2026-12-10T09:00:00+01:00"}
    seen: list = []
    probs = asyncio.run(onboard.blind_verify(_verify_ctx(truth, seen), spec, [unopened, opened], n=1,
                                             opened=["B"]))
    assert probs == [] and seen == ["https://x/t/B"]
    nv = dict(truth, submissionDeadline="nv")
    assert asyncio.run(onboard.blind_verify(_verify_ctx(nv, []), spec, [unopened], n=1)) == []


def test_label_alternatives_and_feedback_across_attempts():
    """Tender pages of one portal label the deadline per procedure type ('Abgabefrist Angebot' for open
    procedures, 'Abgabefrist Teilnahme-Antrag' for two-stage ones): a list of labels, first found wins."""
    fs = {"label": ["Abgabefrist Angebot", "Abgabefrist Teilnahme-Antrag"]}
    open_proc = HTMLParser("<body><label><span>Abgabefrist Angebot:</span></label><div><p>17.12.2026 11:00</p></div>"
                           "</body>")
    two_stage = HTMLParser("<body><label><span>Abgabefrist Teilnahme-Antrag:</span></label><div><p>21.12.2026 13:00"
                           "</p></div></body>")
    assert html_value(open_proc, fs) == "17.12.2026 11:00" and html_value(two_stage, fs) == "21.12.2026 13:00"
    spec = {"tz": BERLIN, "list": {"kind": "html", "url": "https://x/l", "items": "tr"},
            "fields": {"id": {"sel": "td"}, "url": {"sel": "a", "attr": "href"},
                       "submissionDeadline": {"from": "detail", **fs}, "buyerName": {"from": "detail", "label": []}}}
    probs = validate_spec(spec)
    assert len(probs) == 1 and "buyerName: label must be the text" in probs[0]
    hints: dict = {}
    p1 = ["submissionDeadline missing although the page shows '17.12.2026 11:00' (tender 1); the tender page shows "
          "it under the label 'Abgabefrist Angebot': map submissionDeadline as {...}"]
    p2 = ["submissionDeadline missing although the page shows '21.12.2026 13:00' (tender 2); the tender page shows "
          "it under the label 'Abgabefrist Teilnahme-Antrag': map submissionDeadline as {...}"]
    assert onboard._label_feedback(p1, hints) == []
    fb = onboard._label_feedback(p2, hints)
    assert len(fb) == 1 and '"label": ["Abgabefrist Angebot", "Abgabefrist Teilnahme-Antrag"]' in fb[0]


def test_missing_title_points_to_the_list():
    """A title missing on the tender page check must steer back to the list row: details are opened ~60 per read,
    so a detail-sourced title leaves most tenders of a large portal untitled (evergabe-online: 931 of 989)."""
    spec = {"tz": BERLIN, "detail": {"url_from": "url"}, "fields": {}}
    truth = {"id": None, "title": "Heizungstechnik Messe Erfurt", "buyerName": None, "publicationDate": None,
             "submissionDeadline": None, "labels": {}}
    t = {"id": "908809", "url": "https://x/t/908809", "title": None}
    probs = asyncio.run(onboard.blind_verify(_verify_ctx(truth, []), spec, [t], n=1))
    assert len(probs) == 1 and "title missing" in probs[0] and "from the LIST row" in probs[0]


def test_blind_verify_accepts_either_buyer_name_the_page_gives():
    """A page that names the awarding office and the contracting body: the spec may read either one."""
    spec = {"tz": BERLIN, "detail": {"url_from": "url"}, "fields": {}}
    truth = {"id": None, "title": "T", "buyerName": "Bundesamt für Ausrüstung der Bundeswehr",
             "otherBuyerNames": ["Bundeswehrverwaltung"], "publicationDate": None, "submissionDeadline": None}
    t = {"id": "908771", "url": "https://x/t/908771", "title": "T", "buyerName": "Bundeswehrverwaltung"}
    assert asyncio.run(onboard.blind_verify(_verify_ctx(truth, []), spec, [t], n=1)) == []
    wrong = dict(t, buyerName="Bundesrepublik Deutschland")
    probs = asyncio.run(onboard.blind_verify(_verify_ctx(truth, []), spec, [wrong], n=1))
    assert len(probs) == 1 and "buyerName mismatch" in probs[0]
    swapped = dict(t, buyerName="Titel der Ausschreibung")     # not a name the page gives for the buyer
    assert asyncio.run(onboard.blind_verify(_verify_ctx(dict(truth, otherBuyerNames=None), []), spec,
                                            [swapped], n=1))
