"""Mock tender portal for testing attach / notice / repair.

Run:  uvicorn mock_portal.app:app --port 9000
Skins (choose via MOCK_SKIN env or /_admin/config):
  html - server-rendered list with ?page= pagination, detail pages, document links (incl. a ZIP)
  api  - JavaScript shell; data from /api/tenders?offset=&limit= (JSON) and /api/tenders/{id}; tender pages
         /app/tender/{id} are drawn by JavaScript too (mutation detail_js_blank: they stay empty)
  postback - server-rendered list whose pager posts the page's form (CONFIG postback_flavor: "prado" =
         href="javascript:;//id" bound by a script + PRADO_PAGESTATE, "aspnet" = __doPostBack + __VIEWSTATE);
         a session cookie, a results-per-page select (3 / 5) and a search form before the list form
         mutations: pager_ignores_target (every postback re-renders the same page),
                    pager_stops (the next control is gone from page 2 on, although more pages exist)
                    external_row (the last tender links to another platform, unlike every other row)
  Locate skins (same list at /ausschreibungen, harder way there):
  login_root   - '/' redirects to a sign-in page; its public area is one link away
  dead_end     - the home page's best-looking link leads to a dead end; the list is the second choice. The home
                 page also shows a 3-row 'latest tenders' teaser (a weaker candidate)
  sitemap_only - nothing on the home page links to the list; robots.txt names a sitemap that does
  decoy_first  - the first list-looking page has rows a spec cannot read (li rows); the real list is the 2nd choice
Mutations (simulate the portal changing under us), POST /_admin/config {"mutations": [...]}:
  rename_classes   CSS class names change                (html)
  restructure      table -> div cards                    (html)
  pagination       ?page=N -> ?p=N, page size changes    (html, api: offset->start)
  rename_keys      JSON keys renamed                     (api)
  date_format      dates shown as "9 December 2026, 10:00" instead of "09.12.2026 10:00"
  move_urls        detail URLs move from /tender/<id> to /notice/<id>
  swap_columns     buyer and title swap places, SAME shape  (silent wrong data!)
  login_wall       list requires login
  captcha_decor    captcha script shown, data still available
  error500         everything returns 500
  overlay          a notice pop-up (with its own table) sits in front of the list   (html)
  extra_table      a table of recently awarded contracts (same row markup) before the list   (html)
  award_rows       the list gains a notice-type column ("Art") and lists ex-post award notices (no deadline) and
                   one call without any deadline among the calls, like an archive-style portal   (html)
Request log (for the 1 req/s check): GET /_admin/requests
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import time
import zipfile
from datetime import datetime, timedelta
from urllib.parse import parse_qsl

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response

app = FastAPI(title="Mock Vergabeportal")
CONFIG = {"skin": os.environ.get("MOCK_SKIN", "html"), "mutations": set(), "page_size": 5}
REQUESTS: list[tuple[float, str]] = []

BUYERS = ["Stadt Musterhausen", "Landkreis Beispiel", "Universitätsklinikum Nord", "Stadtwerke Süd"]
SUBJECTS = ["Dachsanierung Grundschule", "Lieferung von Laborgeräten", "IT-Rahmenvertrag Server",
            "Reinigungsdienstleistungen Rathaus", "Neubau Feuerwehrhaus", "Winterdienst 2027",
            "Beschaffung Elektrofahrzeuge", "Gebäudereinigung Schulen", "Brückensanierung B12",
            "Druckerpapier Rahmenvertrag", "Catering Kantine", "Glasfaserausbau Gewerbegebiet"]


def _tenders():
    base = datetime(2026, 11, 20, 10, 0)
    out = []
    for i, subj in enumerate(SUBJECTS):
        tid = f"MH-2026-{1000 + i}"
        out.append({
            "id": tid, "ref": f"VG {300 + i}/26", "title": subj, "buyer": BUYERS[i % len(BUYERS)],
            "published": (base - timedelta(days=i)).date(),
            "deadline": base + timedelta(days=20 + i, hours=i % 3),
            "cpv": ["45261000-4", "38000000-5", "48820000-2", "90910000-9"][i % 4],
            "city": ["Musterhausen", "Beispielstadt"][i % 2], "zip": ["12345", "54321"][i % 2],
            "withdrawn": i == 7,
            "docs": [f"{tid}-Leistungsbeschreibung.pdf", f"{tid}-Vergabeunterlagen.zip"],
        })
    return out


TENDERS = _tenders()


def _awards():
    """Mutation award_rows: notices that are not open calls, listed among the calls (ex-post publications of
    contracts already awarded: no deadline, no documents), and one call that shows no deadline anywhere."""
    base = datetime(2026, 11, 18, 9, 0)
    kinds = [("MH-2026-0901", "Bekanntmachung vergebener Aufträge", "Schulbusverkehr 2025"),
             ("MH-2026-0902", "Ex-post-Veröffentlichung", "Wartung Aufzüge Rathaus"),
             ("MH-2026-0903", "Vergabebekanntmachung", "Planung Kita Nord"),
             ("MH-2026-0904", "Ausschreibung", "Beratungsleistungen Digitalisierung")]
    return [{"id": tid, "ref": f"VG {900 + i}/25", "title": subj, "buyer": BUYERS[i % len(BUYERS)],
             "published": (base - timedelta(days=3 * i)).date(), "deadline": None,
             "cpv": "79400000-8", "city": "Musterhausen", "zip": "12345", "withdrawn": False, "docs": [],
             "kind": kind, "procedure": "Offenes Verfahren" if kind == "Ausschreibung" else "Ex post Veröffentlichung"}
            for i, (tid, kind, subj) in enumerate(kinds)]


AWARDS = _awards()
BY_ID = {t["id"]: t for t in TENDERS + AWARDS}


def listed() -> list[dict]:
    """The html list's rows: the calls, and with award_rows the other notices in between."""
    return TENDERS[:6] + AWARDS + TENDERS[6:] if m("award_rows") else TENDERS


def doc_bytes(name: str) -> bytes:
    if name.endswith(".zip"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("Formblatt_Angebot.pdf", b"%PDF-1.4 form " + name.encode())
            z.writestr("Vertragsentwurf.docx", b"PK-docx " + name.encode())
        return buf.getvalue()
    return b"%PDF-1.4 " + name.encode() + b" " + hashlib.sha256(name.encode()).hexdigest().encode()


def m(name: str) -> bool:
    return name in CONFIG["mutations"]


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%-d %B %Y, %H:%M") if m("date_format") else dt.strftime("%d.%m.%Y %H:%M")


def fmt_d(d) -> str:
    return d.strftime("%-d %B %Y") if m("date_format") else d.strftime("%d.%m.%Y")


def detail_path(tid: str) -> str:
    return f"/notice/{tid}" if m("move_urls") else f"/tender/{tid}"


@app.middleware("http")
async def log_and_mutate(request: Request, call_next):
    if not request.url.path.startswith("/_admin"):
        REQUESTS.append((time.time(), request.url.path))
        if m("error500"):
            return Response("Internal Server Error", status_code=500)
    return await call_next(request)


# ------------------------------------------------------------------ admin
@app.post("/_admin/config")
async def set_config(body: dict):
    if "skin" in body:
        CONFIG["skin"] = body["skin"]
    if "mutations" in body:
        CONFIG["mutations"] = set(body["mutations"])
    return {"skin": CONFIG["skin"], "mutations": sorted(CONFIG["mutations"])}


@app.get("/_admin/requests")
async def requests_log():
    REQ = sorted(t for t, _ in REQUESTS)
    gaps = [b - a for a, b in zip(REQ, REQ[1:])]
    return {"count": len(REQ), "min_gap": min(gaps) if gaps else None, "paths": [p for _, p in REQUESTS][-50:]}


@app.post("/_admin/reset_log")
async def reset_log():
    REQUESTS.clear()
    return {"ok": True}


# ------------------------------------------------------------------ html skin
def _shell(body: str, title="Ausschreibungen") -> str:
    captcha = '<script src="https://www.google.com/recaptcha/api.js"></script><div class="g-recaptcha"></div>' \
        if m("captcha_decor") else ""
    return (f"<!doctype html><html lang='de'><head><meta charset='utf-8'><title>{title} - Vergabeportal Musterhausen"
            f"</title></head><body><header><h1>Vergabeportal Musterhausen</h1><nav><a href='/'>Start</a> "
            f"<a href='/ausschreibungen'>Aktuelle Ausschreibungen</a></nav></header>{captcha}<main>{body}</main>"
            f"<footer>Powered by MockVergabe 3.2</footer></body></html>")


@app.get("/static/app.js")
async def app_js():
    off = "start" if m("pagination") else "offset"
    js = ("const base = '/' + ['api', 'tenders'].join('/');"
          "fetch(base + '?" + off + "=0&limit=5').then(r => r.json()).then(d => {"
          "document.getElementById('app').innerHTML = (d.items || d.results).map(t =>"
          " `<div class=t><a href='/app/tender/${t.id || t.notice_id}'>${t.title || t.subject}</a></div>`).join('');});")
    return Response(js, media_type="application/javascript")


def _spa_hidden() -> str:
    return ("<!doctype html><html><head><title>Ausschreibungen</title></head><body><div id='app'></div>"
            "<script src='/static/app.js'></script></body></html>")


def _bare(body: str, title: str) -> str:
    """A page WITHOUT the shell's navigation (whose 'Aktuelle Ausschreibungen' link would make the way trivial)."""
    return (f"<!doctype html><html lang='de'><head><meta charset='utf-8'><title>{title}</title></head><body>"
            f"<header><h1>Vergabeportal Musterhausen</h1></header><main>{body}</main>"
            f"<footer><a href='/kontakt'>Kontakt</a> <a href='/impressum'>Impressum</a></footer></body></html>")


_FILLER = ("<p>Das Vergabeportal ist die zentrale Plattform der Stadt Musterhausen und ihrer Gesellschaften für "
           "öffentliche Aufträge. Unternehmen finden hier Informationen zu Verfahren, Fristen und Unterlagen.</p>")


@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return HTMLResponse(_bare(
        "<h2>Anmeldung</h2><form action='/login' method='post'><input name='user'>"
        "<input type='password' name='pw'><button>Anmelden</button></form>"
        "<p><a href='/registrieren'>Registrieren</a> · <a href='/passwort'>Passwort vergessen</a></p>"
        "<p><a href='/ausschreibungen'>Öffentliche Bekanntmachungen ohne Anmeldung</a></p>", "Anmeldung"))


@app.get("/vergabe", response_class=HTMLResponse)
async def dead_end_page():
    return HTMLResponse(_bare("<h2>Ausschreibungen</h2>" + _FILLER * 3 + "<p>Fragen? <a href='/'>Zur Startseite</a></p>",
                              "Hinweise"))


@app.get("/uebersicht", response_class=HTMLResponse)
async def decoy_list():
    rows = "".join(f"<li class='eintrag'><a href='{detail_path(t['id'])}'>{t['title']}</a> - "
                   f"veröffentlicht {fmt_d(t['published'])}, Frist {fmt_dt(t['deadline'])}</li>" for t in TENDERS[:6])
    return HTMLResponse(_bare(f"<h2>Übersicht</h2><ul class='uebersicht'>{rows}</ul>", "Übersicht"))


@app.get("/downloads/setup")
async def installer():
    """A 6 MB binary without a content type (real portals link installers like this)."""
    return Response(b"MZ\x90\x00" + b"\x00" * 6_000_000)


@app.get("/robots.txt")
async def robots(request: Request):
    if CONFIG["skin"] != "sitemap_only":
        return PlainTextResponse("not found", status_code=404)
    base = str(request.base_url).rstrip("/")
    return PlainTextResponse(f"User-agent: *\nAllow: /\nSitemap: {base}/sitemap.xml\n")


@app.get("/sitemap.xml")
async def sitemap(request: Request):
    if CONFIG["skin"] != "sitemap_only":
        return PlainTextResponse("not found", status_code=404)
    base = str(request.base_url).rstrip("/")
    locs = "".join(f"<url><loc>{base}{p}</loc></url>" for p in ("/", "/ueber-uns", "/kontakt", "/ausschreibungen"))
    return Response(f"<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>"
                    f"{locs}</urlset>", media_type="application/xml")


@app.get("/", response_class=HTMLResponse)
async def home():
    skin = CONFIG["skin"]
    if skin == "spa_post":
        return HTMLResponse(_spa_post())
    if skin == "login_root":
        return RedirectResponse("/login", status_code=302)
    if skin == "dead_end":
        teaser = "".join(f"<tr class='neu'><td><a href='{detail_path(t['id'])}'>{t['title']}</a></td>"
                         f"<td>Frist {fmt_dt(t['deadline'])}</td></tr>" for t in TENDERS[:3])
        return HTMLResponse(_bare(
            f"<nav><a href='/vergabe'>Ausschreibungen</a> <a href='/kontakt'>Kontakt</a> "
            f"<a href='/ausschreibungen'>Bekanntmachungen</a></nav>{_FILLER}<h3>Neu</h3><table>{teaser}</table>",
            "Start"))
    if skin == "decoy_first":
        return HTMLResponse(_bare(
            f"<p><a href='/uebersicht'>Ausschreibungen</a> · <a href='/ausschreibungen'>Bekanntmachungen</a></p>"
            f"{_FILLER}", "Start"))
    if skin == "sitemap_only":
        return HTMLResponse(_bare(f"<p>Willkommen. <a href='/ueber-uns'>Über uns</a></p>{_FILLER}", "Start"))
    if skin == "spa_hidden":
        return HTMLResponse(_spa_hidden())
    if CONFIG["skin"] == "api":
        return HTMLResponse(_spa())
    return HTMLResponse(_shell("<p>Willkommen. Hier finden Sie unsere <a href='/ausschreibungen'>aktuellen "
                               "Ausschreibungen</a>.</p>", "Start"))


@app.get("/ausschreibungen", response_class=HTMLResponse)
async def list_page(request: Request):
    if CONFIG["skin"] == "api":
        return HTMLResponse(_spa())
    if CONFIG["skin"] == "postback":
        resp = HTMLResponse(_postback_page(1, PB_SIZES[0]))
        sid = secrets.token_hex(8)
        PB_SESSIONS.add(sid)
        resp.set_cookie("sid", sid)
        return resp
    if m("login_wall"):
        return HTMLResponse(_shell("<form action='/login' method='post'><input name='user'>"
                                   "<input type='password' name='pw'><button>Anmelden</button></form>"), status_code=200)
    size = 4 if m("pagination") else CONFIG["page_size"]
    key = "p" if m("pagination") else "page"
    page = int(request.query_params.get(key, "1") or 1)
    rows_src = listed()
    items = rows_src[(page - 1) * size: page * size]
    pages = (len(rows_src) + size - 1) // size
    kinds = m("award_rows")
    cls = {"row": "ergebnis-zeile", "title": "bez", "buyer": "vst", "deadline": "frist", "pub": "veroeff"} \
        if m("rename_classes") else {"row": "tender-row", "title": "title", "buyer": "buyer", "deadline": "deadline",
                                     "pub": "published"}
    a, b = "title", "buyer"
    swap = m("swap_columns")

    def cell(field, t):
        val = {"title": t["buyer"] if swap else t["title"], "buyer": t["title"] if swap else t["buyer"]}[field]
        if field == "title":
            return f"<a href='{detail_path(t['id'])}'>{val}</a>"
        return val

    if m("restructure"):
        rows = "".join(
            f"<div class='card {cls['row']}'><h3 class='{cls[a]}'>{cell(a, t)}</h3><p class='{cls[b]}'>{cell(b, t)}"
            f"</p><span class='{cls['pub']}'>{fmt_d(t['published'])}</span> · <span class='{cls['deadline']}'>"
            f"{fmt_dt(t['deadline'])}</span> · <span class='nr'>{t['id']}</span></div>" for t in items)
        body = f"<h2>Aktuelle Ausschreibungen</h2><div class='cards'>{rows}</div>"
    else:
        rows = "".join(
            f"<tr class='{cls['row']}'><td class='nr'>{t['id']}</td><td class='{cls[a]}'>{cell(a, t)}</td>"
            f"<td class='{cls[b]}'>{cell(b, t)}</td><td class='{cls['pub']}'>{fmt_d(t['published'])}</td>"
            f"<td class='{cls['deadline']}'>{fmt_dt(t['deadline']) if t['deadline'] else '-'}</td>"
            + (f"<td class='kind'>{t.get('kind', 'Ausschreibung')}</td>" if kinds else "") + "</tr>" for t in items)
        heads = ["Vergabe-Nr.", "Bezeichnung", "Vergabestelle", "Veröffentlicht", "Angebotsfrist"] + \
            (["Art"] if kinds else [])
        body = ("<h2>Aktuelle Ausschreibungen</h2><table class='results'><thead><tr>" +
                "".join(f"<th>{h}</th>" for h in heads) + f"</tr></thead><tbody>{rows}</tbody></table>")
    if m("extra_table"):    # recently awarded contracts in front of the list, drawn with the same row markup
        done = [("MH-2026-0990", "Schulmöbel Rahmenvertrag", "12.11.2026"),
                ("MH-2026-0991", "Straßenbeleuchtung LED", "10.11.2026"),
                ("MH-2026-0992", "Sanierung Stadtbad", "06.11.2026"), ("MH-2026-0993", "Grünflächenpflege", "02.11.2026")]
        body = ("<h2>Kürzlich vergeben</h2><table class='results'><tbody>" + "".join(
            f"<tr class='{cls['row']}'><td class='nr'>{i}</td><td class='{cls['title']}'><a href='/vergeben/{i}'>{n}"
            f"</a></td><td class='{cls['buyer']}'>Stadt Musterhausen</td><td class='{cls['pub']}'>{d}</td>"
            f"<td class='{cls['deadline']}'>vergeben</td></tr>" for i, n, d in done) + "</tbody></table>" + body)
    if m("overlay"):        # a pop-up dialog placed before the content
        body = ("<div class='modal-backdrop'><div class='modal' role='dialog' aria-modal='true'>"
                "<h2>Wichtiger Hinweis</h2><p>Wegen Wartungsarbeiten sind folgende Zeiten eingeschränkt:</p>"
                "<table class='results'><tbody><tr><td>Wartung</td><td>Portal</td><td>Alle Verfahren</td>"
                "<td>21.11.2026</td><td>22.11.2026 08:00</td></tr></tbody></table>"
                "<button class='close'>Schließen</button></div></div>" + body)
    nav = "".join(f"<a class='pg' href='/ausschreibungen?{key}={i}'>{i}</a> " for i in range(1, pages + 1))
    nxt = f"<a class='next' href='/ausschreibungen?{key}={page + 1}'>Weiter</a>" if page < pages else ""
    body += f"<div class='pager'>Seite {page} von {pages}: {nav}{nxt}</div><p class='total'>{len(rows_src)} Treffer</p>"
    return HTMLResponse(_shell(body))


# ------------------------------------------------------------------ postback skin (PRADO / ASP.NET WebForms style)
PB_SIZES = (3, 5)
PB_SESSIONS: set[str] = set()
PB_POSTS: list[dict] = []          # every form post received (for tests)


def _pb_state(page: int, size: int) -> str:
    return base64.b64encode(json.dumps({"p": page, "s": size}).encode()).decode()


def _pb_read_state(value: str):
    try:
        d = json.loads(base64.b64decode(value or ""))
        return int(d["p"]), int(d["s"])
    except (ValueError, KeyError, TypeError):
        return None


def _postback_page(page: int, size: int) -> str:
    prado = CONFIG.get("postback_flavor", "prado") == "prado"
    pages = (len(TENDERS) + size - 1) // size
    items = TENDERS[(page - 1) * size: page * size]
    state = _pb_state(page, size)
    if prado:
        fields = f"<input type='hidden' name='PRADO_PAGESTATE' id='PRADO_PAGESTATE' value='{state}'>"
    else:
        fields = ("<input type='hidden' name='__EVENTTARGET' id='__EVENTTARGET' value=''>"
                  "<input type='hidden' name='__EVENTARGUMENT' id='__EVENTARGUMENT' value=''>"
                  f"<input type='hidden' name='__VIEWSTATE' id='__VIEWSTATE' value='{state}'>")
    opts = "".join(f"<option value='{n}'{' selected' if n == size else ''}>{n}</option>" for n in PB_SIZES)
    onchange = "" if prado else " onchange=\"javascript:setTimeout('__doPostBack(\\'ctl0$pager$size\\',\\'\\')', 0)\""
    select = f"<select name='ctl0$pager$size' id='ctl0_pager_size'{onchange}>{opts}</select>"

    def link(ctl: str, text: str) -> str:
        if prado:
            return f"<a id='ctl0_pager_{ctl}' href='javascript:;//ctl0_pager_{ctl}'>{text}</a>"
        return f"<a id='ctl0_pager_{ctl}' href=\"javascript:__doPostBack('ctl0$pager${ctl}','')\">{text}</a>"

    def off(ctl: str, text: str) -> str:   # PRADO renders a plain span, ASP.NET a disabled link without href
        return f"<span class='btn'>{text}</span>" if prado else f"<a id='ctl0_pager_{ctl}' class='aspNetDisabled'>{text}</a>"
    has_next = page < pages and not (m("pager_stops") and page >= 2)
    pager = (f"{link('prev', 'Zurück') if page > 1 else off('prev', 'Zurück')} "
             f"{link('next', 'Weiter') if has_next else off('next', 'Weiter')} {link('sort', 'Nach Frist sortieren')}")
    def title_link(t) -> str:   # external_row: the last tender is hosted on another platform (its link goes there)
        if m("external_row") and t["id"] == TENDERS[-1]["id"]:
            return f"<a class='extern' href='https://extern.test/pub/{t['id']}'>{t['title']}</a>"
        return f"<a class='intern' href='{detail_path(t['id'])}'>{t['title']}</a>"
    rows = "".join(
        f"<tr class='tender-row'><td class='nr'>{t['id']}</td><td class='title'>{title_link(t)}</td>"
        f"<td class='buyer'>{t['buyer']}</td><td class='published'>{fmt_d(t['published'])}</td>"
        f"<td class='deadline'>{fmt_dt(t['deadline'])}</td></tr>" for t in items)
    scripts = ""
    if prado:
        binds = [("TDropDownList", "size")] + [("TLinkButton", c) for c in ("prev", "next", "sort")]
        scripts = "<script>" + "".join(
            f"new Prado.WebUI.{k}({{'ID':\"ctl0_pager_{c}\",'EventTarget':\"ctl0$pager${c}\",'FormID':\"mainForm\"}});"
            for k, c in binds) + "</script>"
    body = ("<form action='/suche' method='get'><input name='q'><button>Suchen</button></form>"
            f"<form id='mainForm' method='post' action='/ausschreibungen'>{fields}<h2>Aktuelle Ausschreibungen</h2>"
            f"<div class='pager'>Treffer pro Seite {select} · Seite {page} von {pages} · {pager}</div>"
            f"<table class='results'><tbody>{rows}</tbody></table>"
            f"<p class='total'>{len(TENDERS) + CONFIG.get('pb_total_extra', 0)} Treffer</p></form>{scripts}")
    return _shell(body)   # pb_total_extra: the stated total runs ahead of the rows (a listing that changed mid-read)


@app.post("/ausschreibungen", response_class=HTMLResponse)
async def list_postback(request: Request):
    if CONFIG["skin"] != "postback":
        return HTMLResponse("Method Not Allowed", status_code=405)
    form = dict(parse_qsl((await request.body()).decode("utf-8", "replace"), keep_blank_values=True))
    PB_POSTS.append(form)
    if request.cookies.get("sid") not in PB_SESSIONS:
        return HTMLResponse(_shell("<p>Ihre Sitzung ist abgelaufen.</p>"))
    st = _pb_read_state(form.get("PRADO_PAGESTATE") or form.get("__VIEWSTATE") or "")
    if st is None:
        return HTMLResponse(_shell("<p>Ungültiger Seitenstatus.</p>"), status_code=500)
    page, size = st
    target = form.get("PRADO_POSTBACK_TARGET") or form.get("__EVENTTARGET") or ""
    if not m("pager_ignores_target"):
        if target == "ctl0$pager$next":
            page += 1
        elif target == "ctl0$pager$prev":
            page = max(1, page - 1)
        elif target == "ctl0$pager$size":
            size, page = int(form.get("ctl0$pager$size") or size), 1
    return HTMLResponse(_postback_page(page, size))


def _detail_html(tid: str):
    t = BY_ID.get(tid)
    if not t:
        return HTMLResponse(_shell("<p>Nicht gefunden</p>"), status_code=404)
    status = "<p class='status'>Aufgehoben</p>" if t["withdrawn"] else ""
    docs = "".join(f"<li><a class='doc' href='/download/{n}'>{n}</a></li>" for n in t["docs"])
    title, buyer = t["title"], t["buyer"]
    body = (f"<article><h2 class='tender-title'>{title}</h2>{status}<dl>"
            f"<dt>Vergabe-Nr.</dt><dd class='id'>{t['id']}</dd>"
            f"<dt>Aktenzeichen</dt><dd class='ref'>{t['ref']}</dd>"
            f"<dt>Vergabestelle</dt><dd class='buyer'>{buyer}</dd>"
            f"<dt>Veröffentlicht</dt><dd class='published'>{fmt_d(t['published'])}</dd>"
            + (f"<dt>Angebotsfrist</dt><dd class='deadline'>{fmt_dt(t['deadline'])}</dd>" if t["deadline"] else "")
            + f"<dt>CPV</dt><dd class='cpv'>{t['cpv']}</dd>"
            f"<dt>Ort</dt><dd class='place'>{t['zip']} {t['city']}, Deutschland</dd>"
            f"<dt>Verfahren</dt><dd class='procedure'>{t.get('procedure', 'Offenes Verfahren')}</dd></dl>"
            f"<h3>Vergabeunterlagen</h3><ul class='docs'>{docs}</ul></article>")
    return HTMLResponse(_shell(body, title))


@app.get("/tender/{tid}", response_class=HTMLResponse)
async def detail(tid: str):
    if m("move_urls"):
        return HTMLResponse(_shell("<p>Diese Seite wurde verschoben.</p>"), status_code=404)
    return _detail_html(tid)


@app.get("/pub/{tid}", response_class=HTMLResponse)
async def detail_elsewhere(tid: str):
    """The 'other platform' of the external_row mutation (in tests every host is this app)."""
    return _detail_html(tid)


@app.get("/notice/{tid}", response_class=HTMLResponse)
async def detail_moved(tid: str):
    if not m("move_urls"):
        return HTMLResponse(_shell("<p>Nicht gefunden</p>"), status_code=404)
    return _detail_html(tid)


@app.get("/download/{name}")
async def download(name: str):
    media = "application/zip" if name.endswith(".zip") else "application/pdf"
    return Response(doc_bytes(name), media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ------------------------------------------------------------------ api skin
def _spa() -> str:
    off = "start" if m("pagination") else "offset"
    return ("<!doctype html><html><head><title>Vergabeportal</title></head><body><div id='app'>Lädt…</div>"
            "<script>fetch('/api/tenders?" + off + "=0&limit=5').then(r=>r.json()).then(d=>{"
            "document.getElementById('app').innerHTML=(d.items||d.results).map(t=>`<div class=t>"
            "<a href='/app/tender/${t.id||t.notice_id}'>${t.title||t.subject}</a></div>`).join('')})</script>"
            "</body></html>")


def _json_tender(t, full=False) -> dict:
    title, buyer = (t["buyer"], t["title"]) if (m("swap_columns") and not full) else (t["title"], t["buyer"])
    dl = t["deadline"].isoformat() + "+01:00"
    pub = t["published"].isoformat()
    if m("date_format"):
        dl, pub = fmt_dt(t["deadline"]), fmt_d(t["published"])
    if m("rename_keys"):
        d = {"notice_id": t["id"], "subject": title, "contracting_authority": buyer, "publication": pub,
             "closing": dl, "cancelled": t["withdrawn"]}
    else:
        d = {"id": t["id"], "title": title, "buyer": buyer, "publishedOn": pub, "deadline": dl,
             "withdrawn": t["withdrawn"]}
    if full:
        d.update({"reference": t["ref"], "cpv": [t["cpv"]],
                  "place": {"zip": t["zip"], "city": t["city"], "country": "DE"},
                  "documents": [{"name": n, "href": f"/download/{n}"} for n in t["docs"]]})
    return d


@app.get("/api/tenders")
async def api_list(request: Request, limit: int = 5):
    if m("login_wall"):
        return JSONResponse({"error": "login required"}, status_code=401)
    off_key = "start" if m("pagination") else "offset"
    off = int(request.query_params.get(off_key, "0") or 0)
    if CONFIG.get("max_offset") is not None and off >= CONFIG["max_offset"]:
        return JSONResponse({"error": "Result window is too large"}, status_code=400)  # deep-paging refusal
    if CONFIG.get("max_limit") is not None and limit > CONFIG["max_limit"]:
        return JSONResponse({"error": "limit too large"}, status_code=400)  # page-size cap (TenderNed: 100)
    items = [_json_tender(t) for t in TENDERS[off: off + limit]]
    key = "results" if m("rename_keys") else "items"
    return {key: items, "total": len(TENDERS)}


@app.get("/static/detail.js")
async def detail_js():
    """The tender page of the JavaScript app: built in the browser from /api/tenders/{id}, the same labels as the
    server-rendered page. Mutation detail_js_blank: the app never draws the tender (nothing to verify against)."""
    if m("detail_js_blank"):
        return Response("document.getElementById('app').innerHTML = '';", media_type="application/javascript")
    js = ("const id = location.pathname.split('/').pop();"
          "const day = s => /^\\d{4}-/.test(s) ? s.slice(8, 10) + '.' + s.slice(5, 7) + '.' + s.slice(0, 4) : s;"
          "const at = s => /^\\d{4}-/.test(s) ? day(s) + ' ' + s.slice(11, 16) : s;"
          "fetch('/api/tenders/' + id).then(r => r.json()).then(t => {"
          "document.getElementById('app').innerHTML = `<article><h2>${t.title || t.subject}</h2><dl>"
          "<dt>Vergabe-Nr.</dt><dd>${t.id || t.notice_id}</dd>"
          "<dt>Vergabestelle</dt><dd>${t.buyer || t.contracting_authority}</dd>"
          "<dt>Veröffentlicht</dt><dd>${day(t.publishedOn || t.publication)}</dd>"
          "<dt>Angebotsfrist</dt><dd>${at(t.deadline || t.closing)}</dd></dl></article>`;});")
    return Response(js, media_type="application/javascript")


@app.get("/app/tender/{tid}", response_class=HTMLResponse)
async def app_tender(tid: str):
    """A tender page of the JavaScript app (api skins): an empty shell without a browser."""
    return HTMLResponse("<!doctype html><html><head><meta charset='utf-8'><title>Vergabeportal</title></head><body>"
                        "<div id='app'></div><script src='/static/detail.js'></script></body></html>")


@app.get("/api/tenders/{tid}")
async def api_detail(tid: str):
    t = BY_ID.get(tid)
    if not t:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _json_tender(t, full=True)


# ------------------------------------------------------------------ spa_post skin: a POST-only JSON search
# The app asks POST /api/search with a JSON body; the page number sits inside the body. GET is not allowed (405),
# so replaying the address with GET (as for GET APIs) finds nothing.
def _spa_post() -> str:
    return ("<!doctype html><html><head><title>Suche</title></head><body><div id='app'>Lädt…</div><script>"
            "fetch('/api/search',{method:'POST',headers:{'Content-Type':'application/json'},"
            "body:JSON.stringify({filter:{active:true},page:{number:0,size:5},order:{field:'pub',dir:'desc'}})})"
            ".then(r=>r.json()).then(d=>{document.getElementById('app').innerHTML=d.hits.map(t=>"
            "`<div class=t><a href='/app/tender/${t.id}'>${t.title}</a></div>`).join('')})</script></body></html>")


@app.post("/api/search")
async def api_search(body: dict):
    page = body.get("page") or {}
    n, size = int(page.get("number", 0) or 0), int(page.get("size", 5) or 5)
    if CONFIG.get("max_offset") is not None and (n + 1) * size > CONFIG["max_offset"]:
        return JSONResponse({"error": f"Combination of page and size exceeds limit of {CONFIG['max_offset']}"},
                            status_code=400)       # result window (oeffentlichevergabe: 10,000)
    sel = TENDERS[n * size:(n + 1) * size]
    return {"hits": [_json_tender(t) for t in sel], "totalHits": len(TENDERS), "offset": n * size}


# ------------------------------------------------------------------ script-driven pagers (no browser needed)
# Three list pages whose next page is reached only through JavaScript. Every request is logged in JS_HITS.
#   /jsurl/liste  - the next control's onclick names the address: setTargetAndSubmit('/jsurl/liste?seite=2')
#                   (the function lives in an external script; it points the list form at that address and submits
#                   it). A session cookie is set on the first page; a page asked without it shows page 1 again.
#                   mutation jsurl_no_address: the onclick calls a function without an address.
#   /jsform/liste - the next control calls nextPage(), an inline script that sets the hidden page field of a form
#                   (0-based) and submits it to /jsform/suche. The last page has no next control.
#   /xhr/liste    - the first rows are in the page; more are loaded by $.ajax POST /xhr/api/more (offset = rows
#                   shown, a per-session CSRF token from the page) answering {"items": "<li>..</li>"}; the server
#                   starts one row early (as a live portal does). Row links are relative to the list page.
JS_HITS: list[dict] = []
JS_SIZE = 5
JS_SESSIONS: dict[str, str] = {}      # session id -> csrf token


def _js_rows(items, tag="tr", rel="") -> str:
    if tag == "tr":
        return "".join(
            f"<tr class='tender-row'><td class='nr'>{t['id']}</td><td class='title'><a href='{detail_path(t['id'])}'>"
            f"{t['title']}</a></td><td class='buyer'>{t['buyer']}</td><td class='published'>{fmt_d(t['published'])}"
            f"</td><td class='deadline'>{fmt_dt(t['deadline'])}</td></tr>" for t in items)
    return "".join(
        f"<li id='tender-{t['id']}'><div class='item'><span class='nr'>{t['id']}</span> <span class='title'>"
        f"<a href='{rel}tender/{t['id']}'>{t['title']}</a></span> <span class='buyer'>{t['buyer']}</span> "
        f"<span class='published'>{fmt_d(t['published'])}</span> <span class='deadline'>{fmt_dt(t['deadline'])}"
        f"</span></div></li>" for t in items)


async def _js_log(request: Request) -> dict:
    form = dict(parse_qsl((await request.body()).decode("utf-8", "replace"), keep_blank_values=True))
    hit = {"method": request.method, "path": request.url.path, "query": dict(request.query_params), "form": form,
           "xhr": request.headers.get("x-requested-with"), "cookie": request.headers.get("cookie") or ""}
    JS_HITS.append(hit)
    return hit


@app.get("/static/functions.js")
async def functions_js():
    return Response("function setTargetAndSubmit(target, formname) { var f = document.forms[formname || "
                    "'masterForm']; f.action = target; f.submit(); }", media_type="application/javascript")


@app.api_route("/jsurl/liste", methods=["GET", "POST"], response_class=HTMLResponse)
async def jsurl_list(request: Request):
    await _js_log(request)
    sid = request.cookies.get("JSESSIONID")
    page = int(request.query_params.get("seite") or 1)
    new_sid = None
    if sid not in JS_SESSIONS:
        page, new_sid = 1, secrets.token_hex(8)
        JS_SESSIONS[new_sid] = ""
    pages = (len(TENDERS) + JS_SIZE - 1) // JS_SIZE
    items = TENDERS[(page - 1) * JS_SIZE: page * JS_SIZE]
    if page < pages and not m("jsurl_no_address"):
        nxt = (f"<a class='browseForward' aria-label='nächste Seite' title='nächste Seite' href='javascript:void(0);' "
               f"onclick=\"setTargetAndSubmit('/jsurl/liste?method=showTable&amp;seite={page + 1}')\">&#160;</a>")
    elif page < pages:
        nxt = ("<a class='browseForward' aria-label='nächste Seite' href='javascript:void(0);' "
               "onclick='browse(this)'>&#160;</a>")
    else:
        nxt = ("<a class='browseForwardGhost' aria-disabled='true' aria-label='nächste Seite' "
               "href='javascript:void(0);'>&#160;</a>")
    body = ("<script src='/static/functions.js'></script>"
            "<form name='listForm' method='post' action='/jsurl/liste?method=search' id='masterForm'>"
            "<input type='hidden' name='filterStatus' value='expanded'><input name='searchText' value=''>"
            f"<table class='results'><tbody>{_js_rows(items)}</tbody></table>"
            f"<nav class='browsePages'>{nxt}<div class='browsePagesText'>Seite: {page} von {pages} - "
            f"Gesamteinträge: {len(TENDERS)}</div></nav></form>")
    resp = HTMLResponse(_shell(body))
    if new_sid:
        resp.set_cookie("JSESSIONID", new_sid)
    return resp


def _jsform_page(page0: int) -> str:
    pages = (len(TENDERS) + JS_SIZE - 1) // JS_SIZE
    items = TENDERS[page0 * JS_SIZE:(page0 + 1) * JS_SIZE]
    nxt = ("<span class='btnRect' aria-label='Fiche suivante' onclick='nextPage()' role='link'>&gt;</span>"
           if page0 + 1 < pages else "")
    prev = ("<span class='btnRect' aria-label='Fiche précédente' onclick='prevPage()' role='link'>&lt;</span>"
            if page0 else "")
    script = ("<script>function nextPage() { updateFormValues(); $(\"#nextPrevPage\").val(" + str(page0 + 1) + ");"
              " document.getElementById(\"hiddenForm\").submit(); }\n function prevPage() { $(\"#nextPrevPage\")"
              ".val(" + str(max(page0 - 1, 0)) + "); document.getElementById(\"hiddenForm\").submit(); }</script>")
    body = (f"<h1 class='resultTitle'><span class='count'>{len(TENDERS)}</span> consultations</h1>"
            f"<ul class='pagi'>Page {page0 + 1} / {pages} {prev} {nxt}</ul>"
            f"<table class='results'><tbody>{_js_rows(items)}</tbody></table>"
            "<form action='/jsform/suche' method='post' accept-charset='ISO-8859-15' id='hiddenForm'>"
            "<input type='hidden' name='bean.dateFrom' value='20/11/2026'><input type='hidden' name='bean.procedure' "
            "value='-1'><input type='hidden' name='bean.page' id='nextPrevPage'>"
            "<input type='hidden' name='bean.orderby' id='orderBy'></form>" + script)
    return _shell(body)


@app.get("/jsform/liste", response_class=HTMLResponse)
async def jsform_list(request: Request):
    await _js_log(request)
    return HTMLResponse(_jsform_page(int(request.query_params.get("bean.page") or 0)))


@app.post("/jsform/suche", response_class=HTMLResponse)
async def jsform_search(request: Request):
    hit = await _js_log(request)
    if hit["form"].get("bean.dateFrom") != "20/11/2026":
        return HTMLResponse(_shell("<p>Bitte Suchkriterien angeben.</p>"), status_code=400)
    return HTMLResponse(_jsform_page(int(hit["form"].get("bean.page") or 0)))


@app.get("/xhr/liste", response_class=HTMLResponse)
async def xhr_list(request: Request):
    await _js_log(request)
    sid, token = secrets.token_hex(8), secrets.token_hex(20)
    JS_SESSIONS[sid] = token
    script = ("<script>var loading = false; $(window).scroll(function() { var offset = $('ul.stream > li').length;"
              f" var totalEntries = {len(TENDERS)};\n if (!loading && offset < totalEntries) {{ loading = true;\n"
              " $.ajax({ url: '/xhr/api/more', method: 'POST', dataType: 'json',\n data: { offset: offset, filter: 0, "
              f"search: '', form: {{\"orderBy\": \"created desc\", \"radius\": 30}}, CSRF_TOKEN: '{token}' }},\n"
              " success: function(data) { $('ul.stream').append(data.items); loading = false; } }); } });</script>")
    body = (f"<div class='row'><div class='hits'>hit: {len(TENDERS)}</div></div>"
            f"<div class='col'><ul class='stream'>{_js_rows(TENDERS[:JS_SIZE], 'li', '../')}</ul></div>{script}")
    resp = HTMLResponse(_shell(body))
    resp.set_cookie("XSID", sid)
    return resp


@app.post("/xhr/api/more")
async def xhr_more(request: Request):
    hit = await _js_log(request)
    token = JS_SESSIONS.get(request.cookies.get("XSID") or "")
    if not token or hit["form"].get("CSRF_TOKEN") != token:
        return RedirectResponse("/xhr/liste", status_code=302)      # a rejected token: back to the start page
    off = int(hit["form"].get("offset") or 0)
    start = max(off - 1, 0)                                         # one row early, like the live portal
    return {"success": True, "items": _js_rows(TENDERS[start:start + JS_SIZE], "li", "../")}
