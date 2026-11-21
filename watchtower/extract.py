"""Static extraction: run a spec against fetched pages. No model calls in here.

Spec (v1) - JSON, produced by the Onboarder, versioned per portal:
{
  "tz": "Europe/Berlin",
  "list": {"kind": "html"|"json", "url": "...", "method": "GET", "params": {}, "items": "<css or jmespath>",
           "pagination": {"type": "page"|"offset"|"next_link"|"none", "param": "page", "start": 1,
                          "size": 5, "size_param": "limit", "next_sel": "a.next", "max_pages": 200},
           "total": {"sel"|"path": "...", "regex": "(\\d+)"}},
  "detail": {"kind": "html"|"json", "url_from": "url" | "template", "template": "https://.../api/{id}"},
  "fields": {"id": {"from": "list", "sel": "td.nr"}, "title": {...}, "url": {"from": "list", "sel": "a", "attr": "href"},
             "submissionDeadline": {"from": "detail", "sel": "dd.deadline", "formats": ["%d.%m.%Y %H:%M"]},
             "withdrawn": {"from": "detail", "sel": ".status", "regex": "(?i)aufgehoben|annul|cancel"}, ...},
  "documents": {"from": "detail", "sel": "ul.docs a", "attr": "href"}  |  {"from": "detail", "path": "documents[*].href"},
  "block_markers": ["input[type=password]"]
}
For html, `sel` is a CSS selector relative to the item (list) or page (detail); "" or "." = the item itself.
For json, `path` is a JMESPath relative to the item (list) or the document (detail).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import jmespath
from selectolax.lexbor import LexborHTMLParser as HTMLParser, LexborNode as Node

REQUIRED = ["id", "url", "title", "buyerName", "publicationDate", "submissionDeadline"]
OPTIONAL = ["referenceNumber", "description", "procedureType", "contractNature", "cpvCodes"]
PLACE = ["postalCode", "city", "countryCode"]
FIELD_TYPES = {"publicationDate": "date", "submissionDeadline": "datetime", "url": "url", "cpvCodes": "cpv",
               "withdrawn": "bool", "procedureType": "procedure", "contractNature": "nature",
               "countryCode": "country"}

# Month names in common portal languages -> English (for strptime %B / %b)
_MONTHS = {
    "januar": "January", "jänner": "January", "februar": "February", "märz": "March", "maerz": "March",
    "april": "April", "mai": "May", "juni": "June", "juli": "July", "august": "August", "september": "September",
    "oktober": "October", "november": "November", "dezember": "December",
    "janvier": "January", "février": "February", "fevrier": "February", "mars": "March", "avril": "April",
    "juin": "June", "juillet": "July", "août": "August", "aout": "August", "septembre": "September",
    "octobre": "October", "novembre": "November", "décembre": "December", "decembre": "December",
    "enero": "January", "febrero": "February", "marzo": "March", "abril": "April", "mayo": "May",
    "junio": "June", "julio": "July", "agosto": "August", "septiembre": "September", "setiembre": "September",
    "octubre": "October", "noviembre": "November", "diciembre": "December",
    "januari": "January", "februari": "February", "maart": "March", "mei": "May", "augustus": "August",
    "gennaio": "January", "febbraio": "February", "aprile": "April", "maggio": "May", "giugno": "June",
    "luglio": "July", "settembre": "September", "ottobre": "October", "dicembre": "December",
}
_MONTH_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True)) + r")\b", re.I)

_PROCEDURE = [("negotiated", r"verhandl|négoci|negoci|onderhandel|negotiat"),
              ("restricted", r"nicht\s*offen|beschränkt|restreint|restringid|niet-openbaar|restrict"),
              ("open", r"\boffen|\böffentlich|ouvert|abierto|openbaar|\bopen\b")]
_NATURE = [("works", r"bauleist|travaux|\bobras?\b|werken|\bworks?\b"),
           ("services", r"dienstleist|services?|servicios|diensten"),
           ("supplies", r"lieferl|lieferung|fournitur|suministro|leveringen|suppl")]

# The kind or phase of a notice (fields.noticeType: a column, label, badge or JSON field the portal shows). Only a
# call for competition still taking offers is open; a result/award notice, an ex-post publication, prior information
# only or a closed phase is "closed", a cancellation "withdrawn". Generic procurement vocabulary, several languages.
# A correction (Berichtigung, rectificatif, corrigendum) says nothing about the phase: no signal.
_NOTICE_CALL = re.compile(  # prior information used as the call itself, a call for competition: still a call
    r"aufruf zum wettbewerb|call for competition|appel (?:public )?(?:à|a) (?:la )?concurrence|"
    r"oproep tot mededinging|convocatoria de licitaci|indizione di gara|zaproszenie do", re.I)
_NOTICE_WITHDRAWN = re.compile(
    r"cancel|annul|aufhebung|aufgehoben|widerruf|zur[üu]ckgezogen|unieważni|uniewazni|anulaci|anulad|desiert|"
    r"revoca|intrekking|ingetrokken|be[ëe]indig|infructu|sans suite|withdrawn|discontinued", re.I)
_NOTICE_CLOSED = re.compile(
    r"\baward|\bresults?\b|ex[\s-]?post|ex[\s-]?ante|\bveat\b|prior information|"
    r"vergeben|vergebene|vergabebekanntmachung|zuschlag|ergebnis|vorinformation|"
    r"attribu|r[ée]sultat|pr[ée]-?information|"
    r"gunning|gegund|vooraankondiging|"
    r"adjudicaci|adjudicad|formalizaci|formalizad|resuelt|anuncio previo|informaci[oó]n previa|"
    r"aggiudica|\besito|preinformazione|"
    r"udzieleni|wynik|wst[eę]pne og[lł]oszenie|"
    r"adjudica[çc][ãa]o|pr[ée]-?informa[çc][ãa]o|tilldelning|tildeling|"
    r"\bclosed\b|cl[ôo]tur|cerrad|chius|gesloten|abgelaufen|abgeschlossen|beendet|expired|zako[nń]czon", re.I)


# A procedure type that names a result publication (portals often print "Ex post" where the procedure type stands):
# used when the spec maps no noticeType. Narrow on purpose: a procedure name such as "adjudicación directa" or
# "Freihändige Vergabe" is a procedure, not a result notice.
_RESULT_PROCEDURE = re.compile(
    r"ex[\s-]?post|ex[\s-]?ante|vergebene[rn]?\s+auftr|bereits vergeben|award notice|contract award|"
    r"avis d.attribution|gegunde opdracht|anuncio de adjudicaci|avviso di aggiudicazione|og[lł]oszenie o udzieleniu",
    re.I)


def procedure_notice_status(raw: Any) -> str | None:
    """'closed' when a procedure-type value names a result publication (see _RESULT_PROCEDURE), else None."""
    s = clean_text(str(raw)) if isinstance(raw, (str, int, float)) else None
    return "closed" if s and _RESULT_PROCEDURE.search(s) else None


def notice_field(fields: dict) -> str:
    """The field that tells the notice's kind: noticeType when the spec maps it, else procedureType."""
    return "noticeType" if "noticeType" in (fields or {}) else "procedureType"


def row_notice_status(raw: dict, fields: dict) -> str | None:
    """'closed' | 'withdrawn' | None for one raw row (see notice_status / procedure_notice_status)."""
    if "noticeType" in (fields or {}):
        return notice_status(raw.get("noticeType"), fields.get("noticeType") or {})
    return procedure_notice_status(raw.get("procedureType"))


def notice_status(raw: Any, fspec: dict | None = None) -> str | None:
    """'closed' | 'withdrawn' for a notice type or phase that is not an open call; None = no signal (a call, a
    correction, a value we do not recognise). An explicit fields.noticeType.map is used first."""
    if raw is None or isinstance(raw, (dict, bool)):
        return None
    if isinstance(raw, list):
        found = [notice_status(v, fspec) for v in raw]
        return "withdrawn" if "withdrawn" in found else ("closed" if "closed" in found else None)
    m = (fspec or {}).get("map")
    if isinstance(m, dict) and m:
        c = str(raw).strip()
        v = m.get(c)
        if v is None:
            v = {str(k).strip().lower(): val for k, val in m.items()}.get(c.lower())
        if v in ("open", "closed", "withdrawn"):
            return None if v == "open" else v
    s = (clean_text(str(raw)) or "").lower()
    if not s or _NOTICE_CALL.search(s):
        return None
    if _NOTICE_WITHDRAWN.search(s):
        return "withdrawn"
    if _NOTICE_CLOSED.search(s):
        return "closed"
    return None


def clean_text(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None


# ------------------------------------------------------------------ value parsing
def _normalize_months(s: str) -> str:
    return _MONTH_RE.sub(lambda mt: _MONTHS[mt.group(1).lower()], s)


_TZ_ABBR = {"UTC": 0, "GMT": 0, "Z": 0, "WET": 0, "WEST": 1, "BST": 1, "CET": 1, "CEST": 2, "EET": 2, "EEST": 3,
            "MSK": 3}
_OFFSET_RE = re.compile(r"\s*(?:\(?(?:UTC|GMT)\s*)?([+\-\u2212])(\d{1,2})(?::?(\d{2}))?\)?\s*$|\s+(UTC|GMT|CEST|CET|EEST|EET|WEST|WET|BST|MSK|Z)\s*$")


def split_offset(s: str) -> tuple[str, timezone | None]:
    """Split an explicit trailing UTC offset or zone abbreviation ('+02:00', 'UTC+2', 'CEST') off a date string.
    Returns (rest, tzinfo or None). Only a real offset counts; plain text is left alone."""
    m = _OFFSET_RE.search(s)
    if not m:
        return s, None
    if m.group(4):
        hours, minutes = _TZ_ABBR[m.group(4)], 0
    else:
        sign = -1 if m.group(1) in ("-", "\u2212") else 1
        hours, minutes = sign * int(m.group(2)), sign * int(m.group(3) or 0)
    if abs(hours) > 14:
        return s, None
    return s[:m.start()].strip(), timezone(timedelta(hours=hours, minutes=minutes))


_TZ_WORDS = {"mez": "+01:00", "cet": "+01:00", "mesz": "+02:00", "cest": "+02:00", "wet": "+00:00",
             "west": "+01:00", "eet": "+02:00", "eest": "+03:00", "gmt": "+00:00", "utc": "+00:00"}
_LOCAL_WORDS = re.compile(r"\b(ortszeit|uhr|h|heure locale|heures?|local time|hora local|lokale tijd|"
                          r"godz\.?|ora locale)\b\.?", re.I)
_FALLBACKS = [  # unambiguous day-first / ISO-like shapes; tried only after the spec's own formats
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})[ ,T-]+(\d{1,2})[:.](\d{2})\b"), "dmy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})[ T]+(\d{1,2}):(\d{2})\b"), "ymd"),
]


def _strip_local_words(s: str) -> tuple[str, str | None]:
    """'10.12.2026 09:00 Ortszeit' -> ('10.12.2026 09:00', None); '… 09:00 MESZ' -> ('… 09:00', '+02:00')."""
    off = None
    for word, o in _TZ_WORDS.items():
        if re.search(rf"\b{word}\b", s, re.I):
            off = o
            s = re.sub(rf"\(?\b{word}\b\)?", " ", s, flags=re.I)
    s = _LOCAL_WORDS.sub(" ", s)
    s = re.sub(r"[()]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" ,"), off


def parse_datetime(raw: Any, formats: list[str], tz: str) -> str | None:
    """Like _parse_datetime_strict, but tolerant of 'Ortszeit'/'Uhr'/'MESZ' and similar decorations, and with
    unambiguous fallbacks (dd.mm.yyyy hh:mm, yyyy-mm-dd hh:mm). Never invents a time."""
    v = _parse_datetime_strict(raw, formats, tz)
    if v or raw is None:
        return v
    cleaned, off = _strip_local_words(_normalize_months(clean_text(str(raw)) or ""))
    if not cleaned:
        return None
    v = _parse_datetime_strict(cleaned, formats, tz)
    if v:
        if off:
            dt = datetime.fromisoformat(v)
            h, m = int(off[1:3]), int(off[4:6])
            sign = 1 if off[0] == "+" else -1
            from datetime import timedelta, timezone as _tz
            v = dt.replace(tzinfo=_tz(sign * timedelta(hours=h, minutes=m))).isoformat()
        return v
    for rx, order in _FALLBACKS:
        mt = rx.search(cleaned)
        if not mt:
            continue
        a, b, c, hh, mm = (int(x) for x in mt.groups())
        day, month, year = (a, b, c) if order == "dmy" else (c, b, a)
        try:
            dt = datetime(year, month, day, hh, mm)
        except ValueError:
            return None
        if off:
            from datetime import timedelta, timezone as _tz
            sign = 1 if off[0] == "+" else -1
            return dt.replace(tzinfo=_tz(sign * timedelta(hours=int(off[1:3]), minutes=int(off[4:6])))).isoformat()
        return dt.replace(tzinfo=ZoneInfo(tz)).isoformat()
    return None


def _parse_datetime_strict(raw: Any, formats: list[str], tz: str) -> str | None:
    """Return ISO datetime with offset, or None. Never guesses a time that isn't shown."""
    if raw is None:
        return None
    s = clean_text(str(raw))
    if not s:
        return None
    # ISO input (APIs)
    try:
        v = s.replace("Z", "+00:00")
        if re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", v):
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=ZoneInfo(tz))
            return dt.isoformat()
    except ValueError:
        pass
    s2 = _normalize_months(s)
    rest, off = split_offset(s2)  # an offset shown by the portal is kept, never replaced by the portal tz
    for fmt in formats or []:
        if "%H" not in fmt and "%I" not in fmt:
            continue  # date-only format cannot yield a deadline time
        for candidate in ((s2, None), (rest, off)) if off else ((s2, None),):
            try:
                dt = datetime.strptime(candidate[0], fmt)
            except ValueError:
                continue
            if dt.tzinfo is not None:  # the format itself captured an offset (%z)
                return dt.isoformat()
            return dt.replace(tzinfo=candidate[1] or ZoneInfo(tz)).isoformat()
    return None


def parse_date(raw: Any, formats: list[str]) -> str | None:
    if raw is None:
        return None
    s = clean_text(str(raw))
    if not s:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return date.fromisoformat(m.group(1)).isoformat()
        except ValueError:
            return None
    s2 = _normalize_months(s)
    for fmt in formats or []:
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_cpv(raw: Any) -> list[str] | None:
    vals = raw if isinstance(raw, list) else [raw]
    out = []
    for v in vals:
        for code in re.findall(r"\b(\d{8})(?:-\d)?\b", str(v or "")):
            if code not in out:
                out.append(code)
    return out or None


def _keyword(raw: Any, table) -> str | None:
    s = (clean_text(str(raw)) or "").lower() if raw is not None else ""
    if not s:
        return None
    for value, pattern in table:
        if re.search(pattern, s):
            return value
    return None


_NO_VALUE = re.compile(r"^(?:n\.?\s?v\.?|n\.?\s?a\.?|n/a|k\.?\s?a\.?|keine(?:\s+angabe)?|ohne|entf[äa]llt|none|null|"
                       r"nil|sans objet|néant|ninguno|geen|[-–—/?*]+)$", re.I)


def is_no_value(raw: Any) -> bool:
    """'nv', 'n.v.', '-', 'k.A.', 'n/a', 'keine Angabe' ...: the portal says there is no value (e.g. no deadline).
    That is "not given", not a value we failed to parse."""
    if raw is None or isinstance(raw, (dict, list, bool)):
        return False
    s = clean_text(str(raw))
    return bool(s) and bool(_NO_VALUE.match(s))


def convert(field: str, raw: Any, fspec: dict, tz: str, base_url: str) -> Any:
    kind = fspec.get("type") or FIELD_TYPES.get(field, "text")
    if raw is None:
        return None
    if kind in ("datetime", "date") and is_no_value(raw):
        return None
    if kind == "datetime":
        return parse_datetime(raw, fspec.get("formats", []), tz)
    if kind == "date":
        return parse_date(raw, fspec.get("formats", []))
    if kind == "url":
        s = clean_text(str(raw))
        return urljoin(base_url, s) if s else None
    if kind == "cpv":
        return parse_cpv(raw)
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        return bool(clean_text(str(raw)))
    if kind == "procedure":
        return _keyword(raw, _PROCEDURE)
    if kind == "nature":
        return _keyword(raw, _NATURE)
    if kind == "country":
        s = clean_text(str(raw)) or ""
        m = re.search(r"\b([A-Z]{2})\b", s)
        return m.group(1) if m else None
    if isinstance(raw, (dict, list)):
        return None
    return clean_text(str(raw))


# ------------------------------------------------------------------ raw extraction
def _apply_regex(val: str | None, regex: str | None) -> str | None:
    if val is None or not regex:
        return val
    m = re.search(regex, val)
    if not m:
        return None
    return m.group(1) if m.groups() else m.group(0)


def _loose(s: str | None) -> str:
    """Compare labels and headers by their letters and digits only: 'Angebots- / Teilnahmefrist:' ~ 'angebotsteilnahmefrist'."""
    return re.sub(r"[\W_]+", "", str(s or "").casefold())


def _elements(node: Node) -> list[Node]:
    return [c for c in node.iter() if c.tag not in ("-text", "-comment")]


def _cells(tr: Node) -> list[tuple[int, Node]]:
    """(start column, cell) for the td/th of a row, colspans counted."""
    out, col = [], 0
    for c in _elements(tr):
        if c.tag in ("td", "th"):
            out.append((col, c))
            span = str(c.attributes.get("colspan") or "1")
            col += int(span) if span.isdigit() and int(span) > 0 else 1
    return out


def _table_rows(table: Node) -> list[Node]:
    """Rows of this table only (not of tables nested in its cells)."""
    rows = []
    for c in _elements(table):
        if c.tag == "tr":
            rows.append(c)
        elif c.tag in ("thead", "tbody", "tfoot"):
            rows.extend(r for r in _elements(c) if r.tag == "tr")
    return rows


def _header_cells(table: Node) -> list[tuple[int, Node]]:
    """The header row: the last row of <thead>, else the first row made only of <th>."""
    head = [r for c in _elements(table) if c.tag == "thead" for r in _elements(c) if r.tag == "tr"]
    if head:
        return _cells(head[-1])
    for tr in _table_rows(table)[:3]:
        cells = _cells(tr)
        if cells and all(c.tag == "th" for _, c in cells):
            return cells
    return []


def table_cell(row: Node, header: str) -> Node | None:
    """The cell of a table row that sits in the column headed `header` (found by the header text, so a portal
    that reorders its columns is still read correctly). Exact match first, else the one header containing it."""
    tr = row
    while tr is not None and tr.tag != "tr":
        tr = tr.parent
    table = tr.parent if tr is not None else None
    while table is not None and table.tag != "table":
        table = table.parent
    if tr is None or table is None:
        return None
    want = _loose(header)
    if not want:
        return None
    heads = [(col, _loose(th.text(separator=" "))) for col, th in _header_cells(table)]
    cols = [col for col, h in heads if h == want] or [col for col, h in heads if h and (want in h or h in want)]
    if len(set(cols)) != 1:
        return None                      # unknown or ambiguous header: no value rather than a wrong one
    for col, cell in _cells(tr):
        if col == cols[0]:
            return None if cell.tag == "th" and _loose(cell.text()) == want else cell
    return None


_SKIP_LABEL_TAGS = {"script", "style", "option", "select", "title", "button", "textarea", "head", "noscript"}


def _value_after(label_node: Node, max_up: int = 4) -> Node | str | None:
    """What follows a label: the next non-empty sibling (an element, or bare text as in '<b>Frist:</b> 10.12.'),
    climbing out of wrappers that hold nothing but the label ('<div><h4><span>Frist</span></h4></div><div>…')."""
    label = _loose(label_node.text())
    cur = label_node
    for _ in range(max_up + 1):
        sib = cur.next
        while sib is not None:      # separators such as '<span>:</span>' (only punctuation) are skipped
            if sib.tag == "-text":
                t = (clean_text(sib.text_content) or "").lstrip(":：").strip()
                if _loose(t) or is_no_value(t):
                    return t
            elif sib.tag not in ("-comment", "br", "script", "style"):
                t = clean_text(sib.text(separator=" "))
                if _loose(t) or is_no_value(t):
                    return sib
            sib = sib.next
        parent = cur.parent
        if parent is None or parent.tag in ("body", "html", "table", "tbody") or _loose(parent.text()) != label:
            return None
        cur = parent
    return None


def label_value(scope: Node | HTMLParser, label: str) -> Node | str | None:
    """The value a page shows next to `label` (dt/dd, th/td, heading + block, 'Label: value'). The first
    occurrence with a non-empty value wins."""
    want = _loose(label)
    root = scope if isinstance(scope, Node) else (scope.body or scope.root)
    if not want or root is None:
        return None
    for node in root.traverse(include_text=False):
        if node.tag in _SKIP_LABEL_TAGS or node.tag.startswith("-"):
            continue
        own = clean_text(node.text(deep=False)) or ""
        if not own or not _loose(own).startswith(want):
            continue
        if _loose(own) == want:
            if own.rstrip().endswith((":", "：")) and _loose(node.text()) != want:
                m = re.match(r"^[^:：]*[:：]\s*(.+)$", clean_text(node.text(separator=" ")) or "")
                if m:                                # 'Frist: <b>10.12.2026</b> 09:00' inside one element
                    return m.group(1)
            v = _value_after(node)
            if v is not None:
                return v
            continue
        m = re.match(r"^([^:：]*)[:：]\s*(.+)$", clean_text(node.text(separator=" ")) or "")
        if m and _loose(m.group(1)) == want:          # inline 'Abgabefrist: 10.12.2026 09:00'
            return m.group(2)
    return None


def _as_list(v) -> list[str]:
    return [x for x in (v if isinstance(v, list) else [v]) if isinstance(x, str) and x.strip()]


def html_value(scope: Node | HTMLParser, fspec: dict) -> Any:
    sel = fspec.get("sel") or ""
    attr = fspec.get("attr")
    if fspec.get("column") or fspec.get("label"):
        found = None        # a list of headers/labels = alternatives (pages that label a value differently)
        if fspec.get("column") and isinstance(scope, Node):
            for col in _as_list(fspec["column"]):
                if (found := table_cell(scope, col)) is not None:
                    break
        elif fspec.get("label"):
            for lab in _as_list(fspec["label"]):
                if (found := label_value(scope, lab)) is not None:
                    break
        if found is None:
            return None
        if isinstance(found, str):           # bare text after a label: no element to select in
            val = _apply_regex(clean_text(found), fspec.get("regex"))
            return ([val] if val else []) if fspec.get("all") else val
        scope = found
    if sel in ("", "."):
        nodes = [scope] if isinstance(scope, Node) else []
    else:
        nodes = scope.css(sel)
    if not nodes:
        return None
    if fspec.get("all"):
        vals = [(n.attributes.get(attr) if attr else n.text(separator=" ")) for n in nodes]
        return [v for v in (clean_text(v) for v in vals) if v]
    n = nodes[0]
    val = n.attributes.get(attr) if attr else n.text(separator=" ")
    if fspec.get("type") == "bool" or FIELD_TYPES.get(fspec.get("_field", "")) == "bool":
        val = _apply_regex(clean_text(val), fspec.get("regex"))
        return val
    return _apply_regex(clean_text(val), fspec.get("regex"))


def json_value(scope: Any, fspec: dict) -> Any:
    path = fspec.get("path")
    if not path:
        return None
    try:
        val = jmespath.search(path, scope)
    except jmespath.exceptions.JMESPathError:
        return None
    if isinstance(val, str):
        return _apply_regex(clean_text(val), fspec.get("regex"))
    return val


def list_items(kind: str, body: Any, items_expr: str) -> list:
    if kind == "json":
        try:
            val = jmespath.search(items_expr, body)
        except jmespath.exceptions.JMESPathError:
            return []
        return val if isinstance(val, list) else []
    return body.css(items_expr) if items_expr else []


def field_from(kind: str, scope: Any, field: str, fspec: dict) -> Any:
    fs = dict(fspec)
    fs["_field"] = field
    return json_value(scope, fs) if kind == "json" else html_value(scope, fs)


def fill_field_template(tmpl: str, raw: dict[str, Any]) -> str | None:
    """'https://portal/project/{id}/notice/{publicationId}': {id} and any other list field of the row by name.
    None when a placeholder has no value (never a half-filled address)."""
    missing = False

    def sub(m):
        nonlocal missing
        v = raw.get(m.group(1))
        if v in (None, "") or isinstance(v, (dict, list)):
            missing = True
            return ""
        return str(v).strip()
    out = re.sub(r"\{([A-Za-z_][\w.]*)\}", sub, str(tmpl))
    return None if missing else out


def assemble(raw: dict[str, Any], spec: dict, list_url: str) -> dict:
    """Convert raw field values into an API tender. Required fields: present (null if unknown).
    Optional fields: only included when known."""
    tz = spec.get("tz") or "UTC"
    fields = spec.get("fields", {})
    t: dict[str, Any] = {}
    for f, fs in fields.items():  # template fields, e.g. url = "https://portal/app/tender/{id}"
        if fs.get("template") and raw.get("id") is not None and raw.get(f) is None:
            raw[f] = fill_field_template(fs["template"], raw)
    for f in REQUIRED:
        t[f] = convert(f, raw.get(f), fields.get(f, {}), tz, list_url)
    if t.get("id") is not None:
        t["id"] = str(t["id"])
    for f in OPTIONAL:
        v = convert(f, raw.get(f), fields.get(f, {}), tz, list_url)
        if v not in (None, "", []):
            t[f] = v
    place = {}
    for f in PLACE:
        key = f"placeOfPerformance.{f}"
        v = convert(f, raw.get(key), fields.get(key, {}), tz, list_url)
        if v:
            place[f] = v
    if place:
        t["placeOfPerformance"] = place
    w = raw.get("withdrawn")
    t["status"] = "withdrawn" if (w is True or (isinstance(w, str) and w)) else "open"
    # an explicit status: fields.status = {"path"/"sel": ..., "map": {"<code>": "open"|"closed"|"withdrawn"}}, or a
    # status set by the reader itself (a change feed's tombstone)
    from .feed import status_of
    st = raw.get("_status") or status_of(fields.get("status") or {}, raw.get("status"))
    if t["status"] != "withdrawn" and st in ("closed", "withdrawn"):
        t["status"] = st
    # the notice's kind or phase (an award or result notice, a cancellation, prior information only): not open
    nk = row_notice_status(raw, fields)
    if nk and t["status"] != "withdrawn":
        t["status"] = nk
    return t


# ------------------------------------------------------------------ HTML reduction for prompts
def reduce_html(html: str, max_chars: int = 30000) -> str:
    """Strip scripts/styles, keep structure + class/id/href, collapse long repeated sibling runs."""
    tree = HTMLParser(html)
    for tag in ("script", "style", "svg", "noscript", "iframe", "link", "meta"):
        for n in tree.css(tag):
            n.decompose()
    body = tree.body or tree.root
    if body is None:
        return ""

    def walk(node: Node, depth: int) -> str:
        if node.tag == "-text":
            t = clean_text(node.text_content)
            return (t[:120] if t else "")
        if node.tag in ("-comment",):
            return ""
        attrs = {k: v for k, v in (node.attributes or {}).items() if k in ("class", "id", "href", "name", "type")
                 and v}
        a = "".join(f' {k}="{v[:100]}"' for k, v in attrs.items())
        kids = [c for c in node.iter(include_text=True)]
        parts, prev_sig, run = [], None, 0
        for c in kids:
            sig = (c.tag, (c.attributes or {}).get("class") if c.tag != "-text" else None)
            run = run + 1 if sig == prev_sig and c.tag != "-text" else 1
            prev_sig = sig
            if run == 4:
                parts.append("<!-- …more of the same… -->")
            if run >= 4:
                continue
            parts.append(walk(c, depth + 1))
        inner = "".join(p for p in parts if p)
        return f"<{node.tag}{a}>{inner}</{node.tag}>"

    out = walk(body, 0)
    return out[:max_chars]


def page_text(html: str, max_chars: int = 8000) -> str:
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "svg"):
        for n in tree.css(tag):
            n.decompose()
    root = tree.body or tree.root
    return (clean_text(root.text(separator="\n")) or "")[:max_chars] if root else ""
