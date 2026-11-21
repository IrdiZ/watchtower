"""Attach + repair: the LLM learns how a portal is read and writes a spec. Then a static dry run and a blind
verification decide whether the spec is trusted.

Model calls per attach (typical): 1-3 classify (cheap) + 2 synthesize (strong) + 1 verify (cheap).
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import jmespath
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import jspager, postback, runner, timeutil
from .extract import (REQUIRED, _LOCAL_WORDS, _normalize_months, clean_text, is_no_value, list_items, page_text,
                      reduce_html, split_offset)
from .http import BudgetExceeded
from .llm import LLMUnavailable, ModelBudgetExhausted
from .pipeline import AttachFailed, PortalBlocked


SPEC_DOC = """A spec is JSON describing how to read a tender portal WITHOUT a model. Shape:
{
 "tz": IANA timezone of the portal's local times (e.g. "Europe/Berlin"),
 "list": {"kind": "html" or "json", "url": absolute URL of the tender list (or JSON API), "method": "GET" or "POST",
          "body": the JSON request body when method is POST (copy the portal's own request body as given; write
                  {today} where today's date belongs),
          "order": "deadline_desc" | "published_desc" | null,
          "params": {extra query params, optional},
          "items": CSS selector of ONE tender row (html) or JMESPath to the array of tenders (json),
          "pagination": {"type": "page"|"offset"|"cursor"|"next_link"|"postback"|"form"|"load_more"|"none",
                         "param": query param name,
                         "start": first value, "step": 1, "size": page size (offset), "size_param": optional,
                         "next_sel": CSS of the next-page link (next_link) or of the next-page control (postback),
                         "size_sel": optional CSS of a results-per-page <select> (postback),
                         "body_path": for a POST body, the dotted path of the page/offset/cursor field inside the
                                      body (e.g. "paging.pageNo") instead of "param"; "size_body_path": same for the size,
                         cursor (keyset / "load more" APIs that return a token or the last key instead of page
                         numbers): "param" (or "body_path") = where the cursor is sent, and EITHER "cursor_path" =
                         JMESPath of the next cursor in the response (e.g. "meta.nextToken", "next_cursor") OR
                         "cursor_from_last" = JMESPath of the cursor value on the LAST item (e.g. "id"); optional
                         "more_path" = JMESPath of a has-more flag, "size" = rows per page,
                         "reverse": optional, for page/offset lists whose request has a sort direction: how to ask for
                                    the SAME list in the opposite order, e.g. {"body_path": "sort.dir", "value":
                                    "ASC"} or {"param": "sortOrder", "value": "asc"}. Declare it whenever the request
                                    sorts; it is only used when the portal refuses pages beyond a result window
                                    (e.g. "at most 10,000 results"), to read the rest from the other end. The sort
                                    must then be on a field EVERY row has (e.g. the publication date, then the id):
                                    rows with an empty sort value stay last in both directions},
          "total": optional {"sel": CSS (html) or "path": JMESPath (json), "regex": "(\\\\d+)"} for the stated total},
 "detail": {"kind": "html"|"json", "url_from": "url" (use the tender's url field) or "template",
            "template": "https://.../api/tenders/{id}" (only if url_from=template; {name} of any other list field may
                        be used too, e.g. "https://.../procedure/{id}/notice/{noticeId}" with a list field
                        "noticeId": {"from": "list", "path": "noticeId"}),
            "follow": optional, ONLY when a value (e.g. the deadline with its time) is not on the tender page but on
                      a page it links to (the full notice): {"sel": CSS of that link on the tender page (e.g.
                      "a[href*='/Notice/']"; the first match is used, so the newest notice first), "attr": "href",
                      optional "regex" capturing part of the link (e.g. the notice number "/Notice/([^/?#]+)"),
                      optional "template": an address built from that capture, e.g. the portal's open-data record
                      "https://.../api/notices/{value}", "kind": "html"|"json" (of the linked page)}; fields read
                      there use "from": "detail2" (label/sel for html, path for json)},
 "fields": { FIELD: {"from": "list"|"detail"|"detail2", "sel": CSS relative to the row (list) or page (detail),
                     "column": (html list that is a TABLE with a header row) the header text of the column that
                               holds the value, copied from the page (e.g. "Closing date"); the runner finds the
                               column by its header, so reordered columns stay right; "sel"/"attr" then apply inside
                               that cell (e.g. "column": "Subject", "sel": "a", "attr": "href"),
                     "label": (html) the label printed next to the value, copied from the page (e.g. "Offers due");
                              the runner returns the value that follows the label (dt/dd, th/td, a heading followed
                              by a block, "Label: value"); best for detail pages that show label/value pairs. When
                              tender pages label the same value differently (e.g. per procedure type), give a list:
                              ["Offers due", "Requests to participate due"] (first one found wins),
                     "attr": attribute name or null for text, "regex": optional capture regex,
                     "path": JMESPath (json kinds), "formats": [strptime formats for dates]} },
 "documents": {"from": "detail", "sel": CSS of document links, "attr": "href"} or {"from": "detail", "path": JMESPath to the list of URLs}
              or, when documents come from a separate address per tender:
              {"from": "template", "url": "https://.../tenders/{id}/documents", "kind": "json"|"html",
               "path"/"sel": where the links or file ids are, "link_template": optional "https://.../files/{value}" to
               turn an extracted id into a download URL},
 "block_markers": [CSS selectors that indicate a login wall]
}
FIELDS: id (the portal's own tender identifier, stable; when the portal publishes several notices for one procedure - original, corrections, cancellation, award - and shows a number they all share (a procedure/dossier number printed on each of them, e.g. 'Procedure no. 2027-0413'), use that shared number, so one procedure is one tender), url (tender page link), title, buyerName (the public body),
publicationDate, submissionDeadline (date AND time), withdrawn (marker that the tender was cancelled; regex for words
like aufgehoben/annulé/cancelled), noticeType (the kind or phase of the notice when the portal shows one - a
column, a label, a badge, a JSON field: e.g. "Contract notice", "Award notice", "Ex post", "Vorinformation",
"Bekanntmachung vergebener Aufträge", "avis d'attribution", "Ausschreibung"; it may be the same column or label as
procedureType when the portal shows both in one, e.g. a "Vergabeart"/"type of procedure" value that says "Ex post";
the runner recognises award/result/cancellation/prior-information notices itself; optional "map": {"<code>": "open" |
"closed" | "withdrawn"} for codes such as "ADJ"), referenceNumber, description, procedureType, contractNature, cpvCodes,
placeOfPerformance.postalCode, placeOfPerformance.city, placeOfPerformance.countryCode.
A field may instead be {"from": "list", "template": "https://portal/path/{id}"} when its value is built from the id
(e.g. the url of a tender in a JavaScript app); {name} of any other field read from the list may be used as well.
IMPORTANT - open tenders only: Watchtower must deliver every OPEN tender. Many portals and APIs list all
publications (old, awarded, corrected). If the portal or API offers a filter for open/current tenders (a status
parameter, "deadline from today", an "aktuelle/laufende/en cours/abiertas" view, a facet in the API), put it in
list.url/list.params so the list only contains open tenders; write {today} where today's date belongs (it is filled
in on every read, e.g. "params": {"deadlineFrom": "{today}"}; {today} is YYYY-MM-DD - when the portal's own form
writes dates differently, give its format: "{today:%d/%m/%Y}"). A search form that only works as a POST may be
given as "method": "POST", "form": {its field names and values}. If it can sort, prefer sorting by deadline
descending.
Declare the order you used in list.order: "deadline_desc" | "published_desc" | null (unknown). Never filter in a
way that could drop open tenders.
PAGINATION: "next_link" only when the next-page link has a real address (href="https://..." or "?page=2"). When the
pager's links are JavaScript form posts - href="javascript:...", __doPostBack(...), hidden page state such as
__VIEWSTATE or PRADO_PAGESTATE (ASP.NET WebForms, PRADO and similar frameworks) - use "postback": next_sel is the CSS
of the ONE enabled next-page link or button (e.g. "a#ctl0_pager_next"; on the last page it must match nothing or a
disabled control, so prefer the link's own id with its tag). The runner then submits the page's form with that control
as the event target, like a browser does; a control whose script names an address (onclick="setTargetAndSubmit(
'/list.do?page=2')") works with postback too (its form is submitted to that address). Never put a javascript: href
into a URL or a next link.
When the next control runs a script that sets a page field of a form and submits it: {"type": "form",
"form_sel": CSS of that form, "field": the page field's name, "start": the field's value for the FIRST page, "step": 1,
"next_sel": optional CSS of the enabled next control (the last page has none), "fields": optional fixed overrides}.
When more rows are loaded by a script request (infinite scroll, "load more"): {"type": "load_more", "url": the
request's address, "method": "POST"|"GET", "fields": its form data (or "json": a JSON body) with "{offset}" = rows
already shown, "{page}" = page number (start/step as for page), "{token:NAME}" = a value the list page sets per visit
(CSRF token, session key; read from the page on every read), "html_path": JMESPath of the rows' HTML when the answer
is JSON (omit when the answer is HTML)}; list.items stays the row selector of the first page.
Rules: only map a field if the portal actually shows it. A wrong value is worse than a missing one.
Read id, url and title from the LIST whenever the list shows them: tender pages are opened only a few per read
(about 60), so a title read from the tender page stays empty for most tenders of a large portal for hours.
Give date formats exactly as shown (e.g. "%d.%m.%Y %H:%M"; month names may be in any language - write %B).
submissionDeadline needs the date AND the time: if the list shows only the date, read it from the tender page.
Cell values such as "nv", "n.v.", "-", "k.A." mean "not given" and are read as null (not an error).
Prefer a JSON API if one is given. Prefer stable selectors (classes/ids) over positions; in a table whose cells
have no stable classes use "column" (its header text) instead of positions such as td:nth-child(2), and on a
page of label/value pairs use "label" instead of positions."""


FEED_DOC = """The portal publishes its tenders as an ATOM/RSS CHANGE FEED (its official syndication / open data): each
entry is the current state of ONE procedure at one moment; newer entries for the same procedure supersede older
ones; files are chained toward older entries. You map the fields of ONE entry. Return JSON:
{"tz": IANA timezone of local times in the entries (e.g. "Europe/Madrid"),
 "older": the rel of the link to the next OLDER file of the chain (usually "next"; "prev-archive" for RFC 5005
          archives), or null,
 "fields": {FIELD: {"path": XPath 1.0 relative to the entry element, using the prefixes given in "namespaces"
                    (the default Atom namespace is "atom", e.g. "atom:title", "atom:link/@href"); a LIST of two paths
                    is joined with a space (a date and a time in separate elements: [".../cbc:EndDate",
                    ".../cbc:EndTime"]); optional "regex" with one capture group; optional "all": true for a list
                    of values (cpvCodes)}},
 "status_map": {every status code the entries use: "open" | "closed" | "withdrawn"}}
FIELDS: id (stable per procedure, the same for all entries of one procedure - usually the entry's atom:id or its
numeric tail with "regex": "(\\d+)$"), url (the tender's page link), title, buyerName (the contracting body),
publicationDate (the date the tender was FIRST published, e.g. the contract notice's issue date - NEVER the entry's
updated time, which changes with every update), submissionDeadline (deadline for offers: date AND time),
status (the element holding the procedure's status code; give its codes in status_map: "open" = still accepting
offers, "closed" = evaluation/awarded/resolved/prior-information only, "withdrawn" = cancelled/annulled/deserted),
referenceNumber (the buyer's file number), description, cpvCodes, placeOfPerformance.postalCode,
placeOfPerformance.city, placeOfPerformance.countryCode.
Only map what the entry actually holds; a wrong value is worse than a missing one. Never use text matching in XPath
(contains() on human text); use element paths."""


def feed_spec_problems(spec: dict) -> list[str]:
    """Static checks of a feed spec (paths compile with the feed's namespaces, a known status vocabulary)."""
    from . import feed
    problems = []
    lst = spec["list"]
    if not (isinstance(lst.get("url"), str) and lst["url"].startswith(("http://", "https://"))):
        problems.append("list.url must be the absolute address of the feed")
    ns = lst.get("namespaces") if isinstance(lst.get("namespaces"), dict) else {}
    fields = spec.get("fields")
    if not isinstance(fields, dict):
        return problems + ["fields missing"]
    for name in ("id", "url", "title"):
        if name not in fields:
            problems.append(f"field '{name}' is not mapped")
    for name, fs in fields.items():
        if not isinstance(fs, dict):
            problems.append(f"field {name}: must be an object")
            continue
        if fs.get("from", "list") != "list":
            problems.append(f"field {name}: a feed entry holds the fields (from must be 'list')")
        for p in feed.path_problems(fs.get("path"), ns):
            problems.append(f"field {name}: {p}")
        if fs.get("regex"):
            try:
                re.compile(fs["regex"])
            except re.error:
                problems.append(f"field {name}: invalid regex")
    st = fields.get("status")
    if isinstance(st, dict) and st.get("map") is not None:
        m = st["map"]
        if not isinstance(m, dict) or not m or any(v not in feed.STATUSES for v in m.values()):
            problems.append("fields.status.map must map each status code to open, closed or withdrawn")
    return problems


def _feed_sample(doc, n: int = 3) -> list[str]:
    """Entries for the model: the first ones plus, if present, one of each other root-element shape."""
    from . import feed
    return [feed.entry_xml(e, 16000 if i == 0 else 6000) for i, e in enumerate(doc.entries[:n])]


async def synthesize_feed(ctx, found: dict, feedback: list[str] | None = None, old_spec: dict | None = None) -> dict:
    """One strong call: map the fields of a sample entry. The feed's chain, tombstones and paging are generic."""
    from . import feed
    fd = found["feed"]
    doc = fd["doc"]
    codes_hint = None
    payload = {"portalUrl": ctx.url, "feedUrl": fd["url"], "format": doc.format,
               "namespaces": {k: v for k, v in doc.ns.items()},
               "feedLinks": doc.links, "entries": len(doc.entries), "deletedEntries": len(doc.deleted),
               "sampleEntries": _feed_sample(doc),
               "dateElementsInFirstEntry": feed.date_paths(doc, doc.entries[0]) if doc.entries else []}
    if feedback:
        payload["problemsToFix"] = feedback
    if old_spec:
        payload["previousSpecThatBroke"] = old_spec
    out = await ctx.llm.json_call("synthesize", "You map a tender portal's change feed for a static reader. "
                                  + FEED_DOC, json.dumps(payload, ensure_ascii=False),
                                  portal_id=ctx.portal_id, run_id=ctx.run_id)
    fields = out.get("fields") if isinstance(out.get("fields"), dict) else {}
    fields = {k: ({"path": v} if isinstance(v, (str, list)) else dict(v)) for k, v in fields.items()
              if isinstance(v, (str, list, dict))}
    for fs in fields.values():
        fs["from"] = "list"
    smap = out.get("status_map") or out.get("statusMap")
    if isinstance(smap, dict) and "status" in fields:
        fields["status"]["map"] = {str(k): str(v).lower() for k, v in smap.items()}
    spec = {"version": 1, "tz": out.get("tz") or "UTC",
            "list": {"kind": "feed", "url": fd["url"], "format": doc.format,
                     "namespaces": {k: v for k, v in doc.ns.items()},
                     "pagination": {"type": "feed", "older": out.get("older") or None}},
            "fields": fields, "detail": {}, "documents": {}, "block_markers": [],
            "source": {"kind": "official feed", "url": fd["url"], "foundVia": found.get("path")}}
    if not spec["list"]["pagination"]["older"]:
        spec["list"]["pagination"].pop("older")
    bad = validate_spec(spec)
    if bad:
        raise SpecError("; ".join(bad[:4]))
    return spec


def feed_dry_problems(spec: dict, dry) -> list[str]:
    """The static checks a feed spec must pass on its dry run: ids present, dates parse, status codes known."""
    st = (dry.stats or {}).get("feed") or {}
    problems = []
    if st.get("unknownStatuses"):
        problems.append(f"status codes seen in the feed but not in fields.status.map: {st['unknownStatuses']} - map "
                        f"each to open, closed or withdrawn")
    if st.get("entriesWithoutId"):
        problems.append(f"{st['entriesWithoutId']} entries have no id (fields.id path)")
    ts = dry.tenders
    if ts:
        no_dl = sum(1 for t in ts if t.get("status") == "open" and not t.get("submissionDeadline"))
        n_open = sum(1 for t in ts if t.get("status") == "open")
        if n_open and no_dl * 2 > n_open:
            problems.append(f"{no_dl} of {n_open} open tenders have no parseable submissionDeadline (a deadline needs "
                            f"date and time: give the date and the time elements as a list of two paths)")
        if "status" not in (spec.get("fields") or {}):
            problems.append("fields.status is not mapped: without it a change feed cannot tell open tenders from "
                            "awarded or cancelled ones")
    return problems


class SpecError(Exception):
    """The model wrote a spec that cannot work. The message is fed (briefly) into the next attempt.
    `spec`: the spec that failed (for the run log), when there is one."""

    def __init__(self, msg: str, spec: dict | None = None):
        super().__init__(msg)
        self.spec = spec


def _run_error_problem(e: BaseException, spec: dict | None) -> str:
    """An exception while running a spec, as feedback: a selector the parser rejects is named (which one, where),
    so the next attempt can fix it; anything else stays 'spec could not be run'."""
    if _is_css_parse_error(e):
        named = selector_problems(spec or {})
        if named:
            return "spec problem: " + "; ".join(named[:3])
        return (f"spec problem: a CSS selector in the spec cannot be parsed ({_short(e, 120)}); check every 'sel', "
                f"'items', 'next_sel' and 'total.sel' (escape special characters such as '!', or use an attribute "
                f"selector like [class~=\"...\"])")
    return f"spec could not be run: {_short(e)}"


def _short(e: BaseException, n: int = 220) -> str:
    return f"{type(e).__name__}: {str(e)[:n]}"


def _check_selector(sel, what: str, kind: str, problems: list[str]) -> None:
    if not isinstance(sel, str):
        problems.append(f"{what}: selector/path missing or not a string")
        return
    if sel in ("", "."):
        return
    try:
        if kind == "json":
            jmespath.compile(sel)
        else:
            HTMLParser("<div></div>").css(sel)
    except Exception as e:
        if kind == "json":
            problems.append(f"{what}: invalid JMESPath '{sel[:60]}'")
            return
        problems.append(_css_problem(sel, what, e))


def _css_problem(sel: str, what: str, err: BaseException | str) -> str:
    """A named, fixable problem: which selector, where in the spec, what the parser said, how to write it instead."""
    hint = ""
    if re.search(r"(?<!\\)[!@$%^&/?]", re.sub(r"\[[^\]]*\]|\([^)]*\)", "", sel)):
        hint = (" - match a class with special characters as an attribute, which needs no escaping: "
                "[class~=\"govuk-!-font-weight-bold\"] instead of .govuk-!-font-weight-bold")
    elif re.search(r":(contains|matches|-soup-contains|has-text|text)\b", sel):
        hint = " - CSS cannot match text; give the label or column header instead"
    return f"{what}: invalid CSS selector '{sel[:80]}' (cannot be parsed: {str(err)[:80]}){hint}"


def _is_css_parse_error(e: BaseException) -> bool:
    return type(e).__name__ == "SelectolaxError" or "parse css selector" in str(e).lower()


def selector_problems(spec: Any) -> list[str]:
    """Every CSS selector anywhere in a spec that the parser rejects, named by where it sits. A runner error such as
    'Can't parse CSS selector' then becomes a problem the model can fix, never a run-ending error."""
    out: list[str] = []
    if not isinstance(spec, dict):
        return out
    lst = spec.get("list") if isinstance(spec.get("list"), dict) else {}
    lkind = lst.get("kind", "html")
    dkind = (spec.get("detail") or {}).get("kind", "html") if isinstance(spec.get("detail"), dict) else "html"

    def css(sel, what):
        if isinstance(sel, str) and sel.strip() not in ("", "."):
            _check_selector(sel, what, "html", out)
    if lkind == "html":
        css(lst.get("items"), "list.items")
        css((lst.get("total") or {}).get("sel") if isinstance(lst.get("total"), dict) else None, "list.total.sel")
    pag = lst.get("pagination") if isinstance(lst.get("pagination"), dict) else {}
    for key in ("next_sel", "size_sel", "form_sel"):
        css(pag.get(key), f"list.pagination.{key}")
    for name, fs in (spec.get("fields") or {}).items() if isinstance(spec.get("fields"), dict) else []:
        if isinstance(fs, dict) and not fs.get("template"):
            origin = fs.get("from", "list")
            kind = lkind if origin == "list" else (_source_kind(spec, origin) if origin == "detail2" else dkind)
            if kind == "html":
                css(fs.get("sel"), f"field {name}")
    fol = (spec.get("detail") or {}).get("follow") if isinstance(spec.get("detail"), dict) else None
    if isinstance(fol, dict) and dkind == "html":
        css(fol.get("sel"), "detail.follow.sel")
    docs = spec.get("documents") if isinstance(spec.get("documents"), dict) else {}
    if docs.get("sel") and (docs.get("kind") or dkind) == "html":
        css(docs.get("sel"), "documents.sel")
    for i, m in enumerate(spec.get("block_markers") or [] if isinstance(spec.get("block_markers"), list) else []):
        css(m, f"block_markers[{i}]")
    return list(dict.fromkeys(out))


def _check_request_paging(lst: dict, pag: dict, lkind: str, problems: list[str]) -> None:
    """Static checks for POST list requests, paging inside the body, and cursor paging."""
    method = lst.get("method") or "GET"
    if not isinstance(method, str) or method.upper() not in ("GET", "POST"):
        problems.append("list.method must be GET or POST")
    method = str(method).upper()
    body = lst.get("body")
    if method == "POST" and body is not None and not isinstance(body, (dict, list)):
        problems.append("list.body must be the JSON request body (an object)")
    for key in ("body_path", "size_body_path"):
        bp = pag.get(key)
        if bp is None:
            continue
        if not isinstance(bp, str) or not runner.BODY_PATH_RE.match(bp):
            problems.append(f"pagination.{key} must be a dotted path inside the body, like 'paging.pageNo'")
        elif method != "POST" or not isinstance(body, dict):
            problems.append(f"pagination.{key} needs list.method POST and list.body (a JSON object)")
    rev = pag.get("reverse")
    if rev is not None:
        if not isinstance(rev, dict) or "value" not in rev or len([k for k in ("param", "body_path") if rev.get(k)]) != 1:
            problems.append("pagination.reverse must be {\"param\" or \"body_path\": where the sort direction goes, "
                            "\"value\": the opposite direction}")
        elif pag.get("type") not in ("page", "offset"):
            problems.append("pagination.reverse works with page or offset pagination only")
        elif rev.get("body_path") and (not runner.BODY_PATH_RE.match(str(rev["body_path"])) or method != "POST"
                                       or not isinstance(body, dict)):
            problems.append("pagination.reverse.body_path must be a dotted path inside the POST body")
    if pag.get("type") == "cursor":
        if lkind != "json":
            problems.append("cursor pagination needs a JSON list (list.kind json)")
        if len([k for k in ("cursor_path", "cursor_from_last") if pag.get(k)]) != 1:
            problems.append("cursor pagination needs exactly one of pagination.cursor_path / cursor_from_last")
        for k in ("cursor_path", "cursor_from_last", "more_path"):
            if pag.get(k):
                _check_selector(pag[k], f"pagination.{k}", "json", problems)


def _check_scripted_paging(pag: dict, lkind: str, problems: list[str]) -> None:
    """Static checks for the script-driven pagers: "form" (submit a form with its page field set) and "load_more"
    (the script request that appends rows)."""
    t = pag["type"]
    if lkind != "html":
        problems.append(f"pagination type {t} needs list.kind html (rows read with the list's CSS selectors)")
    for key in ("start", "step"):
        if key in pag and not (isinstance(pag[key], int) and not isinstance(pag[key], bool)):
            problems.append(f"pagination.{key} must be a whole number")
    if t == "form":
        if not pag.get("form_sel"):
            problems.append("pagination.form_sel (CSS of the form the pager submits) is missing")
        if not (isinstance(pag.get("field"), str) and pag["field"].strip()):
            problems.append("pagination.field (name of the form's page field) is missing")
        for key in ("form_sel", "next_sel"):
            if pag.get(key) is not None:
                _check_selector(pag.get(key), f"pagination.{key}", "html", problems)
        if pag.get("fields") is not None and not isinstance(pag["fields"], dict):
            problems.append("pagination.fields must be an object of field name -> value")
        return
    url = pag.get("url")
    if not (isinstance(url, str) and url.strip() and not url.strip().lower().startswith("javascript:")):
        problems.append("pagination.url (the address of the script's request) is missing")
    if str(pag.get("method") or "POST").upper() not in ("GET", "POST"):
        problems.append("pagination.method must be GET or POST")
    if pag.get("fields") is not None and pag.get("json") is not None:
        problems.append("pagination: give fields (form data) or json (a JSON body), not both")
    if pag.get("fields") is not None and not isinstance(pag["fields"], dict):
        problems.append("pagination.fields must be an object (the request's form data)")
    if pag.get("json") is not None and not isinstance(pag["json"], (dict, list)):
        problems.append("pagination.json must be the JSON request body (an object)")
    blob = json.dumps([pag.get("url"), pag.get("fields"), pag.get("json")], ensure_ascii=False)
    if "{offset}" not in blob and "{page}" not in blob:
        problems.append("pagination load_more: put {offset} (rows already shown) or {page} where the request says "
                        "which rows to load next; without it every request asks for the same rows")
    if pag.get("html_path"):
        _check_selector(pag["html_path"], "pagination.html_path", "json", problems)


def _template_problems(tmpl, what: str, spec: dict) -> list[str]:
    """A per-row address template: at least one placeholder, and every {name} is {id} or a field read from the list
    (its value is filled in per row, e.g. {publicationId} next to {id})."""
    fields = spec.get("fields") if isinstance(spec.get("fields"), dict) else {}
    names = re.findall(r"\{([A-Za-z_][\w.]*)\}", str(tmpl or ""))
    if not names:
        return [f"{what} needs {{id}} (or another list field in braces)"]
    out = []
    for name in dict.fromkeys(names):
        fs = fields.get(name)
        if name != "id" and not (isinstance(fs, dict) and fs.get("from", "list") == "list" and not fs.get("template")):
            out.append(f"{what} uses {{{name}}} but '{name}' is not a field read from the list (map it, e.g. "
                       f"\"{name}\": {{\"from\": \"list\", \"path\": ...}}, or use {{id}})")
    return out


def _check_template_fields(spec: dict, problems: list[str]) -> None:
    det = spec.get("detail")
    if isinstance(det, dict) and det.get("url_from") == "template":
        problems.extend(_template_problems(det.get("template"), "detail.template", spec))


_SELF_SELECTORS = {":scope", "&", "self", ".", "this", ":self", "*:scope"}
_ODD_CLASS_RE = re.compile(r"(?<!\\)\.((?:[A-Za-z_-]|\\.)(?:[\w-]|\\.|[!@$%^&?/])*[!@$%^&?/](?:[\w-]|\\.|[!@$%^&?/])*)")


def _repair_class_selector(sel: str) -> str:
    """A class name with special characters written unescaped or over-escaped ('.govuk-!-font-weight-bold',
    '.govuk-\\\\!-x'): when the parser rejects the selector, match those classes as attributes instead
    ([class~="govuk-!-font-weight-bold"]), which needs no escaping. A selector that parses is left as it is."""
    probs: list[str] = []
    _check_selector(sel, "x", "html", probs)
    if not probs:
        return sel
    fixed = _ODD_CLASS_RE.sub(lambda m: '[class~="' + re.sub(r"\\+(.)", r"\1", m.group(1)) + '"]', sel)
    probs2: list[str] = []
    _check_selector(fixed, "x", "html", probs2)
    return fixed if not probs2 else sel


def coerce_spec(spec: dict) -> dict:
    """Repair harmless model sloppiness instead of burning a retry: a field given as a bare string becomes
    {"sel": ...} / {"path": ...}; ':scope'-style 'the item itself' selectors become '' (supported by the runner)."""
    if not isinstance(spec, dict):
        return spec
    lkind = (spec.get("list") or {}).get("kind", "html")
    dkind = (spec.get("detail") or {}).get("kind", "html")
    fields = spec.get("fields")
    if isinstance(fields, dict):
        for name, fs in list(fields.items()):
            if isinstance(fs, str):
                fields[name] = {"path": fs} if lkind in ("json", "feed") else {"sel": fs}
                fs = fields[name]
            if isinstance(fs, dict):
                kind = lkind if fs.get("from", "list") == "list" else dkind
                if kind == "html" and isinstance(fs.get("sel"), str):
                    sel = fs["sel"].strip()
                    if sel in _SELF_SELECTORS:
                        fs["sel"] = ""
                    elif sel.startswith(":scope"):
                        fs["sel"] = sel[len(":scope"):].lstrip(" >").strip()
    lst = spec.get("list") if isinstance(spec.get("list"), dict) else {}
    if lkind == "html" and isinstance(lst.get("total"), dict) and isinstance(lst["total"].get("sel"), str):
        lst["total"]["sel"] = _repair_class_selector(lst["total"]["sel"])
    if lkind == "html" and isinstance(lst.get("items"), str):
        lst["items"] = _repair_class_selector(lst["items"])
    for fs in (fields or {}).values() if isinstance(fields, dict) else []:
        if isinstance(fs, dict) and isinstance(fs.get("sel"), str):
            kind = lkind if fs.get("from", "list") == "list" else dkind
            if kind == "html":
                fs["sel"] = _repair_class_selector(fs["sel"])
    if isinstance(spec.get("block_markers"), list):
        # a login-wall marker the parser rejects can never match: drop it rather than fail the read on it
        spec["block_markers"] = [m for m in spec["block_markers"]
                                 if isinstance(m, str) and m.strip() and not selector_problems({"block_markers": [m]})]
    return spec


def _source_kind(spec: dict, origin: str) -> str:
    """html or json: the kind of page a detail / detail2 field is read from."""
    det = spec.get("detail") if isinstance(spec.get("detail"), dict) else {}
    if origin == "detail2" and isinstance(det.get("follow"), dict):
        return det["follow"].get("kind") or "html"
    return det.get("kind", "html")


def validate_spec(spec: Any, full: bool = False) -> list[str]:
    """Static sanity check of a model-written spec; returns concise problems (empty = usable).
    full=True also checks the detail/documents part."""
    problems: list[str] = []
    if not isinstance(spec, dict):
        return ["spec is not a JSON object"]
    tz = spec.get("tz")
    try:
        ZoneInfo(tz or "UTC")
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        problems.append(f"tz '{tz}' is not an IANA timezone")
    lst = spec.get("list")
    if not isinstance(lst, dict):
        return problems + ["list part missing"]
    if lst.get("kind") == "feed":
        return problems + feed_spec_problems(spec)
    lkind = lst.get("kind", "html")
    if lkind not in ("html", "json"):
        problems.append(f"list.kind must be html or json, got '{lkind}'")
        lkind = "html"
    if not (isinstance(lst.get("url"), str) and lst["url"].startswith(("http://", "https://"))):
        problems.append("list.url must be an absolute http(s) URL")
    if lst.get("items"):
        _check_selector(lst["items"], "list.items", lkind, problems)
    else:
        problems.append("list.items is empty")
    pag = lst.get("pagination") or {"type": "none"}
    if not isinstance(pag, dict) or pag.get("type", "none") not in ("page", "offset", "cursor", "next_link", "postback",
                                                                    "form", "load_more", "none"):
        problems.append("list.pagination.type must be page, offset, cursor, next_link, postback, form, load_more or "
                        "none")
    elif pag.get("type") in ("form", "load_more"):
        _check_scripted_paging(pag, lkind, problems)
    elif pag.get("type") == "next_link":
        _check_selector(pag.get("next_sel"), "pagination.next_sel", "html", problems)
    elif pag.get("type") == "postback":
        if lkind != "html":
            problems.append("pagination type postback needs list.kind html (a pager that posts the page's form)")
        if not pag.get("next_sel"):
            problems.append("pagination.next_sel (CSS of the next-page control) is missing")
        for key in ("next_sel", "size_sel", "form_sel"):
            if pag.get(key) is not None:
                _check_selector(pag.get(key), f"pagination.{key}", "html", problems)
    elif pag.get("type") in ("page", "offset", "cursor") and not pag.get("param") and not pag.get("body_path"):
        problems.append("pagination.param (query parameter name) is missing")
    if isinstance(pag, dict):
        _check_request_paging(lst, pag, lkind, problems)
    tot = lst.get("total")
    if isinstance(tot, dict):
        if lkind == "json" and tot.get("path"):
            _check_selector(tot["path"], "list.total.path", "json", problems)
        elif lkind == "html" and tot.get("sel"):
            _check_selector(tot["sel"], "list.total.sel", "html", problems)
    form = lst.get("form")
    if form is not None and not (isinstance(form, dict) and all(
            isinstance(k, str) and (isinstance(v, (str, int)) or (isinstance(v, list) and all(
                isinstance(x, (str, int)) for x in v))) for k, v in form.items())):
        problems.append("list.form must be the search form's fields as {name: value or [values]}")
    elif form is not None and str(lst.get("method") or "GET").upper() != "POST":
        problems.append("list.form is sent as a form POST: set list.method POST (or put the fields in list.params)")
    fields = spec.get("fields")
    if not isinstance(fields, dict):
        return problems + ["fields missing"]
    for name in ("id", "url"):
        if name not in fields:
            problems.append(f"field '{name}' is not mapped")
    for name, fs in fields.items():
        if not isinstance(fs, dict):
            problems.append(f"field {name}: must be an object")
            continue
        origin = fs.get("from", "list")
        kind = lkind if origin == "list" else _source_kind(spec, origin)
        if origin == "detail2" and not isinstance((spec.get("detail") or {}).get("follow"), dict):
            problems.append(f"field {name}: from detail2 needs detail.follow (the link on the tender page to the "
                            f"page that shows it)")
        if fs.get("template"):
            problems.extend(_template_problems(fs["template"], f"field {name}: template", spec))
            continue
        anchored = kind == "html" and (fs.get("column") is not None or fs.get("label") is not None)
        if anchored:
            for key in ("column", "label"):
                v = fs.get(key)
                texts = v if isinstance(v, list) else [v]
                if key in fs and v is not None and not (texts and all(isinstance(x, str) and x.strip()
                                                                      for x in texts)):
                    problems.append(f"field {name}: {key} must be the text shown on the page (or a list of them)")
            if fs.get("column") is not None and origin != "list":
                problems.append(f"field {name}: column (a table header) works on list rows only; use label")
            if fs.get("sel"):                       # optional: a selector inside the cell / the labelled value
                _check_selector(fs["sel"], f"field {name}", kind, problems)
        else:
            _check_selector(fs.get("path") if kind == "json" else fs.get("sel"), f"field {name}", kind, problems)
        if fs.get("regex"):
            try:
                re.compile(fs["regex"])
            except re.error as e:
                # say which pattern and why: a bare "invalid regex" was repeated unchanged by the next attempt
                problems.append(f"field {name}: invalid regex {str(fs['regex'])[:80]!r} ({e}; Python re syntax)")
        for fmt in fs.get("formats") or []:
            try:
                datetime.strptime("2000-01-01 00:00", fmt)
            except ValueError as e:
                if "does not match format" not in str(e) and "unconverted" not in str(e):
                    problems.append(f"field {name}: invalid date format '{fmt}'")
            except Exception:
                problems.append(f"field {name}: invalid date format '{fmt}'")
    det = spec.get("detail")
    if det is not None:
        if not isinstance(det, dict):
            problems.append("detail must be an object")
        else:
            _check_template_fields(spec, problems)
            fol = det.get("follow")
            if fol is not None:
                if not isinstance(fol, dict):
                    problems.append("detail.follow must be an object ({\"sel\": CSS of the link on the tender page})")
                else:
                    dk = det.get("kind", "html")
                    _check_selector(fol.get("path") if dk == "json" else fol.get("sel"), "detail.follow", dk,
                                    problems)
                    if fol.get("kind", "html") not in ("html", "json"):
                        problems.append("detail.follow.kind must be html or json")
                    if fol.get("template") is not None and "{value}" not in str(fol.get("template")):
                        problems.append("detail.follow.template must contain {value} (what the link gives)")
                    if fol.get("regex"):
                        try:
                            re.compile(fol["regex"])
                        except re.error as e:
                            problems.append(f"detail.follow: invalid regex {str(fol['regex'])[:80]!r} ({e})")
    if full:
        docs = spec.get("documents") or {}
        if docs and isinstance(docs, dict):
            dk = (docs.get("kind") or (det or {}).get("kind") or "html") if docs.get("from") == "template" else \
                (det or {}).get("kind", "html")
            if docs.get("from") == "template":
                durl = str(docs.get("url") or "")
                if "{id}" not in durl and not ("{noticeId}" in durl and (spec.get("fields") or {}).get("noticeId")):
                    problems.append("documents.url must contain {id}")
            if docs.get("sel") or docs.get("path"):
                _check_selector(docs.get("path") if dk == "json" else docs.get("sel"), "documents", dk, problems)
    return problems


# ------------------------------------------------------------------ helpers
_API_RE = re.compile(r"""["'`]((?:https?://[^"'`\s]+)|(?:/[^"'`\s]*?(?:api|json|rest|search|tender|notice|"""
                     r"""ausschreib|avis|licit)[^"'`\s]*))["'`]""", re.I)


def _api_candidates(html: str, base: str) -> list[str]:
    tree = HTMLParser(html)
    blobs = [s.text() for s in tree.css("script") if s.text()]
    found = []
    for b in blobs:
        for m in _API_RE.finditer(b):
            u = m.group(1).split("${")[0]
            u = urljoin(base, u)
            if urlsplit(u).hostname == urlsplit(base).hostname and u not in found:
                found.append(u)
    return found[:5]


# Opendatasoft-style open-data platforms (many public bodies publish their notices there): an app that asked for a
# dataset's metadata (/api/datasets/1.0/<id>/) or its records (/api/records/1.0/search/?dataset=<id>,
# /api/explore/v2.x/catalog/datasets/<id>/...) can be read through the platform's standard records endpoint.
_ODS_RE = re.compile(r"^(https?://[^/?#]+)(?:/[^?#]*)?/api/(?:datasets/1\.0/([A-Za-z0-9_.-]+)|"
                     r"explore/v2\.[01]/catalog/datasets/([A-Za-z0-9_.-]+)|records/1\.0/search)\b")


def open_data_records_candidates(urls: list[str]) -> list[str]:
    """Records addresses of the datasets the app touched, by evidence (the URLs it requested), never by host name."""
    out: list[str] = []
    for u in urls or []:
        m = _ODS_RE.match(str(u))
        if not m:
            continue
        ds = m.group(2) or m.group(3) or dict(parse_qsl(urlsplit(u).query)).get("dataset")
        if not ds or ds in ("search",):
            continue
        cand = f"{m.group(1)}/api/explore/v2.1/catalog/datasets/{ds}/records?limit=20"
        if cand not in out:
            out.append(cand)
    return out[:6]


def _trim_long_strings(obj, n: int = 400, depth: int = 0):
    """A JSON sample for a prompt: very long strings (embedded XML/JSON documents, HTML descriptions) are cut, so
    the sample shows several whole rows with all their field names instead of half of the first one."""
    if depth > 8:
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= n else obj[:n] + "...(cut)"
    if isinstance(obj, list):
        return [_trim_long_strings(x, n, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {k: _trim_long_strings(v, n, depth + 1) for k, v in obj.items()}
    return obj


def _first_list_of_dicts(obj, depth=0):
    if depth > 4:
        return None
    if isinstance(obj, list) and len(obj) >= 2 and all(isinstance(x, dict) for x in obj[:5]):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = _first_list_of_dicts(v, depth + 1)
            if r is not None:
                return r
    return None


_TENDERISH = ("title", "titel", "titre", "titulo", "naam", "name", "deadline", "sluiting", "frist", "date",
              "datum", "buyer", "authority", "aanbesteder", "dienst", "publicat", "notice", "tender", "id", "cpv")


def _score_json(data) -> int:
    rows = _first_list_of_dicts(data)
    if not rows:
        return 0
    keys = " ".join(str(k).lower() for k in rows[0].keys())
    hits = sum(1 for w in _TENDERISH if w in keys)
    if hits < 2:  # a menu or a list of options is not a list of tenders: need at least two tender-like keys
        return 0
    return len(rows) * (1 + hits)


def _js_shell(html: str) -> bool:
    """A page whose visible content is (almost) empty: rendered by JavaScript."""
    text = page_text(html, 4000)
    links = len(HTMLParser(html or "").css("a[href]"))
    return (links < 3 and len(text) < 600) or len(text) < 80


def _evidence_of_block(html: str, status: int) -> str | None:
    if status in (401, 403):
        return f"access denied (HTTP {status}); a login is required"
    tree = HTMLParser(html or "")
    if tree.css_first("input[type=password]") and len(page_text(html, 4000)) < 1500:
        return "a login is required to see the tenders"
    if runner.has_captcha(html) and len(page_text(html, 4000)) < 800:
        return "a captcha blocks the tender list"
    return None


async def _try_json(ctx, url: str):
    try:
        r = await ctx.client.get(url, headers={"Accept": "application/json"})
        if r.status_code >= 400:
            return None
        data = r.json()
        if _first_list_of_dicts(data) is not None:
            return data
    except Exception:
        return None
    return None


def _norm(s) -> str:
    return re.sub(r"\W+", "", str(s or "").casefold())


# ------------------------------------------------------------------ replaying what the browser saw (GET or POST)
def _json_request_body(meta: dict | None):
    """The app's request body as JSON, or None (no body, or not JSON - form posts are another mechanism)."""
    body = (meta or {}).get("body")
    if not isinstance(body, str) or not body.strip():
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    return data if isinstance(data, (dict, list)) else None


async def replay_request(ctx, url: str, meta: dict | None = None):
    """Ask for a captured data address again WITHOUT the browser, the way the app asked: GET, or POST with the same
    JSON body (search APIs are often POST-only; a GET replay loses them). Returns (data, request) where request is
    {"method": "POST", "body": ...} for a POST and None for a GET; data is None when the replay gives no list."""
    if (meta or {}).get("method", "GET") != "POST":
        return await _try_json(ctx, url), None
    body = _json_request_body(meta)
    if body is None:
        return None, None
    try:
        r = await ctx.client.request("POST", url, json=body, headers={"Accept": "application/json"})
        if r.status_code >= 400:
            return None, None
        data = r.json()
    except BudgetExceeded:
        raise
    except Exception:
        return None, None
    if _first_list_of_dicts(data) is None:
        return None, None
    return data, {"method": "POST", "body": body}


def captured_endpoints(cap: dict) -> list[dict]:
    """Tender-like JSON responses the browser saw, best first, one per (address, method, body):
    [{"url", "data", "meta"}]. A POST search and a GET lookup on the same address stay apart."""
    metas = cap.get("json_meta") or []
    rows = [{"url": u, "data": d, "meta": metas[i] if i < len(metas) else None}
            for i, (u, d) in enumerate(cap.get("json") or [])]
    out, seen = [], set()
    for r in sorted(rows, key=lambda r: _score_json(r["data"]), reverse=True):
        m = r["meta"] or {}
        key = (r["url"], m.get("method", "GET"), m.get("body") if m.get("method") == "POST" else None)
        if _score_json(r["data"]) > 0 and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _untried_entry(ep: dict):
    """An endpoint left for later: a bare URL for GET (as before), {"url", "meta"} for a POST."""
    m = ep.get("meta") or {}
    return {"url": ep["url"], "meta": m} if m.get("method") == "POST" else ep["url"]


def _same_endpoint(a: str | None, b: str | None) -> bool:
    pa, pb = urlsplit(a or ""), urlsplit(b or "")
    return (pa.scheme, pa.netloc, pa.path.rstrip("/")) == (pb.scheme, pb.netloc, pb.path.rstrip("/"))


def _apply_captured_request(spec: dict, found: dict) -> None:
    """The list API answered a POST: the spec must ask the same way (the model may forget method or body)."""
    req = found.get("apiRequest")
    if not req or req.get("method") != "POST" or not found.get("api"):
        return
    lst = spec.get("list") or {}
    if _same_endpoint(lst.get("url"), found["api"][0]):
        lst["method"] = "POST"
        if not isinstance(lst.get("body"), (dict, list)):
            lst["body"] = json.loads(json.dumps(req["body"]))


# ------------------------------------------------------------------ main flows
async def locate_list(ctx, start_url: str) -> dict:
    """Walk from the given address to the tender list: best-first search with backtracking, see locate.py.
    Returns {"url", "html", "api": (url, json)|None, "captcha", ...} plus "candidates" (ranked) and "walk" (asks for
    the next-ranked candidate on demand, see locate.next_candidate)."""
    from . import locate
    return await locate.locate_list(ctx, start_url)


async def _why_no_tenders(ctx, spec: dict) -> str:
    """Concrete, actionable feedback when a list spec yields nothing: what did the runner actually get?"""
    lst = spec.get("list") or {}
    try:
        url, resp = await runner.fetch_list_page(ctx, spec, 0)
    except Exception as e:
        return f"the list spec found no tenders: fetching list.url failed ({_short(e)})"
    kind = lst.get("kind", "html")
    items = lst.get("items", "")
    if kind == "json":
        try:
            body = resp.json()
            keys = list(body.keys())[:12] if isinstance(body, dict) else f"array of {len(body)}"
        except ValueError:
            return f"list.url {url} returned HTTP {resp.status_code} but not JSON ({len(resp.text)} chars)"
        return (f"the list spec found no tenders: {url} returned HTTP {resp.status_code}; top-level keys {keys}; "
                f"items path '{items}' matched nothing")
    tree = HTMLParser(resp.text or "")
    title = clean_text(tree.css_first("title").text()) if tree.css_first("title") else ""
    try:
        n = len(tree.css(items)) if items else 0
    except Exception:
        n = 0
    hint = ""
    if _js_shell(resp.text or ""):
        hint = " The page fetched without a browser is (almost) empty: the rows are built by JavaScript."
    elif n == 0:
        hint = (" The selector matches nothing in the HTML we fetched; the rows may come from a search form, "
                "a session, or an XHR request.")
    return (f"the list spec found no tenders: {url} returned HTTP {resp.status_code}, {len(resp.text or '')} chars, "
            f"title '{title[:80]}'; items selector '{items}' matched {n} elements.{hint}")


def _api_sample(data, limit: int) -> str:
    """The API answer for the model, cut to `limit` characters WITHOUT losing its shape: when the rows fill the
    budget, the longest top-level array is shortened to its first rows, so the keys after it (a pagination block
    with the next cursor, a total) stay visible."""
    full = json.dumps(data, ensure_ascii=False)
    if len(full) <= limit or not isinstance(data, dict):
        return full[:limit]
    arrays = [k for k, v in data.items() if isinstance(v, list) and len(v) > 2]
    if not arrays:
        return full[:limit]
    k = max(arrays, key=lambda a: len(json.dumps(data[a], ensure_ascii=False)))
    for n in (5, 3, 2, 1):
        short = {**data, k: data[k][:n]}
        out = json.dumps(short, ensure_ascii=False)
        if len(out) <= limit:
            return out + f"  (the array '{k}' has {len(data[k])} rows; {n} shown)"
    return full[:limit]


async def _probe_read(ctx, spec: dict):
    """One list page with the partial spec, no details. A selector the parser rejects becomes a named SpecError."""
    try:
        return await runner.read(ctx, spec, {}, runner.Limits(max_pages=1, max_details=0, max_docs=0,
                                                              fetch_docs=False))
    except Exception as e:
        if _is_css_parse_error(e):
            raise SpecError(_run_error_problem(e, spec).removeprefix("spec problem: "), spec) from e
        raise


async def _see_blind_json_list(ctx, found: dict, spec: dict) -> bool:
    """The spec reads a JSON list whose rows synthesize never saw (no sample, or a sample of another address).
    Fetch its first page once; if it holds rows, it becomes found["api"] (the sample every later attempt sees).
    True = a sample was added now."""
    lst = spec.get("list") or {}
    if lst.get("kind") != "json" or (found.get("api") and _same_endpoint(found["api"][0], lst.get("url"))):
        return False
    if found.get("blindJsonTried") == lst.get("url"):
        return False
    found["blindJsonTried"] = lst.get("url")
    try:
        url, resp = await runner.fetch_list_page(ctx, spec, 0)
        if resp.status_code >= 400:
            return False
        data = resp.json()
    except (BudgetExceeded, PortalBlocked):
        raise
    except Exception:
        return False
    items, ok = runner._checked_items("json", data, lst.get("items", ""))
    if not (ok and items and isinstance(items[0], dict)) and _first_list_of_dicts(data) is None:
        return False
    found["api"] = (url, data)
    found["apiRequest"] = None
    return True


def _per_visit_tokens(spec: dict, html: str) -> None:
    """A load_more request copied from the page with a per-visit value written out (the CSRF token the page held
    when the model saw it): the next visit has another one. A literal value that is exactly what the page carries
    under that field's name becomes {token:NAME}, read freshly on every read."""
    pag = (spec.get("list") or {}).get("pagination") or {}
    fields = pag.get("fields")
    if pag.get("type") != "load_more" or not isinstance(fields, dict) or not html:
        return
    tree = HTMLParser(html)
    for name, val in list(fields.items()):
        if isinstance(val, str) and len(val) >= 12 and not re.search(r"\s|\{", val) \
                and jspager.page_value(tree, html, name) == val:
            fields[name] = "{token:" + name + "}"


def pager_notes(html: str) -> str | None:
    """What the list page's pager really does, for the model: postback state (postback.pager_hint) and the scripts
    behind its next control or its 'load more' request (jspager.pager_scripts_hint)."""
    notes = [h for h in (postback.pager_hint(html), jspager.pager_scripts_hint(html)) if h]
    return " ".join(notes) if notes else None


# ------------------------------------------------------------------ deterministic evidence over model choice
_NEXT_REL = re.compile(r"\bnext\b", re.I)


def _unique_selector(tree, el) -> str | None:
    """A CSS selector whose first match on this page is `el` (id, else tag + classes, else an attribute)."""
    a = el.attributes or {}
    cands = []
    if re.fullmatch(r"[A-Za-z][\w-]*", a.get("id") or ""):
        cands.append(f"{el.tag}#{a['id']}")
    cls = [c for c in (a.get("class") or "").split() if re.fullmatch(r"[A-Za-z_][\w-]*", c)]
    if cls:
        cands.append(el.tag + "".join("." + c for c in cls[:3]))
    for k in ("rel", "aria-label", "title", "name"):
        if a.get(k) and '"' not in a[k]:
            cands.append(f'{el.tag}[{k}="{a[k]}"]')
    par = el.parent
    pa = (par.attributes or {}) if par is not None else {}
    pcls = [c for c in (pa.get("class") or "").split() if re.fullmatch(r"[A-Za-z_][\w-]*", c)]
    if par is not None and pcls:
        cands.append(f"{par.tag}.{pcls[0]} > {el.tag}")
    for sel in cands:
        try:
            first = tree.css_first(sel)
        except Exception:
            continue
        if first is not None and first.html == el.html:
            return sel
    return None


def _next_controls(tree) -> list:
    """Enabled controls that say 'next page' (text, rel, aria-label, title or class), in page order."""
    out = []
    for el in tree.css("a, button, input[type=submit], input[type=button], input[type=image]"):
        a = el.attributes or {}
        label = " ".join(filter(None, [a.get("aria-label"), a.get("title"), a.get("class"), a.get("rel"),
                                       a.get("value"), a.get("id"), el.text(strip=True)[:40]]))
        if (postback._NEXT_WORDS.search(label) or _NEXT_REL.search(str(a.get("rel") or ""))) \
                and not postback._disabled(el):
            out.append(el)
    return out


_PAGER_BOX = re.compile(r"pag|browse|pager|seiten|blätter", re.I)


def _in_pager(el) -> bool:
    """The control has rel=next or sits in a pager (an ancestor whose class/id/role/label names paging)."""
    if _NEXT_REL.search(str((el.attributes or {}).get("rel") or "")):
        return True
    node, depth = el, 0
    while node is not None and depth < 5:
        a = node.attributes or {}
        if _PAGER_BOX.search(" ".join(str(a.get(k) or "") for k in ("class", "id", "aria-label", "role"))):
            return True
        node, depth = node.parent, depth + 1
    return False


def static_pager(spec: dict, html: str, page_url: str = "") -> str | None:
    """Deterministic evidence over model choice: when the list page's pager is recognisably a postback (a next
    control that posts the page's form: __doPostBack, PRADO, a bound script, an address in its script) or a plain
    next link, and the spec's pagination cannot follow it (none; a next_link whose control is a javascript: action;
    a postback/next_link selector that matches nothing), set it from the page before the dry run. Parameter pagers
    (page, offset, cursor, form, load_more) are left to the dry run. Returns a note when the spec was changed."""
    lst = spec.get("list") or {}
    if lst.get("kind", "html") != "html" or not html:
        return None
    pag = lst.get("pagination") or {}
    ptype = pag.get("type") or "none"
    if ptype not in ("none", "next_link", "postback"):
        return None
    tree = HTMLParser(html)
    sel_now = pag.get("next_sel") or ""
    if ptype == "postback" and sel_now and postback.next_control(tree, sel_now) is not None:
        return None
    if ptype == "next_link" and sel_now:
        try:
            el = tree.css_first(sel_now)
        except Exception:
            el = None
        if el is not None:
            href = (el.attributes.get("href") or "").strip()
            if (href and not href.lower().startswith("javascript:") and href != "#") or jspager.js_url(el):
                return None                 # a real address (or one in its script): the runner follows it
    for el in _next_controls(tree):
        if not _in_pager(el):
            continue                        # a 'next' outside a pager (a carousel, a wizard step) is no evidence
        a = el.attributes or {}
        href = (a.get("href") or "").strip()
        real = bool(href) and not href.lower().startswith("javascript:") and not href.startswith("#")
        sel = _unique_selector(tree, el)
        if not sel:
            continue
        if real:
            new = {"type": "next_link", "next_sel": sel}
        else:
            req = postback.build(tree, html, page_url or lst.get("url") or "", None, el, {})
            if isinstance(req, str):
                continue                    # cannot be clicked without a browser: leave it to the model
            new = {"type": "postback", "next_sel": sel}
        for k in ("size_sel", "form_sel", "target", "argument", "target_field", "max_pages"):
            if pag.get(k) is not None:
                new[k] = pag[k]
        if new["type"] == ptype and new["next_sel"] == sel_now:
            return None
        lst["pagination"] = new
        return (f"pager set from the list page: {new['type']} via {sel} (the spec had {ptype}"
                + (f" '{sel_now}'" if sel_now else "") + ")")
    return None


_DL_LABEL_RE = re.compile(
    r"(deadline|closing|time.?limit|\bdue\b|frist|einreichung|abgabe|schlusstermin|date limite|remise des|"
    r"sluiting|uiterste|plazo|presentaci[oó]n de ofertas|scadenza|termin sk[lł]ad|term[ií]n|hat[aá]rid[oő]|"
    r"t[aä]htaeg|m[aä][aä]r[aä]aika|rok za)", re.I)
_DL_PREFER = re.compile(r"(submission|tender|offer|bid|angebot|abgabe|remise|offres|inschrijving|ofertas|offert|"
                        r"receipt)", re.I)
_DL_AVOID = re.compile(r"(question|frage|clarification|teilnahme|participation|request to participate|"
                       r"demande de participation|award|zuschlag|bindefrist|binding|ausf[uü]hrung|execution)", re.I)
_DL_EXTRA_FORMATS = [
    "%d %B %Y, %I:%M%p", "%d %B %Y %I:%M%p", "%d %B %Y, %I:%M %p", "%d %B %Y %I:%M %p", "%d %B %Y, %I%p",
    "%d %b %Y, %I:%M%p", "%d %b %Y %I:%M%p", "%A %d %B %Y, %I:%M%p", "%A %d %B %Y %I:%M%p", "%d/%m/%Y %I:%M %p",
    "%d/%m/%Y %I:%M%p", "%Y-%m-%d %H:%M", "%d.%m.%Y um %H:%M", "%d.%m.%Y, %H:%M"]
_AMPM_RE = re.compile(r"\d\s?[ap]\.?m\b", re.I)


def _deadline_formats(value: str, tz: str) -> list[str] | None:
    """The date formats that read `value` as a date WITH a time ([] when the built-in fallbacks already do)."""
    from .extract import convert
    if not value or not (_TIME_RE.search(value) or _AMPM_RE.search(value)):
        return None
    if convert("submissionDeadline", value, {"formats": []}, tz, ""):
        return []
    for fmt in _DAY_FIRST[:18] + _MONTH_FIRST[:8] + _DL_EXTRA_FORMATS:
        if convert("submissionDeadline", value, {"formats": [fmt]}, tz, ""):
            return [fmt]
    return None


def deadline_from_page(html: str, tz: str) -> dict | None:
    """Find the submission deadline on a tender page by evidence: a label in the deadline vocabulary whose value
    (read exactly as the runner reads labels) parses as a date WITH a time. Submission/offer labels win over
    question/participation ones. Returns the field spec ({"from": "detail", "label", "formats"}) plus "_shown"
    (the value as printed), or None."""
    from .extract import label_value
    if not html:
        return None
    tree = HTMLParser(html)
    root = tree.body or tree.root
    if root is None:
        return None
    seen: set[str] = set()
    best = None
    for i, node in enumerate(root.traverse(include_text=False)):
        if node.tag in ("script", "style", "option", "select", "title", "head", "noscript") or node.tag.startswith("-"):
            continue
        own = clean_text(node.text(deep=False)) or ""
        if not own or len(own) > 90 or not _DL_LABEL_RE.search(own):
            continue
        label = (re.split(r"[:：]", own, maxsplit=1)[0] if re.search(r"[:：]", own) else own).strip(" :：")
        if not (3 <= len(label) <= 70) or not _DL_LABEL_RE.search(label) or label in seen:
            continue
        seen.add(label)
        found = label_value(tree, label)
        if found is None:
            continue
        val = clean_text(found if isinstance(found, str) else found.text(separator=" ")) or ""
        fmts = _deadline_formats(val[:80], tz)
        if fmts is None:
            continue
        avoid = bool(_DL_AVOID.search(label))
        rank = (0 if _DL_PREFER.search(label) and not avoid else 1 if not avoid else 2, i)
        if best is None or rank < best[0]:
            best = (rank, {"from": "detail", "label": label, **({"formats": fmts} if fmts else {})}, val)
    if best is None:
        return None
    return {**best[1], "_shown": best[2]}


def _apply_deadline_evidence(ctx, spec: dict, detail_html: str, probe_tenders: list[dict]) -> None:
    """The deadline mapped from evidence on the opened tender page when the model left it unmapped, or mapped it
    where it reads nothing: from the list with no deadline on any probed row (e.g. a date without a time), or from
    the tender page under a label that gives no date+time there. The model's mapping stays when it reads a value."""
    from .extract import convert, field_from
    if (spec.get("detail") or {}).get("kind", "html") != "html" or not detail_html:
        return
    tz = spec.get("tz") or "UTC"
    fs = (spec.get("fields") or {}).get("submissionDeadline")
    if isinstance(fs, dict) and fs.get("from", "list") == "list":
        if any(t.get("submissionDeadline") for t in probe_tenders) or not probe_tenders:
            return
    elif isinstance(fs, dict) and fs.get("from") == "detail" and not fs.get("template"):
        try:
            raw = field_from("html", HTMLParser(detail_html), "submissionDeadline", fs)
        except Exception:
            raw = None
        if convert("submissionDeadline", raw, fs, tz, ""):
            return
    elif fs is not None:
        return
    got = deadline_from_page(detail_html, tz)
    if not got:
        return
    shown = got.pop("_shown", "")
    spec["fields"]["submissionDeadline"] = got
    ctx.note(f"submissionDeadline mapped from the tender page by evidence: label '{got['label']}' shows "
             f"'{shown}'" + (f" (the spec had {json.dumps(fs, ensure_ascii=False)[:160]})" if fs else ""))


# ---- second hop: the deadline is on the page the tender page links to (list row -> tender page -> notice page)
_NOTICE_WORDS = re.compile(r"\b(notices?|bekanntmachung(?:en)?|avis|annonces?|aankondiging(?:en)?|anuncios?|"
                           r"og[lł]oszeni[ea]|kung[öo]relse|bando|avviso|ozn[áa]men[íi]|hirdetm[ée]ny|"
                           r"auftragsbekanntmachung|contract notice|tender notice)\b", re.I)
_NOT_NOTICE = re.compile(r"privacy|cookie|datenschutz|legal|impressum|accessibility|terms|confidentialit", re.I)
_OPEN_DATA_WORDS = re.compile(r"\bapi\b|open[ -]?data|developer|donn[ée]es ouvertes|datos abiertos|offene daten|"
                              r"\bocds\b|\bjson\b", re.I)


def _link_selector(tree, href: str) -> str | None:
    """A CSS selector for links like this one on every tender page: the address's leading path segments without
    digits (e.g. '/Notice/110370-2026' -> a[href*='/Notice/']), checked to pick this link first on this page."""
    path = urlsplit(href).path or href
    segs = [s for s in path.split("/") if s]
    lead = []
    for s in segs:
        if re.search(r"\d", s):
            break
        lead.append(s)
    if not lead or "'" in "".join(lead):
        return None
    sel = "a[href*='/" + "/".join(lead) + "/']"
    first = tree.css_first(sel)
    return sel if first is not None and (first.attributes.get("href") or "") == href else None


def notice_links(html: str, page_url: str) -> list[tuple[str, str, str]]:
    """Links on a tender page to a notice of the same site (link text or address names a notice; not the page itself,
    not privacy/cookie notices), in page order: [(absolute url, text, selector)]. Only links a selector can find again
    on other tender pages are kept."""
    tree = HTMLParser(html or "")
    host = urlsplit(page_url).netloc
    out: list[tuple[str, str, str]] = []
    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "#", "tel:")):
            continue
        u = urljoin(page_url, href)
        if urlsplit(u).netloc != host or u.split("#")[0] == page_url.split("#")[0]:
            continue
        txt = clean_text(a.text()) or ""
        words = re.sub(r"[/_\-.]+", " ", urlsplit(u).path)
        if _NOT_NOTICE.search(txt + " " + words) or not (_NOTICE_WORDS.search(txt) or _NOTICE_WORDS.search(words)):
            continue
        sel = _link_selector(tree, href)
        if sel and all(sel != o[2] for o in out):
            out.append((u, txt[:80], sel))
    return out


def _json_date_leaves(obj, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict:
    """JMESPath -> value of every date-time string in a JSON answer, lists included (their first 3 items)."""
    out = {} if out is None else out
    if depth > 7 or len(out) > 60:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            key = k if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) else json.dumps(k)
            p = f"{prefix}.{key}" if prefix else key
            if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", v.strip()):
                out[p] = v
            elif isinstance(v, (dict, list)):
                _json_date_leaves(v, p, depth + 1, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            _json_date_leaves(v, f"{prefix}[{i}]", depth + 1, out)
    return out


async def _open_data_links(ctx, html: str, page_url: str) -> list[str]:
    """The portal's own open-data/API pointers: the tender page's link to them, opened once (1 request), and the API
    links that page names. What the model needs to address a notice's open-data record when its page is refused."""
    tree = HTMLParser(html or "")
    host = urlsplit(page_url).netloc
    doc = None
    for a in tree.css("a[href]"):
        txt = clean_text(a.text()) or ""
        u = urljoin(page_url, a.attributes.get("href") or "")
        if _OPEN_DATA_WORDS.search(txt) and urlsplit(u).netloc == host:
            doc = (txt, u)
            break
    if doc is None:
        return []
    out = [f"{doc[0]} ({doc[1]})"]
    try:
        resp = await ctx.client.get(doc[1])
    except Exception:
        return out
    if resp.status_code >= 400:
        return out
    t2 = HTMLParser(resp.text or "")
    for a in t2.css("a[href]"):
        txt = clean_text(a.text()) or ""
        u = urljoin(str(resp.url), a.attributes.get("href") or "").split("#")[0]
        if urlsplit(u).netloc == host and (_OPEN_DATA_WORDS.search(txt) or _OPEN_DATA_WORDS.search(u)) \
                and u != doc[1].split("#")[0] and f"{txt} ({u})" not in out:
            out.append(f"{txt} ({u})")
        if len(out) >= 8:
            break
    return out


def _reads_deadline(kind: str, body, fs: dict, tz: str) -> str | None:
    from .extract import convert, field_from
    try:
        return convert("submissionDeadline", field_from(kind, body, "submissionDeadline", fs), fs, tz, "")
    except Exception:
        return None


async def _second_hop(ctx, spec: dict, detail_url: str, detail_html: str, probe_tenders: list[dict],
                      found: dict) -> None:
    """The deadline is neither in the list (with a time) nor on the tender page: open the page the tender page links
    to (the full notice; the model's detail.follow, else the first notice link) and find it there by evidence: a
    deadline label with a date+time (html), or the date-time the cheap model labels as the offer deadline (json).
    Records detail.follow + fields.submissionDeadline {"from": "detail2"}. When the linked page is refused, the
    reason (and the portal's open-data pointers) is kept in found["hopNote"] for the next attempt's feedback."""
    fields = spec.get("fields") or {}
    tz = spec.get("tz") or "UTC"
    fs = fields.get("submissionDeadline")
    det = spec.get("detail") or {}
    tree = HTMLParser(detail_html or "")
    if isinstance(fs, dict) and fs.get("from", "list") == "list" and not fs.get("template"):
        if any(t.get("submissionDeadline") for t in probe_tenders):
            return                              # the list gives it
    elif isinstance(fs, dict) and fs.get("from") == "detail":
        if _reads_deadline("html", tree, fs, tz):
            return                              # the tender page gives it
    elif isinstance(fs, dict) and fs.get("from") != "detail2":
        return
    fol = det.get("follow") if isinstance(det.get("follow"), dict) else None
    links = notice_links(detail_html, detail_url)
    if fol is None:
        if not links:
            return
        fol = {"sel": links[0][2], "attr": "href"}
    found.pop("hopNote", None)
    found["noticeLink"] = links[0][0] if links else None
    k2, b2, u2, why = await runner.fetch_follow(ctx, fol, "html", tree, detail_url)
    if why:
        if why != "no link":
            data = await _open_data_links(ctx, detail_html, detail_url)
            found["hopNote"] = {"url": u2 or "", "why": why, "data": data}
            ctx.note(f"second hop: the tender page links to {u2}, which answered {why}"
                     + (f"; the portal's open data: {data[:4]}" if data else ""))
        return
    if isinstance(fs, dict) and fs.get("from") == "detail2" and _reads_deadline(k2, b2, fs, tz):
        spec["detail"]["follow"] = {**fol, "kind": k2}
        ctx.note(f"second hop: submissionDeadline read from the linked page {u2}")
        return
    got = None
    if k2 == "html":
        ev = deadline_from_page(b2.html or "", tz)
        if ev:
            shown = ev.pop("_shown", "")
            got = {**ev, "from": "detail2"}
            why_txt = f"label '{ev['label']}' shows '{shown}'"
    else:
        dates = _json_date_leaves(b2)
        if dates:
            row = {k: v for k, v in (probe_tenders[0] if probe_tenders else {}).items()
                   if k in ("id", "title") and v}
            p = await _deadline_path(ctx, row, dates)
            if p:
                got = {"from": "detail2", "path": p}
                why_txt = f"'{p}' = '{dates[p]}'"
    if got is None:
        found["hopNote"] = {"url": u2, "why": "shows no deadline with a time", "data": []}
        ctx.note(f"second hop: the linked page {u2} shows no deadline with a time")
        return
    spec["detail"]["follow"] = {**fol, "kind": k2}
    fields["submissionDeadline"] = got
    ctx.note(f"submissionDeadline mapped from the linked page by evidence ({u2}): {why_txt}; follow "
             f"{json.dumps(spec['detail']['follow'], ensure_ascii=False)}")


async def _second_hop_safely(ctx, spec: dict, detail_url: str, detail_html: str, probe_tenders: list[dict],
                             found: dict) -> None:
    try:
        await _second_hop(ctx, spec, detail_url, detail_html, probe_tenders, found)
    except (PortalBlocked, ModelBudgetExhausted, LLMUnavailable, BudgetExceeded):
        raise
    except Exception as e:   # best effort: the dry run and the model judge the spec anyway
        ctx.note(f"second hop skipped: {_short(e, 160)}")


def deadline_from_list(spec: dict, html: str) -> dict | None:
    """The deadline column of an html list table, by evidence: a header in the deadline vocabulary whose cells
    (their text, or the title of an element in them, e.g. <abbr title='27.11.2026 um 12:00 Uhr'>27.11.2026</abbr>)
    parse as a date WITH a time for most rows that show a value. Returns the list field spec or None."""
    from .extract import _header_cells, convert, field_from, table_cell
    lst = spec.get("list") or {}
    if lst.get("kind", "html") != "html" or not html or not lst.get("items"):
        return None
    tree = HTMLParser(html)
    try:
        items = list_items("html", tree, lst["items"])
    except Exception:
        return None
    if not items:
        return None
    node = items[0]
    while node is not None and node.tag != "table":
        node = node.parent
    if node is None:
        return None
    tz = spec.get("tz") or "UTC"
    for _col, th in _header_cells(node):
        head = clean_text(th.text(separator=" ")) or ""
        if not head or len(head) > 70 or not _DL_LABEL_RE.search(head):
            continue
        cands = [{"from": "list", "column": head}]
        for it in items[:20]:
            cell = table_cell(it, head)
            for el in (cell.css("[title]") if cell is not None else []):
                fs = {"from": "list", "column": head, "sel": el.tag, "attr": "title"}
                if fs not in cands:
                    cands.append(fs)
        for fs in cands:
            vals = [field_from("html", it, "submissionDeadline", fs) for it in items]
            # values that show a date at all (a title such as 'no deadline for this procedure' does not count)
            shown = [str(v) for v in vals if v and not is_no_value(v) and re.search(r"\d", str(v))]
            fmts = next((f for f in (_deadline_formats(v[:80], tz) for v in shown[:10]) if f is not None), None)
            if not shown or fmts is None:
                continue
            got = {**fs, **({"formats": fmts} if fmts else {})}
            ok = sum(1 for v in shown if convert("submissionDeadline", v, got, tz, ""))
            if ok * 2 >= len(shown):
                return got
    return None


def _apply_list_deadline_evidence(ctx, spec: dict, html: str) -> bool:
    """The list's deadline column mapped by evidence when the model's list mapping reads no deadline WITH a time
    from any row (unmapped, a date-only cell, or the wrong column). A detail-page mapping is left alone."""
    from .extract import convert, field_from
    fs = (spec.get("fields") or {}).get("submissionDeadline")
    if isinstance(fs, dict) and fs.get("from", "list") != "list":
        return False
    lst = spec.get("list") or {}
    if lst.get("kind", "html") != "html" or not html:
        return False
    tz = spec.get("tz") or "UTC"
    if isinstance(fs, dict):
        try:
            items = list_items("html", HTMLParser(html), lst.get("items", ""))
            if any(convert("submissionDeadline", field_from("html", it, "submissionDeadline", fs), fs, tz, "")
                   for it in items):
                return False
        except Exception:
            pass
    got = deadline_from_list(spec, html)
    if not got:
        return False
    spec.setdefault("fields", {})["submissionDeadline"] = got
    ctx.note(f"submissionDeadline mapped from the list by evidence: {json.dumps(got, ensure_ascii=False)}"
             + (f" (the spec had {json.dumps(fs, ensure_ascii=False)[:160]})" if fs else ""))
    return True


def _best_payload(best: dict | None) -> dict | None:
    """The attempt that read the most so far, for the model to refine instead of starting over."""
    if not best or not best.get("spec"):
        return None
    sp = best["spec"]
    return {"spec": {k: sp.get(k) for k in ("tz", "list", "fields", "detail") if sp.get(k) is not None},
            "rowsReadInDryRun": best.get("rows"), "itsProblems": best.get("problems") or [],
            "instruction": "this earlier attempt read the most rows: keep every part of it that works and change "
                           "only what its problems (and problemsToFix) name"}


async def synthesize(ctx, found: dict, feedback: list[str] | None = None, old_spec: dict | None = None,
                     hints: list[dict] | None = None, _blind_retry: bool = False, best: dict | None = None) -> dict:
    """Two strong calls: (1) list + list fields, (2) detail fields + documents from one detail page.
    `best`: the attempt of this learn that read the most so far (refined, not started over)."""
    base = {"portalUrl": ctx.url, "listPageUrl": found["url"]}
    best_part = _best_payload(best)
    if best_part:
        base["bestAttemptSoFar"] = best_part
    if found.get("api"):
        api_url, data = found["api"]
        base["jsonApi"] = {"url": api_url, "sample": _api_sample(_trim_long_strings(data), 12000)}
        if (found.get("apiRequest") or {}).get("method") == "POST":
            # the portal's own search request: the model copies it into list.method/list.body and says where the
            # page number or offset sits in the body (pagination.body_path)
            base["jsonApi"]["method"] = "POST"
            base["jsonApi"]["requestBody"] = found["apiRequest"]["body"]
    else:
        base["listPage"] = reduce_html(found["html"], 30000)
        hint = pager_notes(found["html"])   # the reduced page drops the scripts that bind or drive the pager
        if hint:
            base["pagerNote"] = hint
    if found.get("links"):
        base["linksOnRenderedPage"] = found["links"]
    if old_spec:
        base["previousSpecThatBroke"] = old_spec
    if feedback:
        base["problemsToFix"] = feedback
    if hints:
        base["knownCorrectTenders"] = hints[:5]
    sys1 = ("You write the LIST part of a spec for a tender portal. " + SPEC_DOC +
            "\nNow return JSON with keys tz, list, fields (only fields readable from the list; include id and url), "
            "detail (how to open one tender). Do not include documents yet.")
    part1 = await ctx.llm.json_call("synthesize", sys1, json.dumps(base, ensure_ascii=False),
                                    portal_id=ctx.portal_id, run_id=ctx.run_id)
    spec = {"version": 1, "tz": part1.get("tz") or "UTC", "list": part1.get("list") or {},
            "fields": part1.get("fields") or {}, "detail": part1.get("detail") or {"kind": "html", "url_from": "url"}}
    if not spec["list"].get("url"):
        spec["list"]["url"] = found["api"][0] if found.get("api") else found["url"]
    _apply_captured_request(spec, found)
    coerce_spec(spec)
    list_dl = None          # a list deadline column found by evidence: the detail part does not replace it
    if not found.get("api"):
        _per_visit_tokens(spec, found.get("html") or "")
        if _apply_list_deadline_evidence(ctx, spec, found.get("html") or ""):
            list_dl = spec["fields"]["submissionDeadline"]
    bad = validate_spec(spec)
    if bad:
        raise SpecError("; ".join(bad[:4]), spec)
    if await _see_blind_json_list(ctx, found, spec) and not _blind_retry:
        # The model wrote a JSON list it had never seen (e.g. the portal's open-data API it knows of): show it the
        # real rows, so fields such as the deadline are mapped from the API's own names, not guessed.
        ctx.note(f"the list part names a JSON API that was not sampled ({spec['list'].get('url')}); "
                 f"writing the list part again with its rows")
        return await synthesize(ctx, found, feedback, old_spec, hints, _blind_retry=True, best=best)

    # A JSON list without the deadline: find the per-tender JSON record by evidence (the model cannot see it).
    dl_fs = spec["fields"].get("submissionDeadline") or {}
    discovered = None
    if spec["list"].get("kind") == "json" and not (dl_fs.get("from", "list") == "list" and dl_fs.get("path")) \
            and not ((spec.get("detail") or {}).get("discovered")):
        discovered = await _discover_detail_safely(ctx, spec)
        if discovered:
            found["preferIds"] = discovered.get("prefer") or []
    elif (spec.get("detail") or {}).get("discovered") and old_spec:
        discovered = {"url": None}

    # open one tender's detail page with the partial spec
    probe = await _probe_read(ctx, spec)
    if not probe.tenders and (probe.stats or {}).get("closedDropped"):
        # Page 1 shows only closed rows: look for the portal's own open/current filter by evidence (its search form
        # for an html list, query parameters for a JSON API) before asking the model again.
        if spec["list"].get("kind", "html") == "html":
            narrowed = await _narrow_html_safely(ctx, spec, found)
        else:
            before = json.loads(json.dumps(spec))
            await _narrow_safely(ctx, spec)
            narrowed = spec != before
        if narrowed:
            probe = await _probe_read(ctx, spec)
    detail_url = (discovered or {}).get("url")
    for t in ([] if detail_url else probe.tenders):
        if spec["detail"].get("url_from") == "template" and spec["detail"].get("template") and t.get("id"):
            detail_url = t.get("_detailUrl") or spec["detail"]["template"].replace("{id}", t["id"])
        else:
            detail_url = t.get("url")
        if detail_url:
            break
    if not probe.tenders and (probe.stats or {}).get("closedDropped") and not runner.mixes_closed(spec):
        tried = (found.get("openFilterTried") or "")
        raise SpecError(f"the first list page only shows closed tenders ({probe.stats['closedDropped']} past their "
                        f"deadline): use the portal's filter or sort for open/current tenders (write {{today}} for "
                        f"today's date, {{today:%d/%m/%Y}} for another date format), or point list.url at its "
                        f"open-tenders view; check that the deadline is read from the right column" + tried, spec)
    if not probe.tenders and not (probe.stats or {}).get("heldBack"):
        raise SpecError(await _why_no_tenders(ctx, spec), spec)
    if not detail_url:
        raise SpecError("no tender has a url or detail link (map the url field, or give a detail template)", spec)
    dresp = await ctx.client.get(detail_url)
    if dresp.status_code >= 400:
        raise SpecError(f"the tender link {detail_url} returned HTTP {dresp.status_code}; fix url/detail template",
                        spec)
    dkind = spec["detail"].get("kind", "html")
    try:
        detail_payload = (json.dumps(dresp.json(), ensure_ascii=False)[:15000] if dkind == "json"
                          else reduce_html(dresp.text, 25000))
    except ValueError:
        raise SpecError(f"detail.kind is json but {detail_url} did not return JSON", spec)
    sys2 = ("You write the DETAIL part of a spec for a tender portal. " + SPEC_DOC +
            "\nYou get the list part already written and one tender's detail page. Return JSON with keys "
            "fields (ONLY fields read from the detail page, each with \"from\": \"detail\"), documents, "
            "block_markers, follow (null, or detail.follow when a value is only on a page this one links to; its "
            "fields then use \"from\": \"detail2\"). Map every field the page shows that the list part does not "
            "already cover; when the list part has no submissionDeadline, find it on this page (date AND time, "
            "under its label) - every open tender needs it.")
    best_detail = ({f: fs for f, fs in (best_part["spec"].get("fields") or {}).items()
                    if isinstance(fs, dict) and fs.get("from") == "detail"} if best_part else None)
    part2 = await ctx.llm.json_call("synthesize", sys2, json.dumps(
        {"listPart": spec, "detailUrl": detail_url, "detailPage": detail_payload,
         "problemsToFix": feedback or [], "previousSpecThatBroke": old_spec or None,
         **({"bestAttemptDetailFields": best_detail} if best_detail else {})}, ensure_ascii=False),
        portal_id=ctx.portal_id, run_id=ctx.run_id)
    kept_dl = spec["fields"].get("submissionDeadline") if discovered else None
    for f, fs in (part2.get("fields") or {}).items():
        if not isinstance(fs, dict):
            continue
        if discovered and f in spec["fields"] and spec["fields"][f].get("from", "list") == "list":
            continue                # the discovered record's address uses this list field: it stays a list field
        fs = dict(fs)
        fs["from"] = "detail2" if fs.get("from") == "detail2" else "detail"
        spec["fields"][f] = fs
    fol = part2.get("follow") if isinstance(part2.get("follow"), dict) else (spec["detail"].get("follow") or None)
    if isinstance(fol, dict) and fol.get("sel" if dkind == "html" else "path"):
        spec["detail"]["follow"] = fol
    else:
        spec["detail"].pop("follow", None)
    if kept_dl or list_dl:
        # found by evidence; the model's guess does not replace it
        spec["fields"]["submissionDeadline"] = kept_dl or list_dl
    elif dkind == "html":
        # the deadline on the opened tender page, by evidence, when the model's mapping reads nothing there
        _apply_deadline_evidence(ctx, spec, dresp.text or "", probe.tenders)
        # not on the tender page either: the page it links to (the full notice), by evidence
        await _second_hop_safely(ctx, spec, detail_url, dresp.text or "", probe.tenders, found)
    for f, fs in list(spec["fields"].items()):
        if isinstance(fs, dict) and fs.get("from") == "detail2" and not spec["detail"].get("follow"):
            spec["fields"][f] = {**fs, "from": "detail"}     # no linked page declared: read where it was seen
    spec["documents"] = part2.get("documents") or {}
    spec["block_markers"] = part2.get("block_markers") or []
    coerce_spec(spec)
    bad = validate_spec(spec, full=True)
    if bad:
        raise SpecError("; ".join(bad[:4]), spec)
    return spec


_DAY_FIRST = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y, %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
              "%d/%m/%Y, %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d %B %Y, %H:%M", "%d %B %Y %H:%M",
              "%d %B %Y, %H:%M:%S", "%d %b %Y %H:%M", "%d %b %Y, %H:%M", "%d.%m.%y %H:%M", "%d. %B %Y, %H:%M",
              "%d. %B %Y %H:%M", "%A, %d %B %Y %H:%M", "%A %d %B %Y %H:%M",
              "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%d.%m.%y", "%d. %B %Y", "%A, %d %B %Y"]
_MONTH_FIRST = ["%B %d, %Y %H:%M", "%B %d, %Y, %H:%M", "%b %d, %Y %H:%M", "%b %d, %Y, %H:%M", "%B %d, %Y %I:%M %p",
                "%b %d, %Y %I:%M %p", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M %p", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"]
_TIME_RE = re.compile(r"(?<![\d.:/-])\d{1,2}[:.]\d{2}(?::\d{2})?(?![\d/-])(?!\.\d{4})")


def _shown_interpretations(shown: str, spec_formats: list[str]) -> list[tuple[datetime, bool]]:
    """All readings of a date string copied from the page, as (datetime, has_time). An explicit offset on the page
    makes the datetime aware. Day-first is preferred; month-first only when day-first does not parse
    (a spec format is also allowed for slash dates, which are genuinely ambiguous)."""
    s = " ".join(str(shown).split())
    s = re.sub(r"\s*\b(Uhr|uur|h)\b\.?", "", s, flags=re.I)
    s = re.sub(r"\s+(?:um|at|à|a las|om|kl\.)\s+", " ", s, flags=re.I)
    # '10.12.2026 09:00 Ortszeit' = the portal's local time; MEZ/MESZ are explicit offsets
    s = _LOCAL_WORDS.sub(" ", s)
    s = re.sub(r"\(?\bMESZ\b\)?", " +02:00", re.sub(r"\(?\bMEZ\b\)?", " +01:00", s, flags=re.I), flags=re.I)
    s = " ".join(re.sub(r"\(\s*\)", " ", s).split()).strip(" ,")
    s = _normalize_months(s)
    out: list[tuple[datetime, bool]] = []
    iso = re.match(r"^\d{4}-\d{2}-\d{2}", s)
    if iso:
        try:
            v = s.replace("Z", "+00:00").replace(" ", "T", 1) if len(s) > 10 else s
            out.append((datetime.fromisoformat(v), len(s) > 10))
        except ValueError:
            pass
        return out
    rest, off = split_offset(s)
    for cand, tzi in ((s, None), (rest, off)) if off else ((s, None),):
        for fmts in (_DAY_FIRST, _MONTH_FIRST):
            for fmt in fmts:
                try:
                    dt = datetime.strptime(cand, fmt)
                except ValueError:
                    continue
                if tzi:
                    dt = dt.replace(tzinfo=tzi)
                out.append((dt, "%H" in fmt or "%I" in fmt))
            if out:
                break
        if out:
            break
    if not out or "/" in s:
        for fmt in spec_formats or []:
            for cand, tzi in ((s, None), (rest, off)) if off else ((s, None),):
                try:
                    dt = datetime.strptime(cand, fmt)
                except ValueError:
                    continue
                if tzi:
                    dt = dt.replace(tzinfo=tzi)
                out.append((dt, "%H" in fmt or "%I" in fmt))
    return out


def _date_problem(field: str, shown: Any, got: Any, tz: str, spec_formats: list[str]) -> str | None:
    """Compare a date the page shows with what the spec read. Publication dates must match as full dates,
    deadlines as instants (explicit offset on the page kept; otherwise the portal timezone). A deadline whose
    time the page does not show must be null."""
    if not shown:
        return None  # the model saw no value; nothing to compare
    interps = _shown_interpretations(str(shown), spec_formats)
    try:
        zone = ZoneInfo(tz or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    if field == "publicationDate":
        if not got:
            return f"{field} missing although the page shows '{shown}'"
        try:
            got_d = date.fromisoformat(str(got)[:10])
        except ValueError:
            return f"{field} '{got}' is not a date"
        if interps:
            if any(dt.date() == got_d for dt, _ in interps):
                return None
            return f"{field} mismatch: page shows '{shown}', spec read '{got}'"
        nums = [int(x) for x in re.findall(r"\d+", str(shown))]
        if got_d.year in nums and got_d.day in nums:  # unreadable format: at least year and day must appear
            return None
        return f"{field} mismatch: page shows '{shown}', spec read '{got}'"
    # submissionDeadline
    has_time = any(h for _, h in interps) if interps else bool(_TIME_RE.search(str(shown)))
    if not has_time:
        if got:
            return f"{field}: the page shows no time ('{shown}') but the spec read '{got}'; use null when the time is not shown"
        return None
    if not got:
        return f"{field} missing although the page shows '{shown}'"
    try:
        got_dt = datetime.fromisoformat(str(got).replace("Z", "+00:00"))
    except ValueError:
        return f"{field} '{got}' is not an ISO datetime"
    if got_dt.tzinfo is None:
        got_dt = got_dt.replace(tzinfo=zone)
    timed = [dt for dt, h in interps if h]
    if not timed:
        return None  # unreadable format with a time on it: cannot disprove
    for dt in timed:
        if (dt if dt.tzinfo else dt.replace(tzinfo=zone)) == got_dt:
            return None
    if any(dt.replace(tzinfo=None) == got_dt.replace(tzinfo=None) for dt in timed):
        return (f"{field} offset mismatch: page shows '{shown}', spec read '{got}' "
                f"(wrong timezone/offset; portal tz '{tz}')")
    return f"{field} mismatch: page shows '{shown}', spec read '{got}'"


def _raw_value(t: dict, field: str):
    """The value a field had before conversion (list row values are kept in _listSig, detail values in _raw)."""
    try:
        sig = json.loads(t.get("_listSig") or "{}")
    except (TypeError, ValueError):
        sig = {}
    v = sig.get(field) if isinstance(sig, dict) else None
    return v if v is not None else (t.get("_raw") or {}).get(field)


def _missing_hint(spec: dict, t: dict, field: str, label) -> str:
    """Why a value the page shows was not read, and how to read it: concrete enough for the next attempt."""
    fs = (spec.get("fields") or {}).get(field) or {}
    parts = []
    raw = _raw_value(t, field)
    if field == "submissionDeadline" and fs.get("from", "list") == "list" and raw and not is_no_value(raw) \
            and not _TIME_RE.search(str(raw)):
        parts.append(f"the list shows only the date ('{clean_text(str(raw))}'), without the time, so it cannot "
                     f"give the deadline")
    if field == "title":
        parts.append("read title (like id and url) from the LIST row when the list shows it - e.g. "
                     "{\"from\": \"list\", \"column\": <its header>} - because tender pages are opened only a few "
                     "per read and every listed tender needs its title at once")
    elif isinstance(label, str) and label.strip() and len(label) <= 80:
        lab = clean_text(label).rstrip(":： ")
        parts.append(f"the tender page shows it under the label '{lab}': map {field} as "
                     f"{{\"from\": \"detail\", \"label\": \"{lab}\"}}")
    return ("; " + "; ".join(parts)) if parts else ""


_DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}|^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")


def _date_keys_hint(found: dict | None, spec: dict) -> str:
    """For a JSON list: the row keys whose values are dates, with an example each (from the sample synthesize saw),
    so an unmapped deadline can be mapped by the API's own field name."""
    lst = spec.get("list") or {}
    if lst.get("kind") != "json" or not (found or {}).get("api"):
        return ""
    data = found["api"][1]
    items, ok = runner._checked_items("json", data, lst.get("items", ""))
    rows = items if ok and items and isinstance(items[0], dict) else (_first_list_of_dicts(data) or [])
    seen: dict[str, str] = {}
    for r in rows[:5]:
        for k, v in r.items() if isinstance(r, dict) else []:
            if isinstance(v, str) and _DATE_VALUE_RE.match(v.strip()) and k not in seen:
                seen[k] = v.strip()[:32]
    if not seen:
        return ""
    return "; the list rows have these date fields: " + ", ".join(f"{k}='{v}'" for k, v in list(seen.items())[:8])


_DEADLINE_SHOWN_RE = re.compile(
    r"((?:[A-Za-zÀ-ÿ'-]+ ){0,3}(?:deadline|closing date|closing|frist|date limite|sluitingsdatum|termin|plazo|"
    r"scadenza|einreichung|abgabe|remise)(?: [A-Za-zÀ-ÿ'-]+){0,3})\s*:?\s+"
    r"(\d{1,2}(?:st|nd|rd|th)?[ ./-](?:\d{1,2}|[A-Za-zÀ-ÿ]{3,10})\.?[ ./-]\d{2,4}(?:,? (?:at |à |um )?\d{1,2}[:.h]\d{2}"
    r"(?:\s?[ap]m)?)?)", re.I)


def _deadline_shown_hint(found: dict | None, spec: dict) -> str:
    """For an html list: where the list page prints a deadline (label and value), and whether it has a time."""
    if (spec.get("list") or {}).get("kind", "html") != "html" or not (found or {}).get("html"):
        return ""
    m = _DEADLINE_SHOWN_RE.search(page_text(found["html"], 20000))
    if not m:
        return ""
    label, value = clean_text(m.group(1)) or "", clean_text(m.group(2)) or ""
    if _TIME_RE.search(value):
        return f"; the list rows show it as '{label}: {value}' - map it from the list with that label"
    return (f"; the list rows show '{label}: {value}' - a date WITHOUT a time, so read submissionDeadline from the "
            f"tender page, where it is shown with its time ({{\"from\": \"detail\", \"label\": the label printed "
            f"there}})")


def _hop_hint(found: dict | None, spec: dict) -> str:
    """When the deadline is not on the tender page: what the second hop saw (the tender page's link to the full
    notice; that page refused, or without a deadline) and how to declare detail.follow."""
    found = found or {}
    note, link = found.get("hopNote"), found.get("noticeLink")
    if note and note.get("why") and note["why"].startswith(("HTTP", "not ")):
        h = (f"; the tender page links to its full notice ({note.get('url')}), but that page answered "
             f"{note['why']} to automated requests")
        if note.get("data"):
            h += (f" - the portal publishes its notices as open data: {'; '.join(note['data'][:6])}. Declare "
                  f"detail.follow with \"sel\" of that notice link, \"regex\" capturing the notice number from it, "
                  f"\"template\": the open-data address of ONE notice with {{value}}, \"kind\": \"json\", and map "
                  f"submissionDeadline {{\"from\": \"detail2\", \"path\": JMESPath of the offer deadline there}}")
        return h
    if note:
        return f"; the tender page links to its full notice ({note.get('url')}), which {note.get('why')}"
    if link and not (spec.get("detail") or {}).get("follow"):
        return (f"; the tender page does not show it but links to its full notice ({link}): declare "
                f"detail.follow {{\"sel\": CSS of that link}} and map submissionDeadline {{\"from\": \"detail2\", "
                f"\"label\": the label printed there}}")
    return ""


def _never_found(spec: dict, field: str, tenders: list[dict], found: dict | None = None) -> list[str]:
    """A field the spec read as null for every tender of the dry run: say what the source actually held."""
    hop = _hop_hint(found, spec) if field == "submissionDeadline" else ""
    if field not in (spec.get("fields") or {}):
        hint = _date_keys_hint(found, spec) or (_deadline_shown_hint(found, spec)
                                                if field == "submissionDeadline" else "")
        return [f"{field} is not mapped: every tender needs it - map it where the portal shows it (a list "
                f"column/field, or the tender page's label){hint}{hop}"]
    out = _never_read(spec, field, tenders, found)
    return [out[0] + hop] + out[1:] if out and hop else out


def _never_read(spec: dict, field: str, tenders: list[dict], found: dict | None = None) -> list[str]:
    raws = [_raw_value(t, field) for t in tenders]
    if raws and all(r is not None and is_no_value(r) for r in raws):
        return []                         # the portal says 'nv' / '-' for all of them: not given, not an error
    shown = [clean_text(str(r)) for r in raws if r not in (None, "") and not is_no_value(r)]
    src = ((spec.get("fields") or {}).get(field) or {}).get("from", "list")
    if field == "submissionDeadline" and shown and not any(_TIME_RE.search(s) for s in shown):
        return [f"submissionDeadline is never read: the {src} shows only dates (e.g. '{shown[0]}') without a "
                f"time; a deadline needs date AND time - read it where the time is shown (usually the tender "
                f"page: {{\"from\": \"detail\", \"label\": the label printed there}})"]
    if shown:
        return [f"{field} is never parsed: the {src} values were e.g. {shown[:2]} - check the date formats"]
    return [f"{field} is never found (check selector and date formats, if the portal shows it)"
            f"{_date_keys_hint(found, spec) if src == 'list' else ''}"]


# What the model copies from a tender page WITHOUT seeing the spec (blind verification at attach; audit.py adds keys
# before the closing brace).
VERIFY_PROMPT = (
    "Read this public tender page and copy these values EXACTLY as shown (do not reformat). If a value is "
    "not shown, use null. JSON: {\"id\": the tender identifier/number, \"title\", \"buyerName\": the public "
    "body that buys, \"otherBuyerNames\": other names this page gives for the buying or awarding body "
    "(e.g. an 'awarding office' besides a 'contracting body'), or [], \"publicationDate\", "
    "\"submissionDeadline\": deadline for offers incl. time, "
    "\"labels\": {\"buyerName\", \"publicationDate\", \"submissionDeadline\": the label printed next to "
    "that value on the page, copied exactly (e.g. \"Offers due\"), or null}")
VERIFY_RENDERS = 2              # browser renders per learn for tender pages that only render with JavaScript
VERIFY_RENDER_WAIT_S = 45.0
VERIFY_JS_ONLY = "not possible: tender pages render only with JavaScript"
_CORE = ("id", "title", "buyerName", "publicationDate", "submissionDeadline")


def _needs_browser(html: str) -> bool:
    """A tender page whose content is built by JavaScript: (almost) nothing to read without a browser, and scripts."""
    return _js_shell(html) and HTMLParser(html or "").css_first("script") is not None


def _shows_tender(truth: dict) -> bool:
    """The model found at least one of the core values on the page (an empty render compares nothing)."""
    return any(isinstance(truth.get(f), (str, int)) and str(truth.get(f)).strip() and not is_no_value(truth.get(f))
               for f in _CORE)


def _json_for_tender(cap: dict, t: dict) -> str | None:
    """The JSON the page loaded for this tender (it names the tender's id or title), as text: what the page would
    show once rendered. A single object (the tender's own record) is preferred over a list."""
    needles = [_norm(t.get(k)) for k in ("id", "title") if len(_norm(t.get(k))) >= 4]
    hits = []
    for _, data in cap.get("json") or []:
        txt = json.dumps(data, ensure_ascii=False)
        if needles and any(nd in _norm(txt) for nd in needles):
            hits.append((not isinstance(data, dict), len(txt), txt))
    return min(hits)[2][:8000] if hits else None


def _note(ctx, msg: str) -> None:
    note = getattr(ctx, "note", None)
    if callable(note):
        note(msg)


async def _rendered_text(ctx, url: str, t: dict, renders: dict) -> tuple[str | None, str]:
    """(text, source) of a tender page that renders only with JavaScript: its rendered text, else the JSON the page
    loaded for this tender; (None, why) when neither shows it. At most renders["left"] browser renders (shared by
    every attempt of one learn); each page's outcome is kept, so a retry does not render it again."""
    seen = renders.setdefault("pages", {})
    if url in seen:
        return seen[url]
    if renders.get("left", 0) <= 0:
        return None, "no browser render left"
    renders["left"] -= 1
    from . import browser
    wait = VERIFY_RENDER_WAIT_S
    needles = [str(t[k]) for k in ("id", "title") if t.get(k)]
    try:  # a page that never settles must not hold the attach
        cap = await asyncio.wait_for(browser.capture(ctx, url, max_wait_s=wait, quiet_s=4.0, max_json=10,
                                                     until_text=needles), wait + 30)
    except asyncio.TimeoutError:
        cap = {"html": "", "json": [], "error": "the browser did not finish in time"}
    html = cap.get("html") or ""
    text = page_text(html)
    _note(ctx, f"verify: {url} renders only with JavaScript; browser: {cap.get('requests')} requests in "
               f"{cap.get('seconds')}s, rendered text {len(text)} chars"
               + (f", error {str(cap['error'])[:100]}" if cap.get("error") else ""))
    data = _json_for_tender(cap, t)
    # The rendered page, if it shows the tender (or there is nothing else); else the data it loaded for the tender
    # (the app had not finished drawing it); an empty page compares nothing.
    mentions = any(nd in _norm(text) for nd in (_norm(t.get(k)) for k in ("id", "title")) if len(nd) >= 4)
    if len(text) >= 80 and (mentions or data is None):
        out = (text, "rendered page")
    elif data is not None:
        out = (data, "data the page loaded")
    else:
        out = (None, "the rendered page showed no tender")
    seen[url] = out
    return out


def json_digest(data, max_chars: int = 6000) -> str:
    """A JSON record as 'path = value' lines, date-times first: what a reader needs from a large record (e.g. an
    open-data notice whose deadline sits after kilobytes of parties) within a prompt's size."""
    lines_d, lines_o = [], []

    def walk(o, p, depth):
        if depth > 7 or len(lines_d) + len(lines_o) > 400:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else str(k), depth + 1)
        elif isinstance(o, list):
            for i, v in enumerate(o[:3]):
                walk(v, f"{p}[{i}]", depth + 1)
        elif isinstance(o, (str, int, float)) and not isinstance(o, bool):
            s = str(o).strip()
            if s and len(s) <= 300:
                (lines_d if re.match(r"^\d{4}-\d{2}-\d{2}", s) else lines_o).append(f"{p} = {s}")
    walk(data, "", 0)
    return "\n".join(lines_d + lines_o)[:max_chars]


async def linked_page_text(ctx, spec: dict, url: str) -> str:
    """The text of a tender's linked page (detail.follow), appended to its tender page for a reader: the values the
    spec reads 'from detail2' are shown there. '' when it cannot be read."""
    fol = runner.follow_spec(spec) or {}
    try:
        resp = await ctx.client.get(url)
    except (PortalBlocked, ModelBudgetExhausted, LLMUnavailable, BudgetExceeded):
        raise
    except Exception:
        return ""
    if resp.status_code >= 400:
        return ""
    if fol.get("kind") == "json" or "json" in (resp.headers.get("content-type") or ""):
        try:
            body = json_digest(resp.json())
        except ValueError:
            return ""
    else:
        body = page_text(resp.text, 8000)
    return f"\n\nTHE FULL NOTICE THIS PAGE LINKS TO ({url}):\n{body}"


async def blind_verify(ctx, spec: dict, tenders: list[dict], n: int = 3, opened=None,
                       renders: dict | None = None) -> list[str]:
    """The model reads n tender pages WITHOUT seeing the spec; disagreements on shown fields are failures.
    `opened`: ids whose detail page the dry run read. Detail-sourced fields are only known for those, so when the
    spec reads anything from the detail page, only those tenders are compared.
    A tender page that is an empty JavaScript shell is rendered in a browser (at most `renders["left"]` renders,
    default VERIFY_RENDERS). When no tender page shows anything even then, nothing could be compared: that is not a
    pass - spec["verification"] says so (and the caller must not mark the spec as verified)."""
    problems = []
    checked = shells = 0
    spec.pop("verification", None)
    renders = renders if renders is not None else {"left": VERIFY_RENDERS}
    dkind = (spec.get("detail") or {}).get("kind", "html")
    if opened is not None and any(isinstance(fs, dict) and fs.get("from") in runner.DETAIL_SOURCES
                                  for fs in (spec.get("fields") or {}).values()):
        tenders = [t for t in tenders if t.get("id") in set(opened)] or tenders
    for t in tenders:
        if checked >= n:
            break
        url = t.get("url")
        if spec.get("detail", {}).get("url_from") == "template" and t.get("id"):
            url = t.get("_detailUrl") or spec["detail"]["template"].replace("{id}", t["id"])
        if not url:
            continue
        resp = await ctx.client.get(url)
        if resp.status_code >= 400:
            continue
        source = "page"
        if dkind == "json":
            try:
                text = json.dumps(resp.json(), ensure_ascii=False)[:8000]
            except ValueError:
                continue
        elif _needs_browser(resp.text or ""):
            text, source = await _rendered_text(ctx, url, t, renders)
            if text is None:
                shells += 1
                if renders.get("left", 0) <= 0 and url not in renders.get("pages", {}):
                    break             # the other tender pages are built the same way: no point fetching them
                continue
        else:
            text = page_text(resp.text)
        if t.get("_followUrl") and runner.follow_spec(spec):
            text += await linked_page_text(ctx, spec, t["_followUrl"])
        truth = await ctx.llm.json_call("verify", VERIFY_PROMPT + "}", text, portal_id=ctx.portal_id,
                                        run_id=ctx.run_id)
        if source != "page" and not _shows_tender(truth):
            shells += 1                   # rendered, but the tender itself did not appear: nothing to compare
            continue
        checked += 1
        labels = truth.get("labels") if isinstance(truth.get("labels"), dict) else {}
        for f in ("publicationDate", "submissionDeadline"):
            if is_no_value(truth.get(f)):     # the page says 'nv' / '-': no value shown
                truth[f] = None
        others = [o for o in (truth.get("otherBuyerNames") or []) if isinstance(o, str) and o.strip()] \
            if isinstance(truth.get("otherBuyerNames"), list) else []
        for f in ("title", "buyerName"):
            shown, got = truth.get(f), t.get(f)
            # a page may name the buyer twice (awarding office and contracting body): either one is the buyer
            names = [_norm(s) for s in [shown] + (others if f == "buyerName" else []) if s and _norm(s)]
            g = _norm(got)
            if shown and got and names and g and not any(s in g or g in s for s in names):
                problems.append(f"{f} mismatch on {t.get('id')}: page shows '{shown}', spec read '{got}'")
            elif shown and not got:
                problems.append(f"{f} missing on {t.get('id')} although the page shows '{shown}'"
                                f"{_missing_hint(spec, t, f, labels.get(f))}")
        sid = truth.get("id")
        # Portals often show several identifiers (platform id, file number, notice number). Any stable identifier
        # the portal itself shows for this tender is fine: only complain if ours appears nowhere on the page/URL.
        ours = _norm(t.get("id"))
        if sid and ours and ours not in _norm(sid) and _norm(sid) not in ours \
                and ours not in _norm(text) and ours not in _norm(url):
            problems.append(f"id mismatch: page shows '{sid}', spec read '{t['id']}' (which is not on the page)")
        for f in ("publicationDate", "submissionDeadline"):
            fmts = list((spec.get("fields", {}).get(f) or {}).get("formats") or [])
            msg = _date_problem(f, truth.get(f), t.get(f), spec.get("tz") or "UTC", fmts)
            if msg:
                hint = _missing_hint(spec, t, f, labels.get(f)) if "missing although" in msg else ""
                problems.append(f"{msg} (tender {t.get('id')}){hint}")
    if checked == 0 and shells:
        spec["verification"] = VERIFY_JS_ONLY
        _note(ctx, f"blind verification {VERIFY_JS_ONLY} ({shells} tender page(s) showed nothing to compare)")
    elif checked == 0:
        problems.append("could not open any tender page to verify")
    return problems


# ------------------------------------------------------------------------------- narrowing a JSON API to open tenders
_SIZE_KEYS = ("size", "pageSize", "page_size", "pagesize", "limit", "rows", "per_page", "perPage", "hitsPerPage",
              "maxResults", "max_results")
_FROM_SUFFIXES = ("Vanaf", "From", "_from", "Von", "Ab", "Min", "Start", "After", "[gte]", ".gte")
NARROW_MAX_PROBES = 12


async def _probe_first_page(ctx, spec: dict, extra: dict | None = None) -> dict | None:
    """One request: the first list page with extra params. Counts rows by deadline state (no model call)."""
    s = json.loads(json.dumps(spec))
    s["list"]["params"] = {**(s["list"].get("params") or {}), **(extra or {})}
    pag = s["list"].get("pagination") or {}
    if pag.get("size_param") and pag["size_param"] in (extra or {}):
        pag["size"] = int(extra[pag["size_param"]])     # offset paging sets the size param from pagination.size
    try:
        _url, resp = await runner.fetch_list_page(ctx, s, 0)
    except BudgetExceeded:
        raise
    except Exception:
        return None
    if resp.status_code >= 400:
        return None
    body = runner._parse_body("json", resp)
    if body is None:
        return None
    items, ok = runner._checked_items("json", body, s["list"].get("items", ""))
    if not ok:
        return None
    from .extract import field_from
    fs = (s.get("fields") or {}).get("submissionDeadline") or {}
    today = runner.today_local(s)
    id_fs = (s.get("fields") or {}).get("id") or {}
    open_n = closed_n = undated = 0
    open_ids: list[str] = []
    for it in items:
        dl = runner.row_deadline(s, {"submissionDeadline": field_from("json", it, "submissionDeadline", fs)})
        if dl is None:
            undated += 1
        elif dl.date().isoformat() >= today:
            open_n += 1
            open_ids.append(str(field_from("json", it, "id", id_fs)))
        else:
            closed_n += 1
    total = runner._stated_total("json", body, s)
    guessed = False
    if total is None:
        _path, total = _guess_total(body)
        guessed = total is not None
    ids = [str(field_from("json", it, "id", id_fs)) for it in items]
    return {"items": len(items), "total": total, "guessedTotal": guessed, "open": open_n, "closed": closed_n,
            "undated": undated, "ids": ids, "openIds": open_ids, "sample": items[:3]}


_TOTAL_KEYS = ("totalElements", "total", "totalCount", "total_count", "totalResults", "totalHits", "numFound",
               "recordsTotal", "count", "hits.total", "hits.total.value", "meta.total", "page.totalElements")


def _guess_total(body) -> tuple[str | None, int | None]:
    """A stated-total field the spec did not map (paging APIs nearly always have one)."""
    if not isinstance(body, dict):
        return None, None
    for k in _TOTAL_KEYS:
        try:
            v = jmespath.search(k, body)
        except jmespath.exceptions.JMESPathError:
            continue
        if isinstance(v, int) and not isinstance(v, bool) and v >= 0:
            return k, v
    return None, None


def _filter_candidates_static(spec: dict) -> list[dict]:
    """'<deadline field>From'-style params derived from the API's own deadline field name."""
    path = str(((spec.get("fields") or {}).get("submissionDeadline") or {}).get("path") or "")
    name = re.split(r"[.\[\]|&]", path.strip())[-1] if path else ""
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    if not name:
        return []
    out = [{f"{name}{suf}": "{today}"} for suf in _FROM_SUFFIXES]
    out.append({f"min{name[:1].upper()}{name[1:]}": "{today}"})
    return out


async def _filter_candidates_model(ctx, spec: dict, sample: Any) -> list[dict]:
    """The cheap model proposes query parameters; it never decides - every candidate is probed."""
    try:
        res = await ctx.llm.json_call(
            "narrow",
            "A public tender JSON API lists every publication (old, awarded, corrections). Propose up to 6 sets of "
            "EXTRA query parameters that make it return only OPEN calls for competition, best first. Use the literal "
            "text {today} where today's date (YYYY-MM-DD) belongs, e.g. a 'deadline from today' parameter, or the "
            "API's own 'open/current' preset or status value if the sample suggests one. Do NOT filter by notice "
            "type: some portals publish real calls for competition under other types (e.g. a prior information "
            "notice used as a call), and the portal's own 'open tenders' view keeps them. Guess parameter names "
            "from the API's own field names and language. Each candidate is tested; wrong guesses are harmless. "
            "JSON: {\"candidates\": [{\"params\": {name: value}, \"why\": short}]}",
            json.dumps({"listUrl": spec["list"].get("url"), "currentParams": spec["list"].get("params") or {},
                        "deadlineField": (spec.get("fields") or {}).get("submissionDeadline"),
                        "sample": sample}, ensure_ascii=False)[:9000],
            portal_id=ctx.portal_id, run_id=ctx.run_id)
    except (ModelBudgetExhausted, LLMUnavailable):
        return []
    out = []
    for c in (res.get("candidates") or [])[:6]:
        p = c.get("params") if isinstance(c, dict) else None
        if isinstance(p, dict) and p and all(isinstance(k, str) and isinstance(v, (str, int)) for k, v in p.items()):
            out.append({k: str(v) for k, v in p.items()})
    return out


def _size_key(spec: dict) -> str | None:
    lst = spec["list"]
    pag = lst.get("pagination") or {}
    if pag.get("size_param"):
        return pag["size_param"]
    keys = set((lst.get("params") or {}).keys()) | {k for k, _ in parse_qsl(urlsplit(lst.get("url") or "").query)}
    for k in _SIZE_KEYS:
        if k in keys:
            return k
    return None


async def narrow_to_open(ctx, spec: dict) -> None:
    """Many JSON APIs serve the whole archive, so a read either walks into the portal's paging cap or delivers
    closed notices. Find, by evidence, (1) query params that return only open tenders and (2) the largest page size
    the API accepts. The model only proposes. A candidate passes only if one request shows the listing shrank and
    every row on page 1 is open; of the passing candidates the one keeping the MOST rows wins, so a filter that also
    cuts open tenders (a notice type, a publication date) loses to a plain 'deadline from today'.
    Best effort: any problem leaves the spec unchanged."""
    lst = spec.get("list") or {}
    dl_fs = (spec.get("fields") or {}).get("submissionDeadline") or {}
    if lst.get("kind") == "json" and (lst.get("method") or "GET").upper() == "POST":
        # A POST search carries the portal's own filters in its body (kept as captured); only the page size is tuned.
        await _raise_body_page_size(ctx, spec)
        return
    if lst.get("kind") != "json" or (lst.get("method") or "GET").upper() != "GET" \
            or dl_fs.get("from", "list") != "list" or not dl_fs.get("path"):
        return
    base = await _probe_first_page(ctx, spec)
    if not base or not base["items"]:
        return
    # A guessed total (the spec maps none) is only used to compare candidates, never stored, and only when it is
    # plainly a listing total: a "count" equal to the rows on the page may be a per-page count.
    use_totals = base["total"] is not None and not (base["guessedTotal"] and base["total"] <= base["items"])
    probes = 1
    if base["closed"] or base["undated"]:
        cands = await _filter_candidates_model(ctx, spec, base["sample"]) + _filter_candidates_static(spec)
        tried: set[str] = set()
        passing: list[tuple[dict, dict]] = []
        for cand in cands:
            key = json.dumps(cand, sort_keys=True)
            if key in tried or probes >= NARROW_MAX_PROBES:
                continue
            tried.add(key)
            probes += 1
            got = await _probe_first_page(ctx, spec, cand)
            if not got or not got["items"]:
                continue
            if use_totals and got["total"] is not None:
                shrank = got["total"] < base["total"]
            else:   # no usable total: page 1 must have changed from mixed to all open
                shrank = got["ids"] != base["ids"]
            # Same order, closed rows removed: every row that was open on page 1 must still be on page 1. This rejects
            # filters that also cut open tenders (published today, one notice type) even when no total is stated.
            kept_open = set(base["openIds"]) <= set(got["ids"])
            if shrank and kept_open and got["open"] == got["items"]:
                passing.append((cand, got))
        if passing:
            # most rows kept first; then fewer params; then the proposal order
            ranked = sorted(enumerate(passing), key=lambda ic: (
                -(ic[1][1]["total"] if ic[1][1]["total"] is not None else -1), len(ic[1][0]), ic[0]))
            cand, got = ranked[0][1]
            spec["list"]["params"] = {**(lst.get("params") or {}), **cand}
            spec["list"]["openFilter"] = {
                "params": cand, "undatedOnPage1": base["undated"], "alternativesPassed": len(passing) - 1,
                "evidence": f"stated total {base['total']} -> {got['total']}; all {got['items']} rows on page 1 open "
                            f"(before: {base['closed']} closed, {base['undated']} without a deadline - notices "
                            f"without a deadline are not served once this filter is on)"}
            ctx.note(f"narrowed to open tenders with {cand}: {base['total']} -> {got['total']} listed"
                     + (f"; {base['undated']} of {base['items']} rows on page 1 had no deadline and are excluded"
                        if base["undated"] else ""))
        else:
            ctx.note("no open-only filter found for this API; closed notices are skipped while reading")
    # page size: fewer requests and more rows inside the portal's paging window (failure here keeps the filter)
    try:
        await _raise_page_size(ctx, spec, base, probes)
    except (BudgetExceeded, PortalBlocked):
        raise
    except Exception as e:
        ctx.note(f"page size unchanged ({_short(e, 80)})")


# ------------------------------------------------------------- narrowing an HTML list to open tenders (its search form)
# The HTML twin of narrow_to_open: a list page whose first page shows closed rows usually has a search/filter form
# with the portal's own open/current settings (a status option, a 'deadline from' date). The cheap model reads the
# form and proposes settings in the portal's own option values; each one is submitted and kept only by evidence.
HTML_NARROW_MAX_PROBES = 10
_ADV_SEARCH_WORDS = ("advanced", "advsearch", "erweitert", "avancée", "avancee", "uitgebreid", "avanzada",
                     "avanzata", "zaawansowan", "detailsuche", "expertensuche", "recherche avanc")
_DEADLINE_NAME_RE = re.compile(r"(submission|deadline|closing|frist|limite|sluiting|cierre|scadenz|abgabe|"
                               r"reponse|réponse)", re.I)
_FROM_NAME_RE = re.compile(r"(from|von|start|min|vanaf|desde|after|begin|_du\b|du$|^du|\bab\b|gte)", re.I)
_TOKENISH = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9+/=_-]{16,}$")


def _txt(s) -> str:
    """clean_text that never returns None (empty labels, options and links are common in search forms)."""
    return clean_text(s) or ""


def _label_texts(form) -> dict[str, str]:
    out = {}
    for lab in form.css("label[for]"):
        out[lab.attributes.get("for") or ""] = _txt(lab.text())[:60]
    return out


def describe_forms(html: str, page_url: str) -> list[dict]:
    """The filter/search forms of a page, compact: action, method, fields with their current value, options and
    label. Login forms and forms with too few fields to filter anything are left out."""
    tree = HTMLParser(html or "")
    out = []
    for form in tree.css("form"):
        method = (form.attributes.get("method") or "get").strip().lower()
        if method not in ("get", "post") or form.css_first("input[type=password]"):
            continue
        labels = _label_texts(form)
        fields: list[dict] = []
        submit = None
        for el in form.css("input, select, textarea"):
            a = el.attributes
            name = (a.get("name") or "").strip()
            typ = (a.get("type") or ("select" if el.tag == "select" else "text")).lower()
            if not name or typ in ("file", "reset", "image"):
                continue
            if typ in ("submit", "button"):
                if submit is None:
                    submit = {"name": name, "type": "submit", "value": a.get("value") or ""}
                continue
            f: dict = {"name": name, "type": typ}
            lab = labels.get(a.get("id") or "") or a.get("placeholder") or a.get("title") or a.get("aria-label")
            if lab:
                f["label"] = _txt(lab)[:60]
            if el.tag == "select":
                opts = el.css("option")
                f["options"] = [[o.attributes.get("value") if o.attributes.get("value") is not None
                                 else _txt(o.text()), _txt(o.text())[:50]] for o in opts][:30]
                chosen = [o for o in opts if "selected" in o.attributes] or opts[:1]
                f["value"] = [(o.attributes.get("value") if o.attributes.get("value") is not None
                               else _txt(o.text())) for o in chosen] if "multiple" in a else \
                    ((chosen[0].attributes.get("value") if chosen[0].attributes.get("value") is not None
                      else _txt(chosen[0].text())) if chosen else "")
            elif typ in ("checkbox", "radio"):
                f["value"] = a.get("value") if a.get("value") is not None else "on"
                f["checked"] = "checked" in a
            else:
                f["value"] = a.get("value") or (el.text() if el.tag == "textarea" else "") or ""
            fields.append(f)
        filtering = any(f["type"] in ("select", "checkbox", "radio", "date") or _DEADLINE_NAME_RE.search(f["name"])
                        or "status" in f["name"].lower() for f in fields)
        if len(fields) < 2 or not filtering:
            continue
        action = urljoin(page_url, (form.attributes.get("action") or "").strip() or page_url)
        if not action.startswith(("http://", "https://")):
            continue
        out.append({"i": len(out), "page": page_url, "action": action, "method": method.upper(),
                    "fields": fields[:80], "submit": submit})
    return out


def form_values(form: dict, values: dict) -> dict:
    """What a browser submits: the form's current values (hidden fields, selected options, ticked boxes), with the
    proposed settings on top (a list = several ticked boxes / options of one name)."""
    out: dict = {}
    for f in form["fields"]:
        name = f["name"]
        if f["type"] in ("checkbox", "radio"):
            if f.get("checked"):
                if f["type"] == "checkbox" and name in out:
                    prev = out[name] if isinstance(out[name], list) else [out[name]]
                    out[name] = prev + [f["value"]]
                else:
                    out[name] = f["value"]
            continue
        out.setdefault(name, f.get("value") if f.get("value") is not None else "")
    if form.get("submit") and form["submit"]["name"] not in out:
        out[form["submit"]["name"]] = form["submit"]["value"]
    for k, v in values.items():
        out[k] = [str(x) for x in v] if isinstance(v, list) else str(v)
    return out


def _static_form_candidates(forms: list[dict]) -> list[dict]:
    """'Deadline from today' in the form's own date field, in the date formats forms commonly expect."""
    out = []
    for form in forms:
        for f in form["fields"]:
            n = f["name"]
            if f["type"] in ("checkbox", "radio", "select") or not (_DEADLINE_NAME_RE.search(n)
                                                                   and _FROM_NAME_RE.search(n)):
                continue
            hint = (f.get("label") or "").lower()
            if f["type"] == "date" or "yyyy-mm-dd" in hint:
                fmts = ["{today}"]
            elif "mm/dd" in hint:
                fmts = ["{today:%m/%d/%Y}"]
            elif "." in hint and "dd" in hint:
                fmts = ["{today:%d.%m.%Y}"]
            else:
                fmts = ["{today:%d/%m/%Y}", "{today}", "{today:%d.%m.%Y}"]
            out += [{"form": form["i"], "values": {n: v}} for v in fmts]
    return out[:4]


async def _form_candidates_model(ctx, spec: dict, forms: list[dict], sample_rows: list[str]) -> list[dict]:
    """The cheap model reads the forms and proposes settings; it never decides - every candidate is submitted."""
    compact = [{"form": f["i"], "action": f["action"], "method": f["method"],
                "fields": [{k: v for k, v in fd.items() if k != "checked" or v} for fd in f["fields"]]}
               for f in forms]
    try:
        res = await ctx.llm.json_call(
            "narrow",
            "A public tender portal's list shows closed tenders (deadline passed) on its first page. Below are the "
            "search/filter forms of the portal. Propose up to 6 settings that make the portal's OWN search return "
            "only OPEN/current calls for competition, best first: its own status option for open/current tenders "
            "(e.g. 'Tender Submission', 'active', 'en cours', 'laufend', 'lopend', 'abiertas' - copy the option's "
            "value exactly), and/or its 'deadline from' date field set to today. Write {today} for today's date as "
            "YYYY-MM-DD, or {today:FORMAT} with a strftime format when the field wants another format (e.g. "
            "{today:%d/%m/%Y}). Use only field names and option values from the forms. Do NOT filter by notice "
            "type or publication date: that would drop open tenders. Each candidate is tested; wrong guesses are "
            "harmless. JSON: {\"candidates\": [{\"form\": form number, \"values\": {name: value or [values]}, "
            "\"why\": short}]}",
            json.dumps({"today": runner.today_local(spec), "listUrl": spec["list"].get("url"),
                        "deadlineField": (spec.get("fields") or {}).get("submissionDeadline"),
                        "firstRows": sample_rows[:3], "forms": compact}, ensure_ascii=False)[:12000],
            portal_id=ctx.portal_id, run_id=ctx.run_id)
    except (ModelBudgetExhausted, LLMUnavailable):
        return []
    out = []
    by_i = {f["i"]: f for f in forms}
    for c in (res.get("candidates") or [])[:6]:
        if not isinstance(c, dict) or not isinstance(c.get("values"), dict) or not c["values"]:
            continue
        try:
            fi = int(c.get("form", 0))
        except (TypeError, ValueError):
            continue
        form = by_i.get(fi)
        if form is None:
            continue
        names = {f["name"] for f in form["fields"]}
        vals = {k: v for k, v in c["values"].items() if k in names and isinstance(v, (str, int, list))}
        if vals:
            out.append({"form": fi, "values": vals})
    return out


def _form_trial(spec: dict, form: dict, values: dict, method: str) -> dict:
    """The spec with its list read through the search form (GET: fields in the query; POST: a form body)."""
    s = json.loads(json.dumps(spec))
    lst = s["list"]
    fields = form_values(form, values)
    if method == "GET":
        # a per-visit token (search session id, anti-forgery token) must not be frozen into an address read daily
        hidden = {f["name"] for f in form["fields"] if f["type"] == "hidden"}
        fields = {k: v for k, v in fields.items()
                  if not (k in hidden and k not in values and isinstance(v, str) and _TOKENISH.match(v))}
    lst["url"] = form["action"]
    lst.pop("order", None)               # the search result's own order is unknown
    lst.pop("body", None)
    lst.pop("openFilter", None)
    if method == "GET":
        lst["method"] = "GET"
        lst.pop("form", None)
        lst["params"] = fields
    else:
        lst["method"] = "POST"
        lst["params"] = {}
        lst["form"] = fields
    return s


async def _html_probe(ctx, spec: dict) -> dict | None:
    """One request: the first list page of an html spec, rows counted by deadline state (no model call)."""
    from .extract import field_from as _ff
    try:
        _url, resp = await runner.fetch_list_page(ctx, spec, 0)
    except BudgetExceeded:
        raise
    except Exception:
        return None
    if resp.status_code >= 400 or runner.detect_block(resp.text, resp.status_code, spec.get("block_markers")):
        return None
    tree = HTMLParser(resp.text or "")
    try:
        rows = list_items("html", tree, spec["list"].get("items", ""))
    except Exception:
        return None
    fields = spec.get("fields") or {}
    dl_fs, id_fs = fields.get("submissionDeadline") or {}, fields.get("id") or {}
    today = runner.today_local(spec)      # by date, like the JSON probe: a 'deadline from today' filter keeps today
    open_n = closed_n = undated = 0
    ids, open_ids, sample = [], [], []
    for r in rows:
        rid = str(_ff("html", r, "id", id_fs)) if id_fs else _txt(r.text())[:80]
        ids.append(rid)
        dl = runner.row_deadline(spec, {"submissionDeadline": _ff("html", r, "submissionDeadline", dl_fs)}) \
            if dl_fs and dl_fs.get("from", "list") == "list" else None
        if dl is None:
            undated += 1
        elif dl.date().isoformat() >= today:
            open_n += 1
            open_ids.append(rid)
        else:
            closed_n += 1
        if len(sample) < 3:
            sample.append(_txt(r.text(separator=" | "))[:300])
    try:
        total = runner._stated_total("html", tree, spec)
    except Exception:
        total = None
    return {"items": len(rows), "total": total, "open": open_n, "closed": closed_n, "undated": undated,
            "ids": ids, "openIds": open_ids, "sample": sample, "html": resp.text or "", "url": str(resp.url)}


async def _last_page_probe(ctx, spec: dict, first: dict) -> dict | None:
    """The last page of a numbered/offset list, from its stated total and the rows on page 1 (one request)."""
    pag = (spec.get("list") or {}).get("pagination") or {}
    total, per = first.get("total"), first.get("items") or 0
    if pag.get("type") not in ("page", "offset") or not total or not per or total <= per:
        return None
    last = (int(total) - 1) // per
    s = json.loads(json.dumps(spec))
    try:
        _u, resp = await runner.fetch_list_page(ctx, s, last)
    except BudgetExceeded:
        raise
    except Exception:
        return None
    if resp.status_code >= 400:
        return None
    tree = HTMLParser(resp.text or "")
    fields = spec.get("fields") or {}
    dl_fs = fields.get("submissionDeadline") or {}
    if not dl_fs or dl_fs.get("from", "list") != "list":
        return None
    from .extract import field_from as _ff
    today = runner.today_local(spec)
    rows = list_items("html", tree, spec["list"].get("items", ""))
    closed = 0
    for r in rows:
        dl = runner.row_deadline(spec, {"submissionDeadline": _ff("html", r, "submissionDeadline", dl_fs)})
        if dl is not None and dl.date().isoformat() < today:
            closed += 1
    return {"items": len(rows), "closed": closed}


async def _forms_to_try(ctx, html: str, page_url: str) -> list[dict]:
    """Forms of the list page; when none of them offers a choice (only hidden fields), also the forms of the
    page's own 'advanced search' link (one request)."""
    forms = describe_forms(html, page_url)
    if any(f["type"] in ("select", "checkbox", "radio") for form in forms for f in form["fields"]):
        return forms
    host = urlsplit(page_url).hostname
    for a in HTMLParser(html or "").css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.lower().startswith(("javascript:", "mailto:", "#")):
            continue
        u = urljoin(page_url, href)
        hay = (href + " " + _txt(a.text())).lower()
        if urlsplit(u).hostname == host and any(w in hay for w in _ADV_SEARCH_WORDS):
            try:
                r = await ctx.client.get(u)
            except BudgetExceeded:
                raise
            except Exception:
                break
            if r.status_code < 400:
                more = describe_forms(r.text or "", str(r.url))
                for f in more:
                    f["i"] = len(forms)
                    forms.append(f)
            break
    return forms


async def narrow_html_to_open(ctx, spec: dict, base: dict | None = None, mixed: bool = False) -> bool:
    """Find, by evidence, settings of the portal's own search form that list only open tenders. A candidate passes
    only if one request shows: the stated total shrank (or page 1 changed when no total is stated), no row on page 1
    is closed, and every row that was open on page 1 is still there. Of the passing ones, the most rows win.
    Returns True when the spec was changed. The model only proposes."""
    lst = spec.get("list") or {}
    if lst.get("kind", "html") != "html":
        return False
    base = base or await _html_probe(ctx, spec)
    if not base:
        return False
    deep = None
    if not base["closed"]:
        # Page 1 is all open, but a listing sorted by publication date keeps its closed tenders further back: the
        # last page shows whether the portal's total counts closed ones (then a read walks into the archive).
        deep = await _last_page_probe(ctx, spec, base)
        if (not deep or not deep["closed"]) and not mixed:
            return False
        if not deep or not deep["closed"]:
            deep = None
            ctx.note("page 1 is open but later list pages showed closed tenders; looking for the portal's open "
                     "filter")
        else:
            ctx.note(f"page 1 is open but the last list page shows {deep['closed']} closed of {deep['items']}: the "
                     f"listing ({base['total']}) includes closed tenders; looking for the portal's open filter")
    forms = await _forms_to_try(ctx, base["html"], base["url"])
    if not forms:
        ctx.note("page 1 shows closed tenders and the list page has no search form to narrow it")
        return False
    cands = await _form_candidates_model(ctx, spec, forms, base["sample"]) + _static_form_candidates(forms)
    by_i = {f["i"]: f for f in forms}
    tried: set[str] = set()
    passing: list[tuple[dict, dict, dict]] = []
    probes, rejected = 0, []
    for cand in cands:
        form = by_i[cand["form"]]
        methods = ["GET"] if form["method"] == "GET" else ["GET", "POST"]   # a GET search pages like any list
        for method in methods:
            key = json.dumps([cand["form"], cand["values"], method], sort_keys=True)
            if key in tried or probes >= HTML_NARROW_MAX_PROBES:
                continue
            tried.add(key)
            probes += 1
            trial = _form_trial(spec, form, cand["values"], method)
            got = await _html_probe(ctx, trial)
            if not got or not got["items"]:
                rejected.append(f"{cand['values']} ({method}): no rows")
                continue
            if base["total"] is not None and got["total"] is not None:
                shrank = got["total"] < base["total"]
            else:
                shrank = got["ids"] != base["ids"]
            kept_open = set(base["openIds"]) <= set(got["ids"])
            if shrank and kept_open and not got["closed"] and got["open"]:
                if deep is not None and probes < HTML_NARROW_MAX_PROBES + 2:
                    probes += 1         # the closed rows were at the end: the filtered list's end must be open
                    tail = await _last_page_probe(ctx, trial, got)
                    if tail and tail["closed"]:
                        rejected.append(f"{cand['values']} ({method}): last page still {tail['closed']} closed")
                        continue
                passing.append((cand, got, trial))
                break                       # this setting works; its POST twin need not be tried
            rejected.append(f"{cand['values']} ({method}): total {base['total']} -> {got['total']}, "
                            f"{got['closed']} closed of {got['items']} on page 1")
    if not passing:
        ctx.note(f"no open-only setting of the portal's search form passed ({len(tried)} tried)"
                 + (f": {'; '.join(rejected[:3])}" if rejected else ""))
        return False
    ranked = sorted(enumerate(passing), key=lambda ic: (
        -(ic[1][1]["total"] if ic[1][1]["total"] is not None else -1), len(ic[1][0]["values"]), ic[0]))
    cand, got, trial = ranked[0][1]
    spec["list"] = trial["list"]
    spec["list"]["openFilter"] = {
        "form": by_i[cand["form"]]["action"], "values": cand["values"], "alternativesPassed": len(passing) - 1,
        "evidence": f"stated total {base['total']} -> {got['total']}; all {got['open']} dated rows on page 1 open "
                    f"(before: {base['closed']} closed of {base['items']})"}
    ctx.note(f"narrowed the list to open tenders with the portal's search form {cand['values']} "
             f"({trial['list'].get('method', 'GET')} {trial['list']['url']}): {base['total']} -> {got['total']} listed")
    return True


async def _narrow_html_safely(ctx, spec: dict, found: dict | None = None) -> bool:
    """narrow_html_to_open, best effort: any problem leaves the spec as it was."""
    before = json.loads(json.dumps(spec))
    try:
        changed = await narrow_html_to_open(ctx, spec)
        if found is not None:
            found["openFilterTried"] = "" if changed else (
                " (the portal's own search form was tried: no open-only setting passed the evidence check)")
        return changed
    except (BudgetExceeded, PortalBlocked):
        spec.clear()
        spec.update(before)
        raise
    except Exception as e:
        spec.clear()
        spec.update(before)
        ctx.note(f"search-form narrowing skipped ({_short(e, 120)})")
        if found is not None:
            found["openFilterTried"] = f" (the list page's search form was tried: {_short(e, 80)})"
        return False


async def _raise_body_page_size(ctx, spec: dict) -> None:
    """Page size of a POST search (pagination.size_body_path; if the model gave none, a size-named field next to the
    page field, e.g. PAGE.size beside PAGE.number). Kept only if one probe shows the API returns more rows per page:
    fewer requests, and more of the list inside a result window."""
    lst = spec["list"]
    pag = lst.get("pagination") or {}
    body = lst.get("body")
    if pag.get("type") not in ("page", "offset") or not isinstance(body, dict):
        return
    key = pag.get("size_body_path")
    if not key and pag.get("body_path") and "." in pag["body_path"]:
        parent = pag["body_path"].rsplit(".", 1)[0]
        holder = jmespath.search(parent, body)
        if isinstance(holder, dict):
            key = next((f"{parent}.{k}" for k in _SIZE_KEYS if isinstance(holder.get(k), int)), None)
    if not key:
        return
    base = await _probe_first_page(ctx, spec)
    if not base or not base["items"]:
        return
    cur = base["items"]
    for size in (100, 50):
        if size <= cur:
            continue
        trial = json.loads(json.dumps(spec))
        trial["list"]["pagination"].update(size_body_path=key, size=size)
        got = await _probe_first_page(ctx, trial)
        want = min(size, got["total"]) if got and got["total"] is not None else size
        if got and got["items"] > cur and got["items"] >= want:
            pag.update(size_body_path=key, size=size)
            ctx.note(f"page size {cur} -> {size} ({key} in the request body)")
            return


async def _raise_page_size(ctx, spec: dict, base: dict, probes: int) -> None:
    key = _size_key(spec)
    pag = spec["list"].get("pagination") or {}
    if not key or pag.get("type") not in ("page", "offset"):
        return
    in_url = dict(parse_qsl(urlsplit(spec["list"].get("url") or "").query)).get(key)
    raw = (spec["list"].get("params") or {}).get(key) or in_url or pag.get("size") or base["items"] or 0
    cur = int(str(raw).strip())
    for size in (100, 50):
        if size <= cur or probes >= NARROW_MAX_PROBES + 2:
            continue
        probes += 1
        got = await _probe_first_page(ctx, spec, {key: size})
        want = min(size, got["total"]) if got and got["total"] is not None else size
        if got and got["items"] > cur and got["items"] >= want:
            spec["list"].setdefault("params", {})[key] = size
            if pag.get("type") == "offset":
                pag["size"] = size
            ctx.note(f"page size {cur} -> {size}")
            return


# ---------------------------------------------------------------- documents of a JSON API: per-tender endpoint discovery
# A JSON list API rarely carries document links, and its tender pages are often JavaScript apps that show the model
# nothing. Most of these APIs follow one REST shape: <list address>/{id}/<documents>. Probe that shape (a word list
# in the portal languages), find where the download links sit in the answer, and keep it only if one real file
# downloads. No model call; a miss leaves the spec as it was (documents stay empty, as before).
_DOC_WORDS = ("documenten", "documents", "dokumente", "unterlagen", "docs", "attachments", "files", "bijlagen",
              "documentos", "documenti", "pieces", "dokumenty")
DOC_MAX_PROBES = 30
_LINK_KEYS = ("href", "url", "link", "download", "downloadurl", "download_url", "uri", "path")


def _lists_of_dicts_with_path(obj, path: str = "", depth: int = 0):
    """(JMESPath of a list, the list) for every list of dicts, outermost first."""
    if depth > 4:
        return
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj[:5]):
        yield (f"{path}[]" if path else "[]"), obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                yield from _lists_of_dicts_with_path(v, f"{path}.{k}" if path else k, depth + 1)


def _leaf_strings(d: dict, prefix: str = "", depth: int = 0):
    if depth > 3:
        return
    for k, v in d.items():
        if not (isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)):
            continue
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            yield p, k, v
        elif isinstance(v, dict):
            yield from _leaf_strings(v, p, depth + 1)


def _link_score(key: str, path: str, value: str) -> int:
    v = value.strip()
    if not (v.startswith(("http://", "https://")) or v.startswith("/")):
        return 0
    s = 1
    low = (path + " " + v).lower()
    if "download" in low:
        s += 3
    if re.search(r"/content\b|/file|/document|/attachment|/bijlage", low):
        s += 1
    if re.search(r"\.(pdf|zip|docx?|xlsx?|odt|rtf|txt|xml)(\?|$)", v.lower()):
        s += 2
    if key.lower() in _LINK_KEYS:
        s += 1
    return s


def doc_link_path(data) -> tuple[str, list[str]] | None:
    """Where the download links are in a documents answer: (JMESPath, links), or None.
    An empty document list returns ("", []) so the caller can try another tender on the same endpoint."""
    found_empty = False
    for lpath, items in _lists_of_dicts_with_path(data):
        scores: dict[str, int] = {}
        hits: dict[str, int] = {}
        for it in items[:20]:
            for p, k, v in _leaf_strings(it):
                sc = _link_score(k, p, v)
                if sc:
                    scores[p] = max(scores.get(p, 0), sc)
                    hits[p] = hits.get(p, 0) + 1
        good = [p for p in scores if hits[p] >= max(1, len(items[:20]) // 2)]
        if good:
            best = max(good, key=lambda p: (scores[p], hits[p]))
            expr = f"{lpath}.{best}"
            try:
                links = jmespath.search(expr, data) or []
            except jmespath.exceptions.JMESPathError:
                continue
            return expr, [x for x in links if isinstance(x, str)]
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and not v and any(w in k.lower() for w in ("doc", "bijlag", "attach", "file",
                                                                                 "unterlag", "piece")):
                found_empty = True
    return ("", []) if found_empty else None


def _doc_url_candidates(spec: dict) -> list[str]:
    lst = spec.get("list") or {}
    parts = urlsplit(runner.fill_tokens(lst.get("url") or "", spec))
    base = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
    out = [f"{base}/{{id}}/{w}" for w in _DOC_WORDS]
    det = spec.get("detail") or {}
    if det.get("url_from") == "template" and det.get("kind") == "json" and det.get("template"):
        t = str(det["template"]).rstrip("/")
        out = [f"{t}/{w}" for w in _DOC_WORDS] + out
    seen, uniq = set(), []
    for u in out:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


async def _download_works(ctx, url: str) -> bool:
    """One real file: HTTP 200 and not an HTML/login page (only the first bytes are read)."""
    from .http import detect_document_wall
    resp = await ctx.client.stream("GET", url)
    try:
        if resp.status_code != 200 or "html" in resp.headers.get("content-type", "").lower():
            return False            # discovery wants a file, never a page
        head = b""
        async for chunk in resp.aiter_bytes(1 << 14):
            head = chunk
            break
        return not detect_document_wall(resp.headers.get("content-type", ""), head)
    finally:
        await resp.aclose()


async def discover_documents(ctx, spec: dict, tenders: list[dict]) -> None:
    """Probe <list address>/{x}/<documents word>, where x is the tender id or, when the tender id is a procedure
    number, another id of the listed notice (documents are often attached per notice)."""
    lst = spec.get("list") or {}
    if lst.get("kind") != "json" or (spec.get("documents") or {}).get("from"):
        return
    fields = spec.get("fields") or {}
    id_path = (fields.get("id") or {}).get("path")
    items: list = []
    try:
        _u, r = await runner.fetch_list_page(ctx, spec, 0)
        items, _ok = runner._checked_items("json", r.json(), lst.get("items", ""))
    except (BudgetExceeded, PortalBlocked):
        raise
    except Exception:
        items = []
    # candidate id fields: the tender id first, then other short id-like scalars of the listed notice
    keys: list[str | None] = [None]
    if items and isinstance(items[0], dict):
        for k, v in items[0].items():
            if k != id_path and isinstance(k, str) and "id" in k.lower() and not isinstance(v, bool) \
                    and isinstance(v, (int, str)) and re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", str(v)):
                keys.append(k)
    keys = keys[:3]
    probes = 1
    for tmpl in _doc_url_candidates(spec):
        for key in keys:
            if probes >= DOC_MAX_PROBES:
                ctx.note("no per-tender documents address found for this API (probe budget used)")
                return
            if key is None:
                vals = [t["id"] for t in tenders if t.get("id")][:3]
            else:
                vals = [it.get(key) for it in items[:5] if isinstance(it, dict) and it.get(key) is not None][:3]
            for n, val in enumerate(vals):
                probes += 1
                resp = await ctx.client.get(tmpl.replace("{id}", str(val)))
                if resp.status_code != 200 or "json" not in resp.headers.get("content-type", ""):
                    break                       # not this address/id: next candidate
                try:
                    found = doc_link_path(resp.json())
                except ValueError:
                    break
                if found is None:
                    break
                path, links = found
                if not links:
                    if n + 1 < len(vals) and probes < DOC_MAX_PROBES:
                        continue                # address exists, this one has no files: try the next
                    break
                first = urljoin(str(resp.url), links[0])
                probes += 1
                if not await _download_works(ctx, first):
                    ctx.note(f"documents address {tmpl} lists files but {first} did not download")
                    return
                url = tmpl
                if key is not None:
                    spec["fields"]["noticeId"] = {"from": "list", "path": key}
                    url = tmpl.replace("{id}", "{noticeId}")
                spec["documents"] = {"from": "template", "url": url, "kind": "json", "path": path,
                                     "discovered": f"{len(links)} file(s) for {key or 'tender id'} {val}; "
                                                   f"first download OK"}
                ctx.note(f"found the documents of each {'notice' if key else 'tender'} at {url} ({path}); "
                         f"{len(links)} for {val}")
                return
    ctx.note("no per-tender documents address found for this API")


async def _discover_docs_safely(ctx, spec: dict, tenders: list[dict]) -> None:
    before = json.loads(json.dumps(spec))
    try:
        await discover_documents(ctx, spec, tenders)
        bad = validate_spec(spec, full=True)
        if bad:
            raise ValueError("; ".join(bad[:2]))
    except (BudgetExceeded, PortalBlocked):
        spec.clear()
        spec.update(before)
        raise
    except Exception as e:  # best effort: never cost the attach
        spec.clear()
        spec.update(before)
        ctx.note(f"documents discovery skipped ({_short(e, 120)})")


# ---------------------------------------------------------------- a JSON list without the deadline: find the tender API
DETAIL_MAX_PROBES = 40
_ID_KEY = re.compile(r"(?i)(^id$|id$|_id$|uuid|number$|nr$|key$)")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$")


def _idish_keys(row: dict) -> list[str]:
    """Top-level scalar fields of a list row that look like identifiers (the parts of a REST address)."""
    out = []
    for k, v in row.items():
        if isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) and _ID_KEY.search(k) \
                and not isinstance(v, bool) and isinstance(v, (int, str)) \
                and re.fullmatch(r"[A-Za-z0-9_.|:-]{1,64}", str(v)):
            out.append(k)
    return out


def _segments(key: str) -> list[str]:
    """REST path words a field name suggests: publicationId -> publication, publications, publication-details."""
    stem = re.sub(r"(?i)(_?id|_?uuid|_?number|_?nr|_?key)$", "", key)
    if not stem:
        return []
    kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", stem).lower()
    return list(dict.fromkeys([kebab, kebab + "s", kebab + "-details", kebab + "-detail", stem]))


def _detail_candidates_static(list_url: str, row: dict) -> list[str]:
    """Addresses a REST API typically gives one tender, built from the list address and the row's id fields (as
    {rowField} placeholders): <collection>/{a}, <collection>/{a}/<word from b>/{b}, with neighbouring API versions
    (v2 -> v1, v3) - a search endpoint is often newer than the per-item one."""
    parts = urlsplit(list_url)
    segs = [s for s in parts.path.split("/") if s]
    if len(segs) < 2:
        return []
    bases = ["/".join(segs[:-1])]
    for i, s in enumerate(segs[:-1]):
        m = re.fullmatch(r"v(\d+)", s)
        if m:
            n = int(m.group(1))
            for alt in (n - 1, n + 1):
                if alt >= 1:
                    bases.append("/".join(segs[:i] + [f"v{alt}"] + segs[i + 1:-1]))
    # plain ids first ('id', then '...Id'), numbers and keys later; the cheapest shapes first, every API version each
    keys = sorted(_idish_keys(row), key=lambda k: (k.lower() != "id", not k.endswith(("Id", "_id", "ID"))))
    roots = [f"{parts.scheme}://{parts.netloc}/{b}" for b in bases]
    out = []
    for a in keys:
        out += [r + "/{" + a + "}" for r in roots]
        for k in keys:
            if k != a:
                for w in _segments(k):
                    out += [r + "/{" + a + "}/" + w + "/{" + k + "}" for r in roots]
    return list(dict.fromkeys(out))


def _spec_template(tmpl: str, fields: dict) -> str:
    """A template written with the row's own field names -> the spec's: the field the tender id is read from
    becomes {id}; any other row field becomes a list field of its own (a row field literally named 'id' that is not
    the tender id is read as 'recordId')."""
    id_path = (fields.get("id") or {}).get("path")

    def sub(m):
        n = m.group(1)
        if n == id_path:
            return "{id}"
        name = "recordId" if n == "id" else n
        fs = fields.get(name)
        if not (isinstance(fs, dict) and fs.get("path") == n and fs.get("from", "list") == "list"):
            fields[name] = {"from": "list", "path": n}
        return "{" + name + "}"
    return re.sub(r"\{([A-Za-z_][\w]*)\}", sub, tmpl)


def _date_leaves(obj, prefix: str = "", depth: int = 0, out: dict | None = None) -> dict:
    """JMESPath -> value of every date-like string in a JSON answer (objects only; lists are not descended)."""
    out = {} if out is None else out
    if depth > 5 or not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if not (isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)):
            continue
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str) and _ISO_DATE.match(v.strip()):
            out[p] = v
        elif isinstance(v, dict):
            _date_leaves(v, p, depth + 1, out)
    return out


def _compact_row(row: dict) -> dict:
    """A list row for the model: scalars as they are, nested values shortened."""
    out = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False)[:120]
        else:
            out[k] = v
    return out


def _row_kind(r: dict) -> tuple:
    """A row's short code-like values (a notice type, a status, a procedure): what kind of notice it is."""
    return tuple(sorted((k, v) for k, v in r.items() if isinstance(v, str) and re.fullmatch(r"[a-z][a-z_]{1,30}", v)))


def _diverse(rows: list[dict], n: int) -> list[dict]:
    """Up to n rows, at most 3 per combination of short code-like values (a notice type, a status): the newest rows
    of an archive are often all of one kind (e.g. awards), and the model should see the other kinds too."""
    kind = _row_kind
    per: dict = {}
    out = []
    for r in rows:
        k = kind(r)
        if per.get(k, 0) < 3:
            per[k] = per.get(k, 0) + 1
            out.append(r)
        if len(out) >= n:
            break
    return out


async def _sample_list_items(ctx, spec: dict, pages: int = 3) -> list[dict]:
    """Raw rows of the first list pages (several pages: the newest rows of an archive are often all one kind)."""
    lst = spec["list"]
    pag = lst.get("pagination") or {"type": "none"}
    items_all: list[dict] = []
    page_cursor, used = None, set()
    for i in range(pages):
        _u, resp = await runner.fetch_list_page(ctx, spec, i, page_cursor=page_cursor)
        if resp.status_code >= 400:
            break
        body = runner._parse_body("json", resp)
        items, ok = runner._checked_items("json", body, lst.get("items", "")) if body is not None else ([], False)
        if not ok or not items:
            break
        items_all.extend(it for it in items if isinstance(it, dict))
        ptype = pag.get("type", "none")
        if ptype == "none":
            break
        if ptype == "cursor":
            done, why, page_cursor = runner._cursor_step(pag, body, items, i + 1, used)
            if why:
                break
        elif ptype not in ("page", "offset"):
            break
    return items_all


async def _detail_model(ctx, list_url: str, rows: list[dict]) -> tuple[list[str], list[int]]:
    """The cheap model proposes per-tender JSON addresses and picks the rows most likely to be open calls."""
    try:
        res = await ctx.llm.json_call(
            "detail_api",
            "A public tender portal's JSON list API returns rows WITHOUT the submission deadline. The deadline is in a "
            "per-tender JSON endpoint of the same API. Propose up to 6 URL templates for that endpoint, most likely "
            "first, written with {field} placeholders naming top-level fields of the rows (e.g. "
            "https://host/api/v1/items/{id}/details/{noticeId}). Also give the indexes of up to 3 rows that are most "
            "likely calls for competition still taking offers (not awards, cancellations or corrections). Every "
            "candidate is tested; wrong guesses are harmless. JSON: {\"templates\": [..], \"rows\": [..]}",
            json.dumps({"listUrl": list_url, "rows": [_compact_row(r) for r in rows]}, ensure_ascii=False)[:9000],
            portal_id=ctx.portal_id, run_id=ctx.run_id)
    except (ModelBudgetExhausted, LLMUnavailable):
        return [], []
    tmpls = [t for t in (res.get("templates") or []) if isinstance(t, str) and t.startswith(("http://", "https://"))
             and "{" in t][:6]
    picks = [i for i in (res.get("rows") or []) if isinstance(i, int) and 0 <= i < len(rows)][:3]
    return tmpls, picks


async def _deadline_path(ctx, row: dict, dates: dict[str, str]) -> str | None:
    """The cheap model labels which date of a tender's JSON is its submission deadline; None if none is."""
    try:
        res = await ctx.llm.json_call(
            "detail_api",
            "These are the date fields of one public tender's JSON record (path: value). Which ONE path is the deadline "
            "for submitting offers or requests to participate in this tender? Not a publication, question, opening, "
            "award or contract date. If none is, answer null. JSON: {\"path\": string or null, \"why\": short}",
            json.dumps({"tender": _compact_row(row), "dates": dates}, ensure_ascii=False)[:6000],
            portal_id=ctx.portal_id, run_id=ctx.run_id)
    except (ModelBudgetExhausted, LLMUnavailable):
        return None
    p = res.get("path")
    return p if isinstance(p, str) and p in dates else None


async def discover_detail_json(ctx, spec: dict) -> dict | None:
    """A JSON list that lacks the deadline: find the per-tender JSON endpoint by evidence. Candidates come from the
    list address and the rows' id fields (static) and from the cheap model; one is accepted only when its answer for
    a listed row is JSON that names that row's id and holds a parseable date-time the model labels as the deadline.
    Records detail.kind=json + template and fields.submissionDeadline from the detail. When tender pages of listed
    notices show no deadline or a past one, the listing mixes closed notices (list.mixesClosed). Best effort."""
    lst = spec.get("list") or {}
    if lst.get("kind") != "json":
        return None
    fields = spec.setdefault("fields", {})
    dl = fields.get("submissionDeadline") or {}
    if dl.get("from", "list") == "list" and dl.get("path"):
        return None
    id_path = (fields.get("id") or {}).get("path")
    rows = await _sample_list_items(ctx, spec, pages=8)
    if not rows:
        return None
    from .extract import field_from, fill_field_template

    def rid_of(r):
        v = field_from("json", r, "id", fields.get("id") or {})
        return str(v).strip() if v not in (None, "") else None
    sample = _diverse(rows, 30)
    model_tmpls, picks = await _detail_model(ctx, runner.fill_tokens(lst.get("url") or "", spec), sample)
    order = list(dict.fromkeys(picks + list(range(len(sample)))))
    probe_rows = [sample[i] for i in order[:6]]
    static = _detail_candidates_static(runner.fill_tokens(lst.get("url") or "", spec), rows[0])
    probes = 0
    for tmpl in list(dict.fromkeys(model_tmpls + static)):
        valid = False
        tries = list(probe_rows)
        extended = False
        i = 0
        while i < len(tries):
            row = tries[i]
            i += 1
            if valid and i == len(tries) and not extended:
                # The address answers, but the notices tried so far had no deadline (the newest rows of an archive
                # are often all awards): also try one notice of every other kind in the sampled pages.
                extended = True
                kinds = {_row_kind(r) for r in tries}
                freq: dict = {}
                first: dict = {}
                for r in rows:
                    freq[_row_kind(r)] = freq.get(_row_kind(r), 0) + 1
                    first.setdefault(_row_kind(r), r)
                rare = sorted((k for k in first if k not in kinds), key=lambda k: freq[k])   # rarest kinds first
                tries += [first[k] for k in rare[:10]]
            if probes >= DETAIL_MAX_PROBES:
                ctx.note(f"no per-tender JSON endpoint found ({probes} probes)")
                return None
            rid = rid_of(row)
            url = fill_field_template(tmpl, row)       # the row's own fields by name
            if not url or not rid:
                break
            probes += 1
            resp = await ctx.client.get(url)
            if resp.status_code != 200 or "json" not in resp.headers.get("content-type", "").lower():
                break                       # not this address: next candidate
            try:
                data = resp.json()
            except ValueError:
                break
            dump = json.dumps(data, ensure_ascii=False)
            used = [str(row.get(n)) for n in re.findall(r"\{([A-Za-z_][\w]*)\}", tmpl) if row.get(n) is not None]
            if rid not in dump and not any(len(v) >= 4 and v in dump for v in used):
                break                       # an answer that does not name this tender is not its record
            valid = True
            dates = {p: v for p, v in _date_leaves(data).items() if "T" in v or " " in v.strip()}
            path = await _deadline_path(ctx, row, dates) if dates else None
            if not path:
                continue                    # the address works; this notice has no deadline (an award?)
            try:
                timeutil.parse(dates[path])
            except ValueError:
                continue
            tmpl = _spec_template(tmpl, fields)
            spec["detail"] = {"kind": "json", "url_from": "template", "template": tmpl,
                              "discovered": f"{url} names tender {rid} and gives its deadline at {path} "
                                            f"({dates[path]})"}
            fields["submissionDeadline"] = {"from": "detail", "path": path}
            ctx.note(f"found each tender's JSON record at {tmpl}; deadline at {path} ({dates[path]} for {rid})")
            await _closed_evidence(ctx, spec, rows, path, rid_of)
            return {"url": url, "prefer": [rid]}
        if valid:
            ctx.note(f"{tmpl} answers with each tender's record, but none of {i} notices tried showed a deadline")
    ctx.note(f"no per-tender JSON endpoint found ({probes} probes)")
    return None


async def _closed_evidence(ctx, spec: dict, rows: list[dict], path: str, rid_of, n: int = 4) -> None:
    """Open a few more listed notices' records: one past its deadline or without one shows the listing carries
    closed notices too, so a notice whose record is not read yet must not be served as open."""
    from .extract import field_from
    tmpl = spec["detail"]["template"]
    now = timeutil.now()
    seen = {"open": 0, "closed": 0, "none": 0}
    lf = {f: fs for f, fs in spec["fields"].items() if fs.get("from", "list") == "list" and not fs.get("template")}
    for row in rows[:n]:
        rid = rid_of(row)
        raw = {f: field_from("json", row, f, fs) for f, fs in lf.items()}
        url = runner.fill_row_template(tmpl, raw, rid) if rid else None
        if not url:
            continue
        resp = await ctx.client.get(url)
        if resp.status_code != 200:
            continue
        try:
            v = jmespath.search(path, resp.json())
        except (ValueError, jmespath.exceptions.JMESPathError):
            continue
        try:
            d = timeutil.parse(v) if isinstance(v, str) and v.strip() else None
        except ValueError:
            d = None
        seen["none" if d is None else ("closed" if d <= now else "open")] += 1
    if seen["closed"] or seen["none"]:
        spec["list"]["mixesClosed"] = (f"of {sum(seen.values())} listed notices opened, {seen['closed']} were past "
                                       f"their deadline and {seen['none']} had none")
        ctx.note("the listing also carries closed notices: " + spec["list"]["mixesClosed"])


async def _discover_detail_safely(ctx, spec: dict) -> dict | None:
    before = json.loads(json.dumps(spec))
    try:
        got = await discover_detail_json(ctx, spec)
        bad = validate_spec(spec)
        if bad:
            raise ValueError("; ".join(bad[:2]))
        return got
    except (BudgetExceeded, PortalBlocked):
        spec.clear()
        spec.update(before)
        raise
    except Exception as e:  # best effort: the attach goes on with what the model wrote
        spec.clear()
        spec.update(before)
        ctx.note(f"per-tender JSON discovery skipped ({_short(e, 120)})")
        return None


async def _narrow_safely(ctx, spec: dict, mixed: bool = False) -> None:
    """narrow_to_open, then one static page read to prove the narrowed spec still reads tenders; else undo.
    mixed: the dry run already saw closed rows (on any page), so an html list's search form is tried even when
    page 1 itself is all open."""
    before = json.loads(json.dumps(spec))
    try:
        if (spec.get("list") or {}).get("kind", "html") == "html":
            await narrow_html_to_open(ctx, spec, mixed=mixed)   # rows mix open and closed: the portal's search form
        else:
            await narrow_to_open(ctx, spec)
        if spec != before:
            check = await runner.read(ctx, spec, {}, runner.Limits(max_pages=1, max_details=0, max_docs=0,
                                                                   fetch_docs=False))
            if not check.tenders:
                raise ValueError("the narrowed list read no tenders")
    except (BudgetExceeded, PortalBlocked):
        spec.clear()
        spec.update(before)
        raise
    except Exception as e:  # best effort: a failed narrowing must never cost the attach
        spec.clear()
        spec.update(before)
        ctx.note(f"kept the unnarrowed list ({_short(e, 120)})")


async def postback_page_size(ctx, spec: dict, dry) -> None:
    """A postback pager often has a results-per-page <select> (Atexo: 10/20). Find it without a model (numbers only,
    the selected one equal to the rows shown) and keep it only if a 2-page read with it shows more rows and pages on
    from the bigger page. Fewer pages = fewer requests on every refresh."""
    lst = spec.get("list") or {}
    pag = lst.get("pagination") or {}
    if pag.get("type") != "postback" or pag.get("size_sel") or lst.get("kind", "html") != "html":
        return
    _u, resp = await runner.fetch_list_page(ctx, spec, 0)
    if resp.status_code >= 400:
        return
    tree = HTMLParser(resp.text or "")
    from .extract import list_items
    shown = len(list_items("html", tree, lst.get("items", "")))
    before = int((dry.stats or {}).get("rows") or 0)
    for sel in postback.size_select_candidates(tree, shown)[:2]:
        pag["size_sel"] = sel
        check = await runner.read(ctx, spec, {}, runner.Limits(max_pages=2, max_details=0, max_docs=0,
                                                               fetch_docs=False))
        st = check.stats or {}
        if int(st.get("rows") or 0) > before and (check.complete or st.get("stopReason") == "max_pages"):
            ctx.note(f"results per page raised with {sel}: 2 pages show {st.get('rows')} rows (was {before})")
            return
        pag.pop("size_sel", None)
    if shown:
        ctx.note("no working results-per-page control; reading at the portal's default page size")


async def _postback_size_safely(ctx, spec: dict, dry) -> None:
    before = json.loads(json.dumps(spec))
    try:
        await postback_page_size(ctx, spec, dry)
    except (BudgetExceeded, PortalBlocked):
        spec.clear()
        spec.update(before)
        raise
    except Exception as e:  # best effort: the default page size still reads everything
        spec.clear()
        spec.update(before)
        ctx.note(f"page size unchanged ({_short(e, 120)})")


def _paging_ignored_feedback(spec: dict) -> str:
    lst = spec.get("list") or {}
    pag = lst.get("pagination") or {}
    msg = (f"list page 2 returned the same rows as page 1: the portal ignores pagination "
           f"'{pag.get('param') or pag.get('body_path')}'")
    if (lst.get("method") or "GET").upper() == "POST" and isinstance(lst.get("body"), dict) and not pag.get("body_path"):
        msg += ("; this API is a POST search: the page number/offset is a field INSIDE list.body - give its dotted "
                "path as pagination.body_path (and the size field as size_body_path)")
    elif pag.get("type") in ("page", "offset"):
        msg += ("; if the API answers with a next-page token or 'last item' key instead of page numbers, use "
                "pagination type cursor")
    return msg


async def _next_endpoint(ctx, found: dict) -> bool:
    """Switch to the next captured JSON endpoint that replays and looks like tenders."""
    while found.get("untried"):
        nxt = found["untried"].pop(0)
        url, meta = (nxt["url"], nxt.get("meta")) if isinstance(nxt, dict) else (nxt, None)
        data, req = await replay_request(ctx, url, meta)
        if data is not None and _score_json(data) > 0:
            found["api"] = (url, data)
            found["apiRequest"] = req
            ctx.note(f"trying the next JSON endpoint: {(req or {}).get('method', 'GET')} {url}")
            return True
    return False


def _label_feedback(problems: list[str], label_hints: dict[str, list[str]]) -> list[str]:
    """Labels the verifier saw, collected across attempts (label_hints is updated). Tender pages of one portal
    often label a value differently (e.g. per procedure type); fixing one label per attempt would flip-flop
    between them, so once there are several, ask for all of them at once."""
    for p in problems:
        for lab, fld in re.findall(r"under the label '([^']+)': map (\w+)", p):
            if lab not in label_hints.setdefault(fld, []):
                label_hints[fld].append(lab)
    return [f"the tender pages label {fld} differently ({', '.join(repr(x) for x in labs)}): give all of them, "
            f"e.g. {{\"from\": \"detail\", \"label\": {json.dumps(labs, ensure_ascii=False)}}}"
            for fld, labs in label_hints.items() if len(labs) > 1]


MAX_LOCATION_SWITCHES = 2
EXTRA_TRIES = 2          # more attempts for cheap, fixable problems while the job's model-call budget allows
CALLS_PER_TRY = 4        # model calls one more attempt may need (2 synthesize + open-filter / verify calls)


def _coverage_only(problems: list[str]) -> list[str]:
    """The problems when they are only about coverage (the spec reads tenders correctly but does not follow the
    pager): a working read with incomplete coverage, which Patterno's rule A serves as partial. [] otherwise."""
    core = [p for p in problems if not p.startswith("hint:")]
    # only a spec that never pages ("single page"); a pager that ends early is breakage evidence, not coverage
    ok = core and all("the portal states" in p and "and stopped (single page)" in p for p in core)
    return core if ok else []


def _calls_left(ctx) -> int | None:
    fn = getattr(ctx.llm, "calls_left", None)
    try:
        return fn(ctx.run_id) if callable(fn) else None
    except Exception:
        return None


def _accept_notes(ctx, spec: dict, found: dict, attempt: int) -> None:
    spec["verifiedBlind"] = not spec.get("verification")
    spec["learnedFrom"] = found["url"]
    if found.get("captcha"):
        spec["captchaShownButIgnored"] = True
    ctx.note(f"spec accepted after {attempt} attempt(s)")
    ctx.note("spec: " + json.dumps({k: spec.get(k) for k in ("list", "fields", "detail", "documents")},
                                   ensure_ascii=False)[:4000])


async def learn(ctx, start_url: str, old_spec: dict | None = None, failures: list[str] | None = None,
                hints: list[dict] | None = None, max_tries: int = 3) -> dict:
    """locate -> synthesize (spec validation + first probe) -> dry run -> blind verify, in one retry loop:
    any malformed spec, failing probe or verification problem becomes concise feedback for the next attempt.
    A place that yields no readable tenders (after a retry, a browser look or the other captured endpoints) is
    replaced by the locate walk's next-ranked candidate, which gets attempts of its own."""
    found = await locate_list(ctx, start_url)
    feedback = list(failures or [])
    last_problems: list[str] = []
    label_hints: dict[str, list[str]] = {}
    renders = {"left": VERIFY_RENDERS}      # browser renders of tender pages for blind verification, this learn
    attempt, tries, switches = 0, max_tries, 0
    known_total = None      # (list url, list.total) of an earlier attempt whose dry run read the portal's total
    best: dict | None = None    # the attempt that read the most rows (fewest problems): refined, not started over
    while attempt < tries:
        attempt += 1
        spec = dry = None
        problems: list[str] = []
        try:
            if found.get("feed"):
                spec = await synthesize_feed(ctx, found, feedback, old_spec)
            else:
                spec = await synthesize(ctx, found, feedback, old_spec, hints,
                                        best=best if best and best.get("found") is found else None)
                pager = static_pager(spec, found.get("html") or "", found.get("url") or "") \
                    if not found.get("api") else None
                if pager:
                    ctx.note(pager)
            lst = spec.get("list") or {}
            if known_total and not lst.get("total") and lst.get("url") == known_total[0]:
                # The stated total is what judges a full read complete; a rewrite that drops it would let a read
                # that stops early pass as complete. Keep the selector that read it (the dry run re-checks it).
                lst["total"] = json.loads(json.dumps(known_total[1]))
                ctx.note("kept the earlier attempt's stated-total selector (this attempt gave none)")
            if runner.mixes_closed(spec):
                # most listed notices may be closed: also open the records the discovery found with a deadline
                dry = await runner.read(ctx, spec, {}, runner.Limits(
                    max_pages=5, max_details=6, max_docs=2, prefer_ids=frozenset(found.get("preferIds") or [])))
            else:
                # a deadline read from the tender page is only known for the pages opened: open enough of them
                dl_src = ((spec.get("fields") or {}).get("submissionDeadline") or {}).get("from")
                # (through a linked page every tender costs two pages: the cap counts pages)
                cap = 12 if runner.follow_spec(spec) else 8 if dl_src == "detail" else 5
                dry = await runner.read(ctx, spec, {}, runner.Limits(max_pages=2, max_details=cap, max_docs=2))
            if (dry.stats or {}).get("statedTotal") is not None and lst.get("total"):
                known_total = (lst.get("url"), json.loads(json.dumps(lst["total"])))
        except (PortalBlocked, ModelBudgetExhausted, LLMUnavailable, BudgetExceeded):
            raise
        except SpecError as e:
            problems = [f"spec problem: {str(e)[:400]}"]
            if e.spec is not None:      # what was tried, for whoever reads the run log
                ctx.note(f"attempt {attempt} spec (rejected): " + json.dumps(
                    {"list": e.spec.get("list"), "fields": e.spec.get("fields")}, ensure_ascii=False)[:3000])
        except AttachFailed as e:
            problems = [f"spec problem: {str(e)[:300]}"]
        except Exception as e:
            # e.g. a selector the parser rejects at run time: named, so the next attempt can fix it
            problems = [_run_error_problem(e, spec)]
        if spec is not None and not problems:
            opened = set(dry.verified_ids)
            if not dry.tenders and runner.mixes_closed(spec):
                problems.append(f"no listed notice read in the dry run has an open deadline on its tender record "
                                f"({(dry.stats or {}).get('noDeadline', 0)} without one, "
                                f"{(dry.stats or {}).get('closedDropped', 0)} past it)")
            elif not dry.tenders and (dry.stats or {}).get("closedDropped"):
                problems.append(f"the first list pages only show closed tenders ({dry.stats['closedDropped']} past "
                                f"their deadline): use the portal's filter or sort for open/current tenders")
            elif not dry.tenders:
                problems.append("the spec found no tenders on the list page")
            st = dry.stats or {}
            if not dry.complete and dry.tenders and "repeated earlier results" in str(st.get("stopReason") or ""):
                problems.append(_paging_ignored_feedback(spec))
            # A dry run is cut at 2 pages (capped); one that ended by itself short of the stated total did not follow
            # the pager (no pagination, a parameter the portal ignores, a javascript: next link, a broken postback).
            if (dry.complete or not st.get("capped")) and st.get("statedTotal") \
                    and int(st.get("rows") or 0) < 0.95 * st["statedTotal"]:
                problems.append(f"the portal states {st['statedTotal']} results but the spec read {st.get('rows')} "
                                f"and stopped ({st.get('stopReason')}): the list has more pages - map its pagination "
                                f"(page/offset parameter, next link, postback for a pager that posts a form, form "
                                f"for a script that sets a page field, load_more for rows a script request appends)")
                hint = pager_notes(found.get("html") or "")
                if hint:
                    problems.append(hint)

            def pool(f):  # detail-sourced fields are only known for tenders whose page the dry run opened
                src = (spec.get("fields", {}).get(f) or {}).get("from", "list")
                return [t for t in dry.tenders if t.get("id") in opened] if src in runner.DETAIL_SOURCES \
                    else dry.tenders
            examples = st.get("missingExamples") or {}
            for f in ("id", "url", "title"):
                ts = pool(f)
                miss = sum(1 for t in ts if not t.get(f))
                if miss:
                    problems.append(f"{miss} of {len(ts)} tenders have no {f}"
                                    + (f"; a row without it (the selector must also cover rows like this one): "
                                       f"{examples[f]}" if f in examples else ""))
            errs = [e for e in st.get("errors") or [] if e.startswith(("linked page", "detail"))]
            if runner.follow_spec(spec) and dry.tenders and not st.get("followRead"):
                problems.append(f"no tender's linked page (detail.follow) could be read in the dry run "
                                f"({st.get('followMissing', 0)} tender pages without the link"
                                + (f"; {errs[0]}" if errs else "") + ")" + _hop_hint(found, spec))
            elif dry.tenders and not opened and any(
                    isinstance(fs, dict) and fs.get("from") in runner.DETAIL_SOURCES
                    for f, fs in (spec.get("fields") or {}).items() if f in REQUIRED):
                # nothing read from where the spec says these fields are: they cannot be judged, never a pass
                problems.append("no tender page could be read in the dry run" + (f" ({errs[0]})" if errs else ""))
            for f in ("buyerName", "submissionDeadline", "publicationDate"):
                ts = pool(f)
                if ts and all(t.get(f) is None for t in ts):
                    problems.extend(_never_found(spec, f, ts, found))
            if not problems and found.get("feed"):
                # A feed holds every field in its entries: the static checks are the verification (no tender page is
                # opened: the portal's website may not be readable at all - that is why the feed is used).
                problems = feed_dry_problems(spec, dry)
                if not problems:
                    spec["learnedFrom"] = found["url"]
                    ctx.note(f"feed spec accepted after {attempt} attempt(s)")
                    ctx.note("spec: " + json.dumps({k: spec.get(k) for k in ("list", "fields")},
                                                   ensure_ascii=False)[:4000])
                    return spec
            elif not problems:
                problems = await blind_verify(ctx, spec, dry.tenders, opened=opened, renders=renders)
                if not problems:
                    # dry-run pages that mix open and closed rows: look for the open filter even when page 1 is
                    # all open (before a full read walks into the archive and is judged capped or incomplete)
                    await _narrow_safely(ctx, spec, mixed=bool(st.get("closedDropped")))
                    await _discover_docs_safely(ctx, spec, dry.tenders)
                    await _postback_size_safely(ctx, spec, dry)
                    # only a comparison that happened counts; spec["verification"] says why none could
                    _accept_notes(ctx, spec, found, attempt)
                    return spec
        rows_read = int((dry.stats or {}).get("rows") or len(dry.tenders)) if dry is not None else -1
        if spec is not None and (best is None or best.get("found") is not found
                                 or (rows_read, -len(problems)) > (best["rows"], -len(best["problems"]))):
            best = {"spec": json.loads(json.dumps(spec)), "rows": rows_read, "problems": list(problems),
                    "found": found, "dry": dry, "attempt": attempt}
        if attempt == tries and tries < max_tries + EXTRA_TRIES and problems \
                and not any("no tenders" in p or "could not be run" in p for p in problems):
            left = _calls_left(ctx)
            if left is not None and left >= CALLS_PER_TRY:
                tries += 1
                ctx.note(f"one more attempt for a fixable problem ({left} model calls left in this job)")
        ctx.note(f"attempt {attempt} problems: {problems}")
        if spec is not None:   # what was tried, for whoever reads the run log
            ctx.note(f"attempt {attempt} spec: " + json.dumps(
                {"list": spec.get("list"), "fields": spec.get("fields")}, ensure_ascii=False)[:3000])
        last_problems = problems
        feedback = list(problems)
        feedback.extend(_label_feedback(problems, label_hints))
        if any("missing although the page shows" in p or "invalid CSS selector" in p or "never" in p
               for p in problems):
            feedback.append("hint: CSS cannot match text (no :contains/:matches/:-soup-contains/:has). When a value "
                            "sits next to a label (e.g. 'Offers due' then '03.02.2027 12:00'), give the label "
                            "instead of a selector: {\"from\": \"detail\", \"label\": \"Offers due\", \"formats\": "
                            "[\"%d.%m.%Y %H:%M\"]}; in a list table, give the column header: {\"from\": \"list\", "
                            "\"column\": \"Offers due\"}. Words after the time (a time-zone name, 'local time') "
                            "are fine.")
        # a list whose tender links the portal refuses to us (401/403) is not readable here either: after a retry,
        # the walk's next candidate (e.g. Find a Tender: notice search -> 403, opportunities -> readable)
        refused = any(re.search(r"the tender link \S+ returned HTTP 40[13]\b", p) for p in problems)
        if not refused and not any("no tenders" in p or "could not be run" in p for p in problems):
            continue
        found["noRowFails"] = found.get("noRowFails", 0) + 1
        # Static HTML without rows: the list is probably filled by JavaScript -> look for its data in a browser.
        # (Not when the walk saw the rows in the static HTML: then the spec is wrong, not the page.)
        if (not found.get("api") and not found.get("browserTried") and not found.get("staticRows")
                and not found.get("feed")
                and any("no tenders" in p for p in problems)):
            found["browserTried"] = True
            from . import browser
            cap = await browser.capture(ctx, found["url"])
            ctx.note(f"browser on list page: {cap.get('requests')} requests, {len(cap['json'])} JSON responses")
            for ep in captured_endpoints(cap)[:3]:
                replay, req = await replay_request(ctx, ep["url"], ep["meta"])
                if replay is not None:
                    found["api"] = (ep["url"], replay)
                    found["apiRequest"] = req
                    how = (req or {}).get("method", "GET")
                    feedback = [f"the HTML list had no rows; the page loads its tenders from this JSON API instead: "
                                f"{how} {ep['url']}"]
                    ctx.note(f"found JSON API via browser: {how} {ep['url']}")
                    break
            if found.get("api"):
                continue
        # An endpoint that yields no readable tenders may simply be the wrong one: try the next captured one.
        elif found.get("api") and await _next_endpoint(ctx, found):
            feedback = [f"the previous JSON endpoint did not work ({problems[0][:120]}); a different endpoint is given"]
            continue
        # Still nothing readable here after a retry or a browser look: the walk's next-ranked candidate.
        if (found["noRowFails"] >= 2 or found.get("browserTried")) and switches < MAX_LOCATION_SWITCHES:
            from . import locate
            nxt = await locate.next_candidate(ctx, found)
            if nxt is not None:
                switches += 1
                ctx.note(f"no readable tenders at {found['url']}; trying the next located candidate {nxt['url']}"
                         f" (tier {nxt.get('tier')}: {nxt.get('evidence')})")
                feedback = list(failures or []) + [
                    f"another page ({found['url']}) was tried first and had no readable tenders ({problems[0][:160]}); "
                    f"this is a different page"]
                found = nxt
                label_hints = {}            # labels seen on the other page's tenders do not apply here
                tries = max(tries, attempt + 2)
    partial = await _accept_partial(ctx, best, renders) if old_spec is None else None   # attach only: a repair
    if partial is not None:                                                             # must not hide breakage
        return partial
    raise AttachFailed("could not learn a trustworthy way to read this portal: " + "; ".join(last_problems[:3]))


async def _accept_partial(ctx, best: dict | None, renders: dict) -> dict | None:
    """Patterno rule A, last resort: the best attempt reads tenders correctly and only misses pages (its pager was
    not followed). Blind-verified like any spec, it is served as a CAPPED listing (list.partial: every read says
    'Coverage: partial' and closes nothing) instead of failing the attach."""
    core = _coverage_only((best or {}).get("problems") or [])
    dry = (best or {}).get("dry")
    if not core or dry is None or not dry.tenders:
        return None
    spec = json.loads(json.dumps(best["spec"]))
    try:
        problems = await blind_verify(ctx, spec, dry.tenders, opened=set(dry.verified_ids), renders=renders)
    except (ModelBudgetExhausted, LLMUnavailable) as e:
        ctx.note(f"partial listing not verified ({_short(e, 100)})")
        return None
    if problems:
        ctx.note(f"partial listing not accepted: blind verification found {problems[:2]}")
        return None
    spec["list"]["partial"] = f"the pager is not followed ({core[0][:160]})"
    await _discover_docs_safely(ctx, spec, dry.tenders)
    ctx.note(f"serving the verified spec of attempt {best['attempt']} as a partial listing: {core[0][:200]}")
    _accept_notes(ctx, spec, best["found"], best["attempt"])
    return spec
