"""Static runner: executes a spec. Regular reads make NO model call.

Incremental refresh (architecture v1.1): every read walks all list pages; detail pages are fetched only
for new tenders, tenders whose list row changed, and a small rotating sample; documents are fetched only
when a tender is new or its document links changed.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit, parse_qsl

import httpx
import jmespath
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import jspager, postback
from .extract import REQUIRED, assemble, field_from, list_items, html_value, json_value, reduce_html
from .http import WEB_PAGE_REASON, BudgetExceeded, DocumentBlocked
from .pipeline import PortalBlocked, ReadResult

BLOCK_MARKERS_DEFAULT = ["input[type=password]"]
LIST_RETRY_PAUSE = 3.0     # seconds before the single retry of a list page that failed in transport
CAPTCHA_PAT = re.compile(r"recaptcha|hcaptcha|turnstile|captcha", re.I)


@dataclass
class Limits:
    max_pages: int | None = None        # cap list pages (dry runs)
    max_details: int | None = None      # cap detail fetches (dry runs)
    max_docs: int | None = None         # cap document fetches (dry runs)
    sample: int = 5                     # rotating re-check of known tenders
    fetch_docs: bool = True
    prefer_ids: frozenset = frozenset()  # open these tenders' pages first (dry runs: notices known to be calls)


def _with_params(url: str, params: dict) -> str:
    parts = urlsplit(url)
    if not params:
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))   # exactly as written
    # keep_blank_values: flag parameters like '&AllCons' (Atexo: "all consultations") are the whole list.
    # Repeated parameters (lang=de&lang=fr) stay repeated unless a param replaces them: they are part of the query;
    # a list value is a repeated parameter too (several ticked checkboxes of one search form: status=a&status=b).
    new = {k: v for k, v in params.items() if v is not None}

    def vals(v):
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]
    pairs, done = [], set()
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        if k in new:
            if k not in done:
                pairs += [(k, x) for x in vals(new[k])]
                done.add(k)
            continue
        pairs.append((k, v))
    for k, v in new.items():
        if k not in done:
            pairs += [(k, x) for x in vals(v)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), ""))


def today_local(spec: dict) -> str:
    """Today's date in the portal's timezone (ISO), for '{today}' in list.url / list.params."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    from . import timeutil
    try:
        tz = ZoneInfo(spec.get("tz") or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    return timeutil.now().astimezone(tz).date().isoformat()


TODAY_TOKEN_RE = re.compile(r"\{today(?::([^{}]+))?\}")
_TODAY = TODAY_TOKEN_RE


def fill_tokens(value, spec: dict):
    """A spec is written once but read every day: '{today}' keeps a 'deadline from today' filter current.
    '{today:%d/%m/%Y}' writes the date the way a search form expects it (strftime format)."""
    if isinstance(value, list):
        return [fill_tokens(v, spec) for v in value]
    if isinstance(value, str) and "{today" in value:
        from datetime import date as _date
        iso = today_local(spec)

        def sub(m):
            fmt = m.group(1)
            if not fmt:
                return iso
            try:
                return _date.fromisoformat(iso).strftime(fmt)
            except (ValueError, TypeError):
                return m.group(0)
        return TODAY_TOKEN_RE.sub(sub, value)
    return value


def fill_tokens_deep(value, spec: dict):
    """fill_tokens inside a JSON request body (dicts and lists, any depth)."""
    if isinstance(value, dict):
        return {k: fill_tokens_deep(v, spec) for k, v in value.items()}
    if isinstance(value, list):
        return [fill_tokens_deep(v, spec) for v in value]
    return fill_tokens(value, spec)


# ------------------------------------------------------------------ paging inside a POST body, and cursor paging
BODY_PATH_RE = re.compile(r"^[A-Za-z_$][\w$-]*(?:\.[A-Za-z_$][\w$-]*)*$")


def set_body_path(body: dict, path: str, value) -> None:
    """Write value at a dotted path ("PAGE.number") of a JSON body, creating missing objects on the way."""
    keys = path.split(".")
    cur = body
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _page_value(pag: dict, page_index: int):
    if pag.get("type") == "page":
        return pag.get("start", 1) + page_index * pag.get("step", 1)
    if pag.get("type") == "offset":
        return int(pag.get("start", 0)) + page_index * int(pag.get("size", 20))
    return None


def list_body(spec: dict, page_index: int = 0, page_cursor=None):
    """The JSON body of a POST list request for one page: list.body as captured from the portal's own search, with
    {today} filled in and the page number / offset / cursor written at pagination.body_path (e.g. "PAGE.number"),
    the page size at pagination.size_body_path. None when the spec has no body."""
    lst = spec["list"]
    if lst.get("body") is None:
        return None
    body = fill_tokens_deep(json.loads(json.dumps(lst["body"])), spec)
    pag = lst.get("pagination") or {}
    if not isinstance(body, dict):
        return body
    bp = pag.get("body_path")
    if bp and pag.get("type") in ("page", "offset"):
        set_body_path(body, bp, _page_value(pag, page_index))
    elif bp and pag.get("type") == "cursor" and page_cursor is not None:
        set_body_path(body, bp, page_cursor)
    if pag.get("size_body_path") and pag.get("size"):
        set_body_path(body, pag["size_body_path"], int(pag["size"]))
    return body


def _key_present(body, path: str) -> bool:
    """A plain dotted path whose last key really exists (with whatever value): an explicit 'no next cursor'."""
    if not path or not BODY_PATH_RE.match(path):
        return False
    cur = body
    keys = path.split(".")
    for k in keys[:-1]:
        if not isinstance(cur, dict) or not isinstance(cur.get(k), dict):
            return False
        cur = cur[k]
    return isinstance(cur, dict) and keys[-1] in cur


def next_cursor(pag: dict, body, items: list):
    """Cursor ("keyset", "load more") paging: the value to send for the next page, and what the response says.
    Returns (cursor, "next"), (None, "end") when the portal itself states there is no more (more_path is false, or
    the cursor field is present but null/empty), or (None, "missing") when the cursor is not where the spec says -
    a changed response shape, never evidence that the list ended.
    The cursor is read from the response (pagination.cursor_path, JMESPath on the whole body) or from the last row
    (pagination.cursor_from_last, JMESPath on the last item, for APIs that page by 'after this id/date')."""
    more = pag.get("more_path")
    if more:
        try:
            if jmespath.search(more, body) is False:
                return None, "end"
        except jmespath.exceptions.JMESPathError:
            return None, "missing"
    if pag.get("cursor_from_last"):
        if not items:
            return None, "end"
        try:
            val = jmespath.search(pag["cursor_from_last"], items[-1])
        except jmespath.exceptions.JMESPathError:
            return None, "missing"
        return (val, "next") if val not in (None, "") and not isinstance(val, (dict, list)) else (None, "missing")
    path = pag.get("cursor_path") or ""
    try:
        val = jmespath.search(path, body) if path else None
    except jmespath.exceptions.JMESPathError:
        return None, "missing"
    if val not in (None, "") and not isinstance(val, (dict, list, bool)):
        return val, "next"
    return (None, "end") if val in (None, "") and _key_present(body, path) else (None, "missing")


def row_deadline(spec: dict, raw: dict):
    """The parsed submission deadline of a raw list row, or None (not on the list / not parseable)."""
    from . import timeutil
    from .extract import convert
    fs = (spec.get("fields") or {}).get("submissionDeadline") or {}
    try:
        d = convert("submissionDeadline", raw.get("submissionDeadline"), fs, spec.get("tz") or "UTC", "")
        return timeutil.parse(d) if d else None
    except (ValueError, TypeError):
        return None


def _row_link(node, rid, title) -> str | None:
    """The row's link to its tender page when the spec's url selector found none: an <a href> that is an address
    (not javascript:, mailto: or a bare #); prefer the one whose text matches the title, then one whose address
    contains the id, then the only one. Several unrelated links and no match -> None (no guessing)."""
    try:
        anchors = node.css("a[href]")
    except Exception:
        return None
    links = []
    for a in anchors:
        href = (a.attributes.get("href") or "").strip()
        low = href.lower()
        if not href or low.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        links.append((href, " ".join((a.text() or "").split()).lower()))
    if not links:
        return None
    t = " ".join(str(title or "").split()).lower()
    if t:
        for href, text in links:
            if text and (text in t or t in text) and len(text) >= 8:
                return href
    if rid:
        for href, _text in links:
            if str(rid) in href:
                return href
    distinct = {h for h, _ in links}
    return links[0][0] if len(distinct) == 1 else None


def _merge_notice(spec: dict, kept: dict, new: dict) -> None:
    """Two list rows with the same tender id: keep the newer notice's values (by publication date; on a tie the
    first one listed), and keep 'withdrawn' if either notice says so."""
    from .extract import convert
    fs = (spec.get("fields") or {}).get("publicationDate") or {}

    def pub(r):
        try:
            return convert("publicationDate", r.get("publicationDate"), fs, spec.get("tz") or "UTC", "") or ""
        except (ValueError, TypeError):
            return ""
    from .extract import notice_status
    nfs = (spec.get("fields") or {}).get("noticeType") or {}
    withdrawn = (kept.get("withdrawn") or new.get("withdrawn")
                 or any(notice_status(r.get("noticeType"), nfs) == "withdrawn" for r in (kept, new)))
    notices = int(kept.get("_notices") or 1) + 1
    nids = [n for n in (kept.get("_noticeIds") or [kept.get("noticeId")]) + [new.get("noticeId")] if n is not None]
    nids = list(dict.fromkeys(str(n) for n in nids))
    if pub(new) > pub(kept):
        kept.clear()
        kept.update(new)
    if withdrawn:
        kept["withdrawn"] = withdrawn
    kept["_notices"] = notices
    if nids:
        kept["_noticeIds"] = nids


def _is_closed(deadline_iso, now) -> bool:
    from . import timeutil
    if not deadline_iso:
        return False
    try:
        return timeutil.parse(deadline_iso) <= now
    except ValueError:
        return False   # an unparseable deadline is validate()'s business, not a reason to drop the row


def detect_block(html: str, status: int, markers: list[str] | None = None, use_default: bool = False) -> str | None:
    """Return a human reason if the page is a login/captcha wall, else None.
    A password box alone is not a block (many portals have a login form in the header next to public lists);
    only spec-provided markers, or 401/403, count during reads."""
    if status in (401, 403):
        return f"access denied (HTTP {status}); a login is required"
    tree = HTMLParser(html or "")
    for sel in (markers or []) + (BLOCK_MARKERS_DEFAULT if use_default else []):
        try:
            if tree.css_first(sel):
                return "login required to view the tender list"
        except Exception:
            continue
    return None


def has_captcha(html: str) -> bool:
    return bool(CAPTCHA_PAT.search(html or ""))


async def fetch_list_page(ctx, spec: dict, page_index: int, cursor: str | None = None, post=None,
                          referer: str | None = None, page_cursor=None):
    """One list page. `cursor` = a next link's address, followed like a browser follows it: with the page it was
    on as Referer (stateful apps, e.g. Wicket listener links, answer 403 without one); `post` = a
    postback.Postback (a pager that submits a form); `page_cursor` = the value of a cursor parameter (cursor
    paging: keyset / "load more" APIs)."""
    if post is not None:
        resp = await ctx.client.request(post.method, post.url, content=post.content, headers=post.headers)
        return post.url, resp
    lst = spec["list"]
    pag = lst.get("pagination") or {"type": "none"}
    params = {k: fill_tokens(v, spec) for k, v in (lst.get("params") or {}).items()}
    ptype = pag.get("type", "none")
    in_body = bool(pag.get("body_path"))      # the page/offset/cursor goes into the POST body (list_body)
    if ptype == "page" and not in_body:
        params[pag.get("param", "page")] = pag.get("start", 1) + page_index * pag.get("step", 1)
    elif ptype == "offset" and not in_body:
        size = int(pag.get("size", 20))
        params[pag.get("param", "offset")] = int(pag.get("start", 0)) + page_index * size
        if pag.get("size_param"):
            params[pag["size_param"]] = size
    elif ptype == "cursor" and not in_body and page_cursor is not None:
        params[pag.get("param", "cursor")] = page_cursor
    url = cursor or _with_params(fill_tokens(lst["url"], spec), params)
    method = (lst.get("method") or "GET").upper()
    if method == "POST" and isinstance(lst.get("form"), dict):
        if cursor:      # the result page's own next link (it carries the search) is followed like a browser does
            resp = await ctx.client.get(url, headers={"Referer": referer} if referer else None)
            return url, resp
        # the portal's own search form, submitted the way a browser submits it (form-encoded; {today} filled in)
        form = {k: fill_tokens(v, spec) for k, v in lst["form"].items()}
        resp = await ctx.client.request("POST", url, content=urlencode(form, doseq=True),
                                        headers={"Content-Type": "application/x-www-form-urlencoded"})
    elif method == "POST":
        body = list_body(spec, page_index, page_cursor)
        resp = await ctx.client.request("POST", url, json=body) if body is not None \
            else await ctx.client.request("POST", url)
    else:
        resp = await ctx.client.get(url, headers={"Referer": referer} if cursor and referer else None)
    return url, resp


def _parse_body(kind: str, resp):
    if kind == "json":
        try:
            return resp.json()
        except (json.JSONDecodeError, ValueError):
            return None
    return HTMLParser(resp.text)


def _checked_items(kind: str, body, expr: str) -> tuple[list, bool]:
    """List items plus whether the items path was valid: an empty array is a legitimate empty page, a missing or
    non-array value is not."""
    if kind == "json":
        if not expr:
            return (body, True) if isinstance(body, list) else ([], False)
        try:
            val = jmespath.search(expr, body)
        except jmespath.exceptions.JMESPathError:
            return [], False
        return (val, True) if isinstance(val, list) else ([], False)
    return list_items(kind, body, expr), True


def _stated_total(kind: str, body, spec: dict) -> int | None:
    tot = spec["list"].get("total")
    if not tot or body is None:
        return None
    val = json_value(body, tot) if kind == "json" else html_value(body, tot)
    if val is None:
        return None
    m = re.search(r"\d[\d.,]*", str(val))
    return int(re.sub(r"[.,]", "", m.group(0))) if m else None


def _row_evidence(node) -> str:
    """A list row the spec could not read, compact enough for a model prompt: its structure, then its links (the
    usual culprit: some rows link elsewhere, e.g. to the tender on another platform)."""
    links = []
    for a in node.css("a[href]"):
        h = (a.attributes.get("href") or "").strip()
        if not h or h.startswith("#") or h.lower().startswith("javascript:"):
            continue
        par = a.parent
        title = (par.attributes.get("title") or "") if par is not None else ""
        where = (par.tag if par is not None else "?") + (f" title='{title[:70]}'" if title else "")
        links.append(f"<a class='{a.attributes.get('class') or ''}' href='{h[:160]}'> in <{where}>")
    return (reduce_html(node.html or "", 900) + " | links in this row: "
            + "; ".join(dict.fromkeys(links)))[:2200]


def _drift(total: int) -> int:
    """Rows a live listing may move while every one of its pages is walked (1 + 0.5 %): 2 of 402, 9 of 1,640."""
    return 1 + total // 200


async def _postback_resize(ctx, spec: dict, resp, body):
    """Postback pager with a results-per-page control (pagination.size_sel): choose its largest option once, so the
    walk needs fewer pages. Kept only if the answer shows more rows; otherwise the default page stays."""
    lst = spec["list"]
    req = postback.resize_request(body, resp.text, str(resp.url), resp.encoding, lst["pagination"])
    if req is None:
        return None
    r2 = await ctx.client.request(req.method, req.url, content=req.content, headers=req.headers)
    if r2.status_code >= 400 or detect_block(r2.text, r2.status_code, spec.get("block_markers")):
        ctx.note(f"the page-size control did not work (HTTP {r2.status_code}); reading at the default size")
        return None
    b2 = HTMLParser(r2.text)
    n_before, n_after = len(list_items("html", body, lst["items"])), len(list_items("html", b2, lst["items"]))
    if n_after <= n_before:
        ctx.note(f"the page-size control showed {n_after} rows (default {n_before}); reading at the default size")
        return None
    return r2, b2


INCOMPLETE_SHARE = 0.05        # at most this share of rows may be left out for lacking id/url/title
NOT_A_CALL = "not an open call"   # known_closed.why of a notice whose type is an award, result or prior information
# A new open tender without a deadline is suspicious when most tenders of the read show one: it is not served until
# its tender page shows a deadline. Judged on tenders whose deadline is known to be read (list, or tender page read).
DEADLINE_EXPECTED_SHARE = 0.5
DEADLINE_EXPECTED_MIN = 5

PUBLISHED_WINDOW_DAYS = int(os.environ.get("WATCHTOWER_PUBLISHED_WINDOW_DAYS", "365"))


def _sorted_list_finished(spec: dict, page_rows: list[dict]) -> str | None:
    """Early, evidence-based end of a SORTED listing (spec.list.order), so huge feeds need not be walked fully.
    deadline_desc: a whole page whose deadlines have all passed -> everything after it is closed too.
    published_desc: a whole page that is closed AND older than the publication window -> the rest is older still."""
    order = (spec.get("list") or {}).get("order")
    if order not in ("deadline_desc", "published_desc") or not page_rows:
        return None
    from datetime import timedelta
    from . import timeutil
    from .extract import convert
    tz = spec.get("tz") or "UTC"
    fields = spec.get("fields", {})
    now = timeutil.now()
    deadlines, pubs = [], []
    for r in page_rows:
        d = convert("submissionDeadline", r.get("submissionDeadline"), fields.get("submissionDeadline", {}), tz, "")
        p = convert("publicationDate", r.get("publicationDate"), fields.get("publicationDate", {}), tz, "")
        deadlines.append(timeutil.parse(d) if d else None)
        pubs.append(p)
    if any(d is None for d in deadlines) or not all(d <= now for d in deadlines):
        return None  # only stop on positive evidence for every row on the page
    if order == "deadline_desc":
        return "sorted by deadline: this page and everything after it are closed"
    cutoff = (now - timedelta(days=PUBLISHED_WINDOW_DAYS)).date().isoformat()
    if all(p and p < cutoff for p in pubs):
        return f"sorted by publication date: this page is closed and older than {PUBLISHED_WINDOW_DAYS} days"
    return None


def _cursor_step(pag: dict, body, items: list, pages_read: int, used: set[str]):
    """One cursor-paging decision after a non-empty page: (complete, stop reason or None, next cursor).
    The list only counts as complete on positive evidence (the portal says there is no next cursor, or a page shorter
    than the declared size); a missing or repeated cursor stops the read as incomplete."""
    nxt, state = next_cursor(pag, body, items)
    if state == "end":
        return True, "the portal gives no next cursor", None
    if state == "missing":
        if pag.get("size") and len(items) < int(pag["size"]):
            return True, "short page", None
        where = pag.get("cursor_path") or pag.get("cursor_from_last") or "?"
        return False, f"page {pages_read} has no next cursor at '{where}'", None
    key = json.dumps(nxt, default=str)
    if key in used:
        return False, f"page {pages_read} returned a cursor that was already used", None
    used.add(key)
    return False, None, nxt


SCRIPTED = ("postback", "form", "load_more")     # pagers whose next page is a request built from the page


def _pager_word(ptype: str) -> str:
    return {"postback": "postback", "form": "form submission", "load_more": "load-more request"}.get(ptype, ptype)


def _end_at_total(listed: int, total: int, pages_read: int) -> tuple[bool, str, int]:
    """A pager that shows no more rows, judged by the portal's stated total: (complete, reason, shortfall).
    Complete at the total, or a few rows short of it when the live listing moved while it was read (served, but
    nothing is closed on the strength of such a read); further short is an incomplete read."""
    if listed >= total:
        return True, "reached stated total", 0
    if total - listed <= _drift(total):
        return True, (f"last page reached; read {listed} of a stated {total} (the listing changed while it was "
                      f"read)"), total - listed
    return False, f"the pager ended at page {pages_read} but the portal states {total} (read {listed})", 0


def reverse_allowed(pag: dict, stated_total) -> bool:
    """A result window can be read from both ends only for numbered/offset pages, and only when the portal states
    its total: reaching that total is the evidence that the two ends met."""
    rev = pag.get("reverse")
    return (isinstance(rev, dict) and "value" in rev and bool(rev.get("param") or rev.get("body_path"))
            and pag.get("type") in ("page", "offset") and stated_total is not None)


def reversed_spec(spec: dict) -> dict:
    """The same list in the opposite sort order (pagination.reverse: a query param or a field of the POST body)."""
    s = json.loads(json.dumps(spec))
    lst = s["list"]
    rev = lst["pagination"]["reverse"]
    if rev.get("body_path") and isinstance(lst.get("body"), dict):
        set_body_path(lst["body"], rev["body_path"], rev["value"])
    elif rev.get("param"):
        lst.setdefault("params", {})[rev["param"]] = rev["value"]
    lst.pop("order", None)       # no early stop on a list read backwards
    return s


def fill_row_template(tmpl: str, raw: dict, rid: str | None) -> str | None:
    """A detail address built from the list row: {id} and any other list field by name (e.g. {publicationId}
    when the API addresses one notice of a project). None when a placeholder has no value."""
    from .extract import fill_field_template
    return fill_field_template(tmpl, {**raw, "id": rid})


def mixes_closed(spec: dict) -> bool:
    """The listing carries closed notices too (onboarding saw tender pages of listed notices that were past their
    deadline or had none) and the deadline is only on the tender page: a listed notice is then not open until its
    tender page shows an open deadline."""
    dl = (spec.get("fields") or {}).get("submissionDeadline") or {}
    return dl.get("from") in ("detail", "detail2") and bool((spec.get("list") or {}).get("mixesClosed"))


# ---- second hop: a value that is not on the tender page itself but on a page it links to (the full notice)
DETAIL_SOURCES = ("detail", "detail2")


def follow_spec(spec: dict) -> dict | None:
    """detail.follow when some field is read from the linked page ("from": "detail2"), else None."""
    fol = (spec.get("detail") or {}).get("follow")
    if not isinstance(fol, dict):
        return None
    if not any(isinstance(fs, dict) and fs.get("from") == "detail2" for fs in (spec.get("fields") or {}).values()):
        return None
    return fol


def follow_url(fol: dict, dkind: str, dbody, page_url: str, rid: str | None = None) -> str | None:
    """The address of the linked page, read from the tender page: the link (html "sel", default attribute href;
    json "path"), optionally a "regex" capture of it, optionally put into "template" ({value}, {id})."""
    if dkind == "json":
        v = json_value(dbody, {"path": fol.get("path"), "regex": fol.get("regex")})
    else:
        if not fol.get("sel"):
            return None
        attr = fol.get("attr", "href")      # an explicit null reads the link's text (for a template)
        if attr is None and not fol.get("template"):
            attr = "href"
        v = html_value(dbody, {"sel": fol["sel"], "attr": attr, "regex": fol.get("regex")})
    if isinstance(v, list):
        v = next((x for x in v if isinstance(x, (str, int)) and str(x).strip()), None)
    if v is None or isinstance(v, (dict, list, bool)) or not str(v).strip():
        return None
    v = str(v).strip()
    tmpl = fol.get("template")
    u = tmpl.replace("{value}", v).replace("{id}", rid or "") if isinstance(tmpl, str) and tmpl else urljoin(page_url, v)
    return u if u.startswith(("http://", "https://")) else None


async def fetch_follow(ctx, fol: dict, dkind: str, dbody, page_url: str, rid: str | None = None):
    """Open the linked page. Returns (kind, body, url, None) when it was read, (None, None, None, why) otherwise;
    why = "no link" when the tender page does not link to one (no request is made then)."""
    url = follow_url(fol, dkind, dbody, page_url, rid)
    if not url:
        return None, None, None, "no link"
    resp = await ctx.client.get(url)
    if resp.status_code >= 400:
        return None, None, url, f"HTTP {resp.status_code}"
    kind = fol.get("kind") or ("json" if "json" in (resp.headers.get("content-type") or "") else "html")
    body = _parse_body(kind, resp)
    if body is None:
        return None, None, url, f"not {kind}"
    return kind, body, str(resp.url), None


async def read(ctx, spec: dict, previous: dict[str, dict] | None = None, limits: Limits | None = None,
               known_closed: dict[str, dict] | None = None, state: dict | None = None) -> ReadResult:
    """known_closed: {id: {"sig", "deadline", "why"}} of listed notices whose tender page already showed they are not
    open; their tender page is not opened again while their list row is unchanged.
    `state`: what the previous accepted read left for this one (a change feed's high-water mark); the new state
    is returned in ReadResult.state and stored only when this read is accepted."""
    limits = limits or Limits()
    previous = previous or {}
    known_closed = known_closed or {}
    lst = spec["list"]
    kind = lst.get("kind", "html")
    pag = lst.get("pagination") or {"type": "none"}
    max_pages = min(int(pag.get("max_pages", 200)), limits.max_pages or 10 ** 6)
    fields = spec.get("fields", {})
    list_fields = {f: s for f, s in fields.items() if s.get("from", "list") == "list" and not s.get("template")}

    rows: list[dict] = []          # raw list-level values per item
    seen: set[str] = set()
    seen_rows: set[str] = set()
    url_fallbacks = 0
    by_id: dict[str, dict] = {}
    merged_notices = 0
    complete = False
    stop_reason = "max_pages"
    stated_total = None
    cursor = referer = None
    post = None                    # postback pagination: the form submission that shows the next page
    ptype = pag.get("type", "none")
    latest_total = None            # the stated total on the latest page (a live listing changes while it is read)
    shortfall = 0
    missing_examples: dict[str, str] = {}   # required list field -> one row the spec reads no value from
    page_cursor = None              # cursor paging: the value sent for the next page
    cursors_used: set[str] = set()
    pages_read = 0
    capped = False
    rev_spec, rev_from = None, 0    # second pass from the other end of a windowed list (pagination.reverse)
    feed_read = None
    if kind == "feed":              # an ATOM/RSS change feed: its own walk (feed.py); the rest of the read is shared
        from . import feed
        feed_read = await feed.collect(ctx, spec, previous, limits, state)
        rows, merged_notices, pages_read = feed_read.rows, feed_read.merged, feed_read.pages
        complete, capped, stop_reason = feed_read.complete, feed_read.capped, feed_read.stop_reason
        max_pages = 0
    received = 0                    # rows received so far, as the page counts them ({offset} of load_more)
    lm_page, lm_wrap, lm_tokens = ("", "", None), ("", ""), {}   # load_more: the first page, its row container
    for page_index in range(max_pages):
        cur_spec, cur_index = (rev_spec, page_index - rev_from) if rev_spec else (spec, page_index)
        try:
            url, resp = await fetch_list_page(ctx, cur_spec, cur_index, cursor, post, referer, page_cursor=page_cursor)
        except httpx.TransportError as e:
            # One lost page must not end a walk of a hundred (a postback walk cannot skip a page): retry it once.
            ctx.note(f"list page {pages_read + 1}: {type(e).__name__}; retrying once")
            await asyncio.sleep(LIST_RETRY_PAUSE)
            url, resp = await fetch_list_page(ctx, cur_spec, cur_index, cursor, post, referer,
                                              page_cursor=page_cursor)
        pages_read += 1
        if pages_read % 5 == 1:
            ctx.note(f"reading list page {pages_read} ({len(rows)} tenders so far)")
        if resp.status_code >= 400:
            if page_index > 0 and rows and resp.status_code in (400, 404, 416, 422) and ptype not in SCRIPTED:
                if rev_spec is None and reverse_allowed(pag, stated_total):
                    # The portal shows at most N rows per sort order: read the rest from the other end. Complete
                    # only if both ends together reach the stated total.
                    rev_spec, rev_from = reversed_spec(spec), page_index + 1
                    ctx.note(f"the portal refuses pages beyond {pages_read - 1}; reading the list from the other end")
                    continue
                # Many search backends refuse deep paging (e.g. a 2,000-result window). That is the end of what
                # the portal will show, not a breakage: keep what we have, close nothing.
                stop_reason = f"portal refuses pages beyond {pages_read - 1} (HTTP {resp.status_code})"
                capped = True
                break
            block = detect_block(resp.text if kind != "json" else "", resp.status_code)
            if block:
                raise PortalBlocked(block)
            if ptype in SCRIPTED and page_index > 0:
                stop_reason = f"the {_pager_word(ptype)} for page {pages_read} returned HTTP {resp.status_code}"
                break               # a rejected form post is a broken pager, never the end of the list
            raise RuntimeError(f"list page returned HTTP {resp.status_code}: {url}")
        more_html = None            # load_more: the rows' HTML in the script request's answer
        if ptype == "load_more" and page_index > 0:
            more_html, why = jspager.answer_html(resp, pag)
            if more_html is None:
                stop_reason = f"the load-more answer for page {pages_read} {why}"
                break
        if kind == "html":
            block = detect_block(more_html if more_html is not None else resp.text, resp.status_code,
                                 spec.get("block_markers"))
            if block:
                raise PortalBlocked(block)
        # rows the script appends to the first page's container: read them inside that container, so the
        # spec's selectors match as on the first page (and their links resolve against the list page)
        body = HTMLParser(lm_wrap[0] + more_html + lm_wrap[1]) if more_html is not None else _parse_body(kind, resp)
        if body is None:
            raise RuntimeError(f"list page is not valid {kind}: {url}")
        row_base = lm_page[0] if more_html is not None else str(resp.url)
        if page_index == 0 and ptype == "postback" and pag.get("size_sel") and kind == "html":
            bigger = await _postback_resize(ctx, spec, resp, body)
            if bigger:
                resp, body = bigger     # page 1 again, with more rows (the extra request is in stats.requests)
        page_total = _stated_total(kind, body, spec)
        latest_total = page_total if page_total is not None else latest_total
        if page_index == 0:
            stated_total = page_total
        items, items_valid = _checked_items(kind, body, lst.get("items", ""))
        received += len(items)
        if ptype == "load_more" and page_index == 0 and items:
            lm_page = (str(resp.url), resp.text, body)
            lm_wrap = jspager.container_wrap(items[0])
        if not items_valid:
            # A missing or non-array items path is not "no more results": it looks like a changed response shape.
            if page_index == 0:
                raise RuntimeError(f"list page has no valid items array at '{lst.get('items', '')}': {url}")
            stop_reason = f"page {pages_read} has no valid items array"
            break
        new_on_page = 0
        for it in items:
            raw = {f: field_from(kind, it, f, s) for f, s in list_fields.items()}
            rid = raw.get("id")
            rid = str(rid).strip() if rid is not None else None
            # A JSON row is identified by the whole item, not only the mapped fields: two lots of one notice can share
            # every mapped value and are still two of the rows the portal's total counts.
            sig = json.dumps(it if kind == "json" else raw, sort_keys=True, default=str)
            if sig in seen_rows:
                continue            # the very same row again (a repeated page), not a new notice
            seen_rows.add(sig)
            new_on_page += 1
            raw["_list_url"] = row_base
            for f in ("id", "url", "title"):
                if f in list_fields and raw.get(f) in (None, "", []):
                    # evidence for the model: what a row looks like that the spec cannot read
                    if f not in missing_examples:
                        missing_examples[f] = (_row_evidence(it) if kind == "html"
                                               else json.dumps(it, ensure_ascii=False, default=str)[:1500])
            if kind == "html" and "url" in list_fields and raw.get("url") in (None, "", []):
                # The learned link selector missed this row (e.g. a tender hosted on another platform links out with
                # different markup): use the row's own link to the tender, never a script action. The row stays
                # recorded as evidence above, so a later repair can still learn the better selector.
                fb = _row_link(it, raw.get("id"), raw.get("title"))
                if fb:
                    raw["url"] = fb
                    url_fallbacks += 1
            if rid and rid in seen:
                # Another notice of the same tender (a correction, a termination, an award of the same procedure):
                # one tender, read from its newest notice; a cancellation sticks.
                _merge_notice(spec, by_id[rid], raw)
                merged_notices += 1
                continue
            if rid:
                seen.add(rid)
                by_id[rid] = raw
            rows.append(raw)
        if not items:
            listed = len(rows) + merged_notices
            cur_total = latest_total if latest_total is not None else stated_total
            if page_index == 0 and kind != "json":
                stop_reason = "no items matched on the first page"   # cannot tell "no tenders" from a broken selector
            elif page_index == 0 and stated_total:
                stop_reason = f"first page is empty but the portal states {stated_total}"
            elif ptype == "postback" and not (cur_total is not None and listed >= cur_total - _drift(cur_total)):
                # a postback pager ends by removing its next control; an empty answer is an expired or rejected state
                stop_reason = (f"the postback for page {pages_read} showed no rows (read {listed}"
                               + (f" of a stated {cur_total})" if cur_total is not None else ")"))
            elif ptype in ("form", "load_more") and cur_total is not None:
                # no more rows: complete only at the stated total (a few rows may move while a live list is read)
                complete, stop_reason, shortfall = _end_at_total(listed, cur_total, pages_read)
            elif rev_spec and len(rows) + merged_notices < (stated_total or 0):
                # Both ends of the result window were read and did not meet: what the portal shows is read, the
                # rest it will not show. A capped read (Patterno rule A: served as partial, closes nothing).
                stop_reason = (f"the list read from the other end ended at {len(rows)} of {stated_total}; "
                               f"serving what both ends showed (capped)")
                capped = True
            else:
                complete, stop_reason = True, "empty page"
            break
        if new_on_page == 0:
            if stated_total is not None and len(rows) + merged_notices >= stated_total:
                complete, stop_reason = True, "reached stated total"
            elif rev_spec:
                # Read from the other end, the list repeats what the first pass saw: the reversed sort is no mirror
                # (rows with an empty sort value stay last in both directions, or the sort is not unique). The first
                # pass is still a working read of the result window: served as a CAPPED read (Patterno rule A:
                # coverage partial, closes nothing), never failed and never complete.
                stop_reason = (f"page {pages_read} repeated earlier results: the list read from the other end is not "
                               f"the mirror of the first pass (its sort field is empty for some rows); serving the "
                               f"result window read so far (capped)")
                capped = True
            elif ptype in ("form", "load_more") and (latest_total or stated_total) is not None:
                complete, stop_reason, shortfall = _end_at_total(len(rows) + merged_notices,
                                                                 latest_total or stated_total, pages_read)
            else:
                # The page parameter was probably ignored (or clamped): this is not evidence that the list ended.
                stop_reason = f"page {pages_read} repeated earlier results"
            break
        early = _sorted_list_finished(spec, rows[-new_on_page:]) if new_on_page and not rev_spec else None
        if early:
            cur_total = latest_total if latest_total is not None else stated_total
            if cur_total is not None and len(rows) + merged_notices < 0.95 * cur_total:
                # The sort says the rest is closed, but the portal's own count says most of the list is unread:
                # an early stop the total cannot confirm (the sort may not be what spec.list.order claims). Served
                # as a capped read - nothing beyond it is closed - never as a complete one.
                capped = True
                stop_reason = (f"{early}; read {len(rows) + merged_notices} of a stated {cur_total}, so the "
                               f"early stop is not confirmed (capped)")
            else:
                complete, stop_reason = True, early
            break
        if ptype == "none":
            complete, stop_reason = True, "single page"
            break
        if ptype == "next_link":
            nxt = body.css_first(pag.get("next_sel", "a[rel=next]")) if kind == "html" else None
            href = nxt.attributes.get("href") if nxt else None
            if nxt is not None and not href and jspager.js_url(nxt):
                href = "#"                                 # <a onclick="...('/l.do?page=2')"> without an href
            if not href:
                complete, stop_reason = True, "no next link"
                break
            if href.strip().lower().startswith("javascript:") or href.strip() == "#":
                # the address may sit in the control's script (onclick="setTargetAndSubmit('/l.do?page=2')")
                href = jspager.js_url(nxt)
                if not href:
                    stop_reason = (f"the next link on page {pages_read} is a JavaScript action, not an address "
                                   f"({(nxt.attributes.get('href') or '').strip()[:80]}): the pager posts a form "
                                   f"(pagination type postback)")
                    break
            cursor, referer = urljoin(str(resp.url), href), str(resp.url)
        if ptype == "postback":
            post, why = postback.next_request(body, resp.text, str(resp.url), resp.encoding, pag) \
                if kind == "html" else (None, "postback paging needs an html list")
            if post is None:
                listed = len(rows) + merged_notices
                cur_total = latest_total if latest_total is not None else stated_total
                if why:
                    stop_reason = f"page {pages_read}: {why}"
                elif cur_total is None:
                    complete, stop_reason = True, "no next-page control"
                elif listed >= cur_total:
                    complete, stop_reason = True, "reached stated total"
                elif cur_total - listed <= _drift(cur_total):
                    # Every page was walked, but rows moved while we read (a new notice pushes one row onto the next
                    # page, a closed one pulls one back): the walk is finished, a few rows may be unseen. Served, but
                    # nothing is closed on the strength of this read.
                    complete, shortfall = True, cur_total - listed
                    stop_reason = (f"last page reached; read {listed} of a stated {cur_total} (the listing changed "
                                   f"while it was read)")
                else:
                    stop_reason = f"the pager ended at page {pages_read} but the portal states {cur_total} (read {listed})"
                break
            continue                # the pager's own end signal decides; the total only judges it
        if ptype == "form":
            if pag.get("next_sel") and postback.next_control(body, pag["next_sel"]) is None:
                listed = len(rows) + merged_notices
                cur_total = latest_total if latest_total is not None else stated_total
                if cur_total is None:
                    complete, stop_reason = True, "no next-page control"
                else:
                    complete, stop_reason, shortfall = _end_at_total(listed, cur_total, pages_read)
                break
            post = postback.form_request(body, str(resp.url), resp.encoding, pag,
                                         _page_value({**pag, "type": "page"}, page_index + 1))
            if isinstance(post, str):
                stop_reason, post = f"page {pages_read}: {post}", None
                break
        if ptype == "load_more":
            if page_index == 0:
                lm_tokens, missing = jspager.read_tokens(pag, lm_page[2], lm_page[1])
                if missing:
                    stop_reason = f"the list page has no value named '{missing}' (pagination load_more token)"
                    break
            post = jspager.load_more_request(pag, lm_page[0], _page_value({**pag, "type": "page"}, page_index + 1),
                                             received, lm_tokens, lambda v: fill_tokens(v, spec))
        if ptype == "cursor":
            done, why, page_cursor = _cursor_step(pag, body, items, pages_read, cursors_used)
            if why:
                complete, stop_reason = done, why
                break
        if ptype == "offset" and len(items) < int(pag.get("size", 20)):
            if stated_total is not None and len(rows) + merged_notices < stated_total:
                # The portal gave fewer per page than we asked for; the next offset would skip rows.
                stop_reason = f"short page ({len(items)} < {pag.get('size', 20)}) but the portal states {stated_total}"
            else:
                complete, stop_reason = True, "short page"
            break
        if stated_total is not None and len(rows) + merged_notices >= stated_total:
            complete, stop_reason = True, "reached stated total"
            break

    # ---- open tenders only: a row whose deadline has already passed is a closed procedure (an award, a correction
    # of a closed call, the portal's archive). It is not bad data and it is not ours to serve, so it never reaches
    # detail fetching, downloads or validate(). Rows without a readable deadline stay (validate() judges them).
    from . import timeutil
    read_now = timeutil.now()
    rows_listed = len(rows) + merged_notices    # notices listed (a stated total counts notices)
    before_drop = len(rows)
    if "submissionDeadline" in list_fields:
        # (a change feed's update of a tender already served is applied whatever it says: "_keep")
        rows = [r for r in rows if r.get("_keep") or not ((dl := row_deadline(spec, r)) is not None and dl <= read_now)]
    closed_dropped = before_drop - len(rows)    # tenders (after merging notices), not notices
    if feed_read is not None:
        closed_dropped += feed_read.closed_dropped    # closed/withdrawn by status, never served before
    # A row whose notice type the list shows as not a call (an award or result notice, an ex-post publication, prior
    # information only): not open, same rule as above. A tender already served stays in the read: it closes.
    from .extract import notice_field, row_notice_status
    not_open_notices = 0
    if notice_field(fields) in list_fields:
        kept_rows = []
        for r in rows:
            rid0 = str(r.get("id")) if r.get("id") is not None else None
            if (row_notice_status(r, fields) == "closed" and not r.get("_keep")
                    and (rid0 is None or rid0 not in previous)):
                not_open_notices += 1
                continue
            kept_rows.append(r)
        rows = kept_rows

    # ---- details (incremental): new, changed and pending tenders first, then a small rotating sample
    detail = spec.get("detail") or {}
    d1_fields = {f: s for f, s in fields.items() if s.get("from") == "detail"}
    follow = follow_spec(spec)
    # second hop: fields read from the page the tender page links to (only opened when such a field is mapped)
    d2_fields = {f: s for f, s in fields.items() if s.get("from") == "detail2"} if follow else {}
    detail_fields = {**d1_fields, **d2_fields}
    docs_spec = spec.get("documents") or {}
    docs_mode = docs_spec.get("from")
    need_detail = bool(detail_fields) or docs_mode == "detail"
    errors: list[str] = []

    infos: list[dict] = []
    known_skipped = known_no_deadline = 0
    forget_closed: list[str] = []
    for idx, raw in enumerate(rows):
        rid = str(raw.get("id")) if raw.get("id") is not None else None
        prev = previous.get(rid) if rid else None
        prev_data = prev["data"] if prev else None
        list_sig = json.dumps({k: v for k, v in raw.items() if not k.startswith("_")}, sort_keys=True, default=str)
        kc = known_closed.get(rid) if rid else None
        if kc is not None and prev_data is None:
            if kc.get("sig") == list_sig:
                known_skipped += 1      # its tender page showed it is not open; the row is unchanged since
                if str(kc.get("why") or "").startswith(NOT_A_CALL):
                    not_open_notices += 1
                elif kc.get("deadline"):
                    closed_dropped += 1
                else:
                    known_no_deadline += 1
                continue
            forget_closed.append(rid)   # a new notice or a changed row: open its tender page again
        changed_row = prev_data is None or prev_data.get("_listSig") != list_sig
        detail_url = None
        tmpl_url = (fill_row_template(detail["template"], raw, rid)
                    if detail.get("url_from") == "template" and detail.get("template") and rid else None)
        if need_detail:
            if detail.get("url_from") == "template" and detail.get("template") and rid:
                detail_url = tmpl_url
            else:
                u = raw.get("url")
                detail_url = urljoin(raw["_list_url"], u) if u else None
        infos.append({"idx": idx, "raw": raw, "rid": rid, "prev": prev_data, "sig": list_sig,
                      "changed": changed_row, "pending": bool(prev_data and prev_data.get("_pendingDetail")),
                      "detail_url": detail_url, "tmpl_url": tmpl_url, "detail_ok": False, "cat": None})

    # rotating sample: only ids that are in the current listing, oldest verification first
    for info in infos:
        if info["prev"] is None:
            info["cat"] = 0
        elif info["changed"]:
            info["cat"] = 1
        elif info["pending"]:
            info["cat"] = 2
    current_ids = {i["rid"] for i in infos if i["rid"]}
    sample_pool = sorted(((tid, p.get("last_verified_at") or "") for tid, p in previous.items()
                          if tid in current_ids), key=lambda kv: (kv[1], kv[0]))
    sample_ids: set[str] = set()
    by_rid = {i["rid"]: i for i in infos if i["rid"]}
    for tid, _ in sample_pool:
        if len(sample_ids) >= limits.sample:
            break
        if by_rid[tid]["cat"] is None and by_rid[tid]["detail_url"]:
            sample_ids.add(tid)
            by_rid[tid]["cat"] = 3

    to_fetch = [i for i in infos if need_detail and i["cat"] is not None and i["detail_url"]]
    to_fetch.sort(key=lambda i: (i["rid"] not in limits.prefer_ids, i["cat"], i["idx"]))
    if limits.max_details is not None:
        to_fetch = to_fetch[:max(limits.max_details, 0)]

    verified_ids: list[str] = []
    details_fetched = follow_fetched = follow_read = follow_missing = 0
    for info in to_fetch:
        if d2_fields and limits.max_details is not None and details_fetched >= limits.max_details:
            break       # the cap counts pages: a tender read through its linked page costs two
        if details_fetched % 10 == 0:
            ctx.note(f"opening tender pages: {details_fetched} done")
        raw = info["raw"]
        try:
            dresp = await ctx.client.get(info["detail_url"])
            details_fetched += 1
            if dresp.status_code >= 400:
                errors.append(f"detail HTTP {dresp.status_code}")
                continue
            dkind = detail.get("kind", "html")
            dbody = _parse_body(dkind, dresp)
            if dbody is None:
                errors.append("detail error: invalid body")
                continue
            for f, s in d1_fields.items():
                raw[f] = field_from(dkind, dbody, f, s)
            if docs_mode == "detail":
                vals = (json_value(dbody, docs_spec) if dkind == "json"
                        else html_value(dbody, {**docs_spec, "all": True}))
                info["doc_urls"] = [urljoin(str(dresp.url), v) for v in (vals or []) if isinstance(v, str)]
            if d2_fields:
                k2, b2, u2, why = await fetch_follow(ctx, follow, dkind, dbody, str(dresp.url), info["rid"])
                if why != "no link":
                    details_fetched += 1
                    follow_fetched += 1
                if why == "no link":
                    follow_missing += 1     # this tender page links to no such page: its detail2 values are unknown
                    for f in d2_fields:
                        raw[f] = None
                elif why:
                    errors.append(f"linked page {why}")
                    continue                # not read: stays pending, read again on a later refresh
                else:
                    for f, s in d2_fields.items():
                        raw[f] = field_from(k2, b2, f, s)
                    info["follow_url"] = u2
                    follow_read += 1
            info["detail_ok"] = True
            if info["rid"]:
                verified_ids.append(info["rid"])
        except PortalBlocked:
            raise
        except Exception as e:  # one bad detail page must not kill the read
            errors.append(f"detail error: {type(e).__name__}")

    # ---- documents: every attempt (list requests and downloads, failed or not) counts against max_docs
    doc_attempts = docs_fetched = 0

    def doc_budget() -> bool:
        return limits.fetch_docs and (limits.max_docs is None or doc_attempts < limits.max_docs)

    # Listing a tender's files and downloading them share max_docs; listing may use at most half, so every read
    # also delivers files (2,800 tenders would otherwise spend ~23 reads listing before the first download).
    list_cap = None if limits.max_docs is None else max(1, limits.max_docs // 2)
    list_attempts = 0

    for info in infos:
        prev_data, raw = info["prev"], info["raw"]
        if not info["detail_ok"] and prev_data is not None:
            for f in detail_fields:  # reuse what we knew
                raw[f] = (prev_data.get("_raw") or {}).get(f)
        if "doc_urls" not in info:
            info["doc_urls"] = list(prev_data.get("_docUrls", [])) if prev_data else []
        if docs_mode == "list":
            info["doc_urls"] = [urljoin(raw["_list_url"], v) for v in (raw.get("_docs_list") or [])]
        info["doc_list_pending"] = bool(prev_data and prev_data.get("_docListPending"))
        if docs_mode == "template" and info["rid"] and (
                prev_data is None or info["doc_list_pending"] or info["changed"] or info["detail_ok"]):
            info["doc_list_wanted"] = True

    for info in infos:
        if not info.get("doc_list_wanted"):
            continue
        rid, prev_data = info["rid"], info["prev"]
        if not doc_budget() or (list_cap is not None and list_attempts >= list_cap):
            info["doc_list_pending"] = True
            continue
        # A tender that merges several notices (original + corrections) has documents on each of them when the
        # address is per notice ({noticeId}); the tender's documents are the union.
        tmpl = docs_spec["url"]
        raw = info["raw"]
        if "{noticeId}" in tmpl:
            nids = [str(n) for n in (raw.get("_noticeIds") or [raw.get("noticeId")]) if n is not None]
            list_urls = [tmpl.replace("{noticeId}", n).replace("{id}", rid) for n in nids]
        else:
            list_urls = [tmpl.replace("{id}", rid)]
        urls_found: list[str] = []
        ok = bool(list_urls)
        stop = False
        for lu in list_urls:
            if not doc_budget():
                ok = False
                break
            doc_attempts += 1
            list_attempts += 1
            try:
                dr = await ctx.client.get(lu)
                if dr.status_code >= 400:
                    errors.append(f"document list HTTP {dr.status_code}")
                    ok = False
                    continue
                dk = docs_spec.get("kind", "json")
                db = _parse_body(dk, dr)
                if db is None:
                    errors.append("document list error: invalid body")
                    ok = False
                    continue
                vals = (json_value(db, docs_spec) if dk == "json"
                        else html_value(db, {**docs_spec, "all": True})) or []
                vals = vals if isinstance(vals, list) else [vals]
                lt = docs_spec.get("link_template")
                for v in vals:
                    if v is None:
                        continue
                    u = urljoin(str(dr.url), lt.replace("{value}", str(v)).replace("{id}", rid) if lt else str(v))
                    if u not in urls_found:
                        urls_found.append(u)
            except BudgetExceeded:
                ok, stop = False, True
                break
            except Exception as e:
                errors.append(f"document list error: {type(e).__name__}")
                ok = False
        if ok:
            info["doc_urls"] = urls_found
            info["doc_list_pending"] = False
        else:
            if urls_found:
                info["doc_urls"] = list(dict.fromkeys(list(info.get("doc_urls") or []) + urls_found))
            info["doc_list_pending"] = True
        if stop:
            break

    # per tender: discovered URLs (_docUrls) vs done URLs (_docDone: downloaded, or omitted as over the size cap)
    doc_block_reason: str | None = None
    for info in infos:
        prev_data = info["prev"] or {}
        urls = info["doc_urls"]
        entries_map = {u: list(v) for u, v in (prev_data.get("_docEntries") or {}).items() if u in urls}
        info["fails"] = {u: n for u, n in (prev_data.get("_docFails") or {}).items() if u in urls}
        info["blocked"] = {u: r for u, r in (prev_data.get("_docBlockedUrls") or {}).items() if u in urls}
        web_page = getattr(getattr(ctx, "docs", None), "stored_web_page", None)
        if web_page:
            # A web page stored as a 'document' by an older version is not one: drop it, and say why on the tender.
            for u in list(entries_map):
                kept = [d for d in entries_map[u] if not web_page(d)]
                if len(kept) < len(entries_map[u]):
                    if kept:
                        entries_map[u] = kept
                    else:
                        del entries_map[u]
                        info["blocked"][u] = f"{WEB_PAGE_REASON} ({u})"
        info["entries"] = entries_map
        # documents stored before checkpoints existed have no per-URL record: keep them shown until replaced
        info["legacy"] = list(prev_data.get("documents", [])) if "_docEntries" not in prev_data else []
        if web_page and info["legacy"]:
            info["legacy"] = [d for d in info["legacy"] if not web_page(d)]

    if limits.fetch_docs:
        queue = []
        for info in infos:
            for n, u in enumerate(info["doc_urls"]):
                if u not in info["entries"]:
                    queue.append((info["fails"].get(u, 0), info["idx"], n, u, info))
        queue.sort(key=lambda q: q[:3])      # fresh URLs first; repeated failures go last
        consecutive_blocked = 0
        for _fails, _i, _n, du, info in queue:
            if not doc_budget():
                break
            if consecutive_blocked >= 3 and docs_fetched == 0:
                break     # the portal walls its downloads: stop spending requests
            doc_attempts += 1
            try:
                got = await ctx.docs.fetch(ctx.client, du)
                info["entries"][du] = [e.as_dict() for e in got]
                info["fails"].pop(du, None)
                info["blocked"].pop(du, None)
                docs_fetched += 1
                consecutive_blocked = 0
            except BudgetExceeded as e:
                errors.append(f"document error: {type(e).__name__}")
                break
            except DocumentBlocked as e:
                info["fails"][du] = info["fails"].get(du, 0) + 1
                info["blocked"][du] = str(e)
                consecutive_blocked += 1
                errors.append(f"download blocked: {e}")
            except Exception as e:
                info["fails"][du] = info["fails"].get(du, 0) + 1
                info["blocked"].pop(du, None)
                consecutive_blocked = 0
                errors.append(f"document error: {type(e).__name__}")

    tenders: list[dict] = []
    pending_docs = pending_doc_lists = doc_blocked = 0
    closed_new: dict[str, dict] = {}
    held_back, no_deadline = 0, known_no_deadline
    mixed = mixes_closed(spec)
    dl_on_detail = (fields.get("submissionDeadline") or {}).get("from") == "detail"
    nt_on_detail = (fields.get(notice_field(fields)) or {}).get("from") == "detail"
    built = [(info, assemble(info["raw"], spec, info["raw"]["_list_url"])) for info in infos]

    def deadline_read(info) -> bool:
        """This tender's deadline was looked for: on the list, or on its tender page (this read or an earlier one)."""
        return (not dl_on_detail or info["detail_ok"]
                or (info["prev"] is not None and not info["prev"].get("_pendingDetail")))
    judged = [t for info, t in built if t.get("status") == "open" and deadline_read(info)]
    with_dl = sum(1 for t in judged if t.get("submissionDeadline"))
    expect_dl = len(judged) >= DEADLINE_EXPECTED_MIN and with_dl > DEADLINE_EXPECTED_SHARE * len(judged)
    # tender pages (read now, or remembered as not a call) showed award or result notices among the listed ones
    awards_on_pages = nt_on_detail and (not_open_notices > 0 or any(
        info["detail_ok"] and t.get("status") == "closed" for info, t in built))
    for info, t in built:
        raw, prev_data, rid = info["raw"], info["prev"], info["rid"]
        if _is_closed(t.get("submissionDeadline"), read_now) and not raw.get("_keep"):
            closed_dropped += 1     # deadline known only from the detail page: closed, same rule as above
            if rid and prev_data is None and info["detail_ok"]:
                closed_new[rid] = {"sig": info["sig"], "deadline": t.get("submissionDeadline"),
                                   "why": "deadline passed"}
            continue
        if t.get("status") == "closed" and prev_data is None and not raw.get("_keep"):
            # an award or result notice, an ex-post publication, prior information only (its notice type or status
            # says so): not an open call, never served. A tender already served stays and closes.
            not_open_notices += 1
            if rid and (info["detail_ok"] or not nt_on_detail):
                closed_new[rid] = {"sig": info["sig"], "deadline": t.get("submissionDeadline"),
                                   "why": f"{NOT_A_CALL}: "
                                          f"{str(raw.get(notice_field(fields)) or raw.get('status') or '')[:80]}"}
            continue
        if not t.get("submissionDeadline") and t.get("status") == "open" and prev_data is None:
            seen_all = info["detail_ok"] or not (dl_on_detail or nt_on_detail)
            if seen_all and (mixed or expect_dl):
                # The listing also carries closed notices, or most tenders show a deadline: one whose tender page
                # (or list row) shows none is not an open call
                no_deadline += 1
                if rid:
                    closed_new[rid] = {"sig": info["sig"], "deadline": None,
                                       "why": "no deadline on its tender page" if dl_on_detail
                                       else "no deadline in its list row"}
                continue
            if not seen_all and (mixed or awards_on_pages):
                # tender page not read yet, and the listing is known to carry closed or award notices that only
                # the tender page shows: not served as open unseen, waits for a later refresh
                held_back += 1
                continue
        for k in ("_feedUpdated", "_entryId"):      # change-feed bookkeeping: newest entry wins, tombstones match
            if raw.get(k):
                t[k] = raw[k]
        if t.get("url") is None and info["detail_url"]:
            t["url"] = info["detail_url"]
        if info["tmpl_url"]:
            t["_detailUrl"] = info["tmpl_url"]      # onboarding opens the same address (templates may use any field)
        if d2_fields and (fu := info.get("follow_url") or (prev_data or {}).get("_followUrl")):
            t["_followUrl"] = fu                    # the linked page its detail2 values came from (verify, audit)
        urls, emap = info["doc_urls"], info["entries"]
        documents, seen_sha = [], set()      # one file once, even when several notices or links carry it
        for u in urls:
            for d in emap.get(u, []):
                key = d.get("sha256") or (u, d.get("fileName"))
                if key not in seen_sha:
                    seen_sha.add(key)
                    documents.append(d)
        if info["legacy"] and any(u not in emap for u in urls):
            have = {d.get("sha256") for d in documents}
            documents += [d for d in info["legacy"] if d.get("sha256") not in have]
        t["documents"] = documents
        wants_detail = info["prev"] is None or info["changed"] or info["pending"]
        t["_pendingDetail"] = bool(need_detail and info["detail_url"] and wants_detail and not info["detail_ok"])
        t["_docUrls"] = urls
        t["_docDone"] = [u for u in urls if u in emap]
        t["_docEntries"] = {u: emap[u] for u in urls if u in emap}
        t["_docFails"] = {u: n for u, n in info["fails"].items() if u not in emap}
        t["_docBlockedUrls"] = {u: r for u, r in info["blocked"].items() if u not in emap}
        if t["_docBlockedUrls"] and not documents:
            # Patterno, 20 Nov: deliver the tender without documents, and say so on the tender itself
            t["documentsUnavailable"] = next(iter(t["_docBlockedUrls"].values()))
        t["_docListPending"] = bool(info["doc_list_pending"])
        t["_listSig"] = info["sig"]
        t["_raw"] = {f: raw.get(f) for f in detail_fields}
        pending_docs += sum(1 for u in urls if u not in emap)
        pending_doc_lists += 1 if t["_docListPending"] else 0
        doc_blocked += len(t["_docBlockedUrls"])
        if doc_block_reason is None and t["_docBlockedUrls"]:
            doc_block_reason = next(iter(t["_docBlockedUrls"].values()))
        tenders.append(t)

    # A small minority of rows the spec reads no id/url/title from (e.g. a tender that links out only through a
    # script) must not sink the whole portal: deliver the rest, say how many were left out, close nothing on this read.
    # A larger share means the spec itself is wrong: those rows stay, validate() fails and the model learns again.
    incomplete = [t for t in tenders if not all(t.get(f) for f in ("id", "url", "title"))]
    dropped_incomplete = 0
    if incomplete and len(incomplete) <= len(tenders) * INCOMPLETE_SHARE:
        gone = {id(t) for t in incomplete}
        tenders = [t for t in tenders if id(t) not in gone]
        dropped_incomplete = len(incomplete)
    pending_n = sum(1 for t in tenders if t.get("_pendingDetail")) + held_back
    if not complete and stop_reason == "max_pages":
        capped = True
    if lst.get("partial") and not capped:
        # onboarding accepted this spec as a partial listing (verified rows, pager not followed): Patterno rule A -
        # served as partial, never complete, nothing closed on its strength
        complete, capped = False, True
        stop_reason = f"{stop_reason}; partial listing: {lst['partial']}"
    stats = {"pages": pages_read, "rows": rows_listed, "closedDropped": closed_dropped,
             "mergedNotices": merged_notices, "urlFallbacks": url_fallbacks, "droppedIncomplete": dropped_incomplete,
             "statedTotal": stated_total, "stopReason": stop_reason,
             "capped": capped, "shortfall": shortfall, "bothEnds": bool(rev_spec),
             **({"missingExamples": missing_examples} if missing_examples else {}),
             **({"statedTotalLast": latest_total} if latest_total != stated_total else {}),
             "pendingDetails": pending_n, "pendingDocs": pending_docs, "pendingDocLists": pending_doc_lists,
             "heldBack": held_back, "noDeadline": no_deadline, "knownClosedSkipped": known_skipped,
             "notOpenNotices": not_open_notices, "deadlineShare": [with_dl, len(judged)],
             "docBlocked": doc_blocked, "docBlockReason": doc_block_reason,
             "detailsFetched": details_fetched, "docsFetched": docs_fetched, "docAttempts": doc_attempts,
             **({"followFetched": follow_fetched, "followRead": follow_read, "followMissing": follow_missing}
                if d2_fields else {}),
             "errors": errors[:20], "requests": ctx.client.requests,
             **({"changeFeed": True, "feed": feed_read.stats} if feed_read is not None else {})}
    return ReadResult(tenders=tenders, complete=complete, stats=stats, verified_ids=verified_ids,
                      closed=closed_new, forget_closed=forget_closed,
                      state=feed_read.state if feed_read is not None else None)
