"""Postback pagination: pagers whose links are JavaScript form posts, not addresses.

Server-side UI frameworks (ASP.NET WebForms: __doPostBack -> __EVENTTARGET/__EVENTARGUMENT + __VIEWSTATE; PRADO:
PRADO_POSTBACK_TARGET/PRADO_POSTBACK_PARAMETER + PRADO_PAGESTATE; and look-alikes) keep the page state inside the form.
A click on "next" submits that form, every field as the browser would send it, plus the name of the clicked control
as the event target. The answer is the next page and carries the state for the following click. No JavaScript runs:
the event target is read from the control's href/onclick or from the script that binds it. A control without an event
target whose script names an address (onclick="setTargetAndSubmit('/list.do?page=2')") submits its form to that
address (jspager.js_url). form_request: pagination type "form" (a script sets the form's page field, then submits).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlencode, urljoin, urlsplit, urlunsplit

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import jspager

# (event-target field, argument field, fields whose presence identifies the framework)
_FRAMEWORKS = (
    ("__EVENTTARGET", "__EVENTARGUMENT", ("__EVENTTARGET", "__VIEWSTATE", "__EVENTVALIDATION")),
    ("PRADO_POSTBACK_TARGET", "PRADO_POSTBACK_PARAMETER", ("PRADO_PAGESTATE", "PRADO_POSTBACK_TARGET")),
)
_STATE_FIELDS = ("__VIEWSTATE", "__EVENTTARGET", "PRADO_PAGESTATE")
_Q = r"""\\?['"]"""
_DOPOSTBACK = re.compile(r"__doPostBack\(\s*" + _Q + r"""([^'"\\]*)""" + _Q + r"\s*,\s*" + _Q + r"""([^'"\\]*)""" + _Q)
_WEBFORM = re.compile(r"WebForm_PostBackOptions\(\s*" + _Q + r"""([^'"\\]+)""" + _Q + r"\s*,\s*" + _Q
                      + r"""([^'"\\]*)""" + _Q)
_NEXT_WORDS = re.compile(r"suivant|\bnext\b|weiter|n[äa]chste|volgende|siguiente|successiv|pr[oó]xim|n[äa]sta|"
                         r"seuraava|nast[eę]pn|dal[sš][ií]|k[öo]vetkez", re.I)


@dataclass
class Postback:
    method: str
    url: str
    content: bytes | None
    headers: dict = field(default_factory=dict)


def form_fields(form) -> list[tuple[str, str]]:
    """The fields a browser submits for this form (no submit buttons: only the clicked one is sent)."""
    out: list[tuple[str, str]] = []
    for el in form.css("input, select, textarea"):
        a = el.attributes
        name = a.get("name")
        if not name or "disabled" in a:
            continue
        if el.tag == "input":
            t = (a.get("type") or "text").lower()
            if t in ("submit", "button", "image", "reset", "file"):
                continue
            if t in ("checkbox", "radio"):
                if "checked" in a:
                    out.append((name, a.get("value") or "on"))
                continue
            out.append((name, a.get("value") or ""))
        elif el.tag == "select":
            opts = el.css("option")
            chosen = [o for o in opts if "selected" in o.attributes]
            if not chosen and opts and "multiple" not in a:
                chosen = opts[:1]
            for o in chosen:
                v = o.attributes.get("value")
                out.append((name, v if v is not None else (o.text(strip=True) or "")))
        else:
            out.append((name, el.text() or ""))
    return out


def _disabled(el) -> bool:
    for node in (el, el.parent):
        if node is None:
            continue
        a = node.attributes or {}
        if "disabled" in a or str(a.get("aria-disabled") or "").lower() == "true" \
                or "disabled" in str(a.get("class") or "").lower():
            return True
    a = el.attributes
    return el.tag == "a" and not a.get("href") and not a.get("onclick")


def _clickable(el):
    """The control itself when the selector names a wrapper (e.g. li.next > a)."""
    if el.tag in ("a", "input", "button", "select"):
        return el
    return el.css_first("a, input[type=submit], input[type=image], button") or el


_BOUND_OBJ = re.compile(r"""\{[^{}]*?(?<![A-Za-z])['"]?ID['"]?\s*:\s*['"]([^'"]+)['"][^{}]*\}""")
_EVENT_TARGET = re.compile(r"""['"]?EventTarget['"]?\s*:\s*['"]([^'"]+)['"]""")
_EVENT_PARAM = re.compile(r"""['"]?EventParameter['"]?\s*:\s*['"]([^'"]*)['"]""")


def script_targets(html: str) -> dict[str, tuple[str, str]]:
    """Controls bound by a script, id -> (event target, parameter), in one pass over the page. E.g. PRADO:
    new Prado.WebUI.TLinkButton({'ID':"x_ctl2",'EventTarget':"x$ctl2",...})."""
    out: dict[str, tuple[str, str]] = {}
    for m in _BOUND_OBJ.finditer(html or ""):
        t = _EVENT_TARGET.search(m.group(0))
        if t:
            p = _EVENT_PARAM.search(m.group(0))
            out.setdefault(m.group(1), (t.group(1), p.group(1) if p else ""))
    return out


def control_target(el, bound: dict[str, tuple[str, str]]) -> tuple[str, str, bool] | None:
    """(event target, argument, is_submit_button) for a control, or None when it cannot be determined.
    `bound` = script_targets(page html)."""
    a = el.attributes
    for attr in ("href", "onclick", "onchange"):
        m = _DOPOSTBACK.search(a.get(attr) or "") or _WEBFORM.search(a.get(attr) or "")
        if m:
            return m.group(1), m.group(2), False
    if a.get("id") and a["id"] in bound:
        return bound[a["id"]][0], bound[a["id"]][1], False
    if el.tag in ("input", "button") and a.get("name") and (a.get("type") or "submit").lower() in (
            "submit", "image", "button"):
        return a["name"], a.get("value") or "", True
    return None


def _form_of(el, tree, form_sel: str | None):
    if form_sel:
        return tree.css_first(form_sel)
    p = el.parent
    while p is not None:
        if p.tag == "form":
            return p
        p = p.parent
    for f in tree.css("form"):        # the form that holds the framework's page state
        if any(f.css_first(f'input[name="{n}"]') for n in _STATE_FIELDS):
            return f
    return None


def _target_fields(pairs: list[tuple[str, str]], pag: dict) -> tuple[str | None, str | None]:
    if pag.get("target_field"):
        return pag["target_field"], pag.get("argument_field")
    names = {n for n, _ in pairs}
    for tf, af, markers in _FRAMEWORKS:
        if names & set(markers):
            return tf, af
    return None, None


def _submit(form, pairs: list[tuple[str, str]], page_url: str, encoding: str | None,
            action: str | None = None) -> Postback:
    """The request a browser sends when `form` is submitted with these fields (to `action` when a script set it)."""
    enc = (form.attributes.get("accept-charset") or "").split(" ")[0].strip() or encoding or "utf-8"
    try:
        body = urlencode(pairs, encoding=enc, errors="xmlcharrefreplace")
    except LookupError:
        body = urlencode(pairs, encoding="utf-8")
    action = urljoin(page_url, action if action is not None else (form.attributes.get("action") or "")) or page_url
    if (form.attributes.get("method") or "get").lower() != "post":
        parts = urlsplit(action)
        return Postback("GET", urlunsplit((parts.scheme, parts.netloc, parts.path, body, "")), None,
                        {"Referer": page_url})
    return Postback("POST", action, body.encode("ascii"),
                    {"Content-Type": "application/x-www-form-urlencoded", "Referer": page_url})


def build(tree, html: str, page_url: str, encoding: str | None, control, pag: dict,
          overrides: dict | None = None, default_target: str | None = None) -> Postback | str:
    """The request a browser sends when `control` is clicked (or changed). A string is the reason it cannot be built."""
    resolved = control_target(control, script_targets(html))
    if pag.get("target") and control.tag != "select":
        resolved = (pag["target"], pag.get("argument") or "", False)
    if resolved is None and default_target:
        resolved = (default_target, "", False)
    if resolved is None:
        # No event target, but the control's script names an address (onclick="setTargetAndSubmit('/l.do?p=2')"):
        # the script points its form at that address and submits it - or, with no form, opens it.
        url = jspager.js_url(control)
        if url:
            form = jspager.named_form(control, tree) or _form_of(control, tree, pag.get("form_sel"))
            if form is None:
                return Postback("GET", urljoin(page_url, url), None, {"Referer": page_url})
            return _submit(form, form_fields(form), page_url, encoding, action=url)
    form = _form_of(control, tree, pag.get("form_sel"))
    if form is None:
        return "no form around the next-page control"
    pairs = form_fields(form)
    if overrides:
        pairs = [(n, overrides[n] if n in overrides else v) for n, v in pairs]
    if resolved is None:
        return ("the next-page control has no postback target (no __doPostBack, no bound script, not a button, no "
                "address in its script)")
    target, argument, submitter = resolved
    if submitter:
        pairs.append((target, argument))
    else:
        tf, af = _target_fields(pairs, pag)
        if not tf:
            return "the form has no event-target field (__EVENTTARGET, PRADO_POSTBACK_TARGET)"
        pairs = [(n, v) for n, v in pairs if n not in (tf, af)]
        pairs.append((tf, target))
        if af:
            pairs.append((af, argument))
    return _submit(form, pairs, page_url, encoding)


def form_request(tree, page_url: str, encoding: str | None, pag: dict, value) -> Postback | str:
    """Pagination type "form": the form (pagination.form_sel) of the page just read, submitted with its page field
    (pagination.field) set to `value` - what a "set the page field, submit" script does. Other fields keep the
    values the page gave them (dates, session keys), plus pagination.fields as fixed overrides."""
    try:
        form = tree.css_first(pag.get("form_sel") or "")
    except Exception:
        form = None
    if form is None:
        return f"no form matches '{pag.get('form_sel')}'"
    over = {str(k): "" if v is None else str(v) for k, v in (pag.get("fields") or {}).items()}
    over[str(pag["field"])] = str(value)
    pairs = [(n, over[n] if n in over else v) for n, v in form_fields(form)]
    present = {n for n, _ in pairs}
    pairs += [(n, v) for n, v in over.items() if n not in present]
    return _submit(form, pairs, page_url, encoding)


def next_control(tree, next_sel: str):
    """The first enabled control the selector matches (a disabled 'next' on the last page does not count)."""
    try:
        nodes = tree.css(next_sel)
    except Exception:
        return None
    for n in nodes:
        el = _clickable(n)
        if not _disabled(el):
            return el
    return None


def next_request(tree, html: str, page_url: str, encoding: str | None, pag: dict) -> tuple[Postback | None, str | None]:
    """(request for the next page, None); (None, None) when the page has no enabled next control (the last page);
    (None, reason) when the control is there but cannot be clicked without a browser."""
    el = next_control(tree, pag.get("next_sel") or "")
    if el is None:
        return None, None
    req = build(tree, html, page_url, encoding, el, pag)
    return (None, req) if isinstance(req, str) else (req, None)


def _option_values(sel) -> tuple[int | None, list[tuple[int, str]]]:
    cur, vals = None, []
    opts = sel.css("option")
    for o in opts:
        v = (o.attributes.get("value") if o.attributes.get("value") is not None else o.text(strip=True)) or ""
        if not re.fullmatch(r"\s*\d{1,4}\s*", v):
            return None, []
        vals.append((int(v), v))
        if "selected" in o.attributes:
            cur = int(v)
    if cur is None and vals:
        cur = vals[0][0]
    return cur, vals


def resize_request(tree, html: str, page_url: str, encoding: str | None, pag: dict) -> Postback | None:
    """Choose the largest option of the results-per-page <select> (pagination.size_sel), as its change would post."""
    try:
        sel = tree.css_first(pag.get("size_sel") or "")
    except Exception:
        return None
    if sel is None or sel.tag != "select" or not sel.attributes.get("name"):
        return None
    cur, vals = _option_values(sel)
    if not vals or max(vals)[0] <= (cur or 0):
        return None
    name = sel.attributes["name"]
    req = build(tree, html, page_url, encoding, sel, pag, overrides={name: max(vals)[1]}, default_target=name)
    return None if isinstance(req, str) else req


def size_select_candidates(tree, rows_on_page: int) -> list[str]:
    """Selects that look like a results-per-page control: only numbers, the selected one equal to the rows shown,
    a larger one offered. Returned as CSS selectors, best first."""
    out = []
    for s in tree.css("form select[name]"):
        cur, vals = _option_values(s)
        if len(vals) < 2 or cur != rows_on_page or max(vals)[0] <= cur:
            continue
        sid = s.attributes.get("id") or ""
        out.append(f"select#{sid}" if re.fullmatch(r"[A-Za-z][\w-]*", sid) else f'select[name="{s.attributes["name"]}"]')
    return out


def pager_hint(html: str) -> str | None:
    """Concrete advice when the list page pages through JavaScript postbacks (what the model should declare)."""
    if not html:
        return None
    tree = HTMLParser(html)
    bound = script_targets(html)
    js = []
    for el in tree.css("a, input, button"):
        a = el.attributes
        act = f"{a.get('href') or ''} {a.get('onclick') or ''}"
        if "javascript:" in act.lower() or "__doPostBack" in act or (a.get("id") or "") in bound:
            js.append(el)
    if not js:
        return None
    nexts = [el for el in js if _NEXT_WORDS.search(el.html or "") or "next" in str(el.attributes.get("rel") or "")]
    has_state = any(tree.css_first(f'input[name="{n}"]') for n in _STATE_FIELDS)
    if not nexts and not has_state:
        return None
    ex = ""
    if nexts:
        el = nexts[0]
        sel = f"{el.tag}#{el.attributes['id']}" if re.fullmatch(r"[A-Za-z][\w-]*", el.attributes.get("id") or "") \
            else None
        ex = (f" The next-page control looks like {('next_sel ' + repr(sel)) if sel else 'this'}: "
              f"{(el.html or '')[:160]}")
    return ("hint: this pager does not use addresses: its links are JavaScript form posts (href \"javascript:...\", "
            "__doPostBack, hidden page state such as __VIEWSTATE / PRADO_PAGESTATE). Declare \"pagination\": "
            "{\"type\": \"postback\", \"next_sel\": CSS of the enabled next-page link}; never use a javascript: href "
            "as a next link." + ex)
