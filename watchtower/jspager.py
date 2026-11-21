"""Pagers driven by JavaScript that need no browser to follow - only what the script would send.

Three shapes, each declared by the model in the spec and verified by the dry run (page 2 must show new rows):
  - an address inside the next control's script (onclick="setTargetAndSubmit('/list.do?page=2')",
    location.href='?p=2'): js_url() finds it; postback.build submits the control's form to it (as a
    "set the action, submit" script does), a next_link pager follows it;
  - a form with a page field the script sets before submitting it (pagination type "form",
    postback.form_request);
  - rows appended by a script request ("load more" / infinite scroll): pagination type "load_more" sends the
    request the script sends ({offset} = rows already shown, {page}, {token:NAME} = a per-visit value from the list
    page such as a CSRF token) and reads the rows from the answer (HTML, or HTML inside JSON at html_path), as if
    appended to the list page's container.
For the model, pager_scripts_hint() quotes the scripts behind the pager, which the reduced page it sees leaves out.
"""
from __future__ import annotations

import html as html_lib
import json
import re
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import jmespath
from selectolax.lexbor import LexborHTMLParser as HTMLParser

# A string literal that is an address: absolute, root- or dot-relative, a bare query, or a server page name.
_JS_URL = re.compile(r"""(['"])((?:https?://|/|\./|\.\./|\?)[^'"\s<>]*|[\w.-]+(?:/[\w.-]+)*\.(?:do|action|aspx?|"""
                     r"""php|jsp|html?|cfm|cgi|xhtml|faces)(?:[?#][^'"\s<>]*)?)\1""")
_LITERAL = re.compile(r"""(['"])([^'"\\]{1,120})\1""")


def js_url(el) -> str | None:
    """The address a JavaScript pager control opens: the one address-like string literal in its onclick (or in a
    javascript: href). None when there is none, or several different ones (no guessing)."""
    a = el.attributes or {}
    found = []
    for attr in ("onclick", "href"):
        v = a.get(attr) or ""
        if attr == "href" and not v.strip().lower().startswith("javascript:"):
            continue
        for m in _JS_URL.finditer(v):
            u = m.group(2)
            if u not in ("/", "?", "./", "../", "//"):
                found.append(u)
    distinct = list(dict.fromkeys(found))
    return distinct[0] if len(distinct) == 1 else None


def named_form(el, tree):
    """A form the control's script names in a string literal (setTargetAndSubmit(url, 'searchForm')), or None."""
    a = el.attributes or {}
    names = {m.group(2) for attr in ("onclick", "href") for m in _LITERAL.finditer(a.get(attr) or "")}
    if not names:
        return None
    for f in tree.css("form"):
        fa = f.attributes or {}
        if (fa.get("name") and fa["name"] in names) or (fa.get("id") and fa["id"] in names):
            return f
    return None


# ------------------------------------------------------------------ load_more: the script's request
_TOKEN = re.compile(r"\{token:([^{}]+)\}")


def page_value(tree, html: str, name: str) -> str | None:
    """A per-visit value the list page carries under this name: a form field, a <meta>, or a script assignment
    (NAME: 'v', NAME = "v", "NAME": "v")."""
    for el in tree.css("input[name], meta[name]"):
        a = el.attributes or {}
        if a.get("name") == name:
            v = a.get("value") if el.tag == "input" else a.get("content")
            if v is not None:
                return v
    m = re.search(r"""(?<![\w$])['"]?""" + re.escape(name) + r"""['"]?\s*[:=]\s*(['"])([^'"\n]{0,500})\1""", html or "")
    return m.group(2) if m else None


def _fill(value, values: dict, tokens: dict):
    """Placeholders in a request value: {offset}, {page}, {token:NAME}. A value that is only {offset} or {page}
    stays a number (for JSON bodies)."""
    if isinstance(value, dict):
        return {k: _fill(v, values, tokens) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, values, tokens) for v in value]
    if not isinstance(value, str):
        return value
    for k, v in values.items():
        if value == "{" + k + "}":
            return v
    out = value
    for k, v in values.items():
        out = out.replace("{" + k + "}", str(v))
    return _TOKEN.sub(lambda m: tokens.get(m.group(1), ""), out)


def _flatten(prefix: str, v, out: list) -> None:
    """Form encoding of nested data the way jQuery's $.param writes it: a[b]=1, a[]=x."""
    if isinstance(v, dict):
        for k, x in v.items():
            _flatten(f"{prefix}[{k}]" if prefix else str(k), x, out)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _flatten(f"{prefix}[{i if isinstance(x, (dict, list)) else ''}]", x, out)
    elif v is None:
        out.append((prefix, ""))
    elif isinstance(v, bool):
        out.append((prefix, "true" if v else "false"))
    else:
        out.append((prefix, str(v)))


def tokens_needed(pag: dict) -> list[str]:
    blob = json.dumps([pag.get("url"), pag.get("fields"), pag.get("json")], ensure_ascii=False)
    return list(dict.fromkeys(_TOKEN.findall(blob)))


def read_tokens(pag: dict, tree, html: str) -> tuple[dict, str | None]:
    """The per-visit values the request needs, read from the list page: (values, None) or ({}, missing name)."""
    out = {}
    for name in tokens_needed(pag):
        v = page_value(tree, html, name)
        if v is None:
            return {}, name
        out[name] = v
    return out, None


def load_more_request(pag: dict, page_url: str, page_number: int, received: int, tokens: dict, fill_today):
    """The request the list page's script sends for more rows (a postback.Postback)."""
    from .postback import Postback
    values = {"offset": received, "page": page_number}
    url = urljoin(page_url, _fill(fill_today(str(pag.get("url") or "")), values, tokens))
    method = str(pag.get("method") or "POST").upper()
    headers = {"Referer": page_url, "X-Requested-With": "XMLHttpRequest",
               "Accept": "application/json, text/html, */*; q=0.01"}
    if pag.get("json") is not None:
        body = json.dumps(_fill(pag["json"], values, tokens), ensure_ascii=False).encode("utf-8")
        return Postback(method, url, body, {**headers, "Content-Type": "application/json"})
    pairs: list = []
    _flatten("", _fill(pag.get("fields") or {}, values, tokens), pairs)
    pairs = [(k, fill_today(v)) for k, v in pairs]
    if method == "GET":
        parts = urlsplit(url)
        q = parse_qsl(parts.query, keep_blank_values=True) + pairs
        return Postback("GET", urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), "")), None, headers)
    return Postback(method, url, urlencode(pairs).encode("ascii"),
                    {**headers, "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"})


def answer_html(resp, pag: dict) -> tuple[str | None, str | None]:
    """The rows' HTML in a load-more answer: (html, None) or (None, what is wrong)."""
    path = pag.get("html_path")
    if not path:
        return resp.text or "", None
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None, f"is not JSON (HTTP {resp.status_code}, {len(resp.text or '')} chars)"
    try:
        val = jmespath.search(path, data)
    except jmespath.exceptions.JMESPathError:
        return None, f"has no value at '{path}'"
    if isinstance(val, list) and all(isinstance(x, str) for x in val):
        val = "".join(val)
    if not isinstance(val, str):
        keys = list(data)[:10] if isinstance(data, dict) else type(data).__name__
        return None, f"has no HTML at '{path}' (keys {keys})"
    return val, None


def container_wrap(item) -> tuple[str, str]:
    """The markup around the list's rows on the first page (the row's ancestors, outermost first), so rows the
    script appends to that container match the spec's selectors exactly as on the first page."""
    chain = []
    p = item.parent
    while p is not None and p.tag not in ("-undef", "#document", None):
        if p.tag and not p.tag.startswith("-"):
            chain.append(p)
        p = p.parent
    opens, closes = [], []
    for n in reversed(chain):
        attrs = "".join(f' {k}="{html_lib.escape(v or "", quote=True)}"' if v is not None else f" {k}"
                        for k, v in (n.attributes or {}).items())
        opens.append(f"<{n.tag}{attrs}>")
        closes.append(f"</{n.tag}>")
    return "".join(opens), "".join(reversed(closes))


# ------------------------------------------------------------------ what the pager's scripts do (model hints)
_NEXT = re.compile(r"suivant|\bnext|weiter|n[äa]chste|volgende|siguiente|successiv|pr[oó]xim|n[äa]sta|seuraava|"
                   r"nast[eę]pn|dal[sš][ií]|k[öo]vetkez|forward|vorw[äa]rts", re.I)
_CALLED = re.compile(r"(?:^|[;\s(!])(?:javascript:\s*)?(?:return\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(")
_JS_BUILTINS = {"void", "return", "if", "function", "alert", "confirm", "setTimeout", "event", "this"}
_XHR = re.compile(r"\$\.(?:ajax|post|get|getJSON)\s*\(|\bfetch\s*\(|new\s+XMLHttpRequest|\baxios(?:\.\w+)?\s*\(|"
                  r"\.load\s*\(\s*['\"]")
# the request names the rows it wants: a data key (offset: n, "page": p) or a query parameter (?start=20)
_PAGING_KEY = re.compile(r"""(?:^|[\s{,'"])(?:offset|page\w*|start\w*|skip|limit|cursor|after|from)['"]?\s*:|"""
                         r"""[?&](?:offset|page\w*|start|skip|from|p)=""", re.I)


def _inline_scripts(tree) -> str:
    return "\n".join(s.text() or "" for s in tree.css("script") if not (s.attributes or {}).get("src"))


def _excerpt(text: str, start: int, end: int, cap: int = 1400) -> str:
    lines = [ln.strip() for ln in text[max(0, start):end].splitlines()]
    out = " ".join((ln[:180] + "…") if len(ln) > 180 else ln for ln in lines if ln)
    return out[:cap]


def _block_end(text: str, i: int, limit: int = 2500) -> int:
    """End of the {...} block that starts at the first '{' at or after i (strings are not parsed: good enough for
    a quote in a hint)."""
    j = text.find("{", i)
    if j < 0 or j - i > 200:
        return min(len(text), i + 300)
    depth = 0
    for k in range(j, min(len(text), j + limit)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return min(len(text), j + limit)


def function_source(scripts: str, name: str) -> str | None:
    short = re.escape(name.split(".")[-1])
    for pat in (rf"function\s+{short}\s*\(", rf"(?<![\w$.]){short}\s*[:=]\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|"
                                              rf"[\w$]+\s*=>)"):
        m = re.search(pat, scripts)
        if m:
            return _excerpt(scripts, m.start(), _block_end(scripts, m.start()), 700)
    return None


def _forms_named(src: str, tree) -> list:
    out = []
    for f in tree.css("form"):
        a = f.attributes or {}
        for key in (a.get("id"), a.get("name")):
            if key and re.search(r"""(['"#])""" + re.escape(key) + r"""['"]""", src):
                out.append(f)
                break
    return out


def _form_summary(f) -> str:
    from .postback import form_fields
    a = f.attributes or {}
    sel = f"form#{a['id']}" if re.fullmatch(r"[A-Za-z][\w-]*", a.get("id") or "") else \
        (f'form[name="{a["name"]}"]' if a.get("name") else "form")
    fields = ", ".join(f"{n}={v[:30]!r}" for n, v in form_fields(f)[:25])
    unvalued = [el.attributes.get("name") for el in f.css("input[type=hidden][name]")
                if el.attributes.get("value") is None]
    extra = f"; hidden fields without a value (set by the script): {unvalued[:6]}" if unvalued else ""
    return (f"{sel} (method {(a.get('method') or 'get').upper()}, action {a.get('action') or '(this page)'}; "
            f"fields {fields}{extra})")


def _control_selector(el) -> str:
    a = el.attributes or {}
    if re.fullmatch(r"[A-Za-z][\w-]*", a.get("id") or ""):
        return f"{el.tag}#{a['id']}"
    cls = [c for c in (a.get("class") or "").split() if re.fullmatch(r"[A-Za-z_][\w-]*", c)]
    if cls:
        return el.tag + "".join("." + c for c in cls[:2])
    for k in ("aria-label", "title"):
        if a.get(k) and '"' not in a[k]:
            return f'{el.tag}[{k}="{a[k]}"]'
    return el.tag


def pager_scripts_hint(html: str) -> str | None:
    """What the pager's JavaScript does, quoted for the model (the reduced page it sees has no scripts and no
    onclick): the script request that loads more rows, or the function a next control calls and the form it
    submits. None when there is nothing to say."""
    if not html:
        return None
    tree = HTMLParser(html)
    scripts = _inline_scripts(tree)
    notes = []
    for m in _XHR.finditer(scripts):
        end = _block_end(scripts, m.start(), 60000)
        if not _PAGING_KEY.search(scripts[m.start():end]):
            continue                     # a request that does not say which rows it wants is not a pager
        notes.append("hint: the list page loads more rows with a script request (infinite scroll / 'load more'), "
                     "so rows after the first page have no link. The script: «"
                     + _excerpt(scripts, m.start() - 300, end, 2400) + "». Declare "
                     "\"pagination\": {\"type\": \"load_more\", \"url\": the request's address, \"method\": its "
                     "method, \"fields\": its data (form-encoded; nested objects as objects) with \"{offset}\" where "
                     "the number of rows already shown goes (or \"{page}\" for a page number) and \"{token:NAME}\" "
                     "for a value the page sets per visit (a CSRF token, a session key - read freshly from the list "
                     "page on every read; NAME is its field/variable name on the page), \"html_path\": JMESPath of "
                     "the rows' HTML when the answer is JSON (e.g. \"items\"), omitted when the answer is HTML}. "
                     "list.items stays the selector of one row as on the first page.")
        break
    seen_funcs = set()
    for el in tree.css("[onclick], a[href^='javascript:'], a[href^='JavaScript:']"):
        a = el.attributes or {}
        label = " ".join(filter(None, [a.get("aria-label"), a.get("title"), a.get("class"), a.get("rel"),
                                       el.text(strip=True)[:60]]))
        if not _NEXT.search(label):
            continue
        code = (a.get("onclick") or "") + " " + (a.get("href") or "")
        if "__doPostBack" in code or "WebForm_" in code:
            continue                     # postback.pager_hint covers these
        url = js_url(el)
        ctl = f"{_control_selector(el)}: {(el.html or '')[:220]}"
        if url:
            notes.append(f"hint: the next-page control {ctl} opens the address {url!r} from its script (it is not "
                         f"an href). Declare \"pagination\": {{\"type\": \"postback\", \"next_sel\": CSS of that "
                         f"enabled control}} - the runner submits the control's form to that address as the script "
                         f"does (or \"next_link\" with the same next_sel: the runner follows the address).")
            break
        names = [n for n in _CALLED.findall(code) if n.split(".")[-1] not in _JS_BUILTINS]
        name = next((n for n in names if n not in seen_funcs), None)
        if not name:
            continue
        seen_funcs.add(name)
        src = function_source(scripts, name)
        if not src:
            continue
        forms = _forms_named(src, tree)
        what = (f" It submits {_form_summary(forms[0])}." if forms else "")
        notes.append(f"hint: the next-page control {ctl} runs the script «{src}».{what} If the script sets a page "
                     f"field and submits a form, declare \"pagination\": {{\"type\": \"form\", \"form_sel\": CSS of "
                     f"that form, \"field\": the page field's name, \"start\": the value that field stands for on the "
                     f"FIRST page (the script on page 1 sets start+step to show page 2), \"step\": 1, \"next_sel\": "
                     f"CSS of the enabled next-page control}} - the runner submits the form from each page with the "
                     f"field set; if the list address takes that field as a query parameter, {{\"type\": \"page\", "
                     f"\"param\": the field, \"start\": ...}} works too.")
        break
    return " ".join(notes)[:4000] if notes else None
