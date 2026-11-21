"""Locate: walk from a start address to the tender list (attach/repair only; regular reads never come here).

Best-first search over a frontier of same-site links. The cheap model proposes (the page type and its top 3 links
with a confidence); static evidence decides. A page counts as the list only with evidence: repeated tender-like rows
(a link and a date per row, deadline words) or a tender-like JSON array (the address itself, an API named in the
page source, JSON embedded in the page, or JSON captured from the app in a browser). Login, blocked, empty and
dead-end pages close their branch and the walk backtracks to the next-best open link; a login page still offers its
links (portals often link from the sign-in page to their public area). Extra candidates come from GET search forms
submitted empty and from the site's sitemap (robots.txt `Sitemap:` entries, /sitemap.xml).
Harder ways in: a browser render presses the page's search control (default criteria) when the app loads no
tender-like data by itself, or when its data cannot be replayed without the browser; the address the app moves to
is the candidate. A list page found in one application of the site (first path segment) is also tried under the
application the given address belongs to, and under the one an excerpt's tender forwards into (one tender of an
excerpt is opened for that). Raw pages of a JavaScript app already rendered are frames: rendered, never classified.

The walk keeps its state, so learn() can ask for the next-ranked candidate when the spec fails on the first one.
Budgets per walk (attach or repair): BUDGET below. Every decision goes to ctx.note (prefix 'locate') and to
Walk.trace (structured, the benchmark's dataset). No portal-specific rules: only generic, multilingual signals.
"""
from __future__ import annotations

import asyncio
import gzip
import heapq
import html as htmllib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import runner
from .extract import _normalize_months, clean_text, page_text, reduce_html
from .http import BudgetExceeded
from .llm import LLMUnavailable, ModelBudgetExhausted
from .onboard import (_api_candidates, _evidence_of_block, _first_list_of_dicts, _js_shell, _score_json, _try_json,
                      _untried_entry, captured_endpoints, open_data_records_candidates, replay_request)
from .pipeline import AttachFailed, PortalBlocked

BUDGET = {"pages": 15, "classify": 12, "renders": 3, "probes": 10, "sitemap": 3, "feeds": 3}
MODEL_RESERVE = 8        # model calls the job keeps for synthesis + blind verification after the walk
PUSH_MIN = 0.15          # links ranked below this are not worth a request
SITEMAP_BELOW = 0.35     # read the sitemap once the best open link is weaker than this (or none is left)
ITEM_PRIO = 0.5          # one tender of an excerpt: after the strong links (it costs a page, it is rarely needed)
CONTEXT_PRIO = 0.95      # a list page tried under the application the tenders / the given address belong to
RENDER_WAIT_S = 120.0    # one browser render at most (the app's requests are paced at 1/s)
WALK_SECONDS = float(os.environ.get("WATCHTOWER_LOCATE_SECONDS", "300"))  # soft wall-clock cap for one walk
TIER_RANK = {"A": 3, "B": 2, "C": 1}


# ------------------------------------------------------------------------------------------- text + URL helpers
def _fold(s: str) -> str:
    """Casefold and drop diacritics ('Öffentliche' -> 'offentliche', 'ogłoszenia' -> 'ogloszenia')."""
    s = (s or "").replace("ł", "l").replace("Ł", "L").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


def _url_words(url: str) -> str:
    p = urlsplit(url)
    return re.sub(r"[/_\-.=&?+%;:,]+", " ", _fold(unquote(f"{p.path} {p.query}")))


def _key(url: str) -> str:
    """Visited-set key: scheme/host/'www.' insensitive, no fragment, no session id, sorted query, no trailing '/'."""
    p = urlsplit(url)
    host = (p.hostname or "").lower().removeprefix("www.")
    path = re.sub(r";jsessionid=[^/?#]*", "", p.path, flags=re.I).rstrip("/") or "/"
    q = sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
               if k.lower() not in ("jsessionid", "phpsessid", "sid", "sessionid", "cid"))
    return f"{host}{':' + str(p.port) if p.port else ''}{path}?{urlencode(q)}"


def _site_base(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _context(url: str) -> str | None:
    """The first path segment: on Java/.NET servers the application (context root) a page belongs to."""
    segs = [s for s in urlsplit(url).path.split("/") if s]
    return segs[0] if segs and "." not in segs[0] else None


def _in_context(url: str, ctx: str) -> str | None:
    """The same page under another application of the same site (/A/company/welcome.do -> /B/company/welcome.do)."""
    p = urlsplit(url)
    segs = p.path.split("/")
    rest = [s for s in segs if s]
    if len(rest) < 2 or _context(url) is None:
        return None
    i = next(i for i, s in enumerate(segs) if s)
    segs[i] = ctx
    return urlunsplit((p.scheme, p.netloc, "/".join(segs), p.query, ""))


def _app_sig(html: str) -> tuple | None:
    """The page's script bundle (paths of its external scripts): every route of one JavaScript app shares it."""
    srcs = sorted({urlsplit(s.attributes.get("src") or "").path for s in HTMLParser(html or "").css("script[src]")}
                  - {""})
    return tuple(srcs) if srcs else None


# Multilingual anchor-text / URL signals. Positive: words that lead to tender lists. Negative: sign-in, legal, help,
# news, buyer areas, award archives. Generic procurement vocabulary only - never a portal's own names.
_POS = [
    (1.0, r"ausschreibung|bekanntmachung|tender|opportunit|appels?\s*d.?\s*offres?|avis\s+d.?\s*appel|avis de marche"
          r"|aankondiging|aanbesteding|licitacion|bandi?\b|bandi di gara|gare d.?appalto|appalti|przetarg|ogloszen"
          r"|postepowan|upphandling|udbud|\banbud|hankin|zakazk|zamowien|narocil|kozbeszerz|achizitii|concursos?\b"
          r"|call for (?:tender|competition)|contract notice"),
    (0.7, r"consultation|vergabe|procurement|marches? publics?|\bmarches\b|opdrachten|convocatori|avvisi"
          r"|procedimento|notices?\b|publicaties|veroffentlichung|anuncios?\b|contracts?\b|contrataci"),
    (0.45, r"search|suche\b|recherch|zoek|busqueda|buscador|ricerca|wyszuk|\bsok\b|haku|vyhledav|find\b|finden"),
]
_BOOST = r"\b(?:alle|all|toutes|tous|todas|todos|tutti|wszystkie|alla|kaikki|aktuell\w*|current|open|offen\w*|laufend\w*" \
         r"|en cours|lopend\w*|abiert\w*|in corso|vigent\w*|actuel\w*|biezac\w*)\b"
_NEG = [
    (1.0, r"\blog\s?in\b|\blogon\b|anmeld|einlogg|sign\s?in|sign\s?up|registr|rejestr|inscri|connexion|se connecter"
          r"|identifi|passwor|passwort|wachtwoord|mot de passe|forgot|vergessen"),
    (0.8, r"impressum|imprint|datenschutz|privacy|cookie|barrierefrei|accessib|kontakt|contact|nutzungsbeding"
          r"|\bterms\b|\bagb\b|disclaimer|mentions legales|aviso legal|\blegal\b"),
    (0.6, r"\bfaq\b|hilfe|\bhelp\b|\baide\b|ayuda|support|handbuch|manual|anleitung|leitfaden|tutorial|schulung"
          r"|training|webinar|video|newsletter|presse|\bpress\b|\bnews\b|aktuelles\b|neuigkeiten|\bjobs?\b|karriere"
          r"|career|about|uber uns|qui sommes|over ons|\brss\b"),
    (0.4, r"auftraggeber|vergabestelle|\bbuyers?\b|acheteur|contracting authorit|aanbestedende dienst"
          r"|organo de contratacion|poder adjudicador|\baward|zuschlag|vergebene|attribu|gunning|adjudicac|archiv"),
]
_NO_LOGIN = r"\bohne\s+(?:anmeld|registr|login)|without\s+(?:log|sign|regist)|sans\s+(?:inscri|connexion|identif)" \
            r"|sin\s+(?:registr|identific)|zonder\s+(?:inlog|registr|aanmeld)|senza\s+(?:registr|login)|bez\s+(?:logow|rejestr)"
_FILE_RE = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|zip|rar|7z|gz|tgz|bz2|tar|jpe?g|png|gif|svg|ico|webp|css|js|ics|mp4|mp3"
                      r"|wav|ogg|webm|avi|mov|odt|ods|rtf|exe|msi|dmg|pkg|deb|rpm|apk|iso|img|bin|jar|war|woff2?|ttf"
                      r"|eot|p7s|sig)$", re.I)
MAX_PAGE_BYTES = 5_000_000      # a page larger than this is not a tender list worth parsing (it is a file)
MAX_FEED_BYTES = 40_000_000     # an ATOM/RSS change feed may be large (one file of a chain of ~10 MB files)
# Links to a portal's official syndication / open-data feed: offered when the HTML route does not find a list.
_FEED_HREF = re.compile(r"\.(?:atom|rss)(?:[?#]|$)|[/=_-](?:atom|rss|feed|syndication|sindicacion)(?:[/?#._-]|$)", re.I)
_FEED_TEXT = re.compile(r"\b(?:rss|atom|feed|syndicat\w*|sindicaci\w*|open\s*data|datos\s+abiertos|donnees\s+ouvertes"
                        r"|offene\s+daten|open\s+data|dati\s+aperti)\b", re.I)
FEED_PRIO = 0.32                # below strong list links, above the sitemap's weak guesses


def feed_links(html: str, base: str) -> list[tuple[str, str]]:
    """(url, why) of feeds a page offers: <link rel=alternate type=application/atom+xml|rss+xml>, then links whose
    address or text names a feed / syndication / open data."""
    tree = HTMLParser(html or "")
    out, seen = [], set()
    for ln in tree.css("link[href]"):
        t = (ln.attributes.get("type") or "").lower()
        if "atom+xml" in t or "rss+xml" in t:
            u = urljoin(base, ln.attributes.get("href") or "")
            if u.startswith(("http://", "https://")) and u not in seen:
                seen.add(u)
                out.append((u, f"<link rel=alternate type={t}>"))
    for a in tree.css("a[href]"):
        href = (a.attributes.get("href") or "").strip()
        if not href or href.lower().startswith(("#", "javascript:", "mailto:")):
            continue
        u = urljoin(base, href).split("#")[0]
        text = clean_text(a.text() or a.attributes.get("title") or "") or ""
        if u.startswith(("http://", "https://")) and u not in seen and (_FEED_HREF.search(urlsplit(u).path + "?" +
                                                                                           urlsplit(u).query)
                                                                         or _FEED_TEXT.search(_fold(text))):
            seen.add(u)
            out.append((u, f"feed link '{text[:40]}'"))
    return out[:12]


def keyword_score(text: str, url: str) -> float:
    """-1..1: how strongly an anchor text and its URL point to a tender list (static, no model)."""
    if _FILE_RE.search(urlsplit(url).path or ""):
        return -1.0
    t, u = _fold(text), _url_words(url)
    pos = 0.0
    for w, pat in _POS:
        if re.search(pat, t):
            pos = max(pos, w)
        elif re.search(pat, u):
            pos = max(pos, 0.8 * w)
    if pos and (re.search(_BOOST, t) or re.search(_BOOST, u)):
        pos += 0.25
    negs = _NEG[1:] if re.search(_NO_LOGIN, t) else _NEG   # 'ohne Anmeldung' / 'without login' is a public way in
    neg = max([w for w, pat in negs if re.search(pat, t) or re.search(pat, u)] or [0.0])
    return round(max(-1.0, min(1.0, pos - neg)), 3)


def _prio(conf: float | None, kw: float, depth: int) -> float:
    """Frontier priority: the model's confidence leads, anchor words support or veto, depth costs a little."""
    if conf is not None:
        p = 0.25 + 0.65 * conf + 0.15 * max(kw, 0.0)
    else:
        p = 0.55 * max(kw, 0.0)
    if kw < 0:
        p += 0.5 * kw
    return round(p - 0.03 * depth, 3)


# ---------------------------------------------------------------------------------------- static page evidence
_DATE_RE = re.compile(
    r"(?<!\d)(?:\d{1,2}[./-]\s?\d{1,2}[./-]\s?(?:19|20)\d{2}|(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}"
    r"|\d{1,2}\.?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+(?:19|20)\d{2}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+(?:19|20)\d{2})(?!\d)", re.I)
_DEADLINE_RE = re.compile(
    r"frist|deadline|closing|sluiting|uiterste|date limite|remise des (?:offres|plis)|cloture|plazo|fecha limite"
    r"|\btermin|scadenza|einsendeschluss|ablauf|due date|submission|angebotsabgabe|teilnahmeantr|offertes|prazo"
    r"|lhuta|hatarid|skladania|data otwarcia")
_LINKISH = "a[href], [onclick], [data-href], [data-url], [data-id], button"


def _row_dates(mbr) -> list:
    """The dates of a row-like element that also carries a link (else [])."""
    if not (mbr.tag == "a" or mbr.attributes.get("onclick") or mbr.css_first(_LINKISH) is not None):
        return []
    return _DATE_RE.findall(_normalize_months((mbr.text(separator=" ") or "")[:3000]))


def row_evidence(html: str) -> dict:
    """The strongest group of repeated sibling elements (same tag + class) whose members carry a link and a date.
    strong = >= 3 such rows and deadline words (or two dates per row, e.g. published + deadline)."""
    best = {"rows": 0, "members": 0, "sel": None, "deadline": False, "multiDate": 0}
    best_nodes: list = []
    tree = HTMLParser((html or "")[:3_000_000])
    for tag in ("script", "style", "noscript", "svg", "select", "nav", "header", "footer", "template"):
        for n in tree.css(tag):
            n.decompose()
    body = tree.body
    if body is not None:
        scanned = 0
        for node in body.traverse():
            kids = list(node.iter())
            if len(kids) < 3:
                continue
            scanned += 1
            if scanned > 4000:
                break
            groups: dict[tuple[str, str], list] = {}
            for c in kids:
                cls = " ".join(sorted((c.attributes.get("class") or "").split())[:2])
                groups.setdefault((c.tag, cls), []).append(c)
            for (tag, cls), members in groups.items():
                if len(members) < 3 or tag in ("option", "br", "hr", "img", "input", "meta", "link"):
                    continue
                sample = members[:40]
                hits = multi = 0
                dl = False
                for mbr in sample:
                    dates = _row_dates(mbr)
                    if not dates:
                        continue
                    txt = _normalize_months((mbr.text(separator=" ") or "")[:3000])
                    hits += 1
                    multi += len(set(dates)) >= 2
                    dl = dl or bool(_DEADLINE_RE.search(_fold(txt)))
                rows = round(len(members) * hits / len(sample))
                if rows > best["rows"]:
                    best = {"rows": rows, "members": len(members), "sel": f"{tag}.{cls.split()[0]}" if cls else tag,
                            "deadline": dl, "multiDate": round(len(members) * multi / len(sample))}
                    best_nodes = members
        if best["rows"] and not best["deadline"]:  # the deadline label is often only in the table head
            heads = " ".join((n.text(separator=" ") or "")[:300]
                             for n in tree.css("th, thead, dt, label, [class*=head], [class*=label]")[:300])
            best["deadline"] = bool(_DEADLINE_RE.search(_fold(heads)))
    best["weak"] = best["rows"] >= 3
    # the rows' own links open single tenders: items, not a way to the list
    best["hrefs"] = [a.attributes.get("href") for m in best_nodes[:300] if _row_dates(m)
                     for a in ([m] if m.tag == "a" else m.css("a[href]")) if a.attributes.get("href")] \
        if best["weak"] else []
    best["strong"] = best["weak"] and (best["deadline"] or best["multiDate"] * 2 >= best["rows"])
    return best


def embedded_json(html: str) -> tuple[int, int] | None:
    """(score, rows) of the most tender-like JSON array embedded in the page (application/json, __NEXT_DATA__)."""
    best = None
    for s in HTMLParser(html or "").css("script"):
        typ = (s.attributes.get("type") or "").lower()
        if "json" not in typ and s.attributes.get("id") not in ("__NEXT_DATA__", "__NUXT_DATA__"):
            continue
        txt = s.text() or ""
        if not 20 <= len(txt) <= 3_000_000:
            continue
        try:
            data = json.loads(txt)
        except ValueError:
            continue
        sc = _score_json(data)
        if sc and (best is None or sc > best[0]):
            best = (sc, len(_first_list_of_dicts(data) or []))
    return best


_JS_LOCATION = re.compile(r"""(?:window\.|document\.|top\.|self\.)?location(?:\.href)?\s*=\s*['"]([^'"]+)['"]"""
                          r"""|location\.(?:replace|assign)\(\s*['"]([^'"]+)['"]""")


def soft_redirects(html: str, base: str, shell: bool) -> list[str]:
    """Where a page sends the visitor without a server redirect: meta refresh and frames always; a script that
    sets location only when the page itself is (almost) empty (otherwise it is usually a click handler)."""
    tree = HTMLParser(html or "")
    out = []
    for m in tree.css("meta[http-equiv]"):
        if (m.attributes.get("http-equiv") or "").lower() == "refresh":
            mt = re.search(r"url\s*=\s*['\"]?([^'\";]+)", m.attributes.get("content") or "", re.I)
            if mt:
                out.append(mt.group(1).strip())
    out += [f.attributes.get("src") for f in tree.css("frame[src], iframe[src]") if f.attributes.get("src")]
    if shell:
        for s in tree.css("script"):
            for mt in _JS_LOCATION.finditer(s.text() or ""):
                out.append(mt.group(1) or mt.group(2))
    seen, urls = set(), []
    for u in out:
        full = urljoin(base, u.strip())
        if full.startswith(("http://", "https://")) and _key(full) not in seen and _key(full) != _key(base):
            seen.add(_key(full))
            urls.append(full)
    return urls[:5]


def search_forms(html: str, base: str) -> list[dict]:
    """Search forms as links: the action URL with the form's own default values (an empty search). Many portals
    show the full list only as the result of a search. A POST search form is offered as the same request by GET
    (server frameworks commonly accept both; a GET reads, it changes nothing): after the GET forms, and never a form
    whose name or address points to sign-in, registration, language, consent or contact."""
    gets, posts = [], []
    for form in HTMLParser(html or "").css("form"):
        method = (form.attributes.get("method") or "get").strip().lower()
        if method not in ("get", "post") or form.css_first("input[type=password]"):
            continue
        action = urljoin(base, (form.attributes.get("action") or "").strip() or base)
        if not action.startswith(("http://", "https://")):
            continue
        if method == "post":
            ident = " ".join(form.attributes.get(a) or "" for a in ("name", "id", "class"))
            if keyword_score(ident, action) < 0 or re.search(r"lang|cookie|consent|newsletter|logout", _fold(ident + action)):
                continue
        values: list[tuple[str, str]] = []
        texty = False
        for inp in form.css("input"):
            typ = (inp.attributes.get("type") or "text").lower()
            name = inp.attributes.get("name")
            if not name or typ in ("submit", "button", "image", "reset", "file", "password"):
                continue
            if typ in ("checkbox", "radio"):
                if "checked" in inp.attributes:
                    values.append((name, inp.attributes.get("value") or "on"))
                continue
            values.append((name, inp.attributes.get("value") or ""))
            texty = texty or typ in ("text", "search")
        selects = form.css("select")
        for sel in selects:
            name = sel.attributes.get("name")
            opt = sel.css_first("option[selected]") or sel.css_first("option")
            if name and opt is not None:
                values.append((name, opt.attributes.get("value") if opt.attributes.get("value") is not None
                               else (clean_text(opt.text()) or "")))
        if not (texty or selects):
            continue
        btn = form.css_first("button, input[type=submit]")
        label = clean_text((btn.text() if btn is not None and btn.tag == "button" else
                            (btn.attributes.get("value") if btn is not None else "")) or "") or ""
        parts = urlsplit(action)   # a GET submission replaces the action's query with the form data (HTML spec)
        if method == "get":
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(values), ""))
            gets.append({"text": f"[search form] {label or 'search'}"[:80], "url": url, "form": True})
        else:  # keep the action's own query (e.g. '?method=search'), add the fields
            q = parse_qsl(parts.query, keep_blank_values=True) + values
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), ""))
            posts.append({"text": f"[search form, sent as GET] {label or 'search'}"[:80], "url": url, "form": True,
                          "post": True})
    return (gets + posts)[:3]


# ------------------------------------------------------------------------------------------------- the walk
@dataclass
class Node:
    url: str
    depth: int
    via: str                  # start | model | link | form | sitemap | root
    prio: float
    text: str = ""
    parent: "Node | None" = None
    why: str = ""
    referer: str | None = None    # the page the link was on: sent as a browser does (stateful apps check it)


@dataclass
class Walk:
    ctx: object
    start: str
    budget: dict = field(default_factory=lambda: dict(BUDGET))
    used: dict = field(default_factory=lambda: {k: 0 for k in BUDGET})
    trace: list = field(default_factory=list)
    candidates: list = field(default_factory=list)
    blocks: list = field(default_factory=list)

    def __post_init__(self):
        self._heap: list = []
        self._seq = 0
        self._queued: dict[str, float] = {}
        self._visited: set[str] = set()
        self._probed: set[str] = set()
        self._returned: set[str] = set()
        self._sites = {_site_base(self.start)}
        self._sitemap_done = False
        self._captcha = False
        self._spent = 0.0                 # seconds spent inside next() so far (not the caller's time in between)
        self._t0 = time.monotonic()
        # Applications of the site the list may live in (first path segment): the given address's own, and those a
        # server redirect crossed into (e.g. a tender link of one application forwarding into another). A list found
        # in one is also tried under the others: the same software mounts the same pages in each.
        self._contexts: set[str] = {c for c in [_context(self.start)] if c}
        self._item_tried = False          # one tender of an excerpt is opened: where its page lives tells the app
        self._spa_sigs: set[tuple] = set()  # script bundles of JavaScript apps (their raw pages are empty frames)
        self._push(Node(self.start, 0, "start", 1.0))
        root = urlunsplit((urlsplit(self.start).scheme, urlsplit(self.start).netloc, "/", "", ""))
        if _key(root) != _key(self.start):  # backtracking target of last resort: the site's home page
            self._push(Node(root, 1, "root", 0.2, why="site home"))

    # ----------------------------------------------------------------- bookkeeping
    def note(self, msg: str) -> None:
        self.ctx.note(f"locate: {msg}")

    def same_site(self, url: str) -> bool:
        h = _site_base(url)
        return any(h == s or h.endswith("." + s) for s in self._sites)

    def _push(self, node: Node) -> bool:
        k = _key(node.url)
        if k in self._visited or (node.prio < PUSH_MIN and node.via != "start"):
            return False
        if node.via != "start" and _FILE_RE.search(urlsplit(node.url).path or ""):
            return False        # a file (installer, PDF, archive) is never a list, whoever proposes it
        if self._queued.get(k, -9) >= node.prio:
            return False
        self._queued[k] = node.prio
        self._seq += 1
        heapq.heappush(self._heap, (-node.prio, self._seq, node))
        return True

    def _pop(self) -> Node | None:
        while self._heap:
            _p, _s, node = heapq.heappop(self._heap)
            k = _key(node.url)
            if k in self._visited or self._queued.get(k) != node.prio:
                continue
            return node
        return None

    def _best_open(self) -> float:
        for p, _s, node in sorted(self._heap)[:50]:
            k = _key(node.url)
            if k not in self._visited and self._queued.get(k) == node.prio:
                return -p
        return 0.0

    def _left(self, what: str) -> bool:
        return self.used[what] < self.budget[what]

    def _elapsed(self) -> float:
        return self._spent + (time.monotonic() - self._t0)

    def _model_calls_left(self) -> int | None:
        llm = self.ctx.llm
        try:
            limit = int(getattr(llm.s, "model_max_calls_per_job", 0) or 0)
            used = int(llm._job_calls.get(self.ctx.run_id, 0))
        except (AttributeError, TypeError, ValueError):
            return None
        return None if limit <= 0 else limit - used

    def _block(self, node: Node, url: str, reason: str) -> None:
        self.blocks.append({"url": url, "reason": reason, "via": node.via, "prio": node.prio})

    def _path(self, node: Node | None) -> list[str]:
        out = []
        while node is not None:
            out.append(node.url)
            node = node.parent
        return out[::-1]

    def _record(self, node: Node, final: str, **kw) -> dict:
        rec = {"n": self.used["pages"], "url": node.url, "final": final, "via": node.via, "depth": node.depth,
               "prio": node.prio, **kw}
        self.trace.append(rec)
        return rec

    # ----------------------------------------------------------------- public
    async def next(self) -> dict | None:
        """Search on until a new confirmed list (tier A) turns up or a budget is used up; then hand out the best
        candidate not handed out before (A, else B = rows the model did not call a list, else C = the model's
        'list' without evidence). None when nothing is left."""
        self._t0 = time.monotonic()
        try:
            return await self._next()
        finally:
            self._spent += time.monotonic() - self._t0
            self._t0 = time.monotonic()

    async def _pop_feed(self) -> Node | None:
        """The best queued feed / open-data link (kept for when the HTML route found no usable list)."""
        best = None
        for p, s, node in self._heap:
            k = _key(node.url)
            if node.via == "feed" and k not in self._visited and self._queued.get(k) == node.prio                     and (best is None or (p, s) < best[0]):
                best = ((p, s), node)
        if best is None:
            return None
        self._queued[_key(best[1].url)] = -1.0          # taken out of the heap's view
        return best[1]

    async def _next(self) -> dict | None:
        while True:
            if not self._left("pages") or self._elapsed() > WALK_SECONDS:
                # The page budget is spent without a confirmed list: a few extra pages for the official feed /
                # open-data links the pages offered (the portal's own substitute when its website cannot be read).
                if any(c["tier"] in ("A", "B") for c in self.candidates) or not self._left("feeds")                         or self._elapsed() > WALK_SECONDS + 120:
                    break
                node = await self._pop_feed()
                if node is None:
                    break
                self.used["feeds"] += 1
                self.used["pages"] -= 1      # _visit counts a page; this one is on the feed reserve
                try:
                    new = await self._visit(node)
                except BudgetExceeded as e:
                    self.note(f"request budget used up ({e})")
                    break
                if new is not None and new["tier"] == "A":
                    break
                continue
            if not self._sitemap_done and self._best_open() < SITEMAP_BELOW:
                try:
                    await self._sitemap()
                except BudgetExceeded as e:
                    self.note(f"request budget used up ({e})")
                    break
            node = self._pop()
            if node is None:
                break
            try:
                new = await self._visit(node)
            except BudgetExceeded as e:
                self.note(f"request budget used up ({e})")
                if not self._unreturned():
                    raise
                break
            if new is not None and new["tier"] == "A":
                break
        left = self._unreturned()
        if not left:
            return None
        best = left[0]
        self._returned.add(best["key"])
        others = [f"{c['tier']} {c['url']}" for c in left[1:4]]
        self.note(f"chose {best['url']}" + (f" (API {best['api'][0]})" if best.get("api") else "") +
                  f" [tier {best['tier']}: {best['evidence']}]; path {' -> '.join(best['path'])}; "
                  f"used {self._usage()}" + (f"; other candidates: {others}" if others else ""))
        out = {k: v for k, v in best.items() if k not in ("key", "rank")}
        out["captcha"] = self._captcha
        out["walk"] = self
        out["candidates"] = [{"url": c["url"], "api": c["api"][0] if c.get("api") else None, "tier": c["tier"],
                              "evidence": c["evidence"]} for c in self._ranked()]
        return out

    def failure(self) -> Exception:
        """What to raise when the walk found nothing: a block we can show evidence for, else a dead end."""
        why = f"{self.used['pages']} pages, {self.used['classify']} model looks, {self.used['renders']} browser renders"
        # Blocked only where the list was expected (the given address, or a link the walk ranked high); a login
        # page met on a side branch is not the portal's answer.
        expected = [b for b in self.blocks if b["via"] == "start" or b["prio"] >= 0.5]
        if expected:
            b = expected[0]
            return PortalBlocked(f"{b['reason']} (at {b['url']}; no public tender list found within {why})")
        dead = "; ".join(f"{r['url']}: {r.get('verdict')}" for r in self.trace[-3:] if r.get("verdict"))
        side = f"; a login was required at {self.blocks[0]['url']}" if self.blocks else ""
        return AttachFailed(f"could not find the tender list from {self.start} ({why}{side}; last: {dead})"[:600])

    def _usage(self) -> str:
        return (f"{self.used['pages']} pages, {self.used['classify']} classify, {self.used['renders']} renders, "
                f"{self.used['probes']} API probes, {self.used['sitemap']} sitemap fetches, "
                f"{round(self._elapsed(), 1)}s")

    def _ranked(self) -> list[dict]:
        return sorted(self.candidates, key=lambda c: c["rank"], reverse=True)

    def _unreturned(self) -> list[dict]:
        return [c for c in self._ranked() if c["key"] not in self._returned]

    @staticmethod
    def _pressed(rendered: dict) -> str:
        return f" after pressing its search control '{rendered['submitted']}'" if rendered.get("submitted") else ""

    def _transplant(self, urls: list[str], why: str) -> None:
        """Queue the same pages under the application(s) the list belongs to (see self._contexts)."""
        for u in urls:
            for c in sorted(self._contexts - {_context(u)}):
                v = _in_context(u, c)
                if v and self.same_site(v) and self._push(Node(v, 1, "context", CONTEXT_PRIO, f"[/{c}/]", None,
                                                                f"{why}; the same page under /{c}/")):
                    self.note(f"queued {v}: {why}; the same page under the application /{c}/")

    def _new_context(self, asked: str, final: str) -> None:
        """A server redirect from one application into another (same site), e.g. a tender link of a central portal
        forwarding into the platform that runs the tender: that platform is where the list belongs."""
        a, b = _context(asked), _context(final)
        if not (a and b and a != b and self.same_site(final)) or b in self._contexts:
            return
        self._contexts.add(b)
        self.note(f"{asked} forwards from the application /{a}/ into /{b}/")
        self._transplant([c["url"] for c in self._ranked()], "a list page found in another application")

    def _add_candidate(self, node: Node, url: str, html: str, tier: str, evidence: str, api=None,
                       strength: float = 0.0, **extra) -> dict | None:
        key = _key(api[0]) if api else _key(url)
        if any(c["key"] == key for c in self.candidates):
            return None
        if self._contexts and _context(url) in self._contexts:
            strength += 0.1     # in the application the given address / the tenders belong to
        elif self._contexts:
            self._transplant([url], f"candidate {tier} lies outside the application(s) {sorted(self._contexts)}")
        cand = {"url": url, "html": html, "api": api, "tier": tier, "evidence": evidence, "key": key,
                "path": self._path(node) if _key(node.url) == _key(url) else self._path(node) + [url],
                "rank": (TIER_RANK[tier], round(strength, 3), -len(self.candidates)), **extra}
        self.candidates.append(cand)
        self.note(f"candidate {tier}: {url}" + (f" API {api[0]}" if api else "") + f" ({evidence})")
        return cand

    # ----------------------------------------------------------------- one page
    async def _visit(self, node: Node) -> dict | None:
        ctx = self.ctx
        self.used["pages"] += 1
        self._visited.add(_key(node.url))
        try:
            try:
                resp, refused = await self._fetch(node.url, referer=node.referer)
            except (httpx.TimeoutException, httpx.TooManyRedirects) as e:
                if node.via != "start":
                    raise
                # a slow first answer, or a cookie-check redirect loop (the cookie is set now): once more
                self.note(f"{node.url}: {type(e).__name__}; trying the given address once more")
                resp, refused = await self._fetch(node.url, referer=node.referer)
        except BudgetExceeded:
            raise
        except Exception as e:  # network error, redirect loop: a dead branch, not the end of the walk
            self._record(node, node.url, verdict=f"error {type(e).__name__}")
            self.note(f"p{self.used['pages']} {node.url}: {type(e).__name__} {str(e)[:120]}; backtracking")
            return None
        if refused:
            self._record(node, str(resp.url), status=resp.status_code, verdict=f"dead end ({refused})")
            self.note(f"p{self.used['pages']} {node.url}: {refused}; not read, backtracking")
            return None
        final = str(resp.url)
        self._visited.add(_key(final))
        status = resp.status_code
        ctype = resp.headers.get("content-type", "")
        head = f"p{self.used['pages']} [{node.via} d{node.depth} {node.prio}] {node.url}" + \
            (f" -> {final}" if _key(final) != _key(node.url) else "") + f" HTTP {status}"
        if node.via == "start" and status < 400 and not self.same_site(final):
            self._sites.add(_site_base(final))    # the portal's canonical host (redirect from the given address)
        if status in (401, 403):
            reason = f"access denied (HTTP {status}); a login is required"
            self._block(node, final, reason)
            self._record(node, final, status=status, verdict="blocked")
            self.note(f"{head}: {reason}; branch closed")
            return None
        if status >= 400:
            self._record(node, final, status=status, verdict="dead end")
            self.note(f"{head}: dead end")
            return None
        self._new_context(node.url, final)
        if node.via == "item":   # one tender, opened only to see which application it lives in
            self._record(node, final, status=status, verdict="item (one tender)")
            self.note(f"{head}: one tender of the excerpt; it lives under /{_context(final)}/")
            return None
        # the address may itself be an ATOM/RSS feed (a portal's official syndication of its tenders)
        from . import feed as feedmod
        if feedmod.looks_like_feed(ctype, resp.content[:4000]):
            try:
                doc = feedmod.parse(resp.content)
            except feedmod.FeedError as e:
                doc = None
                self.note(f"{head}: looks like a feed but is not readable ({e})")
            if doc is not None and doc.entries:
                # tender-like entries (deadline / procurement words) make it a list; a news feed is only a fallback
                sample = doc.entries[:10]
                tenderish = sum(1 for e in sample if _DEADLINE_RE.search(_fold(feedmod.entry_xml(e, 30000))))
                tier = "A" if tenderish * 2 >= len(sample) or node.via == "start" else "B"
                self._record(node, final, status=status, verdict=f"list (feed, tier {tier})",
                             feedEntries=len(doc.entries))
                self.note(f"{head}: the address is an {doc.format.upper()} feed ({len(doc.entries)} entries, "
                          f"{tenderish} of {len(sample)} tender-like; links {sorted(doc.links)})")
                return self._add_candidate(node, final, "", tier, f"{doc.format.upper()} feed of {len(doc.entries)} "
                                           f"entries", strength=0.85 if tier == "A" else 0.2,
                                           feed={"url": final, "doc": doc})
            self._record(node, final, status=status, verdict="dead end (feed without entries)")
            return None
        # the address may itself be a JSON API (e.g. a public publications webservice)
        if "json" in ctype:
            try:
                data = resp.json()
            except ValueError:
                data = None
            rows = _first_list_of_dicts(data) if data is not None else None
            if rows is not None and (node.via == "start" or _score_json(data) > 0):
                self._record(node, final, status=status, verdict="list (JSON)", jsonRows=len(rows))
                self.note(f"{head}: the address is a JSON API ({len(rows)} rows)")
                return self._add_candidate(node, final, "", "A", f"JSON array of {len(rows)} rows", api=(final, data),
                                           strength=1.0)
            self._record(node, final, status=status, verdict="dead end (JSON without tenders)")
            self.note(f"{head}: JSON without a tender list; dead end")
            return None
        if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
            self._record(node, final, status=status, verdict=f"dead end ({ctype[:40]})")
            self.note(f"{head}: not a page ({ctype[:40]}); dead end")
            return None
        html = resp.text or ""
        self._captcha = self._captcha or runner.has_captcha(html)
        # a JSON API named in the page source (JavaScript apps)
        cands = [c for c in _api_candidates(html, final) if _key(c) not in self._probed]
        for i, cand in enumerate(cands):
            if not self._left("probes"):
                break
            self.used["probes"] += 1
            self._probed.add(_key(cand))
            data = await _try_json(ctx, cand)
            if data is not None and _score_json(data) > 0:
                self._record(node, final, status=status, verdict="list (API in page source)", api=cand)
                self.note(f"{head}: found JSON API in the page source: {cand}")
                return self._add_candidate(node, final, html, "A", "tender-like JSON from an API in the page source",
                                           api=(cand, data), strength=0.9, untried=cands[i + 1:])
        block = _evidence_of_block(html, status)
        # official feeds the page offers: a fallback route when the HTML list cannot be read (any host: a ministry's
        # open-data page may host the portal's feed)
        for fu, why in feed_links(html, final):
            if self._push(Node(fu, node.depth + 1, "feed", FEED_PRIO - 0.03 * node.depth, why, node, why,
                               referer=final)):
                self.note(f"{head}: offers a feed {fu} ({why}); queued")
        links = self._links([(a.text() or a.attributes.get("title") or a.attributes.get("aria-label") or "",
                              a.attributes.get("href")) for a in HTMLParser(html).css("a[href]")], final)
        rendered = None
        facts: list[str] = []
        shell = _js_shell(html)
        sig = _app_sig(html)
        if not shell and sig in self._spa_sigs:
            # another route of a JavaScript app already rendered: the raw page is only its frame (menus, no
            # content); the model is not asked about it - it is rendered, or closed when the browser budget is gone
            shell = True
            facts.append("a route of the JavaScript app seen before")
        soft = [u for u in soft_redirects(html, final, shell) if self.same_site(u)]
        if soft and (shell or not links):
            # a meta refresh, a frameset or a script redirect on an empty page: follow it before opening a browser
            pushed = [u for u in soft if self._push(Node(u, node.depth + 1, "redirect", 0.9 - 0.03 * node.depth,
                                                         "[redirect]", node, "soft redirect", referer=final))]
            if pushed:
                self._record(node, final, status=status, verdict="soft redirect", pushed=len(pushed))
                self.note(f"{head}: the page forwards to {pushed[:3]}; following")
                return None
        elif soft:  # frames inside a normal page may hold the list: offer them like links
            links += [{"text": "[frame]", "url": u, "kw": max(keyword_score("", u), 0.3)} for u in soft
                      if all(_key(u) != _key(l["url"]) for l in links)]
        if shell:
            if block:
                self._block(node, final, block)
                self._record(node, final, status=status, verdict="blocked (JS shell)")
                self.note(f"{head}: {block}; branch closed")
                return None
            if not self._left("renders"):
                self._record(node, final, status=status, verdict="dead end (JS shell, no render budget)")
                self.note(f"{head}: renders with JavaScript but the browser budget is used up; backtracking")
                return None
            rendered = await self._render(final, head)
            if rendered.get("api"):
                self._record(node, final, status=status, verdict="list (API via browser)", api=rendered["api"][0])
                return self._add_candidate(node, rendered.get("url") or final, rendered["html"], "A",
                                           "tender-like JSON captured in a browser" + self._pressed(rendered),
                                           api=rendered["api"], strength=0.9, links=rendered["links"][:40],
                                           untried=rendered.get("untried", []),
                                           apiRequest=rendered.get("apiRequest"))
            if not rendered.get("html") or _js_shell(rendered["html"]):
                block = _evidence_of_block(rendered.get("html") or "", status)
                if block:
                    self._block(node, final, block)
                verdict = "blocked" if block else "dead end (empty even in a browser)"
                self._record(node, final, status=status, verdict=verdict)
                self.note(f"{head}: {block or 'no content even after loading it in a browser'}; backtracking")
                return None
            html = rendered["html"]
            links = self._links([(l.get("text") or "", l.get("url")) for l in rendered["links"]], final)
            block = _evidence_of_block(html, status)
            facts.append(f"rendered ({rendered.get('captured', 0)} tender-like JSON not replayable)"
                         + self._pressed(rendered))
            if sig and not rendered.get("submitted"):
                self._spa_sigs.add(sig)
        ev = row_evidence(html)
        if ev["hrefs"]:
            row_keys = {_key(urljoin(final, h)) for h in ev["hrefs"]}
            links = [l for l in links if _key(l["url"]) not in row_keys]
        emb = embedded_json(html)
        forms = [f for f in search_forms(html, final) if self.same_site(f["url"])]
        for f in forms:   # a POST form sent as GET is less certain to work: it waits for the model's pick
            f["kw"] = round(max(keyword_score(f["text"], f["url"]), 0.45) - (0.15 if f.get("post") else 0), 3)
        facts.append(f"rows {ev['rows']} ({ev['sel']}, deadline words {'yes' if ev['deadline'] else 'no'})"
                     if ev["rows"] else "no tender-like rows")
        if emb:
            facts.append(f"embedded tender JSON ({emb[1]} rows)")
        if block:
            facts.append("login form")
        # the cheap model proposes: page type + its top 3 links
        offered = self._offer(links + forms)
        ptype, picks, reason = await self._classify(final, html, offered)
        pick_txt = ", ".join(f"#{i} {c:.2f} '{offered[i]['text'][:40]}'" for i, c in picks)
        self.note(f"{head}: {'; '.join(facts)}; model: {ptype or '-'} ({reason[:100]})"
                  + (f"; picks {pick_txt}" if pick_txt else ""))
        # static evidence decides
        listish = ptype in ("list", "search")
        excerpt = ptype == "excerpt"   # 'latest tenders' on a home page: a fallback; its 'all tenders' link is followed
        extra = {"links": rendered["links"][:40]} if rendered else {}
        cand = None
        # A = the model's list confirmed by rows/JSON (or, without a model, strong rows alone); B = rows the model
        # calls an excerpt, or rows with deadline words it does not call a list; dated rows alone may be news
        tier = None
        if emb or ev["weak"]:
            if (listish and not excerpt) or (ptype is None and (emb or ev["strong"])):
                tier = "A"
            elif excerpt or emb or (ev["strong"] and ev["deadline"]):
                tier = "B"
        here = (rendered or {}).get("url") or final    # where the app showed it (after a search press)
        if tier and emb:
            cand = self._add_candidate(node, here, html, tier, f"embedded tender-like JSON ({emb[1]} rows)",
                                       strength=0.8, staticRows=rendered is None, **extra)
        elif tier:
            cand = self._add_candidate(node, here, html, tier, f"{ev['rows']} rows with link and date ({ev['sel']})"
                                       + (", deadline words" if ev["deadline"] else "")
                                       + (", excerpt" if excerpt else "")
                                       + (self._pressed(rendered) if rendered else ""),
                                       strength=min(ev["rows"], 50) / 100 + (0.2 if ev["strong"] else 0),
                                       staticRows=rendered is None, **extra)
            if cand and tier == "B" and not self._item_tried:
                # an excerpt's tenders show where the portal keeps them: open one (a server redirect into another
                # application of the site points to the list there, see _new_context)
                items = [u for u in (urljoin(final, h) for h in ev["hrefs"][:20]) if u.startswith("http")
                         and self.same_site(u) and _key(u) not in self._visited]
                if items and self._push(Node(items[0], node.depth + 1, "item", ITEM_PRIO, "[one tender]", node,
                                             "where the excerpt's tenders live", referer=final)):
                    self._item_tried = True
                    self.note(f"queued one tender of the excerpt ({items[0]}) to see where the tenders live")
        elif not ev["weak"] and (listish or (ptype == "empty" and node.prio >= 0.5)) and not block:
            # the model sees a list (or an empty page where a strong link pointed), the HTML shows no rows: they may
            # be built by JavaScript -> look in a browser
            if rendered is None and self._left("renders"):
                rendered = await self._render(final, head)
                links = links + self._links([(l.get("text") or "", l.get("url")) for l in rendered["links"]], final)
                if sig and not rendered.get("submitted") and \
                        len(page_text(rendered.get("html") or "", 20000)) > 3 * len(page_text(html, 20000)) + 300:
                    self._spa_sigs.add(sig)   # the content only exists after JavaScript ran: an app frame
                    self.note(f"{final}: its content is drawn by JavaScript; other pages with the same scripts are "
                              f"rendered, not shown to the model raw")
                if rendered.get("api"):
                    cand = self._add_candidate(node, rendered.get("url") or final, rendered["html"] or html, "A",
                                               "tender-like JSON captured in a browser" + self._pressed(rendered),
                                               api=rendered["api"],
                                               strength=0.9, links=rendered["links"][:40],
                                               untried=rendered.get("untried", []),
                                               apiRequest=rendered.get("apiRequest"))
                elif rendered.get("html"):
                    ev2 = row_evidence(rendered["html"])
                    if ev2["weak"]:
                        cand = self._add_candidate(node, rendered.get("url") or final, rendered["html"], "A",
                                                   f"{ev2['rows']} rows with link and date after rendering "
                                                   f"({ev2['sel']}){self._pressed(rendered)}",
                                                   strength=min(ev2["rows"], 50) / 100,
                                                   links=rendered["links"][:40])
            if cand is None and rendered and rendered.get("captured"):
                cand = self._add_candidate(node, rendered.get("url") or final, rendered.get("html") or html, "B",
                                           f"the app loads tender-like JSON ({rendered['captured']} responses, not "
                                           f"replayable without the browser)", strength=0.3,
                                           links=rendered["links"][:40])
            if cand is None and listish:
                cand = self._add_candidate(node, final, html, "C", f"model says '{ptype}', no rows seen",
                                           strength=0.0, **extra)
        if block and cand is None:
            self._block(node, final, block)
        verdict = (f"candidate {cand['tier']}" if cand else "blocked" if block else ptype or "page")
        # expand: the model's picks first, then links whose words point to tenders
        pushed = 0
        for i, conf in picks:
            l = offered[i]
            if self._push(Node(l["url"], node.depth + 1, "form" if l.get("form") else "model",
                               _prio(conf, l["kw"], node.depth + 1) - (0.05 if block else 0), l["text"], node,
                               f"model {conf:.2f}", referer=final)):
                pushed += 1
        # a confirmed list's own links are its pages, filters and tenders: the same list again, not an alternative
        for l in ([] if cand and cand["tier"] == "A" else links + forms):
            if self._push(Node(l["url"], node.depth + 1, "form" if l.get("form") else "link",
                               _prio(None, l["kw"], node.depth + 1) - (0.05 if block else 0), l["text"], node,
                               f"words {l['kw']}", referer=final)):
                pushed += 1
        if cand is None and not pushed:
            verdict = "blocked" if block else "dead end"
        self._record(node, final, status=status, verdict=verdict, pageType=ptype, rows=ev["rows"],
                     deadline=ev["deadline"], picks=[offered[i]["url"] for i, _ in picks], pushed=pushed,
                     reason=reason[:120])
        if cand is None:
            self.note(f"{head}: {verdict}; {pushed} links queued; best open {self._best_open():.2f}"
                      + (" (backtracking)" if not pushed else ""))
        return cand

    # ----------------------------------------------------------------- pieces
    async def _fetch(self, url: str, max_bytes: int = MAX_PAGE_BYTES, any_type: bool = False,
                     referer: str | None = None):
        """GET that looks at the answer before reading it: a file (installer, archive, document) is refused unread.
        Returns (response with its body, None) or (response without body, why it was refused)."""
        resp = await self.ctx.client.stream("GET", url, headers={"Referer": referer} if referer else None)
        body = bytearray()
        try:
            ctype = resp.headers.get("content-type", "").lower()
            clen = resp.headers.get("content-length", "")
            if not any_type and ctype and not any(t in ctype for t in ("html", "xml", "text", "json")):
                return resp, f"not a page ({ctype[:40]})"
            if "json" in ctype:
                max_bytes = max(max_bytes, 30_000_000)   # a JSON API may legitimately be big
            if "xml" in ctype:
                max_bytes = max(max_bytes, MAX_FEED_BYTES)   # an ATOM/RSS feed file may be big too
            if clen.isdigit() and int(clen) > max_bytes:
                return resp, f"{int(clen)} bytes, a file"
            async for chunk in resp.aiter_bytes():
                if not body and not any_type and b"\x00" in chunk[:2048]:
                    return resp, "binary content, a file"
                body += chunk
                if len(body) > max_bytes:
                    return resp, f"more than {max_bytes} bytes, a file"
        finally:
            await resp.aclose()
        headers = [(k, v) for k, v in resp.headers.multi_items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(resp.status_code, headers=headers, content=bytes(body), request=resp.request), None

    def _links(self, raw: list[tuple[str, str | None]], base: str) -> list[dict]:
        out, seen = [], set()
        for text, href in raw:
            href = (href or "").strip()
            if not href or href.lower().startswith(("#", "mailto:", "javascript:", "tel:", "data:")):
                continue
            u = urljoin(base, href).split("#")[0]
            if not u.startswith(("http://", "https://")) or not self.same_site(u):
                continue
            k = _key(u)
            if k in seen or k == _key(base):
                continue
            seen.add(k)
            t = (clean_text(text) or "")[:80]
            out.append({"text": t, "url": u, "kw": keyword_score(t, u)})
            if len(out) >= 400:
                break
        return out

    def _offer(self, links: list[dict]) -> list[dict]:
        """At most 80 unvisited links for the model, words-that-point-to-tenders first."""
        fresh = [l for l in links if _key(l["url"]) not in self._visited]
        ranked = [l for l in fresh if l["kw"] > 0] + [l for l in fresh if l["kw"] <= 0]
        return [{**l, "i": i} for i, l in enumerate(ranked[:80])]

    async def _classify(self, url: str, html: str, offered: list[dict]) -> tuple[str | None, list, str]:
        left = self._model_calls_left()
        if not self._left("classify") or (left is not None and left <= MODEL_RESERVE):
            return None, [], "model budget kept for synthesis; static signals only"
        self.used["classify"] += 1
        try:
            v = await self.ctx.llm.json_call(
                "classify", CLASSIFY_SYSTEM,
                json.dumps({"url": url, "page": reduce_html(html, 12000),
                            "links": [{"i": l["i"], "text": l["text"], "url": l["url"]} for l in offered]},
                           ensure_ascii=False),
                portal_id=self.ctx.portal_id, run_id=self.ctx.run_id)
        except (LLMUnavailable, ModelBudgetExhausted, httpx.HTTPError, ValueError) as e:
            # no model answer (network error, bad JSON, budget): the walk goes on with static signals
            return None, [], f"model unavailable ({type(e).__name__}: {str(e)[:60]}); static signals only"
        if not isinstance(v, dict):
            return None, [], "model answer was not an object"
        return parse_verdict(v, len(offered))

    async def _render(self, url: str, head: str, press: bool = False) -> dict:
        """Open the page in a browser: tender-like JSON the app loads (replayed without the browser), else the
        rendered DOM and its links. press: press the search control even when the app loaded data by itself."""
        from . import browser
        ctx = self.ctx
        self.used["renders"] += 1
        wait = max(30.0, min(RENDER_WAIT_S, WALK_SECONDS - self._elapsed()))
        try:  # a page that never settles must not hold the walk: hard stop shortly after the render's own cap
            # a search page shows its results only after its search control is pressed: the browser presses it
            # (default criteria) when the app has loaded no tender-like data by itself
            cap = await asyncio.wait_for(browser.capture(ctx, url, max_wait_s=wait,
                                                         submit_search="always" if press else True), wait + 40)
        except asyncio.TimeoutError:
            cap = {"html": "", "json": [], "links": [], "requests": None, "seconds": wait + 40,
                   "error": "the browser did not finish in time"}
        moved = cap.get("final_url") if cap.get("final_url") and _key(cap["final_url"]) != _key(url) \
            and self.same_site(cap["final_url"]) else None
        self.note(f"browser on {url}: {cap.get('requests')} requests in {cap.get('seconds')}s, {len(cap['json'])} JSON "
                  f"responses" + (f"; pressed the search control '{cap['submitted']}' (default criteria)"
                                  if cap.get("submitted") else "")
                  + (f"; the app moved to {moved}" if moved else "")
                  + (f", error {cap['error'][:100]}" if cap.get("error") else ""))
        calls = cap.get("data_calls") or [{"url": u, "method": "GET"} for u in cap.get("data_requests", [])]
        for dc in calls[:6]:
            self.note(f"  data request: {dc.get('method', 'GET')} {dc['url'][:200]}")
        metas = cap.get("json_meta") or []
        for n, (ju, jd) in enumerate(cap["json"][:8]):
            how = (metas[n] or {}).get("method", "GET") if n < len(metas) else "GET"
            self.note(f"  JSON {how} {ju[:200]} (score {_score_json(jd)})")
        # tender-like responses the app received, best first, a POST search and a GET lookup kept apart; each is
        # replayed the way the app asked (method + JSON body), without the browser
        ranked = captured_endpoints(cap)
        out = {"api": None, "html": cap.get("html") or "", "links": cap.get("links") or [], "captured": len(ranked),
               "url": moved, "submitted": cap.get("submitted")}
        if moved:
            self._visited.add(_key(moved))
        if not cap["json"]:  # the app asked for data but the answer did not arrive in time: try it directly
            for dc in calls[:5]:
                k = f"{dc.get('method', 'GET')} {_key(dc['url'])} {dc.get('body') or ''}"
                if not self._left("probes") or k in self._probed:
                    continue
                self.used["probes"] += 1
                self._probed.add(k)
                data, req = await replay_request(ctx, dc["url"], dc)
                if data is not None and _score_json(data) > 0:
                    self.note(f"found JSON API from the app's data requests: {dc.get('method', 'GET')} {dc['url']}")
                    out.update(api=(dc["url"], data), apiRequest=req)
                    return out
        for i, ep in enumerate(ranked):
            replay, req = await replay_request(ctx, ep["url"], ep["meta"])
            if replay is not None and _score_json(replay) > 0:
                self.note(f"found JSON API via browser: {(req or {}).get('method', 'GET')} {ep['url']}")
                out.update(api=(ep["url"], replay), apiRequest=req,
                           untried=[_untried_entry(e) for e in ranked[i + 1:]])
                return out
            self.note(f"JSON API {ep['url'][:160]} does not work without the browser session")
        # The app reads its data from an open-data platform (its metadata or record requests were seen, but no
        # replayable list of rows): ask the platform's standard records endpoint for the same dataset.
        best = None
        seen_urls = [ju for ju, _ in cap["json"]] + [dc["url"] for dc in calls]
        for u in open_data_records_candidates(seen_urls):
            if not self._left("probes") or u in self._probed:
                continue
            self.used["probes"] += 1
            self._probed.add(u)
            data = await _try_json(ctx, u)
            sc = _score_json(data) if data is not None else 0
            self.note(f"  open-data records {u[:160]} (score {sc})")
            if sc > 0 and (best is None or sc > best[0]):
                best = (sc, u, data)
        if best:
            self.note(f"found JSON API on the portal's open-data platform: GET {best[1]}")
            out.update(api=(best[1], best[2]), apiRequest=None)
        if not best and ranked and not cap.get("submitted") and not press and self._left("renders"):
            # the app loads tender-like data the walk cannot ask for by itself (e.g. behind a session token): the
            # list's own page is then its search result view - press the search control although data arrived
            self.note(f"{url}: the app's tender data cannot be replayed; opening it again to press its search control")
            again = await self._render(url, head, press=True)
            if again.get("api") or again.get("submitted"):
                return again
        return out

    async def _sitemap(self) -> None:
        """robots.txt 'Sitemap:' entries (else /sitemap.xml): same-site URLs whose words point to tenders."""
        self._sitemap_done = True
        ctx = self.ctx
        p = urlsplit(self.start)
        root = f"{p.scheme}://{p.netloc}"
        maps: list[str] = []
        if self._left("sitemap"):
            self.used["sitemap"] += 1
            try:
                r, refused = await self._fetch(root + "/robots.txt", 1_000_000)
                if not refused and r.status_code < 400 and "html" not in r.headers.get("content-type", ""):
                    maps = [m.strip() for m in re.findall(r"(?im)^\s*sitemap\s*:\s*(\S+)", r.text or "")]
            except BudgetExceeded:
                raise
            except Exception:
                pass
        maps = [urljoin(root + "/", m) for m in maps if self.same_site(urljoin(root + "/", m))] or [root + "/sitemap.xml"]
        maps.sort(key=lambda m: -keyword_score("", m))   # a section's own sitemap first (e.g. /notices/sitemap.xml)
        locs: list[str] = []
        # the folder that holds a section sitemap is usually that section's page
        for m in maps:
            folder = urljoin(m, ".")
            if urlsplit(folder).path not in ("", "/") and keyword_score("", folder) > 0:
                locs.append(folder)
        queue = maps[:2]
        while queue and self._left("sitemap"):
            sm = queue.pop(0)
            self.used["sitemap"] += 1
            try:
                r, refused = await self._fetch(sm, 20_000_000, any_type=True)   # .xml.gz is binary on purpose
            except BudgetExceeded:
                raise
            except Exception:
                continue
            if refused or r.status_code >= 400:
                continue
            body = r.content or b""
            if body[:2] == b"\x1f\x8b":
                try:
                    body = gzip.decompress(body)[:20_000_000]
                except OSError:
                    continue
            text = body.decode("utf-8", errors="replace")
            found = [htmllib.unescape(u.strip()) for u in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text, re.I)]
            if re.search(r"<sitemapindex", text, re.I):   # an index: open the most promising sub-sitemap
                subs = sorted((u for u in found if self.same_site(u)), key=lambda u: -keyword_score("", u))
                queue = subs[:1] + queue
                continue
            locs += found
        def segs(u: str) -> int:
            return len([s for s in urlsplit(u).path.split("/") if s])
        # strongest words first; among equals the shorter address (a section page, not one notice)
        scored = sorted({u for u in locs if self.same_site(u)}, key=lambda u: (-keyword_score("", u), segs(u), len(u)))
        pushed = []
        for u in scored[:40]:
            kw = keyword_score("", u)
            if kw <= 0 or len(pushed) >= 8:
                break
            prio = round(0.5 * kw + 0.05 - 0.02 * max(0, segs(u) - 1), 3)
            if self._push(Node(u, 1, "sitemap", prio, "", None, f"sitemap words {kw}")):
                pushed.append(u)
        self.trace.append({"n": self.used["pages"], "url": root, "via": "sitemap", "sitemaps": maps[:3],
                           "locs": len(locs), "pushed": pushed})
        self.note(f"sitemap: {len(locs)} addresses in {maps[:2]}; queued {len(pushed)}: {pushed[:4]}")


CLASSIFY_SYSTEM = (
    "You look at one page of a public-procurement website. The goal is the page that LISTS all open tenders / "
    "contract notices of this portal (or its search page that shows them). Answer JSON: "
    "{\"pageType\": \"list\"|\"excerpt\"|\"search\"|\"tender\"|\"landing\"|\"login\"|\"blocked\"|\"empty\"|\"other\", "
    "\"links\": [{\"i\": link index, \"confidence\": 0..1, \"why\": short}], \"reason\": short}. "
    "list = this page lists the portal's tenders/notices (rows with titles, dates, buyers; paginated is fine); "
    "excerpt = only a few latest or featured tenders, with a link to the full list; search = a tender search form "
    "whose results are not shown yet; tender = one tender; landing = home page or menu; login = sign-in page (it may "
    "still link to a public area); blocked = captcha or access denied. links: up to 3, best first, the links most "
    "likely to lead to the PUBLIC list of ALL open tenders (for example 'Alle Bekanntmachungen', 'Aktuelle "
    "Ausschreibungen', 'Find tenders', 'Consultations en cours', 'Aankondigingen', 'Licitaciones', a '[search form]' "
    "that shows all results). On a list page give no links (pagination is not needed). "
    "Never pick sign-in, registration, help, news, contact or legal links. Empty list if no link fits.")


def parse_verdict(v: dict, n_links: int) -> tuple[str | None, list[tuple[int, float]], str]:
    """(pageType, [(link index, confidence)] best first, reason). Also accepts the older {'nextLink': i} answer."""
    ptype = str(v.get("pageType") or "").strip().lower() or None
    picks: list[tuple[int, float]] = []
    raw = v.get("links") if isinstance(v.get("links"), list) else []
    for p in raw:
        i, conf = (p.get("i", p.get("index")), p.get("confidence", 0.5)) if isinstance(p, dict) else (p, 0.6)
        try:
            i, conf = int(i), float(conf)
        except (TypeError, ValueError):
            continue
        conf = conf / 100 if conf > 1 else conf
        if 0 <= i < n_links and i not in [j for j, _ in picks]:
            picks.append((i, max(0.0, min(1.0, conf))))
    nl = v.get("nextLink")
    if isinstance(nl, int) and not isinstance(nl, bool) and 0 <= nl < n_links and nl not in [j for j, _ in picks]:
        picks.insert(0, (nl, 0.8))
    return ptype, picks[:3], str(v.get("reason") or "")


async def locate_list(ctx, start_url: str) -> dict:
    """The best candidate: {"url", "html", "api": (url, json)|None, "captcha", "tier", "evidence", "path",
    "candidates", "walk", ...}. Raises PortalBlocked (only a block with evidence and nothing public found) or
    AttachFailed."""
    walk = Walk(ctx, start_url)
    found = await walk.next()
    if found is None:
        err = walk.failure()
        walk.note(f"no list found; used {walk._usage()}: {err}")
        raise err
    return found


async def next_candidate(ctx, found: dict) -> dict | None:
    """The next-ranked candidate of the walk that produced `found` (resumes the search within its budget)."""
    walk = found.get("walk")
    if walk is None:
        return None
    return await walk.next()
