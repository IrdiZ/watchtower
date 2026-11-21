"""Same tender, several sources: tenders served by more than one portal (or twice by one portal under two
procedure ids), with the evidence and a confidence for each match, and every field on which the sources disagree
(deadline, title, buyer, status, documents), each source's value next to its link.

Read-only over the store, computed on demand and cached by the API (/_ui/crossref, dashboard only: the contract
endpoints are unchanged). Precision over recall: a pair is linked only on evidence a reviewer can check at a glance.

Evidence, strongest first:
  link       one record links to another portal's tender (its id inside a URL on that portal's host)
  reference  the same buyer reference (normalised), confirmed by a similar buyer or title
  title      near-identical title, similar buyer, and the same deadline to the minute (or, without deadlines,
             the identical title and publication dates within 3 days)
Guards: different numbers on both titles ("Los 1" / "Los 2") never match without a link; a reference shared by
many records is not identifying; different deadlines without a link need the same reference, buyer and title
(flagged as a possible republication, confidence 0.6). Pairs are only compared inside blocks (same reference,
same linked id, same deadline minute, same exact title), so 40k tenders stay well under a second of matching.
"""
from __future__ import annotations

import html
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import unquote

from . import timeutil

SERVED_STATES = ("serving", "healing")
MAX_REF_RECORDS = 5      # a reference shared by more records than this is not identifying
MAX_GROUP = 6            # never grow a group past this (a chain this long is a blocking accident, not one tender)
URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}]+", re.I)

_STOP = set("""
a an and the of for to in on at by with from or as is
de la le les des du d l et en pour par sur au aux un une dans avec
der die das den dem des und fur von zu zur zum im in am an auf mit bei aus ein eine einer eines
van het een voor op te met bij uit naar
del el los las y para por con al un una en
w z i na do dla oraz o od po przez ze
""".split())
_BUYER_STOP = _STOP | set("""
gmbh mbh ag aor kg ev eg se sa sas sasu sarl spa srl sl slu bv nv ltd plc llc inc co sp zoo oo s
""".split())
LOT_WORDS = ("los", "lot", "lote", "lotto", "perceel", "czesc", "teil", "partie")


# ---------------------------------------------------------------------------------------------- normalisation
_FOLD = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss", "æ": "ae",
                       "œ": "oe"})
_FOLD_RE = re.compile("[łŁøØđĐßæœ]")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")


@lru_cache(maxsize=200_000)
def _norm(s: str) -> str:
    if not s.isascii():
        if _FOLD_RE.search(s):
            s = s.translate(_FOLD)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(_NON_ALNUM.sub(" ", s.lower()).split())


def norm_text(value: Any) -> str:
    """Lowercase, strip accents, keep letters and digits only, collapse whitespace."""
    return _norm(html.unescape(str(value))) if value else ""  # some portals serve "&quot;" in titles


def tokens(value: Any, stop: set[str] = _STOP) -> frozenset[str]:
    """Content tokens; digits are always kept ("Los 1" and "Los 2" must stay different)."""
    return _tokens(norm_text(value), stop is _BUYER_STOP)


@lru_cache(maxsize=200_000)
def _tokens(normed: str, buyer: bool) -> frozenset[str]:
    stop = _BUYER_STOP if buyer else _STOP
    return frozenset(t for t in normed.split() if t.isdigit() or (len(t) > 1 and t not in stop))


def ref_key(value: Any) -> str | None:
    """Reference number without case, accents, spaces or punctuation. Needs 4+ characters and a digit."""
    k = norm_text(value).replace(" ", "")
    return k if len(k) >= 4 and any(c.isdigit() for c in k) else None


@lru_cache(maxsize=200_000)
def host_of(url: str | None) -> str:
    m = _HOST_RE.match(url or "")
    h = m.group(1).lower().rstrip(".") if m else ""
    return h[4:] if h.startswith("www.") else h


_HOST_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^/?#@]*@)?([^/?#:\[\]]+)")


@lru_cache(maxsize=200_000)
def url_tokens(url: str) -> frozenset[str]:
    """Id-like parts of a URL after the host (path segments, query values, and their pieces split on . _ -):
    5+ characters with a digit."""
    rest = url.split("//", 1)[-1].lower()
    rest = unquote(rest[rest.find("/"):]) if "/" in rest else ""
    out = set()
    for piece in _URL_SPLIT.split(rest):
        if len(piece) >= 5 and _HAS_DIGIT.search(piece):
            out.add(piece)
            for sub in _PIECE_SPLIT.split(piece):
                if len(sub) >= 5 and _HAS_DIGIT.search(sub):
                    out.add(sub)
    return frozenset(out)


_URL_SPLIT = re.compile(r"[/?&=#;,+ ]+")
_PIECE_SPLIT = re.compile(r"[._~-]+")
_HAS_DIGIT = re.compile(r"\d")


def _sim(a: frozenset[str], b: frozenset[str], containment_min: int) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    jac = inter / len(a | b)
    small = min(len(a), len(b))
    if small >= containment_min:
        return max(jac, 0.9 * inter / small if inter == small else jac)
    return jac


def title_sim(a: "Rec", b: "Rec") -> float:
    if a.title_norm and a.title_norm == b.title_norm:
        return 1.0
    # plain Jaccard: containment would make "Lüftung ..." equal "Schlosserarbeiten (Unterkonstruktion Lüftung) ..."
    s = _sim(a.title_tokens, b.title_tokens, 10**9)
    short, long_ = sorted((a.title_norm, b.title_norm), key=len)
    if len(short) >= 30 and long_.startswith(short):  # one portal truncates the title
        s = max(s, 0.9)
    return s


class Weights(dict):
    """Token -> weight; `top` (the rarest token's weight) for tokens not seen when the weights were built."""
    top = 1.0


def buyer_weights(recs: Iterable["Rec"]) -> Weights:
    """Inverse document frequency of buyer-name tokens: "Stichting", "Stadt", "Onderwijs" weigh little,
    "Veluwe-Vallei" or "Ohrdruf" a lot, so two foundations that share only generic words do not look alike."""
    df: dict[str, int] = defaultdict(int)
    n = 0
    for r in recs:
        n += 1
        for t in r.buyer_tokens:
            df[t] += 1
    w = Weights((t, math.log((n + 1) / (c + 0.5))) for t, c in df.items())
    w.top = max(w.values(), default=1.0)
    return w


def buyer_sim(a: "Rec", b: "Rec", weights: dict[str, float] | None = None) -> float | None:
    """max(weighted Jaccard, weighted containment) of the buyer-name tokens; None if a buyer is missing."""
    A, B = a.buyer_tokens, b.buyer_tokens
    if not A or not B:
        return None
    if weights is None:
        w = len
    else:
        top = getattr(weights, "top", 1.0)
        w = lambda S: sum(weights.get(t, top) for t in S)  # noqa: E731
    inter = w(A & B)
    if not inter:
        return 0.0
    return max(inter / w(A | B), inter / min(w(A), w(B)))


def refs_differ(a: "Rec", b: "Rec") -> bool:
    """Both carry a reference and neither contains the other."""
    return bool(a.ref_key and b.ref_key and a.ref_key not in b.ref_key and b.ref_key not in a.ref_key)


def numbers_conflict(a: "Rec", b: "Rec") -> bool:
    """Both titles carry numbers and each has one the other lacks (lot 1 vs lot 2, 2026 vs 2027)."""
    na = {t for t in a.title_tokens if t.isdigit()}
    nb = {t for t in b.title_tokens if t.isdigit()}
    return bool(na and nb and not (na <= nb or nb <= na))


# ---------------------------------------------------------------------------------------------- time display
def _parse_deadline(raw: Any) -> tuple[datetime | None, bool]:
    """(aware datetime in the offset the source gave, date_only)."""
    if not raw or not isinstance(raw, str):
        return None, False
    return _parse_deadline_str(raw)


@lru_cache(maxsize=100_000)
def _parse_deadline_str(raw: str) -> tuple[datetime | None, bool]:
    try:
        dt = timeutil.parse(raw)
    except ValueError:
        return None, False
    return dt, len(raw.strip()) == 10


def _offset_label(dt: datetime) -> str:
    off = dt.utcoffset() or timedelta(0)
    if off == timedelta(hours=1):
        return "CET"
    if off == timedelta(0):
        return "UTC"
    sign = "+" if off >= timedelta(0) else "-"
    minutes = abs(int(off.total_seconds() // 60))
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def fmt_deadline(raw: Any) -> str | None:
    """'24 Nov 2026 12:00 CET', in the offset the source gave (a date-only deadline stays a date)."""
    dt, date_only = _parse_deadline(raw)
    if dt is None:
        return str(raw) if raw else None
    if date_only:
        return f"{dt.day} {dt.strftime('%b %Y')}"
    return f"{dt.day} {dt.strftime('%b %Y %H:%M')} {_offset_label(dt)}"


def _same_deadline(a: "Rec", b: "Rec") -> bool | None:
    """True/False when both have a deadline; date-only deadlines compare on the date."""
    if a.dl is None or b.dl is None:
        return None
    if a.dl_date_only or b.dl_date_only:
        return a.dl.date() == b.dl.date()
    return a.dl_minute == b.dl_minute


# ---------------------------------------------------------------------------------------------- records
@dataclass
class Rec:
    idx: int
    source: str
    portal: str
    host: str
    tid: str
    url: str | None
    title: str | None
    buyer: str | None
    deadline: str | None
    publication: str | None
    status: str
    ref: str | None
    docs: int
    reads_docs: bool
    links: list[str]
    title_norm: str = ""
    title_tokens: frozenset = frozenset()
    buyer_tokens: frozenset = frozenset()
    ref_key: str | None = None
    dl: datetime | None = None
    dl_date_only: bool = False
    dl_minute: str | None = None
    pub_date: Any = None

    def __post_init__(self):
        self.title_norm = norm_text(self.title)
        self.title_tokens = tokens(self.title)
        self.buyer_tokens = tokens(self.buyer, _BUYER_STOP)
        self.ref_key = ref_key(self.ref)
        self.dl, self.dl_date_only = _parse_deadline(self.deadline)
        if self.dl is not None and not self.dl_date_only:
            self.dl_minute = self.dl.astimezone(timezone.utc).isoformat()[:16]
        if self.publication:
            try:
                self.pub_date = datetime.fromisoformat(str(self.publication)[:10]).date()
            except ValueError:
                self.pub_date = None

    def id_tokens(self) -> set[str]:
        """The tender's own id and the id-like parts of its URL (what another portal's link would contain)."""
        return {str(self.tid).lower()} | (url_tokens(self.url) if self.url else set())

    def ref_out(self) -> dict:
        return {"portal": self.portal, "host": self.host, "tenderId": self.tid, "url": self.url}


def _links(t: dict) -> list[str]:
    out: list[str] = []
    for key in ("_detailUrl",):
        if isinstance(t.get(key), str):
            out.append(t[key])
    for key in ("_docUrls", "_docBlockedUrls", "_docEntries"):
        v = t.get(key)
        if isinstance(v, dict):
            out.extend(k for k in v if isinstance(k, str))
        elif isinstance(v, list):
            out.extend(u for u in v if isinstance(u, str))
    for key in ("description", "title"):
        v = t.get(key)
        if isinstance(v, str) and "http" in v:
            out.extend(URL_RE.findall(v[:20000]))
    for d in t.get("documents") or []:
        if isinstance(d, dict) and isinstance(d.get("url"), str):
            out.append(d["url"])
    return list(dict.fromkeys(u.rstrip(".,;") for u in out if u.lower().startswith(("http://", "https://"))))


def _status(t: dict, now: datetime) -> str:
    """As store.derive_status, from the already parsed deadline."""
    if t.get("status") in ("withdrawn", "closed"):
        return t["status"]
    dl, _ = _parse_deadline(t.get("submissionDeadline"))
    return "closed" if dl is not None and dl <= now else "open"


# ---------------------------------------------------------------------------------------------- aliases
def find_aliases(portals: list[dict], ids: dict[str, set[str]], hosts: dict[str, str]) -> dict[str, dict]:
    """The same portal attached twice (e.g. a website URL and its JSON API): same tender host and at least half of
    the smaller entry's tender ids in common. Maps the later entry to the earlier one (which is kept)."""
    alias: dict[str, dict] = {}
    order = [p["id"] for p in portals]
    for i, a in enumerate(order):
        if a in alias:
            continue
        for b in order[i + 1:]:
            if b in alias or not ids.get(a) or not ids.get(b) or hosts.get(a) != hosts.get(b):
                continue
            shared = len(ids[a] & ids[b])
            if shared >= 0.5 * min(len(ids[a]), len(ids[b])):
                alias[b] = {"portal": b, "sameAs": a, "sharedIds": shared, "tenders": len(ids[b])}
    return alias


def _portal_host(tenders: Iterable[dict], fallback: str) -> str:
    counts: dict[str, int] = defaultdict(int)
    for t in tenders:
        h = host_of(t.get("url"))
        if h:
            counts[h] += 1
    return max(counts, key=counts.get) if counts else host_of(fallback)


# ---------------------------------------------------------------------------------------------- matching
class _UF:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.conf = [1.0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, conf: float) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.size[ra] + self.size[rb] > MAX_GROUP:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.conf[ra] = min(self.conf[ra], self.conf[rb], conf)
        return True


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


@dataclass
class Ctx:
    ref_df: dict[str, int]
    buyer_w: dict[str, float] | None = None
    title_counts: dict[tuple, int] | None = None  # (source, title, deadline minute) -> records

    def title_count(self, r: "Rec") -> int:
        """Records of r's portal with r's title and deadline (several: lots of one project, not one tender)."""
        return 1 if self.title_counts is None else self.title_counts.get((r.source, r.title_norm, r.dl_minute), 1)


def title_counts(recs: Iterable["Rec"]) -> dict[tuple, int]:
    """(source, title, deadline minute) and (source, title) -> number of records."""
    out: dict[tuple, int] = defaultdict(int)
    for r in recs:
        out[(r.source, r.title_norm, r.dl_minute)] += 1
        out[(r.source, r.title_norm)] += 1
    return out


def assess(a: Rec, b: Rec, link: str | None, ctx: Ctx) -> tuple[float, list[dict]] | None:
    """Confidence and evidence that a and b are the same tender, or None."""
    if a.source == b.source and a.tid == b.tid:
        return None
    ref_df = ctx.ref_df
    ts = title_sim(a, b)
    bs = buyer_sim(a, b, ctx.buyer_w)
    same_dl = _same_deadline(a, b)
    nums = numbers_conflict(a, b)
    ref_eq = bool(a.ref_key and a.ref_key == b.ref_key and ref_df.get(a.ref_key, 0) <= MAX_REF_RECORDS)
    ev: list[dict] = []
    if link:
        ev.append({"type": "link", "text": f"{a.host} record links to this {b.host} tender: {link}"})
    if ref_eq:
        ev.append({"type": "reference", "text": f"same reference {a.ref!s}" +
                   (f" / {b.ref!s}" if norm_text(a.ref) != norm_text(b.ref) else "")})
    if ts >= 0.6:
        ev.append({"type": "title", "text": "identical title" if ts == 1.0 else f"titles {_pct(ts)} similar"})
    if bs is not None and bs >= 0.5:
        ev.append({"type": "buyer", "text": "same buyer" if bs == 1.0 and norm_text(a.buyer) == norm_text(b.buyer)
                   else f"buyer names {_pct(bs)} similar"})
    if same_dl:
        ev.append({"type": "deadline", "text": f"same deadline {fmt_deadline(a.deadline)}"})

    if link:
        return (0.98 if ref_eq or ts >= 0.85 else 0.95), ev
    if nums:
        return None
    one_portal = a.source == b.source
    if one_portal and refs_differ(a, b):  # one portal, two references: separate lots or procedures
        return None
    if ref_eq:
        by_buyer = bs is not None and bs >= 0.5
        by_title = ts >= 0.6
        if not (by_buyer or by_title):
            return None
        pair_only = ref_df.get(a.ref_key, 0) == 2
        if one_portal:
            # inside one portal a shared reference often means lots of one project (Kaufbeuren 2020_3: four
            # lots, four deadlines): only an exact pair with the same buyer, title and deadline is a duplicate
            if pair_only and by_buyer and ts >= 0.9 and same_dl:
                return 0.9, ev
            return None
        if same_dl is False:
            if ts >= 0.9 and by_buyer and pair_only:
                ev.append({"type": "caution", "text": "deadlines differ: a correction, a republication or a lot"})
                return 0.6, ev
            return None
        return (0.95 if by_buyer and by_title else 0.85), ev
    if bs is None or bs < 0.8:
        return None
    if one_portal:
        # exactly these two records carry this title and deadline in the portal (three or more: lots)
        if same_dl and a.title_norm == b.title_norm and not (a.ref_key or b.ref_key) \
                and ctx.title_count(a) == 2:
            ev.append({"type": "caution", "text": "no reference to confirm: could be two lots of one project"})
            return 0.7, ev
        return None
    if ctx.title_count(a) > 1 or ctx.title_count(b) > 1:  # which of the lots would it be? do not guess
        return None
    if same_dl and ts >= 0.85:
        return (0.85 if a.title_norm == b.title_norm else 0.8), ev
    if same_dl is None and a.title_norm == b.title_norm and len(a.title_tokens) >= 4 \
            and a.pub_date and b.pub_date and abs((a.pub_date - b.pub_date).days) <= 3:
        ev.append({"type": "date", "text": f"published {a.pub_date.isoformat()} / {b.pub_date.isoformat()}"})
        return 0.7, ev
    if same_dl is False and a.title_norm == b.title_norm and len(a.title_tokens) >= 4 and ctx.title_counts \
            and ctx.title_counts.get((a.source, a.title_norm)) == 1 == ctx.title_counts.get((b.source, b.title_norm)) \
            and abs(a.dl - b.dl) <= timedelta(days=31):
        # the TenderNed / Mercell pattern without the link: one tender, one source not updated after a change
        ev.append({"type": "caution", "text": "same title and buyer, deadlines differ: one source may be out of date"})
        return 0.6, ev
    return None


def _candidate_pairs(recs: list[Rec]) -> tuple[dict[tuple[int, int], str | None], dict[str, int], dict]:
    """Blocked candidate pairs -> link URL (or None). Also the reference frequencies and blocking stats."""
    pairs: dict[tuple[int, int], str | None] = {}

    def add(i: int, j: int, link: str | None = None):
        if i == j:
            return
        key = (i, j) if i < j else (j, i)
        if link or key not in pairs:
            pairs[key] = link or pairs.get(key)

    # 1. links: an id-like URL part of a tender on its own host, unique on that host
    hosts = {r.host for r in recs}
    linked = [(r, u, h) for r in recs for u in r.links for h in (host_of(u),) if h and h != r.host and h in hosts]
    targets = {h for _, _, h in linked}
    owner: dict[tuple[str, str], set[int]] = defaultdict(set)
    for r in recs:
        if r.host in targets:
            for t in r.id_tokens():
                if len(t) >= 5:
                    owner[(r.host, t)].add(r.idx)
    n_links = 0
    for r, u, h in linked:
        for t in url_tokens(u):
            hit = owner.get((h, t))
            if hit and len(hit) == 1:
                j = next(iter(hit))
                if recs[j].source != r.source:
                    add(r.idx, j, u)
                    n_links += 1
    # 2. references
    by_ref: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        if r.ref_key:
            by_ref[r.ref_key].append(r.idx)
    ref_df = {k: len(v) for k, v in by_ref.items()}
    for idxs in by_ref.values():
        if 2 <= len(idxs) <= MAX_REF_RECORDS:
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    add(idxs[x], idxs[y])
    # 3. same deadline minute, sharing a rare title token (prefix filter for Jaccard >= 0.8)
    df: dict[str, int] = defaultdict(int)
    for r in recs:
        for t in r.title_tokens:
            df[t] += 1
    by_minute: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        if r.dl_minute and r.title_tokens and r.buyer_tokens:
            by_minute[r.dl_minute].append(r.idx)
    compared = 0
    for idxs in by_minute.values():
        if len(idxs) < 2:
            continue
        inv: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            toks = sorted(recs[i].title_tokens, key=lambda t: (df[t], t))
            p = len(toks) - math.ceil(0.8 * len(toks)) + 1
            for t in toks[:max(1, p)]:
                inv[t].append(i)
        for lst in inv.values():
            if len(lst) > 200:  # a token this common in one minute cannot be a prefix token of a real pair
                continue
            for x in range(len(lst)):
                for y in range(x + 1, len(lst)):
                    add(lst[x], lst[y])
                    compared += 1
    # 4. identical long titles (a deadline missing on one side, or two different deadlines)
    by_title: dict[str, list[int]] = defaultdict(list)
    for r in recs:
        if len(r.title_norm) >= 25 and r.buyer_tokens:
            by_title[r.title_norm].append(r.idx)
    for idxs in by_title.values():
        if 2 <= len(idxs) <= 20:
            for x in range(len(idxs)):
                for y in range(x + 1, len(idxs)):
                    add(idxs[x], idxs[y])
    return pairs, ref_df, {"linkHits": n_links, "candidatePairs": len(pairs), "titleBlockPairs": compared}


# ---------------------------------------------------------------------------------------------- conflicts
_FIELD_ORDER = {"deadline": 0, "status": 1, "buyer": 2, "documents": 3, "title": 4}


def _value(r: Rec, value: Any, display: Any = None) -> dict:
    return {**r.ref_out(), "value": value, "display": display if display is not None else value}


def conflicts_of(members: list[Rec], buyer_w: dict[str, float] | None = None) -> tuple[list[dict], list[dict]]:
    """Disagreements (two or more sources give different values) and gaps (some sources give no value)."""
    out: list[dict] = []
    gaps: list[dict] = []
    # deadline, to the minute, timezone-aware
    with_dl = [r for r in members if r.dl is not None]
    if len(with_dl) >= 2:
        base = with_dl[0]
        if any(_same_deadline(base, r) is False for r in with_dl[1:]):
            utc = [r.dl.astimezone(timezone.utc) for r in with_dl]
            spread = max(utc) - min(utc)
            minutes = round(spread.total_seconds() / 60)
            days = round(minutes / 1440, 1)
            note = (f"{days:g} day{'s' if days != 1 else ''} apart" if minutes >= 1440 else
                    f"{minutes // 60} h {minutes % 60:02d} min apart" if minutes >= 60 else f"{minutes} min apart")
            out.append({"field": "deadline", "note": note,
                        "values": [_value(r, r.deadline, fmt_deadline(r.deadline)) for r in with_dl]})
    # status
    statuses = {r.status for r in members}
    if len(statuses) > 1:
        out.append({"field": "status", "note": " vs ".join(sorted(statuses)),
                    "values": [_value(r, r.status) for r in members]})
    # buyer: names that do not even overlap
    with_buyer = [r for r in members if r.buyer_tokens]
    if len(with_buyer) >= 2:
        worst = min(buyer_sim(a, b, buyer_w) or 0.0 for i, a in enumerate(with_buyer) for b in with_buyer[i + 1:])
        if worst < 0.5:
            out.append({"field": "buyer", "note": f"names {_pct(worst)} similar",
                        "values": [_value(r, r.buyer) for r in with_buyer]})
    # documents, only between portals whose documents we read
    with_docs = [r for r in members if r.reads_docs]
    if len(with_docs) >= 2 and len({r.docs for r in with_docs}) > 1:
        out.append({"field": "documents", "note": "different number of documents",
                    "values": [_value(r, r.docs, f"{r.docs} file(s)") for r in with_docs]})
    # title
    with_title = [r for r in members if r.title_norm]
    if len({r.title_norm for r in with_title}) > 1:
        norms = sorted({r.title_norm for r in with_title}, key=len)
        truncated = all(norms[-1].startswith(n) for n in norms[:-1])
        worst = min(title_sim(a, b) for i, a in enumerate(with_title) for b in with_title[i + 1:])
        out.append({"field": "title", "note": "one title is cut short" if truncated else f"titles {_pct(worst)} similar",
                    "values": [_value(r, r.title) for r in with_title]})
    # gaps: a value some sources give and others do not
    for fname, has in (("deadline", lambda r: r.dl is not None), ("buyer", lambda r: bool(r.buyer_tokens)),
                       ("referenceNumber", lambda r: bool(r.ref))):
        given = [r for r in members if has(r)]
        missing = [r for r in members if not has(r)]
        if given and missing:
            val = {"deadline": lambda r: fmt_deadline(r.deadline), "buyer": lambda r: r.buyer,
                   "referenceNumber": lambda r: r.ref}[fname]
            gaps.append({"field": fname, "givenBy": [_value(r, val(r)) for r in given],
                         "missingAt": [r.ref_out() for r in missing]})
    return out, gaps


# ---------------------------------------------------------------------------------------------- entry points
def build_records(portals: list[dict], tenders: dict[str, list[dict]], now: datetime,
                  reads_docs: dict[str, bool] | None = None, alias: dict[str, dict] | None = None,
                  ) -> tuple[list[Rec], dict[str, dict], dict[str, str]]:
    """portals: [{id, url, ...}] in attach order; tenders: portal id -> tender dicts (store data, `_` keys kept).
    `alias` (already known self-duplicates) skips the detection; their tenders need not be passed."""
    hosts = {p["id"]: _portal_host(tenders.get(p["id"]) or [], p.get("url") or "") for p in portals}
    if alias is None:
        ids = {pid: {str(t.get("id")) for t in ts} for pid, ts in tenders.items()}
        alias = find_aliases(portals, ids, hosts)
    recs: list[Rec] = []
    for p in portals:
        pid = p["id"]
        if pid in alias:
            continue
        ts = tenders.get(pid) or []
        rd = (reads_docs or {}).get(pid)
        if rd is None:
            rd = any(t.get("documents") or t.get("_docUrls") for t in ts)
        for t in ts:
            if t.get("id") is None:
                continue
            recs.append(Rec(
                idx=len(recs), source=pid, portal=pid, host=host_of(t.get("url")) or hosts[pid], tid=str(t["id"]),
                url=t.get("url"), title=t.get("title"), buyer=t.get("buyerName"),
                deadline=t.get("submissionDeadline"), publication=t.get("publicationDate"),
                status=_status(t, now), ref=t.get("referenceNumber"), docs=len(t.get("documents") or []),
                reads_docs=bool(rd), links=_links(t)))
    return recs, alias, hosts


def compute(portals: list[dict], tenders: dict[str, list[dict]], now: datetime | None = None,
            reads_docs: dict[str, bool] | None = None, max_groups: int = 1000,
            alias: dict[str, dict] | None = None) -> dict:
    started = time.perf_counter()
    now = now or timeutil.now()
    recs, alias, hosts = build_records(portals, tenders, now, reads_docs, alias)
    pairs, ref_df, stats = _candidate_pairs(recs)
    ctx = Ctx(ref_df, buyer_weights(recs), title_counts(recs))

    edges = []
    for (i, j), link in pairs.items():
        a, b = recs[i], recs[j]
        if link and host_of(link) != b.host:  # keep the link's direction: a links to b
            a, b = b, a
        got = assess(a, b, link, ctx)
        if got:
            edges.append((got[0], a.idx, b.idx, got[1]))
    edges.sort(key=lambda e: (-e[0], e[1], e[2]))

    uf = _UF(len(recs))
    kept_edges: list[tuple] = []
    refused = 0
    for conf, i, j, ev in edges:
        if uf.union(i, j, conf):
            kept_edges.append((conf, i, j, ev))
        else:
            refused += 1

    members: dict[int, list[int]] = defaultdict(list)
    for conf, i, j, ev in kept_edges:
        root = uf.find(i)
        for x in (i, j):
            if x not in members[root]:
                members[root].append(x)
    edges_by_root: dict[int, list[tuple]] = defaultdict(list)
    for e in kept_edges:
        edges_by_root[uf.find(e[1])].append(e)

    labels = {p["id"]: hosts.get(p["id"]) or p["id"] for p in portals}
    groups, conflicts = [], []
    for root, idxs in members.items():
        rs = sorted((recs[i] for i in idxs), key=lambda r: (r.portal, r.tid))
        sources = sorted({r.source for r in rs})
        kind = "cross-portal" if len(sources) > 1 else "within-portal"
        confs, gaps = conflicts_of(rs, ctx.buyer_w)
        evidence = [{"confidence": c, "a": recs[i].ref_out(), "b": recs[j].ref_out(), "evidence": ev}
                    for c, i, j, ev in edges_by_root[root]]
        strongest = max((e["type"] for c, i, j, ev in edges_by_root[root] for e in ev),
                        key=lambda t: {"link": 3, "reference": 2}.get(t, 1))
        gid = "g_" + "_".join(f"{r.portal[-4:]}{r.tid[-6:]}" for r in rs[:2])
        g = {"id": gid, "kind": kind, "confidence": round(uf.conf[root], 2), "strongestEvidence":
             {"link": "link", "reference": "reference"}.get(strongest, "title"),
             "sources": [{"portal": s, "host": labels.get(s, s)} for s in sources],
             "records": [{**r.ref_out(), "title": r.title, "buyerName": r.buyer, "submissionDeadline": r.deadline,
                          "deadlineDisplay": fmt_deadline(r.deadline), "status": r.status,
                          "referenceNumber": r.ref, "documents": r.docs if r.reads_docs else None} for r in rs],
             "evidence": evidence, "conflicts": [c["field"] for c in confs], "gaps": gaps}
        groups.append(g)
        for c in confs:
            conflicts.append({"groupId": gid, "kind": kind, "confidence": g["confidence"], **c})

    groups.sort(key=lambda g: (g["kind"] != "cross-portal", -len(g["conflicts"]), -g["confidence"], g["id"]))
    conflicts.sort(key=lambda c: (c["kind"] != "cross-portal", _FIELD_ORDER.get(c["field"], 9), -c["confidence"],
                                  c["groupId"]))
    involved = sorted({r["portal"] for g in groups for r in g["records"]})
    by_field: dict[str, int] = defaultdict(int)
    for c in conflicts:
        by_field[c["field"]] += 1
    by_ev: dict[str, int] = defaultdict(int)
    for g in groups:
        by_ev[g["strongestEvidence"]] += 1
    per_portal = defaultdict(int)
    for r in recs:
        per_portal[r.source] += 1
    summary = {
        "records": len(recs),
        "portals": [{"id": p["id"], "host": labels.get(p["id"]), "url": p.get("url"),
                     "tenders": per_portal.get(p["id"], 0) if p["id"] not in alias else alias[p["id"]]["tenders"],
                     "sameAs": alias.get(p["id"], {}).get("sameAs")} for p in portals],
        "excludedSelfDuplicates": list(alias.values()),
        "groups": len(groups),
        "crossPortalGroups": sum(g["kind"] == "cross-portal" for g in groups),
        "withinPortalGroups": sum(g["kind"] == "within-portal" for g in groups),
        "portalsInvolved": len(involved),
        "crossPortalPortals": len({r["portal"] for g in groups if g["kind"] == "cross-portal" for r in g["records"]}),
        "conflicts": len(conflicts),
        "crossPortalConflicts": sum(c["kind"] == "cross-portal" for c in conflicts),
        "groupsWithConflicts": sum(bool(g["conflicts"]) for g in groups),
        "conflictsByField": dict(by_field),
        "groupsByEvidence": dict(by_ev),
        "edgesRefusedByGroupCap": refused,
        "blocking": stats,
    }
    return {"computedAt": timeutil.iso(timeutil.now()), "elapsedMs": round(1000 * (time.perf_counter() - started)),
            "summary": summary, "conflicts": conflicts, "groups": groups[:max_groups],
            "truncated": len(groups) > max_groups}


def compute_from_store(store, now: datetime | None = None) -> dict:
    """All served portals (serving or healing) of the store, read-only."""
    started = time.perf_counter()
    portals = [p for p in store.list_portals() if p["state"] in SERVED_STATES]
    # self-duplicates from the ids alone (a cheap query), so their full records are never loaded
    ids = {p["id"]: {r["tender_id"] for r in store._q("SELECT tender_id FROM tenders WHERE portal_id=?", (p["id"],))}
           for p in portals}
    alias = find_aliases(portals, ids, {p["id"]: host_of(p["url"]) for p in portals})
    tenders, reads_docs = {}, {}
    for p in portals:
        if p["id"] in alias:
            continue
        tenders[p["id"]] = [json.loads(r["data"]) for r in
                            store._q("SELECT data FROM tenders WHERE portal_id=?", (p["id"],))]
        spec = store.current_spec(p["id"])
        docs_spec = (spec[1].get("documents") or {}) if spec else {}
        reads_docs[p["id"]] = bool(docs_spec.get("from") or docs_spec.get("sel") or docs_spec.get("path"))
    out = compute(portals, tenders, now, reads_docs, alias=alias)
    out["elapsedMs"] = round(1000 * (time.perf_counter() - started))
    return out


class Cache:
    """The result is kept for `ttl` seconds. Computation runs in a worker thread (the event loop stays free), one at a
    time. After `ttl` the previous result is still returned at once while a fresh one is computed in the
    background; only the very first request, or `force`, waits."""

    def __init__(self, store, ttl: float = 300.0):
        self.store, self.ttl = store, ttl
        self._value: dict | None = None
        self._at = 0.0
        self._lock = None
        self._task = None

    async def _compute(self) -> dict:
        import asyncio
        async with self._lock:
            self._value = await asyncio.to_thread(compute_from_store, self.store)
            self._at = time.monotonic()
            return self._value

    async def get(self, force: bool = False) -> dict:
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        if force or self._value is None:
            return await self._compute()
        if time.monotonic() - self._at > self.ttl and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._compute())
        return self._value
