"""Change feeds (ATOM / RSS): a portal's official syndication of tender changes, read as list kind "feed".

A change feed is a log, not a listing. Each entry is the state of one procedure at one moment; files are chained
with rel="next" (or another rel the spec names) toward OLDER entries. So:
- the first load is a bounded backfill: follow the chain until entries are older than `backfill_days`, at most
  `max_files` files per read; an unfinished backfill is continued on the next reads (pending segment);
- later reads load only the delta: from the newest file down to the high-water mark (the newest entry time already
  applied), with a conditional GET (ETag / Last-Modified) on the newest file, so an unchanged feed costs one 304;
- one tender per procedure: the newest entry for an id wins, also against what is already stored;
- nothing is closed for being absent from a read: only an explicit status (fields.status.map -> closed/withdrawn),
  a tombstone (RFC 6721 at:deleted-entry -> withdrawn) or a passed deadline closes a tender.

Fields are XPath 1.0 expressions relative to the entry, with the feed's own namespace prefixes (the default Atom
namespace is "atom", e.g. "atom:link/@href"), or simple dotted element names ("ContractFolderStatus.ContractFolderID",
"link@href") matched by local name. A list of paths is joined with a space (a date and a time in two elements).
No portal-specific code: only the feed formats.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from lxml import etree

from . import timeutil
from .extract import clean_text

ATOM = "http://www.w3.org/2005/Atom"
TOMBSTONES = "http://purl.org/atompub/tombstones/1.0"
NS_DEFAULT = {"atom": ATOM, "at": TOMBSTONES, "dc": "http://purl.org/dc/elements/1.1/",
              "content": "http://purl.org/rss/1.0/modules/content/"}
STATUSES = ("open", "closed", "withdrawn")

BACKFILL_DAYS = int(os.environ.get("WATCHTOWER_FEED_BACKFILL_DAYS", "30"))
MAX_FILES = int(os.environ.get("WATCHTOWER_FEED_MAX_FILES", "20"))   # feed files per read (~10 MB each on big portals)

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, remove_comments=True, remove_pis=True,
                          huge_tree=False, recover=False)
_SIMPLE = re.compile(r"^[A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)*(?:@[A-Za-z_][\w:-]*)?$")


class FeedError(Exception):
    """The answer is not a readable ATOM/RSS feed."""


def looks_like_feed(content_type: str, head: bytes | str) -> bool:
    """Cheap test before parsing: a feed media type, or an XML body whose root is <feed>, <rss> or <rdf:RDF>."""
    ct = (content_type or "").lower()
    if "atom+xml" in ct or "rss+xml" in ct:
        return True
    if isinstance(head, bytes):
        head = head[:4000].decode("utf-8", errors="replace")
    h = re.sub(r"<\?xml[^>]*\?>|<!--.*?-->|<!DOCTYPE[^>]*>", "", head[:4000], flags=re.S).lstrip()
    return bool(re.match(r"<(?:[\w-]+:)?(?:feed|rss|RDF)\b", h))


@dataclass
class FeedDoc:
    format: str                          # atom | rss
    root: Any
    ns: dict[str, str]
    entries: list
    links: dict[str, str]                # rel -> href (self, first, next, prev, next-archive, ...)
    deleted: list[tuple[str, str | None, str | None]] = field(default_factory=list)   # (ref, when, comment)


def parse(content: bytes) -> FeedDoc:
    try:
        root = etree.fromstring(content, _PARSER)
    except (etree.XMLSyntaxError, ValueError) as e:
        raise FeedError(f"not well-formed XML ({str(e)[:120]})") from None
    ns = dict(NS_DEFAULT)
    for p, uri in (root.nsmap or {}).items():
        if p and uri:
            ns.setdefault(p, uri)
    tag = etree.QName(root).localname.lower()
    links: dict[str, str] = {}
    if tag == "feed":
        entries = root.findall(f"{{{ATOM}}}entry")
        for ln in root.findall(f"{{{ATOM}}}link"):
            rel, href = ln.get("rel") or "alternate", ln.get("href")
            if href and rel not in links:
                links[rel] = href.strip()
        deleted = [(d.get("ref"), d.get("when"),
                    next((c.get("type") or clean_text(c.text) for c in d.findall(f"{{{TOMBSTONES}}}comment")), None))
                   for d in root.findall(f"{{{TOMBSTONES}}}deleted-entry") if d.get("ref")]
        return FeedDoc("atom", root, ns, entries, links, deleted)
    if tag in ("rss", "rdf"):
        channel = root.find("channel")
        if channel is None:
            channel = next((c for c in root if etree.QName(c).localname == "channel"), root)
        entries = [c for c in (root.iter() if tag == "rdf" else channel) if isinstance(c.tag, str)
                   and etree.QName(c).localname == "item"]
        for ln in channel.findall(f"{{{ATOM}}}link"):        # RFC 5005 paging inside RSS
            rel, href = ln.get("rel") or "alternate", ln.get("href")
            if href and rel not in links:
                links[rel] = href.strip()
        return FeedDoc("rss", root, ns, entries, links, [])
    raise FeedError(f"the XML root is <{etree.QName(root).localname}>, not an ATOM <feed> or RSS <rss>")


# ------------------------------------------------------------------------------------------- values from an entry
def _to_xpath(path: str) -> str:
    """'ContractFolderStatus.ContractFolderID' / 'link@href' -> local-name steps (no prefixes needed)."""
    if not _SIMPLE.match(path) or ":" in path.split("@")[0]:
        return path
    elems, _, attr = path.partition("@")
    steps = "/".join(f"*[local-name()='{e}']" for e in elems.split("."))
    return steps + (f"/@{attr}" if attr else "")


_COMPILED: dict[tuple, Any] = {}


def compile_path(path: str, ns: dict[str, str]):
    key = (path, tuple(sorted(ns.items())))
    xp = _COMPILED.get(key)
    if xp is None:
        xp = etree.XPath(_to_xpath(path), namespaces=ns)
        if len(_COMPILED) > 2000:
            _COMPILED.clear()
        _COMPILED[key] = xp
    return xp


def _text(node) -> str | None:
    if isinstance(node, etree._Element):
        return clean_text(" ".join(t for t in node.itertext()))
    if isinstance(node, bool):
        return None if node is False else "true"
    if isinstance(node, float):
        return None if node != node else (str(int(node)) if node.is_integer() else str(node))
    return clean_text(str(node)) if node is not None else None


def _eval(entry, path: str, ns: dict[str, str], all_values: bool = False):
    try:
        res = compile_path(path, ns)(entry)
    except (etree.XPathError, etree.XPathEvalError, etree.XPathSyntaxError):
        return [] if all_values else None
    vals = res if isinstance(res, list) else [res]
    out = [v for v in (_text(v) for v in vals) if v]
    return out if all_values else (out[0] if out else None)


def entry_value(entry, fspec: dict, ns: dict[str, str]) -> Any:
    """A field of one entry: XPath (or dotted names); a list of paths is joined with a space; regex applies."""
    from .extract import _apply_regex
    path = fspec.get("path")
    if not path:
        return None
    if fspec.get("all"):
        vals = []
        for p in (path if isinstance(path, list) else [path]):
            vals += _eval(entry, p, ns, all_values=True)
        return [v for v in (_apply_regex(v, fspec.get("regex")) for v in vals) if v]
    if isinstance(path, list):
        parts = [_eval(entry, p, ns) for p in path if isinstance(p, str)]
        if not parts or parts[0] is None:
            return None
        val = " ".join(p for p in parts if p)
    else:
        val = _eval(entry, path, ns)
    return _apply_regex(val, fspec.get("regex")) if isinstance(val, str) else val


def path_problems(path, ns: dict[str, str] | None = None) -> list[str]:
    """Static check of a field path (compiles; prefixes resolvable with the feed's namespaces when given)."""
    out = []
    for p in (path if isinstance(path, list) else [path]):
        if not isinstance(p, str) or not p.strip():
            out.append("path must be an XPath string (or a list of them)")
            continue
        try:
            xp = etree.XPath(_to_xpath(p), namespaces={**NS_DEFAULT, **(ns or {})})
            if ns is not None:
                xp(etree.fromstring(b"<x/>"))          # an unknown prefix only fails at evaluation
        except etree.XPathError as e:
            out.append(f"invalid XPath '{p[:80]}' ({str(e)[:60]})")
    return out


def _parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        return timeutil.parse(s)
    except (ValueError, TypeError):
        pass
    try:
        dt = parsedate_to_datetime(s)        # RSS pubDate (RFC 822)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None


def entry_updated(doc: FeedDoc, entry, spec_path: str | None = None) -> datetime | None:
    """When this entry was written: list.updated (a path), else atom:updated / atom:published / RSS pubDate."""
    paths = [spec_path] if spec_path else (["atom:updated", "atom:published"] if doc.format == "atom"
                                           else ["pubDate", "dc:date", "atom:updated"])
    for p in paths:
        dt = _parse_time(_eval(entry, p, doc.ns))
        if dt is not None:
            return dt
    return None


def entry_id(doc: FeedDoc, entry) -> str | None:
    return _eval(entry, "atom:id" if doc.format == "atom" else "guid", doc.ns) or _eval(entry, "link", doc.ns)


def status_of(fspec: dict, code) -> str | None:
    """open | closed | withdrawn for a status code by fields.status.map (exact, then case-insensitive); None when
    the code is not mapped (unknown)."""
    m = fspec.get("map") if isinstance(fspec, dict) else None
    if not isinstance(m, dict) or code in (None, ""):
        return None
    c = str(code).strip()
    v = m.get(c)
    if v is None:
        low = {str(k).strip().lower(): val for k, val in m.items()}
        v = low.get(c.lower())
    return v if v in STATUSES else None


def entry_xml(entry, max_chars: int = 14000) -> str:
    """One entry as compact XML for a model prompt: code-list attributes dropped, runs of more than two siblings
    with the same tag cut, long texts shortened (a 25 KB procurement entry becomes a few KB)."""
    import copy
    e = copy.deepcopy(entry)
    for el in e.iter():
        if not isinstance(el.tag, str):
            continue
        for a in [a for a in el.attrib if etree.QName(a).localname in ("listURI", "languageID", "listAgencyID",
                                                                        "listVersionID")]:
            del el.attrib[a]
        kids = [c for c in el if isinstance(c.tag, str)]
        run, prev = 0, None
        for c in kids:
            run = run + 1 if c.tag == prev else 1
            prev = c.tag
            if run > 2:
                el.remove(c)
        if el.text and len(el.text) > 200:
            el.text = el.text[:200] + "..."
    s = etree.tostring(e, encoding="unicode")
    s = re.sub(r">\s+<", "><", s)
    s = re.sub(r'\s+xmlns(?::[\w-]+)?="[^"]*"', "", s)
    return s[:max_chars]


def date_paths(doc: FeedDoc, entry, limit: int = 20) -> list[str]:
    """Paths (with the feed's prefixes) of the elements of an entry whose text is a date: a map for the model."""
    rev = {}
    for p, uri in doc.ns.items():
        rev.setdefault(uri, p)
    out = []
    for el in entry.iter():
        if not isinstance(el.tag, str) or not el.text or not re.match(r"\s*\d{4}-\d{2}-\d{2}", el.text):
            continue
        parts, cur = [], el
        while cur is not None and cur is not entry:
            q = etree.QName(cur)
            parts.append(f"{rev.get(q.namespace, '')}:{q.localname}" if q.namespace in rev else q.localname)
            cur = cur.getparent()
        path = "/".join(reversed(parts))
        if not any(o.startswith(path + " = ") for o in out):
            out.append(f"{path} = {el.text.strip()[:30]}")
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------------------------------------ the walk
@dataclass
class FeedRead:
    rows: list[dict]
    merged: int
    complete: bool
    capped: bool
    stop_reason: str
    pages: int
    stats: dict
    state: dict
    closed_dropped: int = 0


def _older_rel(spec: dict, doc: FeedDoc) -> str | None:
    pag = (spec.get("list") or {}).get("pagination") or {}
    rel = pag.get("older") or pag.get("rel")
    if rel:
        return doc.links.get(rel)
    for r in ("next", "prev-archive", "next-archive"):
        if doc.links.get(r):
            return doc.links[r]
    return None


async def _fetch(ctx, url: str, headers: dict | None = None):
    return await ctx.client.get(url, headers=headers or None)


async def walk(ctx, spec: dict, start: str, until: datetime | None, files_left: int, *, conditional: dict | None = None,
               sink=None) -> dict:
    """Read files from `start` toward older entries until a file reaches `until` (entries at or before it are not
    needed), the chain ends, or `files_left` files were read. `sink(doc, entry, updated, file_url)` gets every entry
    newer than `until` and `sink_deleted(...)` every tombstone. Returns a summary of the segment."""
    out = {"files": 0, "entries": 0, "newest": None, "oldest": None, "next": None, "stop": "", "reached": False,
           "notModified": False, "etag": None, "lastModified": None, "ended": False}
    url, seen = start, set()
    upd_path = (spec.get("list") or {}).get("updated")
    while url:
        if url in seen:
            out["stop"] = f"the feed links back to a file already read ({url})"
            break
        if files_left <= 0:
            out["next"], out["stop"] = url, "file cap for this read"
            break
        seen.add(url)
        resp = await _fetch(ctx, url, conditional if out["files"] == 0 else None)
        if resp.status_code == 304:
            out.update(notModified=True, reached=True, stop="not modified since the last read")
            break
        if resp.status_code >= 400:
            if out["files"] == 0:
                raise FeedError(f"feed file returned HTTP {resp.status_code}: {url}")
            out["stop"] = f"HTTP {resp.status_code} for the next feed file {url}"
            break
        doc = parse(resp.content)
        out["format"] = doc.format
        if out["files"] == 0:
            out["etag"], out["lastModified"] = resp.headers.get("etag"), resp.headers.get("last-modified")
        out["files"] += 1
        files_left -= 1
        if out["files"] % 5 == 1:
            ctx.note(f"feed file {out['files']}: {len(doc.entries)} entries ({url})")
        file_oldest = None
        for e in doc.entries:
            upd = entry_updated(doc, e, upd_path)
            if upd is None:
                continue
            file_oldest = upd if file_oldest is None or upd < file_oldest else file_oldest
            if until is not None and upd < until:
                continue
            out["entries"] += 1
            out["newest"] = upd if out["newest"] is None or upd > out["newest"] else out["newest"]
            out["oldest"] = upd if out["oldest"] is None or upd < out["oldest"] else out["oldest"]
            if sink:
                sink(doc, e, upd, str(resp.url))
        for ref, when, comment in doc.deleted:
            w = _parse_time(when)
            if sink and (until is None or w is None or w >= until):
                sink(doc, None, w, str(resp.url), deleted=(ref, comment))
        if until is not None and file_oldest is not None and file_oldest <= until:
            out.update(reached=True, stop="reached the stop point")
            break
        url = _older_rel(spec, doc)
        if not url:
            out.update(reached=True, ended=True, stop="the feed has no older file")
            break
        from urllib.parse import urljoin
        url = urljoin(str(resp.url), url)
    return out


def _iso(dt: datetime | None) -> str | None:
    """UTC, fixed width: feed times are compared as strings (newest entry wins)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00") if dt else None


async def collect(ctx, spec: dict, previous: dict[str, dict], limits, state: dict | None) -> FeedRead:
    """All rows a read of the feed yields (raw list values, newest entry per tender), plus the new feed state."""
    lst = spec["list"]
    pag = lst.get("pagination") or {}
    fields = spec.get("fields", {})
    list_fields = {f: s for f, s in fields.items() if s.get("from", "list") == "list" and not s.get("template")}
    docs_spec = spec.get("documents") or {}
    state = dict(state or {})
    now = timeutil.now()
    days = int(pag.get("backfill_days") or BACKFILL_DAYS)
    files_left = int(pag.get("max_files") or MAX_FILES)
    if limits is not None and limits.max_pages:
        files_left = min(files_left, int(limits.max_pages))
    head = lst["url"]
    hwm = _parse_time(state.get("hwm"))
    bound = now - timedelta(days=days)
    mode = "delta" if hwm else "backfill"

    by_id: dict[str, dict] = {}
    tomb: list[tuple[str, datetime | None, str | None]] = []
    merged = 0
    unknown: dict[str, int] = {}
    no_id = 0

    spec_ns = lst.get("namespaces") if isinstance(lst.get("namespaces"), dict) else {}

    def sink(doc, e, upd, file_url, deleted=None):
        nonlocal merged, no_id
        if deleted is not None:
            tomb.append((deleted[0], upd, deleted[1]))
            return
        ns = {**doc.ns, **spec_ns}      # the prefixes the spec was written with (same URIs if the feed renames them)
        raw = {f: entry_value(e, s, ns) for f, s in list_fields.items()}
        rid = raw.get("id")
        rid = str(rid).strip() if rid not in (None, "") else None
        if rid is None:
            no_id += 1
            if no_id > 50:
                return
        raw["_list_url"] = file_url
        raw["_entryId"] = entry_id(doc, e)
        raw["_feedUpdated"] = _iso(upd)
        if docs_spec.get("from") == "list" and docs_spec.get("path"):
            raw["_docs_list"] = entry_value(e, {**docs_spec, "all": True}, ns)
        code = raw.get("status")
        if code not in (None, "") and "map" in (fields.get("status") or {}) \
                and status_of(fields["status"], code) is None:
            unknown[str(code)] = unknown.get(str(code), 0) + 1
        key = rid or f"_noid{no_id}"
        kept = by_id.get(key)
        if kept is not None:
            merged += 1
            if (kept.get("_feedUpdated") or "") >= raw["_feedUpdated"]:
                return
        by_id[key] = raw

    segments = []
    cond = None
    if mode == "delta":
        cond = {k: v for k, v in (("If-None-Match", state.get("etag")),
                                  ("If-Modified-Since", state.get("lastModified"))) if v}
    seg = await walk(ctx, spec, head, hwm if mode == "delta" else bound, files_left, conditional=cond, sink=sink)
    segments.append(("head", seg))
    files_left -= seg["files"]
    pending = list(state.get("pending") or [])
    if seg["next"]:      # the walk from the newest file did not reach its stop point: continue it next time
        pending.insert(0, {"next": seg["next"], "until": _iso(hwm if mode == "delta" else bound),
                           "oldest": _iso(seg["oldest"])})
    rest = []
    for p in (pending[1:] if seg["next"] else pending):
        if files_left <= 0:
            rest.append(p)
            continue
        until = _parse_time(p.get("until"))
        s2 = await walk(ctx, spec, p["next"], until, files_left, sink=sink)
        segments.append(("pending", s2))
        files_left -= s2["files"]
        if s2["next"]:
            rest.append({**p, "next": s2["next"], "oldest": _iso(s2["oldest"]) or p.get("oldest")})
        elif not s2["reached"]:
            ctx.note(f"feed: the older part could not be continued ({s2['stop']}); coverage stays as it is")
    if seg["next"]:
        rest.insert(0, pending[0])

    # tombstones: a deleted entry withdraws its tender (matched by entry id, in this read or among stored tenders)
    by_entry = {r.get("_entryId"): k for k, r in by_id.items() if r.get("_entryId")}
    prev_by_entry = {p["data"].get("_entryId"): tid for tid, p in previous.items() if p["data"].get("_entryId")}
    withdrawn_n = 0
    for ref, when, comment in tomb:
        w = _iso(when) or ""
        if ref in by_entry:
            r = by_id[by_entry[ref]]
            if w >= (r.get("_feedUpdated") or ""):
                r["_status"] = "withdrawn"
                r["_statusWhy"] = comment
                withdrawn_n += 1
        elif ref in prev_by_entry:
            tid = prev_by_entry[ref]
            pd = previous[tid]["data"]
            if w >= (pd.get("_feedUpdated") or "") and pd.get("status") != "withdrawn":
                import json
                try:
                    raw = json.loads(pd.get("_listSig") or "{}")
                except ValueError:
                    raw = {}
                raw.update({"id": tid, "_list_url": head, "_entryId": ref, "_feedUpdated": w or pd.get("_feedUpdated"),
                            "_status": "withdrawn", "_statusWhy": comment})
                by_id[tid] = raw
                withdrawn_n += 1

    # newest wins against stored tenders too; closed/withdrawn tenders are only kept to update stored ones
    rows, stale, closed_unknown = [], 0, 0
    for key, raw in by_id.items():
        rid = raw.get("id")
        rid = str(rid).strip() if rid not in (None, "") else None
        prev = previous.get(rid) if rid else None
        if prev is not None:
            if (prev["data"].get("_feedUpdated") or "") > (raw.get("_feedUpdated") or "") and not raw.get("_status"):
                stale += 1
                continue
            raw["_keep"] = True       # a known tender: its new state is applied whatever it is
        st = raw.get("_status") or status_of(fields.get("status") or {}, raw.get("status"))
        if st in ("closed", "withdrawn") and prev is None:
            closed_unknown += 1
            continue
        rows.append(raw)

    newest_list = [s["newest"] for _, s in segments if s["newest"]]
    newest = max(newest_list) if newest_list else None
    hwms = [d for d in (hwm, seg["newest"]) if d]
    new_hwm = max(hwms) if hwms else None
    if mode == "backfill":
        if rest:
            reach = _iso(seg["oldest"])                       # so far; deepened on the next reads
        else:
            reach = _iso(seg["oldest"]) if seg["ended"] else _iso(bound)
    else:
        reach = state.get("reach")
        if state.get("pending") and not rest:
            reach = state["pending"][-1].get("until") or reach  # the unfinished backfill finished in this read
        elif rest:
            reach = rest[-1].get("oldest") or reach
    new_state = {"hwm": _iso(new_hwm) or state.get("hwm"), "reach": reach, "pending": rest,
                 "backfillDays": days, "since": state.get("since") or _iso(now), "head": head,
                 "etag": seg["etag"] if seg["files"] else state.get("etag"),
                 "lastModified": seg["lastModified"] if seg["files"] else state.get("lastModified")}
    files = sum(s["files"] for _, s in segments)
    complete = not rest
    stop = seg["stop"] if complete else (f"backfill continues on the next read ({len(rest)} part(s) left; "
                                         f"reached {reach or '?'})")
    fmt = next((s["format"] for _, s in segments if s.get("format")), None)
    stats = {"format": fmt, "url": head, "mode": mode, "files": files,
             "entries": sum(s["entries"] for _, s in segments), "tenders": len(rows), "deleted": len(tomb),
             "withdrawn": withdrawn_n, "stale": stale, "closedNotServed": closed_unknown,
             "newest": _iso(newest), "hwm": new_state["hwm"], "reach": reach, "backfillDays": days,
             "pending": len(rest), "notModified": bool(seg["notModified"]), "stop": stop,
             **({"unknownStatuses": unknown} if unknown else {}),
             **({"entriesWithoutId": no_id} if no_id else {})}
    return FeedRead(rows=rows, merged=merged, complete=complete, capped=not complete, stop_reason=stop,
                    pages=files, stats=stats, state=new_state, closed_dropped=closed_unknown)


def coverage_note(stats: dict) -> str:
    """The honest coverage sentence for a read from a change feed (goes into the portal's reason)."""
    fd = stats.get("feed") or {}
    reach = fd.get("reach")
    when = (reach[:16].replace("T", " ") + " UTC") if isinstance(reach, str) else "?"
    src = f"the portal's official change feed ({fd.get('url')})"
    if fd.get("pending"):
        return (f"Coverage: partial - read from {src}; the backfill reaches back to {when} so far and continues on "
                f"the next reads (goal: {fd.get('backfillDays')} days); tenders whose last change is older are not "
                f"included yet; nothing is closed for being absent.")
    return (f"Coverage: complete for changes since {when} - read from {src}; backfill of "
            f"{fd.get('backfillDays')} days, later reads load only the changes; tenders not changed in that window "
            f"are not included; nothing is closed for being absent.")
