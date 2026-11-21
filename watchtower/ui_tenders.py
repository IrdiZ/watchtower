"""All-tenders explorer: `/_ui/tenders` (JSON) and `/_ui/tenders/view` (page).

Requests are answered from an in-memory index of the tenders table, so no request parses stored JSON. The index is
built once; after that the per-portal row counts are compared at most once a minute, on a side thread, and a change
rebuilds it (as does an age of ten minutes).

Counting rule (the dashboard headline uses the same one): a portal is its host without a leading "www.". When several
registrations share a host, only the registration with the most tenders is served here.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from urllib.parse import urlsplit

from fastapi import Query
from fastapi.responses import HTMLResponse

from . import timeutil

PAGE = os.path.join(os.path.dirname(__file__), "tenders.html")
CHECK_SECONDS = 60.0        # how often the row counts are compared; a change rebuilds the index
MAX_AGE_SECONDS = 600.0     # rebuilt anyway at this age (edits that change no count, such as a moved deadline)
BACKGROUND = True           # refresh on a side thread; tests switch this off to refresh in the calling thread
MAX_SIZE = 100

# Country from the host's ending. Hosts on a generic ending are listed by name.
COUNTRY_SUFFIXES = ((".gov.uk", "UK"), (".gov.wales", "UK"), (".de", "DE"), (".fr", "FR"), (".bzh", "FR"),
                    (".nl", "NL"), (".pl", "PL"), (".es", "ES"), (".ie", "IE"), (".ch", "CH"), (".fi", "FI"),
                    (".no", "NO"), (".lu", "LU"), (".mt", "MT"), (".pt", "PT"), (".lt", "LT"), (".bg", "BG"),
                    (".at", "AT"), (".be", "BE"), (".it", "IT"), (".se", "SE"), (".dk", "DK"), (".cz", "CZ"))
COUNTRY_HOSTS = {"achatpublic.com": "FR", "alsacemarchespublics.eu": "FR"}
SORTS = ("deadline", "published", "title")
STATUSES = ("open", "closed", "withdrawn", "all")


def host_of(url: str) -> str:
    host = (urlsplit(url or "").hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def country_of(host: str) -> str:
    if host in COUNTRY_HOSTS:
        return COUNTRY_HOSTS[host]
    for suffix, code in COUNTRY_SUFFIXES:
        if host.endswith(suffix):
            return code
    return ""


def served_registrations(portals: list[dict], counts: dict[str, int]) -> tuple[list[dict], dict[str, str]]:
    """Apply the counting rule. Returns the served registrations (one per host, with tenders) and a map from every
    registration id to the id that is served for its host. Ties go to the earliest registration."""
    best: dict[str, dict] = {}
    for p in portals:  # list_portals() is ordered by created_at, so the first of equals wins
        host = host_of(p["url"])
        if not host:
            continue
        n = counts.get(p["id"], 0)
        if host not in best or n > best[host]["tenders"]:
            best[host] = {"id": p["id"], "host": host, "country": country_of(host), "tenders": n}
    alias = {p["id"]: best[host_of(p["url"])]["id"] for p in portals if host_of(p["url"]) in best}
    return [b for b in best.values() if b["tenders"] > 0], alias


def _docs(raw) -> list[tuple]:
    """Downloaded documents of a tender as (file name, bytes, local download path); only files we hold."""
    import json
    try:
        items = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except ValueError:
        return []
    out = []
    for d in items if isinstance(items, list) else []:
        if isinstance(d, dict) and isinstance(d.get("path"), str) and d["path"].startswith("/files/"):
            out.append((str(d.get("fileName") or "document"), d.get("sizeBytes"), d["path"]))
    return out


class _Row:
    __slots__ = ("portal", "host", "country", "tender_id", "title", "buyer", "ref", "deadline", "deadline_ts",
                 "published", "status", "url", "hay", "docs")


class TenderIndex:
    def __init__(self, store):
        self.store = store
        self._lock = threading.Lock()
        self._built_at = 0.0
        self._checked_at = 0.0
        self._signature: tuple | None = None
        self._refreshing = False
        self.snapshot: tuple = ([], {s: [] for s in SORTS}, [], {})  # rows, orders, portals, alias
        self.builds = 0

    def _read(self, sql: str, args=()) -> list:
        """Read on a connection of our own: a full pass over the table must not hold the store's lock (the worker
        and every other page share it)."""
        db = sqlite3.connect(os.path.join(self.store.data_dir, "watchtower.db"), check_same_thread=False)
        try:
            return db.execute(sql, tuple(args)).fetchall()
        finally:
            db.close()

    def _counts(self) -> dict[str, int]:
        rows = self._read("SELECT portal_id, COUNT(*), MAX(last_changed_at) FROM tenders GROUP BY portal_id")
        # the latest change per portal joins the signature: an edit that changes no count (documents dropped, a
        # moved deadline) must also rebuild the index
        self._marks = {r[0]: r[2] for r in rows}
        return {r[0]: r[1] for r in rows}

    def ensure(self) -> None:
        """The first call builds the index. Later calls answer from the index in memory and, at most once every
        CHECK_SECONDS, start a refresh on the side: rebuild when the row counts changed or the index is older than
        MAX_AGE_SECONDS. Reading the whole table takes seconds on a large database, so no request waits for it."""
        if self._signature is None:
            with self._lock:
                if self._signature is None:
                    self._build(self._counts())
                    self._built_at = self._checked_at = time.monotonic()
            return
        now = time.monotonic()
        if now - self._checked_at < CHECK_SECONDS or self._refreshing:
            return
        self._checked_at = now
        if BACKGROUND:
            self._refreshing = True
            threading.Thread(target=self._refresh, daemon=True, name="tender-index").start()
        else:
            self._refresh()

    def _refresh(self) -> None:
        try:
            with self._lock:
                counts = self._counts()
                if self._sig(counts) != self._signature or time.monotonic() - self._built_at >= MAX_AGE_SECONDS:
                    self._build(counts)
                    self._built_at = time.monotonic()
        except Exception:  # keep serving the index we have; the next check tries again
            pass
        finally:
            self._refreshing = False

    def _sig(self, counts: dict[str, int]) -> tuple:
        marks = tuple(sorted((getattr(self, "_marks", None) or {}).items()))
        return (tuple(sorted(counts.items())), marks, tuple((p["id"], p["url"]) for p in self.store.list_portals()))

    def _build(self, counts: dict[str, int]) -> None:
        signature = self._sig(counts)
        portals, alias = served_registrations(self.store.list_portals(), counts)
        by_id = {p["id"]: p for p in portals}
        rows: list[_Row] = []
        if by_id:
            marks = ",".join("?" * len(by_id))
            sql = ("SELECT portal_id, tender_id, json_extract(data,'$.title'), json_extract(data,'$.buyerName'),"
                   " json_extract(data,'$.referenceNumber'), json_extract(data,'$.submissionDeadline'),"
                   " json_extract(data,'$.publicationDate'), json_extract(data,'$.status'),"
                   " json_extract(data,'$.url'), json_extract(data,'$.documents')"
                   f" FROM tenders WHERE portal_id IN ({marks})")
            for pid, tid, title, buyer, ref, deadline, published, status, url, docs in self._read(sql, list(by_id)):
                p = by_id[pid]
                r = _Row()
                r.portal, r.host, r.country, r.tender_id = pid, p["host"], p["country"], tid
                r.title = title if isinstance(title, str) else ""
                r.buyer = buyer if isinstance(buyer, str) else None
                r.ref = str(ref) if ref not in (None, "") else None
                r.deadline = deadline if isinstance(deadline, str) else None
                r.deadline_ts = None
                if r.deadline:
                    try:
                        r.deadline_ts = timeutil.parse(r.deadline).timestamp()
                    except (ValueError, TypeError):
                        pass
                r.published = published if isinstance(published, str) else None
                r.status = status if status in ("withdrawn", "closed") else "open"
                r.url = url if isinstance(url, str) else None
                r.hay = " ".join(x for x in (r.title, r.buyer, r.ref, tid) if x).casefold()
                r.docs = _docs(docs)
                rows.append(r)
        # One base order (host, id) and stable sorts on top of it: equal keys always come out in the same order.
        rows.sort(key=lambda r: (r.host, r.tender_id))
        idx = range(len(rows))
        orders = {
            "deadline": sorted(idx, key=lambda i: (rows[i].deadline_ts is None, rows[i].deadline_ts or 0.0)),
            "published": sorted(idx, key=lambda i: (rows[i].published is not None, rows[i].published or ""),
                                reverse=True),
            "title": sorted(idx, key=lambda i: (not rows[i].title, rows[i].title.casefold())),
        }
        # reverse=True keeps equal keys in base order (sort stability holds under reverse).
        # Swapped in one assignment: a request running meanwhile sees the old index or the new one, never a mix.
        self.snapshot = (rows, orders, portals, alias)
        self._signature = signature
        self.builds += 1

    # ------------------------------------------------------------------ query
    def query(self, q: str = "", portals: list[str] | None = None, country: list[str] | None = None,
              status: str = "open", closing: int | None = None, sort: str = "deadline", page: int = 1,
              size: int = 25, now: datetime | None = None, docs: bool = False) -> dict:
        self.ensure()
        rows, orders, served_portals, alias = self.snapshot
        order = orders[sort if sort in SORTS else "deadline"]
        at = now or timeutil.now()
        now_ts = at.timestamp()
        until_ts = (at + timedelta(days=closing)).timestamp() if closing else None
        words = (q or "").casefold().split()
        want_portals = {alias.get(p, p) for p in (portals or []) if p}
        want_countries = {c.upper() for c in (country or []) if c}
        if status not in STATUSES:
            status = "open"

        by_portal = {p["id"]: 0 for p in served_portals}
        by_country: dict[str, int] = {}
        for p in served_portals:
            by_country.setdefault(p["country"], 0)
        served = {"open": 0, "closed": 0, "withdrawn": 0}
        with_docs = 0          # rows matching every other filter that have documents (the "With documents" chip)
        start, total, items = (page - 1) * size, 0, []
        for i in order:
            r = rows[i]
            st = r.status
            if st == "open" and r.deadline_ts is not None and r.deadline_ts <= now_ts:
                st = "closed"  # deadlines that passed since the last read close on the fly, as on GET /tenders
            served[st] += 1
            if status != "all" and st != status:
                continue
            if until_ts is not None and (st != "open" or r.deadline_ts is None or r.deadline_ts > until_ts):
                continue
            if words and not all(w in r.hay for w in words):
                continue
            in_portal = not want_portals or r.portal in want_portals
            in_country = not want_countries or r.country in want_countries
            if r.docs and in_portal and in_country:
                with_docs += 1
            if docs and not r.docs:
                continue
            # Each facet is counted under every filter except its own, so its chips show what a click would give.
            if in_country:
                by_portal[r.portal] += 1
            if in_portal:
                by_country[r.country] += 1
            if in_portal and in_country:
                if start <= total < start + size:
                    items.append({
                        "title": r.title, "buyerName": r.buyer, "submissionDeadline": r.deadline,
                        "publicationDate": r.published, "status": st, "referenceNumber": r.ref, "url": r.url,
                        "tenderId": r.tender_id, "portal": r.portal, "host": r.host, "country": r.country,
                        "documentCount": len(r.docs),
                        "documents": [{"name": n, "size": z, "url": u} for n, z, u in r.docs]})
                total += 1
        return {
            "total": total, "page": page, "size": size, "sort": sort if sort in SORTS else "deadline",
            "status": status, "items": items,
            "selectedPortals": sorted(want_portals),  # as served: a left-out registration maps to its host's one
            "facets": {
                "portals": [{"id": p["id"], "host": p["host"], "country": p["country"], "count": by_portal[p["id"]]}
                            for p in sorted(served_portals, key=lambda p: (-by_portal[p["id"]], p["host"]))],
                "countries": [{"country": c, "count": n}
                              for c, n in sorted(by_country.items(), key=lambda kv: (-kv[1], kv[0]))],
            },
            "served": {**served, "all": len(rows), "portals": len(served_portals)},
            "withDocuments": with_docs, "docs": bool(docs),
        }


def register(app, store) -> TenderIndex:
    index = TenderIndex(store)
    app.state.tender_index = index
    threading.Thread(target=index.ensure, daemon=True, name="tender-index").start()  # warm before the first visit

    @app.get("/_ui/tenders")
    def ui_tenders(q: str = Query(default=""), portal: list[str] = Query(default=[]),
                   country: list[str] = Query(default=[]), status: str = Query(default="open"),
                   closing: int | None = Query(default=None, ge=1, le=3650),
                   sort: str = Query(default="deadline"), page: int = Query(default=1, ge=1),
                   size: int = Query(default=25, ge=1, le=MAX_SIZE), docs: bool = Query(default=False)):
        """Every served tender, filtered, sorted and paged, with per-portal and per-country counts."""
        return index.query(q=q, portals=portal, country=country, status=status, closing=closing, sort=sort,
                           page=page, size=size, docs=docs)

    @app.get("/_ui/tenders/view", response_class=HTMLResponse)
    def ui_tenders_view():
        with open(PAGE, encoding="utf-8") as f:
            return HTMLResponse(f.read())

    return index
