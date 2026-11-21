"""SQLite persistence: portals, runs, specs, tenders (committed snapshots), files, LLM ledger.

Reads that fail validation are never written here: only `commit_snapshot` changes served tenders,
which is what keeps "the state it reports" equal to "what it delivers".
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Iterable

from . import timeutil

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS portals (
  id TEXT PRIMARY KEY, url TEXT NOT NULL, norm_url TEXT NOT NULL UNIQUE,
  state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL, last_success_at TEXT, last_attempt_at TEXT,
  failed_streak INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, portal_id TEXT NOT NULL, kind TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT, result TEXT,
  summary TEXT NOT NULL DEFAULT '', detail TEXT NOT NULL DEFAULT '{}',
  requests INTEGER NOT NULL DEFAULT 0, llm_usd REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS runs_portal ON runs(portal_id, started_at);
CREATE TABLE IF NOT EXISTS specs (
  portal_id TEXT NOT NULL, version INTEGER NOT NULL, spec TEXT NOT NULL,
  created_at TEXT NOT NULL, source_run TEXT, PRIMARY KEY (portal_id, version)
);
CREATE TABLE IF NOT EXISTS tenders (
  portal_id TEXT NOT NULL, tender_id TEXT NOT NULL, data TEXT NOT NULL,
  content_hash TEXT NOT NULL, last_changed_at TEXT NOT NULL, first_seen_at TEXT NOT NULL,
  last_verified_at TEXT, PRIMARY KEY (portal_id, tender_id)
);
CREATE TABLE IF NOT EXISTS known_closed (
  portal_id TEXT NOT NULL, tender_id TEXT NOT NULL, list_sig TEXT NOT NULL, deadline TEXT,
  why TEXT NOT NULL, recorded_at TEXT NOT NULL, PRIMARY KEY (portal_id, tender_id)
);
CREATE TABLE IF NOT EXISTS files (
  sha256 TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS read_state (
  portal_id TEXT PRIMARY KEY, state TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state_changes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, portal_id TEXT NOT NULL, at TEXT NOT NULL,
  from_state TEXT, to_state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', tenders INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS state_changes_at ON state_changes(at);
CREATE TABLE IF NOT EXISTS audits (
  id TEXT PRIMARY KEY, portal_id TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT, spec_version INTEGER, result TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS audits_portal ON audits(portal_id, started_at);
CREATE TABLE IF NOT EXISTS llm_calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT, portal_id TEXT, run_id TEXT, purpose TEXT, model TEXT,
  tokens_in INTEGER, tokens_out INTEGER, usd REAL, at TEXT
);
"""

# Fields whose change should never happen silently on the same tender (used by Notice/Repair checks).
IMMUTABLE_FIELDS = ("title", "buyerName", "publicationDate")
# Fields excluded from the content hash (bookkeeping, not portal content).
_HASH_EXCLUDE = {"lastChangedAt", "status"}
CLOSE_AFTER_MISSED_READS = 2   # complete reads in a row a tender must be missing from before it is closed


def new_run_id() -> str:
    return "r_" + uuid.uuid4().hex[:12]


def content_hash(tender: dict[str, Any]) -> str:
    body = {k: v for k, v in tender.items() if k not in _HASH_EXCLUDE and not k.startswith("_")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def derive_status(tender: dict[str, Any], at: datetime) -> str:
    if tender.get("status") == "withdrawn":
        return "withdrawn"
    if tender.get("status") == "closed":
        return "closed"
    deadline = tender.get("submissionDeadline")
    if deadline:
        try:
            if timeutil.parse(deadline) <= at:
                return "closed"
        except ValueError:
            pass
    return "open"


class Store:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files_dir = os.path.join(data_dir, "files")
        os.makedirs(self.files_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(os.path.join(data_dir, "watchtower.db"), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    # ------------------------------------------------------------------ helpers
    def _q(self, sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, tuple(args)))

    def _x(self, sql: str, args: Iterable = ()) -> None:
        with self._lock:
            self._db.execute(sql, tuple(args))
            self._db.commit()

    # ------------------------------------------------------------------ portals
    def get_portal(self, pid: str) -> dict | None:
        rows = self._q("SELECT * FROM portals WHERE id=?", (pid,))
        return dict(rows[0]) if rows else None

    def list_portals(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM portals ORDER BY created_at")]

    def create_portal(self, pid: str, url: str, norm_url: str) -> bool:
        """Returns True if newly created, False if it already existed."""
        with self._lock:
            try:
                self._db.execute(
                    "INSERT INTO portals (id, url, norm_url, state, reason, created_at) VALUES (?,?,?,?,?,?)",
                    (pid, url, norm_url, "integrating", "Being attached.", timeutil.iso(timeutil.now())),
                )
                self._db.execute("INSERT INTO state_changes (portal_id, at, from_state, to_state, reason) "
                                 "VALUES (?,?,?,?,?)", (pid, timeutil.iso(timeutil.now()), None, "integrating",
                                                        "Being attached."))
                self._db.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def set_state(self, pid: str, state: str, reason: str) -> None:
        """Set the state; a change of state is also logged (with the tenders served at that moment) for the timeline."""
        with self._lock:
            row = self._db.execute("SELECT state FROM portals WHERE id=?", (pid,)).fetchone()
            self._db.execute("UPDATE portals SET state=?, reason=? WHERE id=?", (state, reason, pid))
            if row is not None and row["state"] != state:
                n = self._db.execute("SELECT COUNT(*) AS n FROM tenders WHERE portal_id=?", (pid,)).fetchone()["n"]
                self._db.execute("INSERT INTO state_changes (portal_id, at, from_state, to_state, reason, tenders) "
                                 "VALUES (?,?,?,?,?,?)", (pid, timeutil.iso(timeutil.now()), row["state"], state,
                                                          reason, n))
            self._db.commit()

    def state_changes(self, pid: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM state_changes", ()
        if pid:
            sql, args = sql + " WHERE portal_id=?", (pid,)
        return [dict(r) for r in self._q(sql + " ORDER BY id", args)]

    def note_attempt(self, pid: str, failed: bool) -> None:
        if failed:
            self._x("UPDATE portals SET last_attempt_at=?, failed_streak=failed_streak+1 WHERE id=?",
                    (timeutil.iso(timeutil.now()), pid))
        else:
            self._x("UPDATE portals SET last_attempt_at=?, failed_streak=0 WHERE id=?",
                    (timeutil.iso(timeutil.now()), pid))

    def delete_portal(self, pid: str) -> None:
        with self._lock:
            for table, col in (("tenders", "portal_id"), ("runs", "portal_id"), ("specs", "portal_id"),
                               ("known_closed", "portal_id"), ("read_state", "portal_id"), ("state_changes", "portal_id"),
                               ("audits", "portal_id"),
                               ("portals", "id")):
                self._db.execute(f"DELETE FROM {table} WHERE {col}=?", (pid,))
            self._db.commit()

    # ------------------------------------------------------------------ known-closed notices
    # Listed notices whose tender page showed they are not open (deadline passed, or no deadline in a listing that
    # also carries closed notices). Not served; remembered so later reads skip their tender page until the list row
    # changes (a new notice, a corrected deadline). Written only after a read was accepted.
    def known_closed(self, pid: str) -> dict[str, dict]:
        rows = self._q("SELECT tender_id, list_sig, deadline, why FROM known_closed WHERE portal_id=?", (pid,))
        return {r["tender_id"]: {"sig": r["list_sig"], "deadline": r["deadline"], "why": r["why"]} for r in rows}

    def remember_closed(self, pid: str, closed: dict[str, dict], forget: Iterable[str] = ()) -> None:
        if not closed and not forget:
            return
        at = timeutil.iso(timeutil.now())
        with self._lock:
            for tid in forget:
                self._db.execute("DELETE FROM known_closed WHERE portal_id=? AND tender_id=?", (pid, str(tid)))
            for tid, c in closed.items():
                self._db.execute(
                    "INSERT OR REPLACE INTO known_closed (portal_id, tender_id, list_sig, deadline, why, recorded_at)"
                    " VALUES (?,?,?,?,?,?)", (pid, str(tid), c["sig"], c.get("deadline"), c.get("why") or "", at))
            self._db.commit()

    def tender_count(self, pid: str) -> int:
        return self._q("SELECT COUNT(*) AS n FROM tenders WHERE portal_id=?", (pid,))[0]["n"]

    # ------------------------------------------------------------------ runs
    def create_run(self, pid: str, kind: str, run_id: str | None = None) -> str:
        rid = run_id or new_run_id()
        self._x("INSERT INTO runs (id, portal_id, kind, started_at) VALUES (?,?,?,?)",
                (rid, pid, kind, timeutil.iso(timeutil.now())))
        return rid

    def finish_run(self, rid: str, result: str, summary: str = "", detail: dict | None = None,
                   requests: int = 0) -> None:
        usd = self._q("SELECT COALESCE(SUM(usd),0) AS s FROM llm_calls WHERE run_id=?", (rid,))[0]["s"]
        self._x("UPDATE runs SET finished_at=?, result=?, summary=?, detail=?, requests=?, llm_usd=? WHERE id=?",
                (timeutil.iso(timeutil.now()), result, summary, json.dumps(detail or {}), requests, usd, rid))

    def progress_run(self, rid: str, requests: int, note: str) -> None:
        """Live progress for a running run (finished runs are never touched)."""
        self._x("UPDATE runs SET requests=?, summary=? WHERE id=? AND finished_at IS NULL", (requests, note, rid))

    def list_runs(self, pid: str) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM runs WHERE portal_id=? ORDER BY started_at, rowid", (pid,))]

    def latest_run(self, pid: str, result: str | None = None) -> dict | None:
        """The newest run (optionally with this result), same order as list_runs, without loading every run."""
        sql = "SELECT * FROM runs WHERE portal_id=?" + (" AND result=?" if result else "")
        rows = self._q(sql + " ORDER BY started_at DESC, rowid DESC LIMIT 1", (pid, result) if result else (pid,))
        return dict(rows[0]) if rows else None

    def tender_fingerprint(self, pid: str) -> tuple:
        """Cheap change marker for a portal's served tenders: every write happens in a run that ends 'ok'."""
        r = self._q("SELECT COUNT(*) AS n, MAX(last_changed_at) AS m FROM tenders WHERE portal_id=?", (pid,))[0]
        ok = self._q("SELECT id FROM runs WHERE portal_id=? AND result='ok' ORDER BY started_at DESC, rowid DESC "
                     "LIMIT 1", (pid,))
        return (r["n"], r["m"], ok[0]["id"] if ok else None)

    def get_run(self, rid: str) -> dict | None:
        rows = self._q("SELECT * FROM runs WHERE id=?", (rid,))
        return dict(rows[0]) if rows else None

    # ------------------------------------------------------------------ specs
    def current_spec(self, pid: str) -> tuple[int, dict] | None:
        rows = self._q("SELECT version, spec FROM specs WHERE portal_id=? ORDER BY version DESC LIMIT 1", (pid,))
        return (rows[0]["version"], json.loads(rows[0]["spec"])) if rows else None

    def list_specs(self, pid: str | None = None) -> list[dict]:
        """Every promoted spec version (oldest first): portal_id, version, created_at, source_run, spec (parsed)."""
        sql, args = "SELECT * FROM specs", ()
        if pid:
            sql, args = sql + " WHERE portal_id=?", (pid,)
        out = []
        for r in self._q(sql + " ORDER BY portal_id, version", args):
            d = dict(r)
            d["spec"] = json.loads(d["spec"])
            out.append(d)
        return out

    def promote_spec(self, pid: str, spec: dict, source_run: str | None) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(MAX(version),0) AS v FROM specs WHERE portal_id=?",
                                   (pid,)).fetchone()
            version = row["v"] + 1
            self._db.execute("INSERT INTO specs (portal_id, version, spec, created_at, source_run) VALUES (?,?,?,?,?)",
                             (pid, version, json.dumps(spec), timeutil.iso(timeutil.now()), source_run))
            self._db.commit()
            return version

    # ------------------------------------------------------------------ tenders
    def get_tenders(self, pid: str) -> dict[str, dict]:
        rows = self._q("SELECT * FROM tenders WHERE portal_id=?", (pid,))
        return {r["tender_id"]: {"data": json.loads(r["data"]), "content_hash": r["content_hash"],
                                 "last_changed_at": r["last_changed_at"], "first_seen_at": r["first_seen_at"],
                                 "last_verified_at": r["last_verified_at"]} for r in rows}

    def commit_snapshot(self, pid: str, tenders: list[dict], complete: bool,
                        verified_ids: Iterable[str] = ()) -> dict:
        """Merge a VALIDATED read into the served snapshot.

        - New or changed tenders get lastChangedAt = now.
        - Tenders missing from the read are closed only if the read was complete (pagination finished).
        - Status is derived (withdrawn > closed-by-deadline > open).
        Returns counts for the run summary.
        """
        at = timeutil.now()
        at_iso = timeutil.iso(at)
        verified = set(verified_ids)
        existing = self.get_tenders(pid)
        seen: set[str] = set()
        counts = {"new": 0, "changed": 0, "closed_missing": 0, "unchanged": 0}
        with self._lock:
            for t in tenders:
                tid = str(t["id"])
                seen.add(tid)
                t = dict(t)
                t["status"] = derive_status(t, at)
                h = content_hash(t)
                prev = existing.get(tid)
                if prev is None:
                    counts["new"] += 1
                    t["lastChangedAt"] = at_iso
                    self._db.execute(
                        "INSERT INTO tenders (portal_id, tender_id, data, content_hash, last_changed_at, first_seen_at,"
                        " last_verified_at) VALUES (?,?,?,?,?,?,?)",
                        (pid, tid, json.dumps(t, ensure_ascii=False), h, at_iso, at_iso,
                         at_iso if tid in verified else None))
                    continue
                changed = h != prev["content_hash"] or t["status"] != prev["data"].get("status")
                t["lastChangedAt"] = at_iso if changed else prev["last_changed_at"]
                counts["changed" if changed else "unchanged"] += 1
                self._db.execute(
                    "UPDATE tenders SET data=?, content_hash=?, last_changed_at=?,"
                    " last_verified_at=COALESCE(?, last_verified_at) WHERE portal_id=? AND tender_id=?",
                    (json.dumps(t, ensure_ascii=False), h, t["lastChangedAt"],
                     at_iso if tid in verified else None, pid, tid))
            if complete:
                # A tender closes when it is missing from CLOSE_AFTER_MISSED_READS complete reads in a row (or when
                # its deadline passes): one odd read (rows pushed out by injected rows, a glitch) never closes real
                # open tenders. The first miss is only remembered.
                for tid, prev in existing.items():
                    if tid in seen or prev["data"].get("status") in ("closed", "withdrawn"):
                        continue
                    d = dict(prev["data"])
                    missed = int(d.get("_missedReads") or 0) + 1
                    if missed < CLOSE_AFTER_MISSED_READS:
                        d["_missedReads"] = missed
                        counts["missing_once"] = counts.get("missing_once", 0) + 1
                        self._db.execute("UPDATE tenders SET data=? WHERE portal_id=? AND tender_id=?",
                                         (json.dumps(d, ensure_ascii=False), pid, tid))
                        continue
                    d.pop("_missedReads", None)
                    d["status"] = "closed"
                    d["lastChangedAt"] = at_iso
                    counts["closed_missing"] += 1
                    self._db.execute(
                        "UPDATE tenders SET data=?, last_changed_at=? WHERE portal_id=? AND tender_id=?",
                        (json.dumps(d, ensure_ascii=False), at_iso, pid, tid))
            self._db.execute("UPDATE portals SET last_success_at=? WHERE id=?", (at_iso, pid))
            self._db.commit()
        return counts

    def has_documents(self, pid: str) -> bool:
        """Whether any served tender of the portal has at least one downloaded document (stops at the first)."""
        return bool(self._q("SELECT 1 FROM tenders WHERE portal_id=? AND instr(data, '\"sha256\"') > 0 LIMIT 1",
                            (pid,)))

    def served_tenders(self, pid: str, changed_since: datetime | None = None) -> list[dict]:
        """Tenders as served. Deadlines that passed since the last read flip to closed on the fly."""
        at = timeutil.now()
        out = []
        for row in self._q("SELECT data, last_changed_at FROM tenders WHERE portal_id=? ORDER BY first_seen_at,"
                           " tender_id", (pid,)):
            t = json.loads(row["data"])
            last_changed = timeutil.parse(row["last_changed_at"])
            if t.get("status") == "open" and derive_status(t, at) == "closed":
                t["status"] = "closed"
                deadline = timeutil.parse(t["submissionDeadline"])
                last_changed = max(last_changed, deadline)
            t["lastChangedAt"] = timeutil.iso(last_changed)
            t = {k: v for k, v in t.items() if not k.startswith("_")}
            if changed_since is not None and not (last_changed > changed_since):
                continue
            out.append(t)
        return out

    # ------------------------------------------------------------------ read state (change-feed position)
    def get_read_state(self, pid: str) -> dict | None:
        rows = self._q("SELECT state FROM read_state WHERE portal_id=?", (pid,))
        return json.loads(rows[0]["state"]) if rows else None

    def set_read_state(self, pid: str, state: dict | None) -> None:
        """Written only together with an accepted read (worker._accept), so a rejected read never moves it."""
        if state is None:
            self._x("DELETE FROM read_state WHERE portal_id=?", (pid,))
            return
        self._x("INSERT INTO read_state (portal_id, state, updated_at) VALUES (?,?,?) ON CONFLICT(portal_id)"
                " DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at",
                (pid, json.dumps(state), timeutil.iso(timeutil.now())))

    # ------------------------------------------------------------------ audits (continuous verification)
    # One row per audit: what was sampled, field-by-field counts, every disagreement with both values. Reports only:
    # an audit never changes tenders, specs or the portal state.
    def create_audit(self, pid: str, spec_version: int | None) -> str:
        aid = "a_" + uuid.uuid4().hex[:12]
        self._x("INSERT INTO audits (id, portal_id, started_at, spec_version) VALUES (?,?,?,?)",
                (aid, pid, timeutil.iso(timeutil.now()), spec_version))
        return aid

    def finish_audit(self, aid: str, status: str, result: dict) -> None:
        usd = self._q("SELECT COALESCE(SUM(usd),0) AS s, COUNT(*) AS n FROM llm_calls WHERE run_id=?", (aid,))[0]
        result = {**result, "costUsd": round(usd["s"], 5), "modelCalls": usd["n"]}
        self._x("UPDATE audits SET finished_at=?, status=?, result=? WHERE id=?",
                (timeutil.iso(timeutil.now()), status, json.dumps(result, ensure_ascii=False), aid))

    def list_audits(self, pid: str, limit: int = 20, finished: bool = True) -> list[dict]:
        """Newest first."""
        rows = self._q("SELECT * FROM audits WHERE portal_id=?" + (" AND finished_at IS NOT NULL" if finished else "")
                       + " ORDER BY started_at DESC, rowid DESC LIMIT ?", (pid, limit))
        return [{**dict(r), "result": json.loads(r["result"] or "{}")} for r in rows]

    def latest_audit(self, pid: str) -> dict | None:
        rows = self.list_audits(pid, 1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ files
    def file_path(self, sha256: str) -> str:
        return os.path.join(self.files_dir, sha256)

    def register_file(self, sha256: str, size: int) -> None:
        self._x("INSERT OR IGNORE INTO files (sha256, size_bytes, created_at) VALUES (?,?,?)",
                (sha256, size, timeutil.iso(timeutil.now())))

    def has_file(self, sha256: str) -> bool:
        return bool(self._q("SELECT 1 FROM files WHERE sha256=?", (sha256,))) and os.path.exists(self.file_path(sha256))

    # ------------------------------------------------------------------ LLM ledger
    def log_llm_call(self, portal_id: str | None, run_id: str | None, purpose: str, model: str,
                     tokens_in: int, tokens_out: int, usd: float) -> None:
        self._x("INSERT INTO llm_calls (portal_id, run_id, purpose, model, tokens_in, tokens_out, usd, at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (portal_id, run_id, purpose, model, tokens_in, tokens_out, usd, timeutil.iso(timeutil.now())))

    def llm_by_run(self, pid: str | None = None) -> dict[str, dict]:
        """Model calls per run: {run_id: {"calls", "usd", "purposes": {purpose: n}}}."""
        sql, args = ("SELECT run_id, purpose, COUNT(*) AS n, COALESCE(SUM(usd),0) AS usd FROM llm_calls"
                     " WHERE run_id IS NOT NULL"), ()
        if pid:
            sql, args = sql + " AND portal_id=?", (pid,)
        out: dict[str, dict] = {}
        for r in self._q(sql + " GROUP BY run_id, purpose", args):
            e = out.setdefault(r["run_id"], {"calls": 0, "usd": 0.0, "purposes": {}})
            e["calls"] += r["n"]
            e["usd"] += r["usd"]
            e["purposes"][r["purpose"] or "?"] = r["n"]
        return out

    def llm_totals(self, pid: str) -> dict:
        r = self._q("SELECT COUNT(*) AS calls, COALESCE(SUM(tokens_in),0) AS tin, COALESCE(SUM(tokens_out),0) AS tout,"
                    " COALESCE(SUM(usd),0) AS usd, SUM(CASE WHEN usd IS NULL THEN 1 ELSE 0 END) AS unpriced"
                    " FROM llm_calls WHERE portal_id=?", (pid,))[0]
        return {"calls": r["calls"], "tokensIn": r["tin"], "tokensOut": r["tout"], "usd": round(r["usd"], 4),
                "unpricedCalls": r["unpriced"] or 0}
