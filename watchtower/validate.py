"""Notice: checks run on every staged read before it may be committed. No model calls.

Field profiles (architecture v1.1 §2) catch same-shape drift: a column swap or a reformatted date keeps
counts and structure intact but breaks the statistical shape of the values (kind, length, uniqueness).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from . import timeutil
from .pipeline import Verdict
from .store import derive_status

PROFILED = ["id", "url", "title", "buyerName", "publicationDate", "submissionDeadline", "referenceNumber"]
IMMUTABLE = ["title", "buyerName", "publicationDate"]


def _kind(v) -> str:
    if v is None:
        return "none"
    s = str(v)
    if re.match(r"^\d{4}-\d{2}-\d{2}T", s):
        return "datetime"
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return "date"
    if re.match(r"^https?://", s):
        return "url"
    if re.search(r"\d", s) and len(s) <= 40 and not re.search(r"\s{1}\w+\s\w+\s\w+", s):
        return "code"
    return "text"


def build_profile(tenders: list[dict]) -> dict:
    prof = {}
    n = len(tenders) or 1
    for f in PROFILED:
        vals = [t.get(f) for t in tenders]
        present = [v for v in vals if v not in (None, "")]
        kinds: dict[str, int] = {}
        for v in present:
            k = _kind(v)
            kinds[k] = kinds.get(k, 0) + 1
        dominant = max(kinds, key=kinds.get) if kinds else "none"
        lengths = [len(str(v)) for v in present]
        prof[f] = {
            "present": round(len(present) / n, 3),
            "kind": dominant,
            "kindShare": round(kinds.get(dominant, 0) / len(present), 3) if present else 0,
            "medianLen": statistics.median(lengths) if lengths else 0,
            "unique": round(len(set(map(str, present))) / len(present), 3) if present else 0,
            "n": len(tenders),
        }
    return prof


def check_profile(profile: dict, tenders: list[dict], variety: bool = True) -> list[str]:
    """variety=False: skip the distinct-values comparison (a change feed's delta is a small, biased sample: how
    many distinct buyers 30 changes have says nothing about the spec)."""
    if not profile or not tenders:
        return []
    now = build_profile(tenders)
    failures = []
    for f, old in profile.items():
        new = now.get(f)
        if not new or old["present"] < 0.5:
            continue
        if new["present"] < old["present"] - 0.3:
            failures.append(f"{f}: found in {new['present']:.0%} of tenders (was {old['present']:.0%})")
            continue
        if new["present"] == 0:
            continue
        if new["kind"] != old["kind"] and old["kindShare"] >= 0.8:
            failures.append(f"{f}: values look like '{new['kind']}', expected '{old['kind']}'")
            continue
        if old["kind"] == "text" and old["medianLen"] and new["medianLen"]:
            ratio = new["medianLen"] / old["medianLen"]
            if ratio > 2.5 or ratio < 0.4:
                failures.append(f"{f}: typical length changed {old['medianLen']:.0f}→{new['medianLen']:.0f} chars")
                continue
        if variety and old["n"] >= 4 and new["n"] >= 4 and abs(new["unique"] - old["unique"]) > 0.45:
            failures.append(f"{f}: value variety changed ({old['unique']:.0%}→{new['unique']:.0%} distinct)")
    return failures


def validate(spec: dict, result, previous: dict[str, dict], mode: str = "read") -> Verdict:
    """Hard checks. `previous` = committed tenders (store.get_tenders)."""
    failures: list[str] = []
    warnings: list[str] = []
    tenders = result.tenders
    stats = result.stats or {}
    now = timeutil.now()
    # Effective status at read time: a stored "open" whose deadline has passed is no longer open.
    prev_open = {tid: p["data"] for tid, p in previous.items() if derive_status(p["data"], now) == "open"}
    # A change feed read after the first one is a delta: it holds what changed, not the whole listing. Absent tenders
    # are not gone, and no changes at all is a normal answer.
    feed_delta = bool(stats.get("changeFeed")) and bool(previous)

    held = int(stats.get("heldBack") or 0)
    if not tenders and held and (int(stats.get("noDeadline") or 0) or int(stats.get("closedDropped") or 0)
                                 or int(stats.get("notOpenNotices") or 0)):
        # Every tender page read so far was a closed notice; the rest wait for later refreshes (said in the reason).
        warnings.append(f"no open tender confirmed yet; {held} listed notices wait for their tender page")
    elif not tenders:
        if feed_delta and mode != "repair":
            warnings.append("no changes in the feed since the last read")
        elif mode == "attach" and not previous:
            failures.append("no tenders found")
        elif mode == "repair":
            failures.append("the repaired read found no tenders")
        elif prev_open:
            failures.append(f"no tenders found (previously {len(prev_open)} open and not yet due)")
        elif previous:
            # Every known tender is closed or past its deadline: an empty listing is plausible, but only if the
            # list page itself still looks like a list page (a missing selector also yields zero rows).
            total = stats.get("statedTotal")
            dropped = int(stats.get("closedDropped") or 0)
            if total and dropped and dropped >= int(stats.get("rows") or 0) > 0:
                warnings.append(f"the portal lists {total} but all {dropped} rows read are past their deadline")
            elif total:
                failures.append(f"no tenders read but the portal states {total}")
            elif stats.get("selectorMissing") or any("selector" in str(e).lower() for e in stats.get("errors", [])):
                failures.append("the list selector matched nothing (page structure changed)")
            elif (spec.get("list") or {}).get("total") and total is None:
                failures.append("no tenders read and the stated-total element is gone (page structure changed)")
            elif not result.complete and not str(stats.get("stopReason") or "").startswith(
                    "no items matched on the first page"):
                failures.append("no tenders read and the read did not finish")
            else:
                warnings.append(f"listing is empty; all {len(previous)} known tenders are past their deadline")
    ids = [t.get("id") for t in tenders]
    for f in ("id", "url", "title"):
        missing = sum(1 for t in tenders if not t.get(f))
        if missing:
            failures.append(f"{missing} tender(s) without {f}")
    if len(set(ids)) != len(ids):
        failures.append("duplicate tender ids")

    # implausible dates (closed rows were already dropped by the runner: what is left should be open or undated).
    # Compared as parsed dates, never as strings; one count per tender.
    bad_dates = 0
    bad_examples: list[str] = []
    tomorrow = (now + timedelta(days=1)).date()
    for t in tenders:
        pub, dl = t.get("publicationDate"), t.get("submissionDeadline")
        why = None
        try:
            pub_d = date.fromisoformat(str(pub)[:10]) if pub else None
            dl_dt = timeutil.parse(dl) if dl else None
            if dl_dt is not None and not (2000 <= dl_dt.year <= 2100):
                why = f"deadline year {dl_dt.year}"
            elif pub_d is not None and pub_d > tomorrow:
                why = f"published in the future ({pub_d})"
            elif pub_d is not None and dl_dt is not None and dl_dt.date() < pub_d:
                why = f"deadline {dl_dt.date()} before publication {pub_d}"
        except (ValueError, TypeError):
            why = f"unparseable date ({pub!r}, {dl!r})"
        if why:
            bad_dates += 1
            if len(bad_examples) < 3:
                bad_examples.append(f"{t.get('id')}: {why}")
    if tenders and bad_dates / len(tenders) > 0.2:
        failures.append(f"{bad_dates} tender(s) with implausible dates (e.g. {'; '.join(bad_examples)})")

    # completeness vs stated total (hard only if it was reliable at attach). The stated total counts every row the
    # portal lists, including closed rows the runner skipped, so compare it with the rows listed.
    total = stats.get("statedTotal")
    listed = int(stats.get("rows") or len(tenders))
    if total is not None and spec.get("totalReliable") and result.complete and listed < total * 0.95:
        failures.append(f"read {listed} tenders but the portal states {total}")
    elif total is not None and mode in ("attach", "repair") and result.complete and listed < total * 0.95:
        # A read that believes it is complete but sees a fraction of the portal's own count is a partial delivery
        # in disguise (typically: pagination not followed). Never let that become "serving".
        failures.append(f"read {listed} tenders as if complete but the portal states {total}: the list has more "
                        f"pages that the spec does not follow (stop reason: {stats.get('stopReason') or '?'})")
    elif total is not None and listed < total * 0.95:
        warnings.append(f"read {listed} of a stated {total}")

    # disappearance of open, not-yet-due tenders (effective status; open tenders without a deadline count too)
    ids_set = set(ids)
    if len(prev_open) >= 4 and tenders and not feed_delta:
        gone = [tid for tid in prev_open if tid not in ids_set]
        if len(gone) / len(prev_open) > 0.5:
            failures.append(f"{len(gone)} of {len(prev_open)} open tenders disappeared at once")

    # same tender, fields that should not change
    overlap = [t for t in tenders if t.get("id") in previous]
    if stats.get("changeFeed"):
        # A change feed lists a known tender again because it changed (older copies are dropped by the feed
        # reader), so a new title or publication date there is an amendment. Only the same entry version coming
        # back with different values says the reader broke.
        overlap = [t for t in overlap if t.get("_feedUpdated")
                   and t.get("_feedUpdated") == (previous[t["id"]].get("data") or {}).get("_feedUpdated")]
    drift, swapped = _drift(overlap, previous)
    n_ov = len(overlap)
    if n_ov >= 4:
        if drift / n_ov > 0.3:
            failures.append(f"{drift} of {n_ov} known tenders changed title/buyer/publication date at once")
    elif n_ov >= 2:
        # few tenders: a lower bar (a single amended title is normal, most of a handful changing is not)
        if drift / n_ov > 0.5:
            failures.append(f"{drift} of {n_ov} known tenders changed title/buyer/publication date at once")
    if swapped and not any("title/buyer" in f for f in failures):
        failures.append(f"title and buyer look swapped on {swapped} known tender(s)")

    # field profiles (same-shape drift)
    if not (feed_delta and len(tenders) < 20):          # a handful of changes is too few to judge a profile
        failures.extend(check_profile(spec.get("profile", {}), tenders, variety=not feed_delta))

    # document health
    errs = [e for e in stats.get("errors", []) if e.startswith("document")]
    if errs and stats.get("docsFetched", 0) == 0:
        failures.append("documents could not be downloaded")
    if stats.get("errors"):
        warnings.extend(stats["errors"][:5])

    # repair needs evidence (architecture v1.1 §5)
    if mode == "repair" and previous and not failures:
        agree = sum(1 for t in overlap for f in IMMUTABLE
                    if previous[t["id"]]["data"].get(f) is None or t.get(f) == previous[t["id"]]["data"].get(f))
        share = agree / (n_ov * len(IMMUTABLE)) if n_ov else 1.0
        if n_ov >= 5:
            if share < 0.8:
                failures.append(f"repaired read agrees with the last good data on only {share:.0%} of fields")
        else:
            blind = bool(spec.get("verifiedBlind"))
            if n_ov and share < 0.8:
                msg = f"repaired read agrees with the last good data on only {share:.0%} of fields ({n_ov} shared)"
                (warnings if blind else failures).append(msg)
            if not blind:
                warnings.append(f"only {n_ov} tender(s) shared with the last good data and no blind verification "
                                f"against the source is recorded")
    return Verdict(ok=not failures, failures=failures, warnings=warnings)


def _drift(overlap: list[dict], previous: dict[str, dict]) -> tuple[int, int]:
    """(tenders whose immutable fields changed, tenders whose title and buyer traded places)."""
    drift = swapped = 0
    for t in overlap:
        old = previous[t["id"]]["data"]
        if any(t.get(f) != old.get(f) for f in IMMUTABLE if old.get(f) is not None):
            drift += 1
        if old.get("title") and old.get("buyerName") and old["title"] != old["buyerName"] \
                and t.get("title") == old["buyerName"] and t.get("buyerName") == old["title"]:
            swapped += 1
    return drift, swapped


# ---------------------------------------------------------------------------------------------- acceptance gate
@dataclass
class Decision:
    """Outcome of the single acceptance gate (attach, refresh and repair all go through `decide`)."""
    commit: bool
    state: str                      # serving | failed | healing (what the portal should report)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    close_missing: bool = False     # a tender missing from the read may be closed (complete, uncapped read)
    notes: list[str] = field(default_factory=list)   # honest caveats that go into the reason when serving

    @property
    def failure_text(self) -> str:
        return "; ".join(self.failures) or "validation failed"

    def reason(self, n_tenders: int, n_open: int | None = None) -> str:
        head = f"Delivering {n_tenders} tenders" + (f" ({n_open} open)" if n_open is not None else "") + "."
        return " ".join([head] + self.notes)


def read_flags(result) -> dict:
    """The runner->worker contract, read defensively (a missing field means 'nothing to report')."""
    st = getattr(result, "stats", None) or {}
    return {
        "complete": bool(getattr(result, "complete", False)),
        "capped": bool(st.get("capped")),
        "stopReason": st.get("stopReason") or "",
        "pendingDetails": int(st.get("pendingDetails") or 0),
        "pendingDocs": int(st.get("pendingDocs") or 0),
        "docBlocked": int(st.get("docBlocked") or 0),
        "docBlockReason": st.get("docBlockReason"),
        "docsFetched": int(st.get("docsFetched") or 0),
        "droppedIncomplete": int(st.get("droppedIncomplete") or 0),
        "closedDropped": int(st.get("closedDropped") or 0),
        "heldBack": int(st.get("heldBack") or 0),
        "noDeadline": int(st.get("noDeadline") or 0),
        "notOpenNotices": int(st.get("notOpenNotices") or 0),
        "shortfall": int(st.get("shortfall") or 0),
        "rows": int(st.get("rows") or 0),
        "statedTotal": st.get("statedTotalLast", st.get("statedTotal")),
        "errors": list(st.get("errors") or []),
        "changeFeed": bool(st.get("changeFeed")),
    }


def decide(verdict: Verdict, result, *, mode: str, first: bool, serve_capped: bool = False,
           spec: dict | None = None) -> Decision:
    """Commit or reject a staged read.

    mode:  attach | read | repair.   first: no snapshot has ever been published for this portal.
    - a failed verdict rejects;
    - an unfinished read whose pagination was NOT capped is breakage: rejected (never closes anything, never green
      on first attach); a capped read (page cap / portal refuses deeper pages) may be published, closes nothing;
    - pending details/documents and blocked documents do not block publishing, but the reason names them;
    - so does a spec whose blind verification could not compare anything (spec["verification"]).
    """
    f = read_flags(result)
    failures = list(verdict.failures) if not verdict.ok else []
    if not verdict.ok and not failures:
        failures = ["validation failed"]
    legit_empty = (not failures and not first and not result.tenders
                   and (f["stopReason"] or "").startswith("no items matched on the first page"))
    # validate() already confirmed every known tender is past its deadline and no total is stated: an empty
    # listing is then the portal's real state, not a broken selector.
    if not failures and not f["complete"] and not f["capped"] and not legit_empty:
        why = f["stopReason"] or "pagination did not finish"
        failures.append(f"read incomplete ({why}) and not a known cap; nothing was published or closed")
    if not failures and f["capped"] and not serve_capped:
        failures.append(f"the portal shows only part of its listing ({f['stopReason'] or 'capped'}) and it could not "
                        f"be narrowed to open tenders; delivery would be partial "
                        f"(set WATCHTOWER_SERVE_CAPPED=1 to serve a capped listing)")
    if failures:
        state = "failed" if first and mode != "repair" else "healing"
        return Decision(False, state, failures, list(verdict.warnings))
    # Integration health (the state) and data coverage are separate (Patterno, 20 Nov): a read that works but sees
    # only part of the listing serves, and says so in a fixed, recognisable form ("Coverage: ...", "Documents: ...").
    notes = []
    if f["changeFeed"]:
        from .feed import coverage_note
        notes.append(coverage_note(getattr(result, "stats", None) or {}))
    elif not f["complete"] or f["capped"]:
        notes.append(f"Coverage: partial - the portal shows more than we can read ({f['stopReason'] or 'capped'}); "
                     f"tenders beyond it may be missing; unseen tenders kept, not closed.")
    elif f["heldBack"]:
        # The list was read to its end, but the deadline is only on each tender page and the listing also carries
        # closed notices: a notice whose page is not read yet is not served (it might be closed). Not complete.
        notes.append(f"Coverage: partial - {f['heldBack']} listed notices are not served yet: their deadline is only "
                     f"on their tender page, which is read on later refreshes (this listing also carries closed "
                     f"notices or most tenders show a deadline, so none is served as open unseen).")
    elif f["shortfall"]:
        notes.append(f"Coverage: partial - the listing changed while it was read ({f['rows']} of a stated "
                     f"{f['statedTotal']} seen); tenders missing from this read are kept, not closed.")
    elif f["droppedIncomplete"]:
        notes.append(f"Coverage: partial - {f['droppedIncomplete']} listed tenders could not be read completely "
                     f"(no id, link or title) and are not delivered.")
    else:
        notes.append("Coverage: complete.")
    if f["pendingDetails"]:
        notes.append(f"Details for {f['pendingDetails']} tenders still pending; they fill in on later refreshes.")
    if f["pendingDocs"]:
        notes.append(f"{f['pendingDocs']} documents found but not yet downloaded.")
    if f["docBlocked"]:
        why = f" ({f['docBlockReason']})" if f["docBlockReason"] else ""
        if not f["docsFetched"]:
            notes.append(f"Documents: not available from this portal{why}; tenders are delivered without documents.")
        else:
            notes.append(f"Documents: partial - {f['docBlocked']} documents are blocked{why} and not delivered.")
    if f["closedDropped"] and f["changeFeed"]:
        notes.append(f"{f['closedDropped']} tenders in the feed were already closed, withdrawn or past their deadline "
                     f"and are not served.")
    elif f["closedDropped"]:
        notes.append(f"{f['closedDropped']} listed notices were already past their deadline and are not served.")
    if (spec or {}).get("verification"):
        notes.append(f"Verification: {str(spec['verification']).rstrip('.')}.")
    if f["notOpenNotices"]:
        notes.append(f"{f['notOpenNotices']} listed notices are award, result or prior-information notices (by their "
                     f"notice type or status) and are not served as open.")
    if f["noDeadline"]:
        notes.append(f"{f['noDeadline']} listed notices show no submission deadline on their tender page (awards, "
                     f"cancellations, other notices) and are not served as open.")
    # A change feed is never a full listing: a tender closes on its status, a tombstone or its deadline, never
    # because a read did not mention it.
    return Decision(True, "serving", [], list(verdict.warnings),
                    close_missing=f["complete"] and not f["capped"] and not f["shortfall"]
                    and not f["droppedIncomplete"] and not f["changeFeed"], notes=notes)
