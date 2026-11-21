"""Job manager + state machine.

States:  integrating -> serving <-> healing -> failed  (failed retries on refresh / backoff)

Invariants (from the brief and architecture v1.1):
- At most one active job (read or repair) per portal; refresh while busy returns that job's runId.
- Only a validated read changes served tenders (Store.commit_snapshot). While healing, the last
  correct snapshot keeps being served.
- A partial read (pagination capped) never turns a first attach green and never closes tenders.
- Blocks (login/captcha) -> failed with the reason in words.
- DELETE cancels the job immediately; no further requests are made for that portal.
"""
from __future__ import annotations

import asyncio
import logging
import re
import traceback

import httpx

from . import timeutil
from .http import BudgetExceeded, PoliteClient
from .llm import LLMUnavailable
from .pipeline import AttachFailed, JobContext, PortalBlocked
from .validate import Decision, decide
from .urls import normalize_url, portal_id_for

log = logging.getLogger("watchtower.worker")

FAILED_BACKOFF = [600, 1800, 7200]  # seconds between automatic retries of a failed portal
RELEARN_EXTRA_CALLS = 16             # one bounded top-up (~$0.30 worst case) for the single re-learn at attach


class Manager:
    def __init__(self, settings, store, pipeline, llm, limiter, docs, transport=None):
        self.transport = transport  # tests: route HTTP to an in-process mock portal
        self.s = settings
        self.store = store
        self.pipeline = pipeline
        self.llm = llm
        self.limiter = limiter
        self.docs = docs
        self.active: dict[str, tuple[str, asyncio.Task]] = {}
        self._deleted: set[str] = set()
        self._scheduler: asyncio.Task | None = None
        self.audits: dict[str, tuple[str, asyncio.Task]] = {}   # running audit per portal (reports only)
        self._audit_loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        for p in self.store.list_portals():
            if p["state"] in ("integrating", "healing"):
                self._start_job(p["id"])  # resume work interrupted by a restart
        if self.s.refresh_interval > 0:
            self._scheduler = asyncio.create_task(self._schedule_loop())
        if getattr(self.s, "audit_interval", 0) > 0:
            self._audit_loop_task = asyncio.create_task(self._audit_loop())

    async def stop(self) -> None:
        for t in (self._scheduler, self._audit_loop_task):
            if t:
                t.cancel()
        for _, task in list(self.active.values()) + list(self.audits.values()):
            task.cancel()

    # ------------------------------------------------------------------ public API
    def add(self, url: str) -> tuple[str, bool]:
        norm = normalize_url(url)
        pid = portal_id_for(url)
        created = self.store.create_portal(pid, url.strip(), norm)
        if created:
            self._deleted.discard(pid)
            self._start_job(pid)
        return pid, created

    def refresh(self, pid: str) -> str:
        busy = self.active.get(pid)
        if busy and not busy[1].done():
            return busy[0]
        return self._start_job(pid)

    def delete(self, pid: str) -> None:
        self._deleted.add(pid)
        for jobs in (self.active, self.audits):
            busy = jobs.pop(pid, None)
            if busy:
                busy[1].cancel()
        self.store.delete_portal(pid)

    # ------------------------------------------------------------------ audits (continuous verification)
    def audit(self, pid: str) -> str:
        """Start an audit of what the portal serves (or return the running one's id). Reports only."""
        busy = self.audits.get(pid)
        if busy and not busy[1].done():
            return busy[0]
        cur = self.store.current_spec(pid)
        aid = self.store.create_audit(pid, cur[0] if cur else None)
        task = asyncio.create_task(self._audit_job(pid, aid))
        self.audits[pid] = (aid, task)

        def done(t, pid=pid):
            if self.audits.get(pid, (None, None))[1] is t:
                self.audits.pop(pid, None)
        task.add_done_callback(done)
        return aid

    async def _audit_job(self, pid: str, aid: str) -> None:
        from . import audit as audit_mod
        portal = self.store.get_portal(pid)
        cur = self.store.current_spec(pid)
        ctx = self._ctx(pid, portal["url"] if portal else "", aid, self.s.audit_request_budget)
        ctx.mode = "audit"
        result: dict = {}
        try:
            tenders = [{**e["data"], "_firstSeenAt": e["first_seen_at"]}
                       for e in self.store.get_tenders(pid).values()]
            if not cur or not tenders:
                result = {"sampled": 0, "status": "not possible: nothing is served yet"}
            else:
                result = await audit_mod.audit_portal(ctx, cur[1], tenders, k=self.s.audit_sample,
                                                      renders=self.s.audit_renders,
                                                      degraded_below=self.s.audit_degraded_below)
        except asyncio.CancelledError:
            if self._alive(pid):
                self.store.finish_audit(aid, "not possible: cancelled", {"log": ctx.log})
            raise
        except Exception as e:
            result = {"status": f"not possible: audit error {type(e).__name__}: {str(e)[:200]}",
                      "trace": traceback.format_exc()[-1500:]}
        finally:
            await ctx.client.aclose()
        if self._alive(pid):
            result["log"] = ctx.log
            self.store.finish_audit(aid, result.get("status") or "not possible: no result", result)

    def _audit_due(self, p: dict, now) -> bool:
        if p["state"] not in ("serving", "healing") or not p["last_success_at"] or p["id"] in self.audits:
            return False
        last = self.store.latest_audit(p["id"])
        if last is None:
            return True
        cur = self.store.current_spec(p["id"])
        if cur and last.get("spec_version") is not None and cur[0] != last["spec_version"]:
            return True                                  # a repaired / new spec: measure it soon
        age = (now - timeutil.parse(last["finished_at"])).total_seconds()
        return age >= self.s.audit_interval

    async def _audit_loop(self) -> None:
        """One audit at a time across portals (small, steady cost)."""
        await asyncio.sleep(min(120.0, self.s.audit_interval))
        while True:
            try:
                if not any(not t.done() for _, t in self.audits.values()):
                    now = timeutil.now()
                    due = [p for p in self.store.list_portals() if self._audit_due(p, now)]
                    if due:
                        self.audit(due[0]["id"])
                await asyncio.sleep(60.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("audit scheduler tick failed")
                await asyncio.sleep(60.0)

    # ------------------------------------------------------------------ jobs
    def _start_job(self, pid: str) -> str:
        rid = self.store.create_run(pid, "read")
        task = asyncio.create_task(self._read_job(pid, rid))
        self.active[pid] = (rid, task)
        task.add_done_callback(lambda t, pid=pid: self._job_done(pid, t))
        return rid

    def _job_done(self, pid: str, task: asyncio.Task) -> None:
        cur = self.active.get(pid)
        if cur and cur[1] is task:
            self.active.pop(pid, None)
        if not task.cancelled() and task.exception():
            log.error("job crashed for %s: %s", pid, task.exception())

    def _ctx(self, pid: str, url: str, rid: str, budget: int | None) -> JobContext:
        client = PoliteClient(self.limiter, self.s.user_agent, self.s.http_timeout, budget, transport=self.transport,
                              respect_robots=self.s.respect_robots)
        return JobContext(pid, url, rid, client, self.docs, self.llm, self.store)

    def _progress(self, rid: str, ctx: JobContext) -> asyncio.Task:
        async def tick():
            while True:
                await asyncio.sleep(3)
                last = ctx.log[-1] if ctx.log else "working…"
                self.store.progress_run(rid, ctx.client.requests, f"running · {ctx.client.requests} requests · {last}"[:300])
        return asyncio.create_task(tick())

    def _alive(self, pid: str) -> bool:
        return pid not in self._deleted and self.store.get_portal(pid) is not None

    async def _accept(self, pid: str, rid: str, ctx: JobContext, spec: dict, result, *, mode: str,
                      promote: bool, first: bool, attempts: list | None = None, was: list | None = None,
                      label: str = "") -> Decision:
        """THE acceptance gate: attach, refresh and repair all publish through here and nowhere else.

        validate -> decide (validate.decide) -> on commit: promote the spec (attach/repair), merge the snapshot
        (closing missing tenders only after a complete, uncapped read), finish the run, set state and reason.
        On rejection nothing is written: the last correct snapshot (and spec) stay exactly as they were.
        """
        previous = self.store.get_tenders(pid)
        verdict = await self.pipeline.validate(ctx, spec, result, previous)
        decision = decide(verdict, result, mode=mode, first=first, serve_capped=self.s.serve_capped, spec=spec)
        if attempts is not None:
            attempts.append({"attempt": len(attempts) + 1, "ok": decision.commit, "failures": decision.failures})
        if not decision.commit or not self._alive(pid):
            return decision
        version = self.store.promote_spec(pid, spec, rid) if promote else None
        counts = self.store.commit_snapshot(pid, result.tenders, decision.close_missing, result.verified_ids)
        # notices whose tender page showed they are not open: later reads skip that page while the row is unchanged
        self.store.remember_closed(pid, getattr(result, "closed", None) or {},
                                   getattr(result, "forget_closed", None) or [])
        if getattr(result, "state", None) is not None and hasattr(self.store, "set_read_state"):
            self.store.set_read_state(pid, result.state)     # e.g. a change feed's high-water mark: only when accepted
        summary = (f"Repaired (spec v{version}){label}. " if mode == "repair" else "") + _summary(counts, result, decision)
        self.store.finish_run(rid, "ok", summary,
                              {"log": ctx.log, "counts": counts, "stats": result.stats, "warnings": decision.warnings,
                               **({"attempts": attempts, "was": was} if attempts is not None else {})},
                              ctx.client.requests)
        self.store.note_attempt(pid, failed=False)
        n = self.store.tender_count(pid)
        n_open = sum(1 for t in self.store.served_tenders(pid) if t.get("status") == "open")
        reason = decision.reason(n, n_open)
        if mode == "repair":
            reason += f" Repaired at {timeutil.iso(timeutil.now())}."
        self.store.set_state(pid, "serving", reason)
        return decision

    async def _read_job(self, pid: str, rid: str) -> None:
        portal = self.store.get_portal(pid)
        if portal is None:
            return
        state = portal["state"]
        current = self.store.current_spec(pid)
        attaching = current is None
        first = attaching or portal["last_success_at"] is None
        budget = self.s.attach_request_budget if attaching else None
        ctx = self._ctx(pid, portal["url"], rid, budget)
        ctx.mode = "attach" if attaching else "read"
        failures: list[str] = []
        ticker = self._progress(rid, ctx)
        try:
            spec = current[1] if current else await self.pipeline.attach(ctx)
            result = await self.pipeline.read(ctx, spec)
            decision = await self._accept(pid, rid, ctx, spec, result, mode=ctx.mode, promote=attaching, first=first)
            if decision.commit:
                return
            again = await self._relearn_once(ctx, spec, result, decision) if attaching and self._alive(pid) else None
            if again is not None:
                spec, result = again
                decision = await self._accept(pid, rid, ctx, spec, result, mode=ctx.mode, promote=True, first=first)
                if decision.commit:
                    return
            failures = decision.failures
            self.store.finish_run(rid, "failed", "; ".join(failures), {"log": ctx.log, "failures": failures,
                                  "stats": result.stats}, ctx.client.requests)
        except asyncio.CancelledError:
            self._finish_quietly(rid, "cancelled", ctx)
            raise
        except PortalBlocked as e:
            self.store.finish_run(rid, "failed", str(e), {"log": ctx.log}, ctx.client.requests)
            self._fail_terminal(pid, f"Blocked: {e}")
            return
        except AttachFailed as e:
            self.store.finish_run(rid, "failed", str(e), {"log": ctx.log}, ctx.client.requests)
            self._fail_terminal(pid, f"Could not attach: {e}")
            return
        except BudgetExceeded as e:
            failures = [_budget_reason(e, ctx, "attach" if attaching else "read")]
            self.store.finish_run(rid, "failed", failures[0], {"log": ctx.log}, ctx.client.requests)
        except Exception as e:  # transport errors, parser crashes, ... -> treated as a failed read
            failures = [f"read error: {type(e).__name__}: {e}"]
            self.store.finish_run(rid, "failed", failures[0], {"log": ctx.log,
                                  "trace": traceback.format_exc()[-2000:]}, ctx.client.requests)
        finally:
            ticker.cancel()
            await ctx.client.aclose()

        # ---- the read was rejected (or crashed); the last snapshot stays untouched ----
        if first:
            self._fail_terminal(pid, f"Could not attach: {'; '.join(failures)}" if attaching
                                else f"First read rejected: {'; '.join(failures)}")
            return
        if not self._alive(pid):
            return
        # serving (or failed with a known spec) -> heal while the last correct snapshot keeps being served
        await self._repair_job(pid, current[1], failures, was_state=state)

    async def _relearn_once(self, ctx: JobContext, spec: dict, result, decision: Decision):
        """The model learned from one page; the first full read sees every row. When that read shows rows the spec
        reads no id/url/title from, learn once more with such a row as evidence (same job, same budgets). Returns
        (spec, read) to judge again, or None (then the original rejection stands)."""
        examples = (getattr(result, "stats", None) or {}).get("missingExamples") or {}
        fixable = [f for f in decision.failures if re.search(r"tender\(s\) without (id|url|title)\b", f)]
        if not fixable or not examples or not hasattr(self.pipeline, "repair"):
            return None
        feedback = fixable + [f"a listed row without {k} (the selector must also read rows like this one): {v}"
                              for k, v in examples.items()]
        ctx.note(f"the first full read was rejected ({'; '.join(fixable)}); learning once more from such a row")
        if hasattr(ctx.llm, "grant"):   # its own bounded budget: the attach's learn may have used the job's calls
            ctx.llm.grant(ctx.run_id, RELEARN_EXTRA_CALLS)
        try:
            new_spec = await self.pipeline.repair(ctx, spec, feedback)
            return new_spec, await self.pipeline.read(ctx, new_spec)
        except (AttachFailed, LLMUnavailable, httpx.HTTPError) as e:   # incl. the model budget, a network error:
            ctx.note(f"learning again did not work: {type(e).__name__}: {e}")   # the first rejection stands
            return None

    async def _repair_job(self, pid: str, old_spec: dict, failures: list[str], was_state: str) -> None:
        portal = self.store.get_portal(pid)
        if portal is None:
            return
        rid = self.store.create_run(pid, "repair")
        task = asyncio.current_task()
        self.active[pid] = (rid, task)  # refresh during repair returns the repair's runId
        n_serving = self.store.tender_count(pid)
        self.store.set_state(pid, "healing", f"Broken: {'; '.join(failures)}. Repair running; "
                                             f"serving the last correct {n_serving} tenders.")
        ctx = self._ctx(pid, portal["url"], rid, self.s.repair_request_budget)
        ctx.mode = "repair"
        attempts_log = []
        ticker = self._progress(rid, ctx)
        try:
            for attempt in range(1, self.s.repair_max_attempts + 1):
                if not self._alive(pid):
                    return
                new_spec = await self.pipeline.repair(ctx, old_spec, failures)
                result = await self.pipeline.read(ctx, new_spec)
                decision = await self._accept(pid, rid, ctx, new_spec, result, mode="repair", promote=True,
                                              first=False, attempts=attempts_log, was=failures,
                                              label=f" after {attempt} attempt(s)")
                if decision.commit:
                    return
                failures = decision.failures or failures
            msg = f"Repair gave up after {self.s.repair_max_attempts} attempts: {'; '.join(failures)}"
            self.store.finish_run(rid, "failed", msg, {"log": ctx.log, "attempts": attempts_log}, ctx.client.requests)
            self._fail_terminal(pid, msg + ". A person is needed; serving the last correct tenders.")
        except asyncio.CancelledError:
            self._finish_quietly(rid, "cancelled", ctx)
            raise
        except PortalBlocked as e:
            self.store.finish_run(rid, "failed", str(e), {"log": ctx.log, "attempts": attempts_log}, ctx.client.requests)
            self._fail_terminal(pid, f"Blocked: {e}. Serving the last correct tenders.")
        except BudgetExceeded as e:
            msg = _budget_reason(e, ctx, "repair")
            self.store.finish_run(rid, "failed", msg, {"log": ctx.log, "attempts": attempts_log}, ctx.client.requests)
            self._fail_terminal(pid, f"Repair stopped, delivery is incomplete: {msg}. "
                                     f"A person is needed; serving the last correct tenders.")
        except Exception as e:
            msg = f"Repair failed: {type(e).__name__}: {e}"
            self.store.finish_run(rid, "failed", msg, {"log": ctx.log, "attempts": attempts_log,
                                  "trace": traceback.format_exc()[-2000:]}, ctx.client.requests)
            self._fail_terminal(pid, msg + ". A person is needed; serving the last correct tenders.")
        finally:
            ticker.cancel()
            await ctx.client.aclose()

    def _fail_terminal(self, pid: str, reason: str) -> None:
        """Every terminal failure: stamp the attempt (starts the retry backoff) and report `failed`."""
        if not self._alive(pid):
            return
        self.store.note_attempt(pid, failed=True)
        self.store.set_state(pid, "failed", reason)

    def _finish_quietly(self, rid: str, why: str, ctx: JobContext) -> None:
        try:
            if self.store.get_run(rid):
                self.store.finish_run(rid, "failed", why, {"log": ctx.log}, ctx.client.requests)
        except Exception:
            pass

    # ------------------------------------------------------------------ scheduler
    async def _schedule_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(min(30.0, self.s.refresh_interval))
                now = timeutil.now()
                for p in self.store.list_portals():
                    if p["id"] in self.active:
                        continue
                    last = timeutil.parse(p["last_attempt_at"]) if p["last_attempt_at"] else None
                    age = (now - last).total_seconds() if last else 1e9
                    if p["state"] == "serving" and age >= self.s.refresh_interval:
                        self.refresh(p["id"])
                    elif p["state"] == "failed":
                        wait = FAILED_BACKOFF[min(max(p["failed_streak"] - 1, 0), len(FAILED_BACKOFF) - 1)]
                        if age >= wait:
                            self.refresh(p["id"])
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scheduler tick failed")


def _summary(counts: dict, result, decision) -> str:
    s = (f"{len(result.tenders)} tenders read: {counts['new']} new, {counts['changed']} changed, "
         f"{counts['closed_missing']} closed (no longer listed).")
    if not decision.close_missing:
        s += " Partial read: missing tenders not closed."
    return s


def _budget_reason(e: Exception, ctx: JobContext, what: str) -> str:
    return (f"the {what} used up its request budget ({ctx.client.requests} requests; {e}) before it could finish; "
            f"delivery would be incomplete")
