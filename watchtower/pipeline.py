"""The pluggable brain: attach (learn a spec), read (static), validate, repair.

H1 ships the interfaces plus a placeholder implementation. The worker/state machine only talks to
this interface, so Onboarder, Runner, Validator and Repairer can be built in parallel by different
people and swapped in without touching the API or the state machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class PortalBlocked(Exception):
    """Login / captcha / access wall: the data cannot be obtained without help -> state `failed`."""


class AttachFailed(Exception):
    """Could not learn how to read the portal -> state `failed` with this reason."""


@dataclass
class JobContext:
    portal_id: str
    url: str
    run_id: str
    client: Any          # http.PoliteClient (rate limited, budgeted)
    docs: Any            # http.DocumentStore
    llm: Any             # llm.LLMRouter
    store: Any           # store.Store
    log: list[str] = field(default_factory=list)
    mode: str = "read"   # read | attach | repair
    browser_cache: dict = field(default_factory=dict)   # scripts already fetched by this job's browser renders

    def note(self, msg: str) -> None:
        self.log.append(msg)


@dataclass
class ReadResult:
    tenders: list[dict]
    complete: bool                      # pagination reached its stop condition (not a cap)
    stats: dict = field(default_factory=dict)
    verified_ids: list[str] = field(default_factory=list)
    # listed notices whose tender page showed they are not open: {id: {"sig", "deadline", "why"}}; remembered by the
    # worker after an accepted read so later reads skip their tender page (forget: ids whose row changed since)
    closed: dict = field(default_factory=dict)
    forget_closed: list = field(default_factory=list)
    state: dict | None = None           # portal read state to keep if this read is accepted (change-feed position)


@dataclass
class Verdict:
    ok: bool
    failures: list[str] = field(default_factory=list)   # hard checks that failed, in words
    warnings: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.failures) or "ok"


class Pipeline(Protocol):
    async def attach(self, ctx: JobContext) -> dict: ...
    async def read(self, ctx: JobContext, spec: dict) -> ReadResult: ...
    async def validate(self, ctx: JobContext, spec: dict, result: ReadResult, previous: dict[str, dict]) -> Verdict: ...
    async def repair(self, ctx: JobContext, old_spec: dict, failures: list[str]) -> dict: ...


class PlaceholderPipeline:
    """Until the Onboarder lands: attaching fails honestly with a clear reason."""

    async def attach(self, ctx: JobContext) -> dict:
        raise AttachFailed("Onboarder not implemented yet.")

    async def read(self, ctx: JobContext, spec: dict) -> ReadResult:
        return ReadResult(tenders=[], complete=False)

    async def validate(self, ctx, spec, result, previous) -> Verdict:
        return Verdict(ok=False, failures=["validator not implemented"])

    async def repair(self, ctx, old_spec, failures) -> dict:
        raise AttachFailed("Repairer not implemented yet.")
