"""The real pipeline, auto-loaded by api.create_app.

attach  -> onboard.learn (LLM)       read -> runner.read (static, no LLM)
validate-> validate.validate (static) repair -> onboard.learn seeded with the broken spec + last good tenders
"""
from __future__ import annotations

from selectolax.lexbor import LexborHTMLParser as HTMLParser

from . import onboard, runner
from .pipeline import AttachFailed, JobContext, ReadResult, Verdict
from .validate import build_profile, validate


class Engine:
    def __init__(self, settings=None):
        self.settings = settings

    async def attach(self, ctx: JobContext) -> dict:
        if not ctx.llm.available:
            raise AttachFailed("no model configured (set OPENAI_API_KEY, MODEL_STRONG, MODEL_CHEAP)")
        spec = await onboard.learn(ctx, ctx.url)
        await self._prune_markers(ctx, spec)
        return spec

    # Work cap per read: the whole list is walked every time, but at most this many tender pages / documents are
    # opened per read (1 request/second). The rest is filled in by the following refreshes.
    # First fill of a big portal: more detail pages per read (still 1 request/s per host), so buyer/deadline from
    # tender pages arrive in about an hour instead of six; documents likewise.
    MAX_DETAILS_PER_READ = 240
    MAX_DOCS_PER_READ = 240

    async def read(self, ctx: JobContext, spec: dict) -> ReadResult:
        # A repair reads afresh (a delta could be empty and prove nothing); other reads continue where the last
        # accepted read stopped (a change feed's high-water mark).
        state = None if ctx.mode == "repair" or not hasattr(ctx.store, "get_read_state") \
            else ctx.store.get_read_state(ctx.portal_id)
        return await runner.read(ctx, spec, ctx.store.get_tenders(ctx.portal_id),
                                 runner.Limits(max_details=self.MAX_DETAILS_PER_READ, max_docs=self.MAX_DOCS_PER_READ),
                                 known_closed=ctx.store.known_closed(ctx.portal_id), state=state)

    async def validate(self, ctx: JobContext, spec: dict, result: ReadResult, previous: dict) -> Verdict:
        mode = ctx.mode
        if not spec.get("profile"):
            mode = "repair" if ctx.mode == "repair" else "attach"
        verdict = validate(spec, result, previous, mode=mode)
        if verdict.ok and not spec.get("profile") and result.tenders:
            # first trusted read of this spec: remember what normal looks like
            spec["profile"] = build_profile(result.tenders)
            total = (result.stats or {}).get("statedTotal")
            listed = int((result.stats or {}).get("rows") or len(result.tenders))   # closed rows dropped still count
            spec["totalReliable"] = bool(total is not None and result.complete and total == listed)
        return verdict

    async def repair(self, ctx: JobContext, old_spec: dict, failures: list[str]) -> dict:
        if not ctx.llm.available:
            raise AttachFailed("no model configured for repair")
        previous = ctx.store.get_tenders(ctx.portal_id)
        hints = [{k: v for k, v in p["data"].items() if not k.startswith("_") and k != "documents"}
                 for p in list(previous.values())[:5]]
        start = old_spec.get("learnedFrom") or ctx.url
        # the profile and the blind-verification flag belong to the old spec; the new spec must earn its own
        seed = {k: v for k, v in old_spec.items()
                if k not in ("profile", "verifiedBlind", "verification", "totalReliable")}
        try:
            spec = await onboard.learn(ctx, start, old_spec=seed, failures=failures, hints=hints)
        except AttachFailed:
            if start == ctx.url:
                raise
            spec = await onboard.learn(ctx, ctx.url, old_spec=seed, failures=failures, hints=hints)
        await self._prune_markers(ctx, spec)
        return spec

    async def _prune_markers(self, ctx: JobContext, spec: dict) -> None:
        """Drop block markers that match the working list page (e.g. a login box in the header)."""
        markers = spec.get("block_markers") or []
        if not markers or spec["list"].get("kind") != "html":
            return
        _, resp = await runner.fetch_list_page(ctx, spec, 0)
        tree = HTMLParser(resp.text)
        keep = []
        for sel in markers:
            try:
                if not tree.css_first(sel):
                    keep.append(sel)
            except Exception:
                continue
        spec["block_markers"] = keep
