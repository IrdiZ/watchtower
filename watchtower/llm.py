"""Task-routed LLM client (OpenAI-compatible Chat Completions) with a cost ledger.

- Each call names a *task* (classify, synthesize, verify, audit, triage, repair...). The model is
  chosen per task from env (MODEL_<TASK>, else MODEL_CHEAP / MODEL_STRONG), so different models can
  be interleaved per feature.
- `gather` runs independent calls in parallel under a concurrency cap.
- Every call is logged with tokens and estimated USD. Prices come from env
  (PRICE_IN_<MODEL>, PRICE_OUT_<MODEL>, USD per 1M tokens; model name upper-cased, '-' and '.' -> '_').
  Without prices, tokens are still logged and the cost is stored as NULL (unknown), never as $0.
- One model budget per job (run_id): at most MODEL_MAX_CALLS_PER_JOB calls (default 20), then
  `ModelBudgetExhausted` (an `AttachFailed`) so a confused attach or repair stops spending money.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

from .config import Settings
from .pipeline import AttachFailed


class LLMUnavailable(Exception):
    pass


class ModelBudgetExhausted(AttachFailed):
    """The job used up its model-call budget."""


def _price(model: str, kind: str) -> float | None:
    """USD per 1M tokens, or None when no price is configured (unknown, not free)."""
    key = f"PRICE_{kind}_{model.upper().replace('-', '_').replace('.', '_')}"
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


class LLMRouter:
    def __init__(self, settings: Settings, store, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self.store = store
        self._sem = asyncio.Semaphore(max(1, settings.llm_concurrency))
        self._client = httpx.AsyncClient(timeout=180, transport=transport)
        self._no_reasoning_param: set[str] = set()  # models that rejected reasoning_effort
        self._job_calls: dict[str, int] = {}  # model calls started per job (run_id)

    @property
    def available(self) -> bool:
        return bool(self.s.openai_api_key and (self.s.model_strong or self.s.model_cheap))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def json_call(self, task: str, system: str, user: str | list[dict], schema: dict | None = None,
                        portal_id: str | None = None, run_id: str | None = None,
                        schema_name: str = "result") -> dict[str, Any]:
        """One structured-output call. Returns the parsed JSON object."""
        model = self.s.model_for(task)
        if not (self.s.openai_api_key and model):
            raise LLMUnavailable(f"no model configured for task '{task}' (set OPENAI_API_KEY and MODEL_*)")
        self._charge_budget(run_id)
        content = user if isinstance(user, list) else [{"type": "text", "text": user}]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": ({"type": "json_schema",
                                 "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
                                if schema else {"type": "json_object"}),
        }
        effort = self.s.reasoning_for(task)
        if effort and model not in self._no_reasoning_param:
            body["reasoning_effort"] = effort
        async with self._sem:
            resp = await self._post(body)
            if resp.status_code == 400 and "reasoning" in resp.text and "reasoning_effort" in body:
                self._no_reasoning_param.add(model)  # model doesn't take this effort level: retry without
                body.pop("reasoning_effort")
                resp = await self._post(body)
        if resp.status_code >= 400:
            raise LLMUnavailable(f"LLM error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        p_in, p_out = _price(model, "IN"), _price(model, "OUT")
        usd = None if (p_in is None or p_out is None) else tin / 1e6 * p_in + tout / 1e6 * p_out
        self.store.log_llm_call(portal_id, run_id, task, model, tin, tout, usd)
        text = data["choices"][0]["message"].get("content") or "{}"
        return json.loads(text)

    def grant(self, run_id: str | None, extra: int) -> None:
        """A bounded, one-off top-up of a job's model-call budget (e.g. the single re-learn after an attach read
        showed rows the first spec could not read). Never more than `extra` calls."""
        if run_id and extra > 0:
            self._job_calls[run_id] = max(0, self._job_calls.get(run_id, 0) - int(extra))

    def calls_left(self, run_id: str | None) -> int | None:
        """Model calls this job may still make (None: not budgeted)."""
        limit = int(getattr(self.s, "model_max_calls_per_job", 0) or 0)
        if not run_id or limit <= 0:
            return None
        return max(0, limit - self._job_calls.get(run_id, 0))

    def _charge_budget(self, run_id: str | None) -> None:
        """Count one model call against the job's budget (calls with no run_id are not budgeted)."""
        if not run_id:
            return
        limit = int(getattr(self.s, "model_max_calls_per_job", 0) or 0)
        used = self._job_calls.get(run_id, 0)
        if limit > 0 and used >= limit:
            raise ModelBudgetExhausted(f"model budget exhausted: this job already made {used} model calls "
                                       f"(limit {limit}, MODEL_MAX_CALLS_PER_JOB)")
        self._job_calls[run_id] = used + 1
        if len(self._job_calls) > 500:  # forget the oldest jobs
            for k in list(self._job_calls)[:100]:
                self._job_calls.pop(k, None)

    async def _post(self, body: dict) -> httpx.Response:
        return await self._client.post(f"{self.s.openai_base_url.rstrip('/')}/chat/completions",
                                       headers={"Authorization": f"Bearer {self.s.openai_api_key}"}, json=body)

    async def gather(self, *calls):
        """Run independent LLM coroutines in parallel (bounded by LLM_CONCURRENCY)."""
        return await asyncio.gather(*calls, return_exceptions=True)
