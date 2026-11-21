"""W3: one model budget per job; unpriced usage is unknown (NULL), not $0."""
import asyncio

import httpx
import pytest

from watchtower.config import Settings
from watchtower.llm import LLMRouter, ModelBudgetExhausted
from watchtower.pipeline import AttachFailed
from watchtower.store import Store


def _router(tmp_path, **kw):
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    store = Store(str(tmp_path))
    s = Settings(data_dir=str(tmp_path), openai_api_key="k", model_strong="m-unpriced", model_cheap="m-unpriced", **kw)
    return store, LLMRouter(s, store, transport=httpx.MockTransport(handler))


def test_budget_exhausted_is_a_clear_attach_failure(tmp_path):
    store, llm = _router(tmp_path, model_max_calls_per_job=3)

    async def go():
        for _ in range(3):
            await llm.json_call("classify", "s json", "u", portal_id="p", run_id="job1")
        with pytest.raises(AttachFailed, match="model budget exhausted"):
            await llm.json_call("classify", "s json", "u", portal_id="p", run_id="job1")
        await llm.json_call("classify", "s json", "u", portal_id="p", run_id="job2")  # another job has its own budget
        await llm.aclose()
    asyncio.run(go())
    assert issubclass(ModelBudgetExhausted, AttachFailed)
    assert store.llm_totals("p")["calls"] == 4


def test_default_budget_and_env(monkeypatch):
    assert Settings().model_max_calls_per_job == 20
    monkeypatch.setenv("MODEL_MAX_CALLS_PER_JOB", "7")
    assert Settings().model_max_calls_per_job == 7


def test_unpriced_usage_is_null_not_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("PRICE_IN_M_UNPRICED", raising=False)
    store, llm = _router(tmp_path)
    asyncio.run(llm.json_call("classify", "s json", "u", portal_id="p", run_id="j"))
    rows = store._q("SELECT usd FROM llm_calls WHERE portal_id='p'")
    assert rows[0]["usd"] is None


def test_grant_tops_up_a_job_budget_once():
    from watchtower.llm import LLMRouter
    r = LLMRouter.__new__(LLMRouter)
    r._job_calls = {"r1": 20}
    r.grant("r1", 16)
    assert r._job_calls["r1"] == 4
    r.grant("r1", 16)
    assert r._job_calls["r1"] == 0          # never below zero
    r.grant(None, 5)                         # no run id: nothing to top up
