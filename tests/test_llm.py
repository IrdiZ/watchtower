"""LLM router: per-task model routing, reasoning-effort fallback, cost ledger."""
import asyncio, json

import httpx

from watchtower.config import Settings
from watchtower.llm import LLMRouter
from watchtower.store import Store


def test_routing_fallback_and_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PRICE_IN_STRONG_M", "2")
    monkeypatch.setenv("PRICE_OUT_STRONG_M", "10")
    monkeypatch.setenv("MODEL_VERIFY", "verify-m")
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append((body["model"], body.get("reasoning_effort")))
        if body["model"] == "strong-m" and "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value for reasoning_effort"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}],
                                         "usage": {"prompt_tokens": 1000, "completion_tokens": 100}})
    store = Store(str(tmp_path))
    s = Settings(data_dir=str(tmp_path), openai_api_key="k", model_strong="strong-m", model_cheap="cheap-m")
    llm = LLMRouter(s, store, transport=httpx.MockTransport(handler))

    async def go():
        a = await llm.json_call("synthesize", "sys json", "u", portal_id="p1")
        b = await llm.json_call("verify", "sys json", "u", portal_id="p1")
        c = await llm.json_call("classify", "sys json", "u", portal_id="p1")
        await llm.aclose()
        return a, b, c
    a, b, c = asyncio.run(go())
    assert a == b == c == {"ok": True}
    assert seen[0] == ("strong-m", "low") and seen[1] == ("strong-m", None)   # retried without effort
    assert seen[2] == ("verify-m", "none") and seen[3] == ("cheap-m", "none")
    tot = store.llm_totals("p1")
    assert tot["calls"] == 3 and abs(tot["usd"] - 0.003) < 1e-9   # 1000 in * $2/M + 100 out * $10/M
