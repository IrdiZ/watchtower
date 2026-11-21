"""Browser discovery against a real HTTP mock (JS app whose API URL is only built at runtime).
Skipped automatically when Chromium is not installed."""
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from mock_portal import app as mock
import tests.fake_llm as fake_llm
from tests.fake_llm import FakeLLM
from watchtower.api import create_app
from watchtower.config import Settings


def _chromium_ok():
    try:
        import asyncio
        from playwright.async_api import async_playwright

        async def m():
            async with async_playwright() as p:
                b = await p.chromium.launch(args=["--no-sandbox"])
                await b.close()
        asyncio.run(m())
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _chromium_ok(), reason="chromium not installed")
def test_js_app_attach_via_browser(tmp_path, monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(mock.app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(fake_llm, "BASE", base)
    mock.CONFIG.update(skin="spa_hidden", mutations=set())
    mock.REQUESTS.clear()
    try:
        s = Settings(data_dir=str(tmp_path), refresh_interval=0, min_request_interval=0.2)
        app = create_app(s, llm=FakeLLM)
        with TestClient(app) as client:
            pid = client.post("/portals", json={"url": base + "/"}).json()["id"]
            end = time.time() + 90
            while time.time() < end:
                p = client.get(f"/portals/{pid}").json()
                if p["state"] in ("serving", "failed"):
                    break
                time.sleep(0.2)
            runs = client.get(f"/portals/{pid}/runs").json()
            assert p["state"] == "serving", (p, runs)
            assert p["tenderCount"] == 12
        stamps = sorted(t for t, _ in mock.REQUESTS)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert min(gaps) >= 0.19, min(gaps)  # browser traffic was rate limited too
        assert "/static/app.js" in [p for _, p in mock.REQUESTS]
    finally:
        server.should_exit = True
        th.join(timeout=5)
        mock.CONFIG.update(skin="html", mutations=set())
