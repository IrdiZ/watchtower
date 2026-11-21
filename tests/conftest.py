"""Shared test setup. Live behaviour (robots.txt respected) is the default in production; mock portals in tests answer
arbitrary paths, so the suite runs with robots.txt off except where a test turns it on explicitly."""
import os

import pytest

os.environ.setdefault("WATCHTOWER_RESPECT_ROBOTS", "0")


@pytest.fixture(autouse=True)
def _fresh_robots_cache():
    from watchtower import http as wh
    wh._ROBOTS.clear()
    yield
    wh._ROBOTS.clear()
