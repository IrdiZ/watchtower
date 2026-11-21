"""Rate limiter, redirect accounting, document store (sha256, ZIP unpack, size cap)."""
import asyncio, hashlib, io, time, zipfile

import httpx

from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient, BudgetExceeded
from watchtower.store import Store


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


INNER = _zip({"c.txt": b"ccc"})
ROUTES = {
    "/a.pdf": (200, {"content-disposition": 'attachment; filename="Lastenheft.pdf"'}, b"%PDF-1.4 hello"),
    "/bundle": (200, {}, _zip({"x/one.pdf": b"one", "two.docx": b"two", "inner.zip": INNER})),
    "/big": (200, {}, b"x" * 5000),
    "/redir": (302, {"location": "/a.pdf"}, b""),
}


def handler(request: httpx.Request):
    status, headers, body = ROUTES[request.url.path]
    return httpx.Response(status, headers=headers, content=body)


def run(coro):
    return asyncio.run(coro)


def test_documents_and_zip(tmp_path):
    store = Store(str(tmp_path))
    docs = DocumentStore(store, max_bytes=1000, max_zip_total=10_000)
    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(handler))
        a = await docs.fetch(c, "https://h.example/a.pdf")
        z = await docs.fetch(c, "https://h.example/bundle")
        big = await docs.fetch(c, "https://h.example/big")
        r = await docs.fetch(c, "https://h.example/redir")
        await c.aclose()
        return a, z, big, r, c.requests
    a, z, big, r, n = run(go())
    assert a[0].fileName == "Lastenheft.pdf" and a[0].sha256 == hashlib.sha256(b"%PDF-1.4 hello").hexdigest()
    assert open(store.file_path(a[0].sha256), "rb").read() == b"%PDF-1.4 hello"
    assert sorted(e.fileName for e in z) == ["c.txt", "one.pdf", "two.docx"]  # nested zip unpacked
    assert big == []                       # over the cap -> left out
    assert r[0].sha256 == a[0].sha256      # redirect followed, deduped
    assert n == 5                          # redirect hop counted


def test_rate_limit_spacing():
    async def go():
        lim = HostRateLimiter(0.2)
        t0 = time.monotonic()
        for _ in range(4):
            await lim.wait("h.example")
        return time.monotonic() - t0
    assert run(go()) >= 0.6


def test_budget():
    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, budget=2, transport=httpx.MockTransport(handler))
        await c.get("https://h.example/a.pdf")
        try:
            await c.get("https://h.example/redir")  # 2 hops -> exceeds
        except BudgetExceeded:
            return True
        finally:
            await c.aclose()
        return False
    assert run(go())


# ------------------------------------------------------------------ cookies across redirect hops
def _cookie_check_portal(seen: list):
    """A cookie-checked portal (set a cookie, redirect, expect it back; then a session cookie, redirect again).
    Without the cookie on the next hop it redirects back to the start: a loop."""
    def h(request: httpx.Request):
        cookies = dict(p.strip().split("=", 1) for p in request.headers.get("cookie", "").split(";") if "=" in p)
        q = request.url.query.decode()
        seen.append((request.url.host, request.url.path, q, cookies))
        if request.url.host == "other.example":
            return httpx.Response(200, text="elsewhere")
        if request.url.path == "/away":
            return httpx.Response(302, headers={"location": "https://other.example/x"})
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"location": "/loop?again", "set-cookie": "S=new; Path=/"})
        if request.url.path == "/list" and "cookieCheck" not in q:
            return httpx.Response(302, headers={"location": "/list?cookieCheck",
                                                "set-cookie": "SUPPORTED=yes; Path=/; Secure; HttpOnly"})
        if request.url.path == "/list" and q == "cookieCheck":
            if cookies.get("SUPPORTED") != "yes":
                return httpx.Response(302, headers={"location": "/list"})
            return httpx.Response(302, headers={"location": "/list?0&cookieCheck",
                                                "set-cookie": "JSESSIONID=abc; Path=/; Secure; HttpOnly"})
        if request.url.path == "/list" and q == "0&cookieCheck":
            if cookies.get("JSESSIONID") != "abc":
                return httpx.Response(302, headers={"location": "/list"})
            return httpx.Response(200, text="<table><tr><td>T-1</td></tr></table>")
        return httpx.Response(404)
    return h


def test_cookie_check_redirects_keep_cookies():
    """Set-Cookie of every hop is stored and sent on the next hop (and on later requests of the same job)."""
    seen: list = []

    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(_cookie_check_portal(seen)))
        r = await c.get("https://h.example/list")
        again = await c.get("https://h.example/list?0&cookieCheck")   # the session holds for the next request
        await c.aclose()
        return r, again, c.requests
    r, again, n = run(go())
    assert r.status_code == 200 and "T-1" in r.text
    assert str(r.url) == "https://h.example/list?0&cookieCheck"
    assert again.status_code == 200
    assert n == 4                                      # 3 hops + 1, every hop counted
    assert seen[1][3] == {"SUPPORTED": "yes"}          # cookie of hop 1 sent on hop 2
    assert seen[2][3] == {"SUPPORTED": "yes", "JSESSIONID": "abc"}


def test_own_cookie_header_does_not_hide_the_jar():
    """A caller's Cookie header (the browser's) is merged with the jar on every hop instead of replacing it, and
    it is not sent to another host after a cross-host redirect."""
    seen: list = []

    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(_cookie_check_portal(seen)))
        r = await c.request("GET", "https://h.example/list", headers={"cookie": "consent=1", "accept": "text/html"})
        seen.clear()
        away = await c.request("GET", "https://h.example/away", headers={"Cookie": "consent=1"})
        await c.aclose()
        return r, away
    r, away = run(go())
    assert r.status_code == 200                        # without the merge the browser's header hides the jar: loop
    assert away.status_code == 200
    assert seen[0][3].get("consent") == "1" and seen[0][3].get("JSESSIONID") == "abc"   # same host: merged
    assert seen[1][0] == "other.example" and seen[1][3] == {}                         # other host: nothing


def test_redirect_loop_message_says_what_happened():
    async def go():
        c = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(_cookie_check_portal([])))
        try:
            await c.get("https://h.example/loop")
        except httpx.TooManyRedirects as e:
            return str(e), c.requests
        finally:
            await c.aclose()
        return "", c.requests
    msg, n = run(go())
    assert n == 10
    assert "/loop?again -> /loop?again" in msg and "cookies the site set were sent back" in msg


def test_default_user_agent_is_not_classified_as_a_crawler(monkeypatch):
    """Servers commonly give user agents matching /bot/i no session (cookie-checked portals then loop)."""
    from watchtower.config import Settings
    monkeypatch.delenv("WATCHTOWER_USER_AGENT", raising=False)
    ua = Settings().user_agent
    assert "bot" not in ua.lower() and "watchtower" in ua.lower() and "automated" in ua.lower()


def test_robots_txt_is_respected_and_documents_become_blocked(monkeypatch):
    from watchtower import http as wh
    wh._ROBOTS.clear()
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, content=b"%PDF ok", headers={"content-type": "application/pdf"})

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "Watchtower/0.1", 5, transport=httpx.MockTransport(handler),
                          respect_robots=True)
        try:
            ok = await pc.get("http://r.test/public/list")
            try:
                await pc.get("http://r.test/private/list")
                blocked = None
            except wh.RobotsDisallowed as e:
                blocked = str(e)
            return ok.status_code, blocked
        finally:
            await pc.aclose()
    code, blocked = asyncio.run(go())
    assert code == 200 and "robots.txt of r.test disallows" in blocked
    assert seen.count("/robots.txt") == 1 and "/private/list" not in seen      # fetched once, never requested


def test_429_slows_the_host_down_and_retries_once():
    from watchtower import http as wh
    calls = []

    def handler(req):
        calls.append(time.monotonic())
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, text="ok")
    lim = HostRateLimiter(0)

    async def go():
        pc = PoliteClient(lim, "Watchtower/0.1", 5, transport=httpx.MockTransport(handler))
        try:
            return await pc.get("http://slow.test/x")
        finally:
            await pc.aclose()
    assert asyncio.run(go()).status_code == 200 and len(calls) == 2
    assert lim.interval("slow.test") >= 2.0
    assert wh.parse_host_intervals("www.a.test=2.5, b.test=x") == {"a.test": 2.5}
