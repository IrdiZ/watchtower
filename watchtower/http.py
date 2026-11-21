"""Polite HTTP: one shared per-host limiter (<= 1 req/s), request accounting, document fetching.

Every outbound request from the system - static reads, downloads, and (later) Playwright via
route interception - must go through `HostRateLimiter.wait(host)`.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from urllib.robotparser import RobotFileParser

from .pipeline import PortalBlocked
from .urls import host_of


class BudgetExceeded(Exception):
    pass


class DocumentBlocked(Exception):
    """A document URL answered with a login/captcha/access wall instead of the file. Nothing was stored."""


_CAPTCHA_PAT = re.compile(r"recaptcha|hcaptcha|turnstile|captcha", re.I)
_WALL_HEAD_BYTES = 256 * 1024     # a wall page is small; a larger HTML body is a real document
_WALL_TEXT_CHARS = 800            # "little text"


def _looks_like_html(content_type: str, head: bytes) -> bool:
    if "html" in (content_type or "").lower():
        return True
    if (content_type or "").strip() and "octet-stream" not in content_type.lower() and "text/plain" not in content_type.lower():
        return False
    return head.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html"))


WEB_PAGE_REASON = "the document link opens a web page, not a file (the files need a click or a session there)"


def is_web_page(content_type: str, head: bytes) -> bool:
    """A download that is an HTML page (a portal's own page, a notice page, a viewer) is not a document file.
    Measured on the live system: 3,503 of 11,911 stored 'documents' were the platforms' own pages."""
    h = head.lstrip()[:512].lower()
    if h.startswith((b"%pdf", b"pk\x03\x04", b"\xd0\xcf\x11\xe0", b"{\\rtf")):
        return False                       # a real file, whatever the Content-Type header says
    return h.startswith((b"<!doctype html", b"<html")) or ("html" in (content_type or "").lower() and b"<html" in h)


def detect_document_wall(content_type: str, head: bytes) -> str | None:
    """Positive evidence that a downloaded 'document' is a login or captcha page: HTML with a password field or
    a captcha widget AND very little visible text. Plain HTML without such evidence is not a wall."""
    if not _looks_like_html(content_type, head):
        return None
    html = head.decode("utf-8", errors="replace")
    tree = HTMLParser(html)
    for bad in tree.css("script, style, noscript"):
        bad.decompose()
    text = re.sub(r"\s+", " ", (tree.body.text() if tree.body else tree.text()) or "").strip()
    if len(text) >= _WALL_TEXT_CHARS:
        return None
    if tree.css_first("input[type=password]"):
        return "the download is a login page (password field), not the document"
    if _CAPTCHA_PAT.search(html):
        return "the download is a captcha page, not the document"
    return None


class HostRateLimiter:
    """Global per-host spacing. Shared by all portals that live on the same host.
    A host may get a slower floor (a published API limit, configured as data: WATCHTOWER_HOST_INTERVALS
    "host=seconds,host=seconds"), and a host that answers 429 is slowed down by itself (backoff)."""

    MAX_BACKOFF_INTERVAL = 10.0
    # This machine (e.g. the built-in demo portal): a configured interval may be FASTER than the default floor.
    # Every other host only ever gets a slower floor.
    LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})

    def __init__(self, min_interval: float, host_intervals: dict[str, float] | None = None):
        self.min_interval = min_interval
        self.host_intervals = dict(host_intervals or {})
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def interval(self, host: str) -> float:
        h = (host or "").lower().removeprefix("www.")
        if h in self.LOOPBACK and h in self.host_intervals:
            return self.host_intervals[h]
        return max(self.min_interval, self.host_intervals.get(h, 0.0))

    def slow_down(self, host: str, retry_after: float | None = None) -> float:
        """The host said 'too many requests': at least double its spacing (and honour Retry-After), up to a cap."""
        h = (host or "").lower().removeprefix("www.")
        new = min(self.MAX_BACKOFF_INTERVAL, max(self.interval(host) * 2, retry_after or 0.0, 2.0))
        self.host_intervals[h] = new
        return new

    async def wait(self, host: str) -> None:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last.get(host)
            if last is not None:
                delay = self.interval(host) - (time.monotonic() - last)
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last[host] = time.monotonic()


def parse_host_intervals(raw: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in (raw or "").split(","):
        if "=" in part:
            h, v = part.split("=", 1)
            try:
                out[h.strip().lower().removeprefix("www.")] = float(v)
            except ValueError:
                continue
    return out


class RobotsDisallowed(PortalBlocked):
    """The site's robots.txt disallows this address for our user agent."""


# robots.txt per site (scheme://host), shared by all jobs: (valid until, parser)
_ROBOTS: dict[str, tuple[float, RobotFileParser]] = {}
ROBOTS_TTL = 6 * 3600
ROBOTS_ERROR_TTL = 600


@dataclass
class DocEntry:
    fileName: str
    sizeBytes: int
    sha256: str
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {"fileName": self.fileName, "sizeBytes": self.sizeBytes, "sha256": self.sha256, "path": self.path}


def _cookie_pairs(header: str | None) -> dict[str, str]:
    """'a=1; b=2' -> {'a': '1', 'b': '2'} (first value of a repeated name wins, as servers usually read it)."""
    out: dict[str, str] = {}
    for part in (header or "").split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name.strip():
            out.setdefault(name.strip(), value.strip())
    return out


def _loop_message(url: str, hops: list[str], had_cookies: bool) -> str:
    """Say what the redirect chain did, so a failed attach explains itself (e.g. a cookie check that never ends)."""
    def short(u: str) -> str:
        p = urlsplit(u)
        return (p.path or "/") + (f"?{p.query}" if p.query else "")
    revisits = len(hops) - len(set(hops))
    chain = " -> ".join(short(u) for u in hops[-4:])
    msg = f"too many redirects for {url} (last hops: {chain}"
    if revisits and had_cookies:
        msg += "; the chain repeats although the cookies the site set were sent back"
    return msg + ")"


class PoliteClient:
    """Per-job HTTP client: rate limited, request-counted, budgeted. One cookie jar per job (a session)."""

    def __init__(self, limiter: HostRateLimiter, user_agent: str, timeout: float, budget: int | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, respect_robots: bool = False):
        self.limiter = limiter
        self.budget = budget
        self.requests = 0
        self.user_agent = user_agent
        self.respect_robots = respect_robots
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent, "Accept-Language": "en,de;q=0.8,fr;q=0.7,nl;q=0.6,es;q=0.5"},
            timeout=timeout, follow_redirects=False, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send(self, method: str, url: str, stream: bool = False, **kw) -> httpx.Response:
        """Follow redirects manually so every hop is rate limited and counted.

        Cookies: the client's jar stores the Set-Cookie of every response, redirects included, and sends what
        applies on the next hop, so a cookie check ("set a cookie, redirect, expect it back") passes. A caller's
        own Cookie header (the browser's, during discovery) would hide the jar from httpx; it is merged with the
        jar instead (the jar wins on a name both carry: it holds what the server set last) and only sent to the
        host it was meant for."""
        own: dict[str, str] = {}
        first_host = host_of(url)
        if kw.get("headers"):
            headers = httpx.Headers(kw["headers"])
            if "cookie" in headers:
                own = _cookie_pairs(headers["cookie"])
                del headers["cookie"]
                kw = {**kw, "headers": headers}
        hops: list[str] = []
        retried_429 = retried_transport = False
        for _ in range(10):
            if self.budget is not None and self.requests >= self.budget:
                raise BudgetExceeded(f"request budget of {self.budget} exhausted")
            if self.respect_robots and not await self._robots_allow(url):
                parts = urlsplit(url)
                raise RobotsDisallowed(f"robots.txt of {parts.netloc} disallows automated access to "
                                       f"{parts.path or '/'}")
            await self.limiter.wait(host_of(url))
            self.requests += 1
            req = self._client.build_request(method, url, **kw)
            if own and host_of(url) == first_host:
                merged = {**own, **_cookie_pairs(req.headers.get("cookie"))}
                req.headers["cookie"] = "; ".join(f"{k}={v}" for k, v in merged.items())
            hops.append(url)
            try:
                resp = await self._client.send(req, stream=stream)
            except (httpx.RemoteProtocolError, httpx.ReadError):
                # "Server disconnected without sending a response": typically a kept-alive connection the server
                # had already closed. A read is safe to send once more (paced and counted like any request).
                if method.upper() not in ("GET", "HEAD") or retried_transport:
                    raise
                retried_transport = True
                await asyncio.sleep(1.0)
                continue
            if resp.status_code == 429 and not retried_429:
                # Too many requests: slow down for this host (Retry-After if given), wait, and try once more.
                ra = resp.headers.get("retry-after", "")
                wait_s = self.limiter.slow_down(host_of(url), float(ra) if ra.strip().isdigit() else None)
                await resp.aclose()
                await asyncio.sleep(min(wait_s, 30.0))
                retried_429 = True
                continue
            if resp.is_redirect and "location" in resp.headers:
                await resp.aclose()
                url = str(resp.url.join(resp.headers["location"]))
                if resp.status_code in (301, 302, 303):
                    method, kw = "GET", {k: v for k, v in kw.items() if k not in ("data", "json", "content")}
                continue
            return resp
        raise httpx.TooManyRedirects(_loop_message(url, hops, bool(self._client.cookies.jar)))

    async def _robots_allow(self, url: str) -> bool:
        """RFC 9309: robots.txt of the site, fetched once per 6 h through the same pacing. A missing robots.txt
        (4xx) allows everything; a server error (5xx) disallows for 10 minutes; an unreachable one allows (we read
        the site itself fine)."""
        parts = urlsplit(url)
        if parts.path == "/robots.txt" or parts.scheme not in ("http", "https"):
            return True
        site = f"{parts.scheme}://{parts.netloc}"
        now = time.time()
        cached = _ROBOTS.get(site)
        if cached is None or cached[0] < now:
            rp = RobotFileParser()
            ttl = ROBOTS_TTL
            try:
                await self.limiter.wait(host_of(url))
                self.requests += 1
                r = await self._client.get(site + "/robots.txt", follow_redirects=True)
                if r.status_code >= 500:
                    rp.disallow_all, ttl = True, ROBOTS_ERROR_TTL
                elif r.status_code >= 400 or "html" in r.headers.get("content-type", "").lower():
                    rp.allow_all = True       # no robots.txt (an HTML error/landing page is not a robots file)
                else:
                    rp.parse(r.text.splitlines())
            except httpx.HTTPError:
                rp.allow_all = True
            _ROBOTS[site] = (now + ttl, rp)
            cached = _ROBOTS[site]
        return cached[1].can_fetch(self.user_agent, url)

    async def get(self, url: str, **kw) -> httpx.Response:
        resp = await self._send("GET", url, **kw)
        await resp.aread()
        return resp

    async def request(self, method: str, url: str, **kw) -> httpx.Response:
        resp = await self._send(method, url, **kw)
        await resp.aread()
        return resp

    async def stream(self, method: str, url: str, **kw) -> httpx.Response:
        return await self._send(method, url, stream=True, **kw)


_PACKAGE_EXTS = (".docx", ".docm", ".dotx", ".xlsx", ".xlsm", ".xltx", ".pptx", ".pptm", ".potx", ".vsdx",
                 ".odt", ".ods", ".odp", ".odg", ".ott", ".epub", ".xps", ".oxps", ".jar", ".apk")


def _filename_from(resp: httpx.Response, fallback_url: str) -> str:
    cd = resp.headers.get("content-disposition", "")
    for part in cd.split(";"):
        part = part.strip()
        if part.lower().startswith("filename*="):
            v = part.split("=", 1)[1].strip().strip('"')
            if "''" in v:
                v = v.split("''", 1)[1]
            return unquote(v)
    for part in cd.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    name = os.path.basename(urlsplit(str(resp.url) or fallback_url).path)
    return unquote(name) or "document"


class DocumentStore:
    """Streams documents to content-addressed storage; unpacks ZIPs (one nested level)."""

    def __init__(self, store, max_bytes: int, max_zip_total: int):
        self.store = store
        self.max_bytes = max_bytes
        self.max_zip_total = max_zip_total

    def _save_bytes(self, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        if not self.store.has_file(sha):
            with open(self.store.file_path(sha), "wb") as f:
                f.write(data)
            self.store.register_file(sha, len(data))
        return sha

    def _save_temp(self, tmp_path: str, sha: str, size: int) -> None:
        if self.store.has_file(sha):
            os.remove(tmp_path)
            return
        os.replace(tmp_path, self.store.file_path(sha))
        self.store.register_file(sha, size)

    def stored_web_page(self, entry: dict) -> bool:
        """A stored document entry whose file is an HTML page (kept by older versions): not a document."""
        sha = entry.get("sha256") if isinstance(entry, dict) else None
        if not sha:
            return False
        cache = self.__dict__.setdefault("_web_page_cache", {})
        if sha not in cache:
            try:
                with open(self.store.file_path(sha), "rb") as f:
                    cache[sha] = is_web_page("", f.read(1024))
            except OSError:
                cache[sha] = False
        return cache[sha]

    @staticmethod
    def is_plain_archive(path_or_file, name: str = "") -> bool:
        """A ZIP that holds documents (a tender dossier), not a document that happens to be a ZIP container:
        .docx/.xlsx/.pptx (OOXML), .odt/.ods (OpenDocument), .epub ... must be delivered as the file itself."""
        if name.lower().endswith(_PACKAGE_EXTS):
            return False
        try:
            if not zipfile.is_zipfile(path_or_file):
                return False
            with zipfile.ZipFile(path_or_file) as zf:
                names = set(zf.namelist()[:200])
        except (zipfile.BadZipFile, OSError):
            return False
        finally:
            if hasattr(path_or_file, "seek"):
                path_or_file.seek(0)
        return not ({"[Content_Types].xml", "mimetype", "META-INF/manifest.xml"} & names)

    def _unpack(self, data_or_path, depth: int = 0) -> list[DocEntry]:
        out: list[DocEntry] = []
        total = 0
        with zipfile.ZipFile(data_or_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or info.file_size > self.max_bytes:
                    continue
                total += info.file_size
                if total > self.max_zip_total:
                    break
                data = zf.read(info)
                name = os.path.basename(info.filename) or info.filename
                if depth == 0 and name.lower().endswith(".zip") and self.is_plain_archive(io.BytesIO(data), name):
                    out.extend(self._unpack(io.BytesIO(data), depth + 1))
                    continue
                sha = self._save_bytes(data)
                out.append(DocEntry(name, len(data), sha, f"/files/{sha}"))
        return out

    async def fetch(self, client: PoliteClient, url: str, method: str = "GET", **kw) -> list[DocEntry]:
        """Download one document URL. Returns one entry, or the unpacked entries of a ZIP.
        Returns [] when the file is larger than the cap (allowed to be left out).
        Raises DocumentBlocked for a 401/403 or a login/captcha page (nothing is stored); other errors are
        transient and may be retried."""
        try:
            resp = await client.stream(method, url, **kw)
        except RobotsDisallowed as e:
            raise DocumentBlocked(str(e))
        tmp = None
        try:
            if resp.status_code in (401, 403):
                raise DocumentBlocked(f"access denied (HTTP {resp.status_code}) for {url}")
            resp.raise_for_status()
            length = resp.headers.get("content-length")
            if length and length.isdigit() and int(length) > self.max_bytes:
                return []
            name = _filename_from(resp, url)
            h = hashlib.sha256()
            size = 0
            head = bytearray()
            fd, tmp = tempfile.mkstemp(dir=self.store.files_dir, prefix=".dl-")
            with os.fdopen(fd, "wb") as f:
                async for chunk in resp.aiter_bytes(1 << 16):
                    size += len(chunk)
                    if size > self.max_bytes:
                        return []
                    if len(head) <= _WALL_HEAD_BYTES:
                        head.extend(chunk)
                    h.update(chunk)
                    f.write(chunk)
            if size <= _WALL_HEAD_BYTES:
                wall = detect_document_wall(resp.headers.get("content-type", ""), bytes(head))
                if wall:
                    raise DocumentBlocked(f"{wall} ({url})")
            if is_web_page(resp.headers.get("content-type", ""), bytes(head)):
                raise DocumentBlocked(f"{WEB_PAGE_REASON} ({url})")
            if self.is_plain_archive(tmp, name):
                return self._unpack(tmp)
            sha = h.hexdigest()
            self._save_temp(tmp, sha, size)
            return [DocEntry(name, size, sha, f"/files/{sha}")]
        finally:
            await resp.aclose()
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
