"""Browser discovery for JavaScript apps (attach/repair only - regular reads never start a browser).

Opens the page in headless Chromium, lets the app load, and records the JSON responses it fetches. Every
browser request goes through the same per-host rate limiter and counts against the job's request budget.
Images, fonts, stylesheets and media are blocked; service workers are blocked so no request bypasses routing.
"""
from __future__ import annotations

import asyncio
import json
import logging


log = logging.getLogger("watchtower.browser")
BLOCKED_TYPES = {"image", "font", "stylesheet", "media"}
_SKIP_REQ_HEADERS = {"host", "content-length", "accept-encoding", "connection", "user-agent"}
_SKIP_RESP_HEADERS = {"content-encoding", "content-length", "transfer-encoding", "connection"}


async def _close_quietly(page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def _safe(aw) -> None:
    try:
        await aw
    except Exception:  # page closed / request already handled
        pass


MAX_BODY_CHARS = 20000


def request_meta(req, with_url: bool = False) -> dict:
    """How the app asked for a response: method, request body (text, None for GET) and its content type."""
    try:
        body = req.post_data
    except Exception:       # binary body
        body = None
    try:
        ctype = (req.headers or {}).get("content-type")
    except Exception:
        ctype = None
    meta = {"method": (req.method or "GET").upper(),
            "body": body if body is None or len(body) <= MAX_BODY_CHARS else None, "contentType": ctype}
    if with_url:
        meta["url"] = req.url
    return meta


def _tenderish(data) -> bool:
    from .onboard import _score_json
    return _score_json(data) > 0


MAX_CACHED_SCRIPT = 3_000_000
MAX_SCRIPT_CACHE = 40_000_000


def _script_cache(ctx) -> dict | None:
    """Scripts fetched by earlier renders of this job (url -> (status, headers, body)). Routing every request
    disables Chromium's own HTTP cache, so without this each render of a JavaScript app downloads its whole
    bundle again (one request per second per host)."""
    cache = getattr(ctx, "browser_cache", None)
    return cache if isinstance(cache, dict) else None


BOOT_QUIET_S = 20.0   # an app that has drawn nothing yet is still booting (a large bundle parses slowly): wait longer

# The page's search control: a button / submit input whose label, title, icon or class says "search" (multilingual),
# visible, not in a sign-in form, not a reset / consent / newsletter / login control. Returns a short description of
# the clicked control, or null. A click with the page's own default criteria = an empty search (a read).
_SEARCH_JS = r"""
() => {
  const pos = /search|such|recherch|zoek|busc|cerca|ricerc|szuk|wyszuk|hled|vyhled|s[oö]k|s[oø]g|haku|etsi|keres|pesquis|loupe|magnif|find\b|finden|anzeigen|afficher|show results|zeige/i;
  const neg = /login|log in|anmeld|sign|regist|inscri|passw|reset|zur[uü]ck|clear|l[oö]schen|effacer|cookie|consent|akzept|accept|newsletter|abonn|subscribe|merk|save|speicher|export|download|drucken|print|share|teilen|contact|kontakt|lang|sprache/i;
  const apply = /apply|anwenden|[uü]bernehmen|appliquer|aplicar|applica|zastosuj|toepassen|\bok\b|\bgo\b|submit/i;
  const els = Array.from(document.querySelectorAll('button, input[type=submit], input[type=button], input[type=image], [role=button], a.btn, a.button'));
  let best = null, bestScore = 0;
  for (const el of els) {
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height || el.disabled) continue;
    const form = el.closest('form');
    if (form && form.querySelector('input[type=password]')) continue;
    const text = [el.innerText, el.value, el.getAttribute('aria-label'), el.getAttribute('title')].join(' ').slice(0, 200);
    const ico = el.querySelector('[class*=icon], mat-icon, i, svg');
    const use = ico && ico.querySelector('use');
    const icon = ico ? ['class', 'svgicon', 'data-mat-icon-name', 'title', 'aria-label', 'data-icon']
      .map(a => ico.getAttribute(a) || '').concat([ico.textContent || '', use ? (use.getAttribute('href') || use.getAttribute('xlink:href') || '') : ''])
      .join(' ').slice(0, 300) : '';
    const ident = [el.id, el.getAttribute('name'), el.getAttribute('class')].join(' ').slice(0, 200);
    if (neg.test(text) || neg.test(ident)) continue;
    let s = 0;
    if (pos.test(text)) s += 2;
    if (/search|magnif|loupe|such/i.test(icon)) s += 2;
    if (pos.test(ident)) s += 1;
    if (apply.test(text)) s += 1;
    if ((el.type || '').toLowerCase() === 'submit' && form && form.querySelector('input:not([type=hidden]), select')) s += 1;
    if (s > bestScore) { best = el; bestScore = s; }
  }
  if (!best || bestScore < 2) return null;
  const desc = (best.innerText || best.value || best.getAttribute('aria-label') || best.getAttribute('title') || best.id || best.tagName).trim().slice(0, 60);
  best.click();
  return desc || best.tagName;
}
"""


async def capture(ctx, url: str, max_wait_s: float = 180.0, quiet_s: float = 6.0, max_json: int = 30,
                  until_text: list[str] | None = None, submit_search: bool | str = False) -> dict:
    """Returns {"html": rendered DOM, "json": [(url, data), ...], "json_meta": [{"method", "body", "contentType"},
    ...] (same order as "json": how the app asked for each response, so a POST search can be replayed),
    "data_calls": [{"url", "method", "body", "contentType"}, ...] (the app's XHR/fetch requests), "links": [...],
    "final_url": the page's address at the end (an app may route elsewhere), "submitted": the search control that was
    clicked or None, "error": str|None}.
    until_text: stop waiting shortly after the rendered page shows any of these strings (a tender page that has
    rendered its tender), instead of waiting for tender-like data or a quiet network.
    submit_search: when the app has loaded no tender-like data once it settles, press its search control with the
    page's default criteria (many portals show the list only as a search result) and wait for the answer again;
    "always": press it even when data arrived (the caller could not use that data)."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {"html": "", "json": [], "json_meta": [], "data_calls": [], "links": [],
                "error": "playwright not installed"}
    found: list[tuple[str, object]] = []
    out = {"html": "", "json": found, "json_meta": [], "links": [], "error": None, "requests": 0, "seconds": 0.0,
           "types": {}, "data_requests": [], "data_calls": [], "final_url": None, "submitted": None}
    import time
    state = {"last": time.monotonic(), "n": 0, "inflight": 0}
    pending: set[asyncio.Task] = set()
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-background-networking",
                                                                  "--disable-component-update", "--no-pings"])
            try:
                context = await browser.new_context(service_workers="block", user_agent=ctx.client._client.headers
                                                    .get("User-Agent"), accept_downloads=False)
                main_page = await context.new_page()
                page = main_page

                def close_extra(p):
                    if p is not main_page:  # popups / window.open: their first request is routed (and paced) below
                        asyncio.ensure_future(_close_quietly(p))
                context.on("page", close_extra)
                page.on("popup", lambda p: asyncio.ensure_future(_close_quietly(p)))

                async def route(r):
                    req = r.request
                    out["types"][req.resource_type] = out["types"].get(req.resource_type, 0) + 1
                    if not req.url.startswith(("http://", "https://")):
                        await r.continue_()  # data:, blob:, about: - no network request
                        return
                    if req.resource_type in ("xhr", "fetch") and len(out["data_requests"]) < 20:
                        out["data_requests"].append(req.url)
                        out["data_calls"].append(request_meta(req, with_url=True))
                    if req.resource_type == "stylesheet":
                        # An empty stylesheet, answered locally (no request to the portal). Aborting it is not
                        # harmless: bundlers (Vite's preload helper) treat a failed CSS preload as a failed page
                        # import, and the app never mounts the page that asks for the data.
                        await _safe(r.fulfill(status=200, content_type="text/css", body=""))
                        return
                    if req.resource_type in BLOCKED_TYPES:
                        await r.abort()
                        return
                    cache = _script_cache(ctx)
                    cacheable = cache is not None and req.resource_type == "script" and req.method == "GET"
                    if cacheable and req.url in cache:
                        status, hdrs, body = cache[req.url]
                        await _safe(r.fulfill(status=status, headers=hdrs, body=body))
                        return
                    # Every request is made by PoliteClient (per-host limiter, counts every redirect hop, budget)
                    # and handed to the browser with fulfill(): nothing leaves the browser on its own.
                    # Reserve budget for requests already in flight (the client only checks before it sends).
                    if ctx.client.budget is not None and ctx.client.requests + state["inflight"] >= ctx.client.budget:
                        await _safe(r.abort("failed"))
                        return
                    state["inflight"] += 1
                    try:
                        headers = {k: v for k, v in (await req.all_headers()).items()
                                   if k.lower() not in _SKIP_REQ_HEADERS}
                        kw: dict = {"headers": headers}
                        if req.post_data_buffer is not None:
                            kw["content"] = req.post_data_buffer
                        resp = await ctx.client.request(req.method, req.url, **kw)
                    except Exception as e:  # budget exhausted, network error, too many redirects
                        log.info("browser request %s failed: %s", req.url, e)
                        await _safe(r.abort("failed"))
                        return
                    finally:
                        state["inflight"] -= 1
                    state["n"] += 1
                    state["last"] = time.monotonic()
                    out_headers: dict[str, str] = {}
                    for k, v in resp.headers.multi_items():
                        if k.lower() in _SKIP_RESP_HEADERS:
                            continue
                        out_headers[k] = f"{out_headers[k]}\n{v}" if k in out_headers else v
                    if cacheable and resp.status_code == 200 and len(resp.content) <= MAX_CACHED_SCRIPT \
                            and sum(len(b) for _, _, b in cache.values()) + len(resp.content) <= MAX_SCRIPT_CACHE:
                        cache[req.url] = (resp.status_code, out_headers, resp.content)
                    await _safe(r.fulfill(status=resp.status_code, headers=out_headers, body=resp.content))
                await context.route("**/*", route)

                async def grab(resp):
                    try:
                        ct = resp.headers.get("content-type", "")
                        if "json" not in ct or resp.status >= 400:
                            return
                        body = await resp.body()
                        if len(body) > 3_000_000:
                            return
                        data = json.loads(body)
                        found.append((resp.url, data))
                        out["json_meta"].append(request_meta(resp.request))
                    except Exception:
                        return

                def on_response(resp):
                    if len(found) < max_json:
                        t = asyncio.ensure_future(grab(resp))
                        pending.add(t)
                        t.add_done_callback(pending.discard)
                page.on("response", on_response)
                start = time.monotonic()
                try:
                    await page.goto(url, wait_until="commit", timeout=60000)
                except Exception as e:
                    log.info("browser goto: %s", e)
                wanted = [w.casefold() for w in (until_text or []) if isinstance(w, str) and len(w.strip()) >= 3]

                async def drawn() -> bool:
                    try:
                        return bool(await page.evaluate(
                            "() => !!document.body && (document.body.innerText || '').trim().length >= 80"))
                    except Exception:   # navigating: not settled
                        return False

                async def shows_wanted() -> bool:
                    try:
                        shown = await page.evaluate("() => document.body ? document.body.innerText : ''")
                    except Exception:       # navigating / page closed
                        shown = ""
                    return any(w in (shown or "").casefold() for w in wanted)

                async def settle(t0: float) -> bool:
                    """Requests are spaced 1/s, so apps load slowly. Wait until tender-like data has arrived (or,
                    with until_text, the page shows the wanted text) plus a short grace period, or the page has gone
                    quiet, or the time cap is reached. Quiet on a page that shows nothing yet is a booting app (a big
                    bundle parses for seconds): it gets longer. True when the awaited content arrived."""
                    got_data_at = None
                    seen = len(found)
                    while time.monotonic() - start < max_wait_s:
                        await asyncio.sleep(0.5)
                        now = time.monotonic()
                        if got_data_at is None and wanted and now - t0 > 1 and await shows_wanted():
                            got_data_at = now - 1.5      # the page shows it: a short grace period for the rest
                        if got_data_at is None and not wanted and any(_tenderish(d) for _, d in list(found)[seen:]):
                            got_data_at = now
                        if got_data_at and now - got_data_at > 3:
                            return True
                        idle = now - max(state["last"], t0)
                        if idle > quiet_s and now - t0 > 8 and (idle > BOOT_QUIET_S or await drawn()):
                            return False
                    return got_data_at is not None

                got = await settle(start)
                if submit_search and (not got or submit_search == "always"):
                    try:
                        clicked = await page.evaluate(_SEARCH_JS)
                    except Exception as e:
                        clicked = None
                        log.info("browser search click: %s", e)
                    if clicked:
                        out["submitted"] = clicked
                        await settle(time.monotonic())
                out["seconds"] = round(time.monotonic() - start, 1)
                out["final_url"] = page.url
                if pending:
                    await asyncio.wait(pending, timeout=5)
                out["html"] = await page.content()
                out["links"] = await page.eval_on_selector_all(
                    "a[href]", "els => els.slice(0, 120).map(e => ({text: (e.innerText || e.textContent || '').trim()"
                    ".slice(0, 80), url: (e.href && e.href.baseVal !== undefined) ? e.href.baseVal : e.href}))")
            finally:
                await browser.close()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["requests"] = state["n"]
    return out
