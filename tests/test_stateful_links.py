"""Stateful server apps (e.g. Wicket): a cookie check on the first request, then links that only work inside the
session and with the page they were on as Referer (403 otherwise) - for the link to the list and for the pager."""
import asyncio
import re
from types import SimpleNamespace

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from tests.test_w1_runner import run_read
from watchtower import onboard
from watchtower.http import HostRateLimiter, PoliteClient

HOST = "https://w.test"
TENDERS = [(f"T-{i}", f"Ausschreibung {i}", f"0{1 + i % 9}.12.30, 09:00") for i in range(25)]


def wicket_portal(log: list):
    """Pages need the session cookie; listener links (?0-N.-...) need a same-site Referer."""
    def page(n: int, session: str) -> str:
        rows = "".join(f"<tr><td><a href='./tenderdetails.html?id={tid}'>{title}</a></td><td>{dl}</td></tr>"
                       for tid, title, dl in TENDERS[n * 10:(n + 1) * 10])
        nxt = (f"<a rel='next' class='next' href='./search.html?0-{n + 1}.-navigator-next'></a>"
               if (n + 1) * 10 < len(TENDERS) else "")
        return (f"<html><body><p>Zeige {n * 10 + 1} bis {min(len(TENDERS), n * 10 + 10)} von {len(TENDERS)}</p>"
                f"<table id='datatable'><thead><tr><th>Bezeichnung</th><th>Nächste Frist</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>{nxt}</body></html>")
    state = {"page": 0}

    def h(req: httpx.Request):
        cookies = req.headers.get("cookie", "")
        q = req.url.query.decode()
        log.append((req.url.path, q, req.headers.get("referer")))
        if "SESSION=s1" not in cookies:
            if "cookieCheck" in q:
                return httpx.Response(302, headers={"location": "/start.html"})
            return httpx.Response(302, headers={"location": f"{req.url.path}?cookieCheck",
                                                "set-cookie": "SESSION=s1; Path=/; Secure; HttpOnly"})
        listener = re.match(r"^0-(\d+)\.-", q)
        if listener and not (req.headers.get("referer") or "").startswith(HOST + "/"):
            return httpx.Response(403, text="Forbidden")
        if req.url.path == "/start.html" and listener:      # 'Alle Ausschreibungen' listener link
            state["page"] = 0
            return httpx.Response(302, headers={"location": "/search.html?0"})
        if req.url.path == "/start.html":
            return httpx.Response(200, html="<html><body><h1>Startseite</h1><p>" + "Willkommen auf der "
                                            "Vergabeplattform. " * 40 + "</p><a href='/hilfe.html'>Hilfe</a>"
                                            "<a href='./start.html?0-1.-showAllProceduresLink'>Alle Ausschreibungen</a>"
                                            "<a href='/kontakt.html'>Kontakt</a>"
                                            "<a href='/impressum.html'>Impressum</a></body></html>")
        if req.url.path == "/search.html" and listener:
            state["page"] = int(listener.group(1))
            return httpx.Response(302, headers={"location": "/search.html?0"})
        if req.url.path == "/search.html":
            return httpx.Response(200, html=page(state["page"], "s1"))
        return httpx.Response(404)
    return h


def test_locate_list_follows_a_session_link_with_referer():
    log: list = []

    async def classify(task, system, user, **kw):
        import json
        d = json.loads(user)
        if "datatable" in d["page"]:
            return {"pageType": "list", "nextLink": None}
        return {"pageType": "landing", "nextLink": next(l["i"] for l in d["links"] if "Alle" in l["text"])}

    async def go():
        client = PoliteClient(HostRateLimiter(0), "t", 5, transport=httpx.MockTransport(wicket_portal(log)))
        ctx = SimpleNamespace(client=client, llm=SimpleNamespace(json_call=classify), note=lambda *a: None,
                              portal_id="p", run_id="r", url=HOST)
        try:
            return await onboard.locate_list(ctx, HOST + "/start.html")
        finally:
            await client.aclose()
    found = asyncio.run(go())
    assert found["url"] == HOST + "/search.html?0" and "Zeige 1 bis 10 von 25" in found["html"]
    listener_hits = [r for r in log if r[1].startswith("0-1.-")]
    assert listener_hits and listener_hits[-1][2].startswith(HOST + "/start.html")   # the page the link was on


def test_next_link_pager_sends_referer_and_reads_every_page(setup):
    client, _ = setup
    log: list = []
    spec = {"tz": "Europe/Berlin",
            "list": {"kind": "html", "url": HOST + "/search.html?0", "items": "#datatable tbody tr",
                     "pagination": {"type": "next_link", "next_sel": "a[rel=next]"},
                     "total": {"sel": "p", "regex": r"von (\d+)"}},
            "fields": {"id": {"sel": "a", "attr": "href", "regex": r"id=([\w-]+)"},
                       "url": {"sel": "a", "attr": "href"}, "title": {"column": "Bezeichnung"},
                       "submissionDeadline": {"column": "Nächste Frist", "formats": ["%d.%m.%y, %H:%M"]}},
            "detail": {"kind": "html", "url_from": "url"}}
    res = run_read(client, wicket_portal(log), spec)
    assert len(res.tenders) == 25 and res.complete, res.stats
    assert res.tenders[0]["submissionDeadline"] == "2030-12-01T09:00:00+01:00"
    pager = [r for r in log if re.match(r"^0-\d+\.-", r[1])]
    assert len(pager) == 2 and all((r[2] or "").startswith(HOST + "/search.html") for r in pager)   # page it was on
