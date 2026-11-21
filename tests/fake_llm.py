"""A deterministic stand-in for the model, good enough to read the mock portal like a careful LLM would.
It only sees what the real model sees (the prompt payloads), so the plumbing is exercised end to end."""
from __future__ import annotations

import json
import re

BASE = "http://mock.test"


class FakeLLM:
    def __init__(self, store):
        self.store = store
        self.calls: list[str] = []
        self.available = True

    async def aclose(self):
        pass

    async def json_call(self, task, system, user, schema=None, portal_id=None, run_id=None, schema_name="r"):
        self.calls.append(task)
        self.store.log_llm_call(portal_id, run_id, task, "fake", 1000, 200, 0.001)
        payload = user if isinstance(user, str) else user[0]["text"]
        if task == "classify":
            return self._classify(json.loads(payload))
        if task == "synthesize":
            data = json.loads(payload)
            return self._detail(data) if "detailPage" in data else self._list(data)
        if task == "verify":
            return self._verify(payload)
        if task == "narrow":
            return {"candidates": [{"params": {"deadlineFrom": "{today}"}, "why": "mock api deadline filter"}]}
        raise AssertionError(task)

    # ---------------------------------------------------------------- classify
    def _classify(self, d):
        """Page type + the top 3 links, like the real prompt asks: links naming 'Ausschreibungen' first, then
        'Bekanntmachungen', in page order (so a portal can make the first choice a dead end)."""
        page = d["page"]

        def picks(words):
            out = []
            for w in words:
                for l in d["links"]:
                    if w in l["text"] and l["i"] not in [p["i"] for p in out]:
                        out.append({"i": l["i"], "confidence": round(0.9 - 0.2 * len(out), 2), "why": w})
            return out[:3]
        if 'type="password"' in page:
            return {"pageType": "login", "links": picks(["Bekanntmachungen", "Ausschreibungen"]), "reason": "login form"}
        if "tender-row" in page or "ergebnis-zeile" in page or "eintrag" in page or 'class="stream"' in page:
            return {"pageType": "list", "links": [], "reason": "rows of tenders"}
        found = picks(["Ausschreibungen", "Bekanntmachungen"])
        return {"pageType": "landing", "links": found, "reason": "links to tenders" if found else "?"}

    # ---------------------------------------------------------------- list part
    def _list(self, d):
        if "jsonApi" in d and d["jsonApi"].get("method") == "POST":
            return self._list_post(d)
        if "jsonApi" in d:
            sample = d["jsonApi"]["sample"]
            url = d["jsonApi"]["url"]
            renamed = '"results"' in sample
            off = "start" if "start=" in url else "offset"
            k = (lambda a, b: b if renamed else a)
            return {"tz": "Europe/Berlin",
                    "list": {"kind": "json", "url": f"{BASE}/api/tenders", "items": k("items", "results"),
                             "pagination": {"type": "offset", "param": off, "start": 0, "size": 5,
                                            "size_param": "limit"},
                             "total": {"path": "total"}},
                    "fields": {"id": {"path": k("id", "notice_id")}, "title": {"path": k("title", "subject")},
                               "buyerName": {"path": k("buyer", "contracting_authority")},
                               "publicationDate": {"path": k("publishedOn", "publication"),
                                                   "formats": ["%d %B %Y", "%d.%m.%Y"]},
                               "submissionDeadline": {"path": k("deadline", "closing"),
                                                      "formats": ["%d %B %Y, %H:%M", "%d.%m.%Y %H:%M"]},
                               "withdrawn": {"path": k("withdrawn", "cancelled")},
                               "url": {"template": f"{BASE}/app/tender/{{id}}"}},
                    "detail": {"kind": "json", "url_from": "template", "template": f"{BASE}/api/tenders/{{id}}"}}
        page = d["listPage"]
        if "PRADO_PAGESTATE" in page or "__VIEWSTATE" in page:
            return self._postback_list(d)
        if 'class="stream"' in page:
            return self._stream_list(d)
        list_url = d.get("listPageUrl") or f"{BASE}/ausschreibungen"
        self.use_detail_names = any(re.search(r"title|buyer", x) for x in d.get("problemsToFix", []))
        renamed = "ergebnis-zeile" in page
        cards = 'class="card' in page
        param = "p" if "?p=" in page else "page"
        c = (lambda a, b: b if renamed else a)
        row = c("tender-row", "ergebnis-zeile")
        cell = "span" if cards else "td"
        title_sel = (f"h3.{c('title', 'bez')} a" if cards else f"td.{c('title', 'bez')} a")
        buyer_sel = (f"p.{c('buyer', 'vst')}" if cards else f"td.{c('buyer', 'vst')}")
        months = bool(re.search(r"\d{1,2} (January|February|March|April|May|June|July|August|September|October|"
                                r"November|December) \d{4}", page))
        dfmt = ["%d %B %Y, %H:%M"] if months else ["%d.%m.%Y %H:%M"]
        pfmt = ["%d %B %Y"] if months else ["%d.%m.%Y"]
        if 'class="kind"' in page and not self.use_detail_names:
            # the list shows each notice's type (a column "Art"): mapped as noticeType, like the spec doc asks
            return {"tz": "Europe/Berlin",
                    "list": {"kind": "html", "url": list_url, "items": f"tr.{row}",
                             "pagination": {"type": "page", "param": param, "start": 1},
                             "total": {"sel": "p.total", "regex": r"(\d+)"}},
                    "fields": {"id": {"sel": "td.nr"}, "title": {"sel": title_sel},
                               "url": {"sel": title_sel, "attr": "href"}, "buyerName": {"sel": buyer_sel},
                               "publicationDate": {"sel": f"td.{c('published', 'veroeff')}", "formats": pfmt},
                               "submissionDeadline": {"sel": f"td.{c('deadline', 'frist')}", "formats": dfmt},
                               "noticeType": {"column": "Art"}},
                    "detail": {"kind": "html", "url_from": "url"}}
        return {"tz": "Europe/Berlin",
                "list": {"kind": "html", "url": list_url,
                         "items": f"{'div' if cards else 'tr'}.{row}",
                         "pagination": {"type": "page", "param": param, "start": 1},
                         "total": {"sel": "p.total", "regex": r"(\d+)"}},
                "fields": {"id": {"sel": f"{cell}.nr"},
                           "title": {"sel": title_sel},
                           "url": {"sel": title_sel, "attr": "href"},
                           "buyerName": {"sel": buyer_sel},
                           "publicationDate": {"sel": f"{cell}.{c('published', 'veroeff')}", "formats": pfmt},
                           "submissionDeadline": {"sel": f"{cell}.{c('deadline', 'frist')}", "formats": dfmt}},
                "detail": {"kind": "html", "url_from": "url"}} if not self.use_detail_names else {
                "tz": "Europe/Berlin",
                "list": {"kind": "html", "url": list_url, "items": f"{'div' if cards else 'tr'}.{row}",
                         "pagination": {"type": "page", "param": param, "start": 1},
                         "total": {"sel": "p.total", "regex": r"(\d+)"}},
                "fields": {"id": {"sel": f"{cell}.nr"}, "url": {"sel": title_sel, "attr": "href"},
                           "publicationDate": {"sel": f"{cell}.{c('published', 'veroeff')}", "formats": pfmt},
                           "submissionDeadline": {"sel": f"{cell}.{c('deadline', 'frist')}", "formats": dfmt}},
                "detail": {"kind": "html", "url_from": "url"}}

    def _postback_list(self, d):
        """First the mistake a real model made on Atexo (the javascript: href as a next link); once the feedback
        names postback, the postback pager."""
        self.use_detail_names = False
        problems, old = d.get("problemsToFix", []), json.dumps(d.get("previousSpecThatBroke") or {})
        fixed = any("postback" in x for x in problems) or '"postback"' in old
        pag = {"type": "postback" if fixed else "next_link", "next_sel": "a#ctl0_pager_next"}
        # rows linking to another platform are only seen once a row without url is shown as evidence
        extern = any("row without url" in x and "extern.test" in x for x in problems) or "a.extern" in old
        url_sel = "td.title a.intern, td.title a.extern" if extern else "td.title a.intern"
        return {"tz": "Europe/Berlin",
                "list": {"kind": "html", "url": f"{BASE}/ausschreibungen", "items": "tr.tender-row", "pagination": pag,
                         "total": {"sel": "p.total", "regex": r"(\d+)"}},
                "fields": {"id": {"sel": "td.nr"}, "title": {"sel": "td.title a"},
                           "url": {"sel": url_sel, "attr": "href"}, "buyerName": {"sel": "td.buyer"},
                           "publicationDate": {"sel": "td.published", "formats": ["%d.%m.%Y"]},
                           "submissionDeadline": {"sel": "td.deadline", "formats": ["%d.%m.%Y %H:%M"]}},
                "detail": {"kind": "html", "url_from": "url"}}

    def _stream_list(self, d):
        """Rows a script appends (mock /xhr/liste). The reduced page shows no pager, so first the mistake a real model
        made (pagination none); once the feedback quotes the script, its request - with the CSRF token copied as
        the page showed it (onboarding turns it into a per-visit token)."""
        self.use_detail_names = False
        note = " ".join(d.get("problemsToFix", []))
        pag = {"type": "none"}
        if "load_more" in note:
            tok = re.search(r"CSRF_TOKEN: '(\w+)'", note)
            pag = {"type": "load_more", "url": "/xhr/api/more", "method": "POST", "html_path": "items",
                   "fields": {"offset": "{offset}", "filter": 0, "search": "", "form": {"orderBy": "created desc"},
                              "CSRF_TOKEN": tok.group(1) if tok else ""}}
        f = {"id": {"sel": "span.nr"}, "title": {"sel": "span.title a"},
             "url": {"sel": "span.title a", "attr": "href"}, "buyerName": {"sel": "span.buyer"},
             "publicationDate": {"sel": "span.published", "formats": ["%d.%m.%Y"]},
             "submissionDeadline": {"sel": "span.deadline", "formats": ["%d.%m.%Y %H:%M"]}}
        lst = {"kind": "html", "url": d.get("listPageUrl"), "items": "ul.stream > li[id^='tender-']",
               "pagination": pag, "total": {"sel": "div.hits", "regex": r"hit:\s*([\d.]+)"}}
        if pag["type"] == "load_more":
            del lst["total"]            # the live model dropped the total selector while rewriting the pager
        return {"tz": "Europe/Berlin", "list": lst, "fields": f, "detail": {"kind": "html", "url_from": "url"}}

    def _list_post(self, d):
        """A POST search (mock skin spa_post): the model reads the page field's path from the request body. Like a
        sloppy model it leaves out list.method and list.body - onboarding must keep the captured request."""
        body = d["jsonApi"].get("requestBody") or {}
        assert "page" in body, body
        return {"tz": "Europe/Berlin",
                "list": {"kind": "json", "url": d["jsonApi"]["url"], "items": "hits",
                         "pagination": {"type": "page", "body_path": "page.number", "start": 0,
                                        "size_body_path": "page.size", "size": 5},
                         "total": {"path": "totalHits"}},
                "fields": {"id": {"path": "id"}, "title": {"path": "title"}, "buyerName": {"path": "buyer"},
                           "publicationDate": {"path": "publishedOn"},
                           "submissionDeadline": {"path": "deadline"},
                           "withdrawn": {"path": "withdrawn"},
                           "url": {"template": f"{BASE}/app/tender/{{id}}"}},
                "detail": {"kind": "json", "url_from": "template", "template": f"{BASE}/api/tenders/{{id}}"}}

    # ---------------------------------------------------------------- detail part
    def _detail(self, d):
        if d["listPart"]["detail"].get("kind") == "json":
            return {"fields": {"referenceNumber": {"path": "reference"}, "cpvCodes": {"path": "cpv"},
                               "placeOfPerformance.postalCode": {"path": "place.zip"},
                               "placeOfPerformance.city": {"path": "place.city"},
                               "placeOfPerformance.countryCode": {"path": "place.country"}},
                    "documents": {"from": "detail", "path": "documents[*].href"}, "block_markers": []}
        names = {"title": {"sel": "h2.tender-title"}, "buyerName": {"sel": "dd.buyer"}} \
            if getattr(self, "use_detail_names", False) else {}
        return {"fields": {**names, "referenceNumber": {"sel": "dd.ref"}, "cpvCodes": {"sel": "dd.cpv"},
                           "procedureType": {"sel": "dd.procedure"},
                           "placeOfPerformance.postalCode": {"sel": "dd.place", "regex": r"(\d{5})"},
                           "placeOfPerformance.city": {"sel": "dd.place", "regex": r"\d{5}\s+([^,]+)"},
                           "withdrawn": {"sel": "p.status", "regex": "(?i)aufgehoben"}},
                "documents": {"from": "detail", "sel": "ul.docs a.doc", "attr": "href"},
                "block_markers": ["form[action='/login']"]}

    # ---------------------------------------------------------------- blind verification
    def _verify(self, text):
        try:
            j = json.loads(text)
            return {"id": j.get("id") or j.get("notice_id"), "title": j.get("title") or j.get("subject"),
                    "buyerName": j.get("buyer") or j.get("contracting_authority"),
                    "publicationDate": j.get("publishedOn"), "submissionDeadline": j.get("deadline")}
        except json.JSONDecodeError:
            pass
        # page_text is one line: "<nav> <title> [Aufgehoben] Vergabe-Nr. <id> Aktenzeichen <ref> Vergabestelle ..."
        labels = ["Vergabe-Nr.", "Aktenzeichen", "Vergabestelle", "Veröffentlicht", "Angebotsfrist", "CPV", "Ort",
                  "Verfahren", "Vergabeunterlagen", "Powered by"]
        alt = "|".join(re.escape(x) for x in labels)

        def after(label):
            hit = re.search(rf"(?:^|\s){re.escape(label)}\s+(.*?)(?=\s(?:{alt})(?:\s|$)|$)", text)
            return hit.group(1).strip() or None if hit else None
        head = text.split(" Vergabe-Nr. ")[0] if " Vergabe-Nr. " in f" {text} " else None
        title = None
        if head is not None:
            title = head.split("Aktuelle Ausschreibungen", 1)[-1].strip()
            title = re.sub(r"\s*Aufgehoben$", "", title).strip() or None
        return {"id": after("Vergabe-Nr."), "title": title, "buyerName": after("Vergabestelle"),
                "publicationDate": after("Veröffentlicht"), "submissionDeadline": after("Angebotsfrist")}
