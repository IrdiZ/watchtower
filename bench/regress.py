"""Live regression check, run after every merge + rebuild.

For every portal in bench/live-baseline.json: refresh it (a static read, no model calls), wait for the run, and check
  - still serving, last run ok, read complete and not capped;
  - tender count within TOLERANCE of the last good count (open tenders come and go daily);
  - no served tender without id/url/title; no "open" tender whose deadline has passed;
  - documents look like documents (no Office package parts such as [Content_Types].xml).
Exit code 1 if any portal fails. `--update` writes the new counts as the baseline when everything passes.
Stdlib only:  python bench/regress.py [--api http://localhost:8080] [--update] [--no-refresh]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASELINE = Path(__file__).with_name("live-baseline.json")
TOLERANCE = 0.05
OOXML_PARTS = {"[Content_Types].xml", ".rels", "document.xml", "workbook.xml", "styles.xml", "settings.xml"}


def call(api: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(api + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def check(api: str, base: dict, refresh: bool, wait_s: int) -> list[dict]:
    portals = {p["url"]: p for p in call(api, "/portals")}
    started = {}
    for url in base:
        p = portals.get(url)
        if p and refresh:
            started[url] = call(api, f"/portals/{p['id']}/refresh", "POST")["runId"]
    deadline = time.time() + wait_s
    while refresh and time.time() < deadline:
        pending = [u for u, rid in started.items()
                   if not any(r["id"] == rid and r["finishedAt"] for r in call(api, f"/portals/{portals[u]['id']}/runs"))]
        if not pending:
            break
        time.sleep(15)
    out = []
    now = datetime.now(timezone.utc)
    for url, want in base.items():
        res = {"url": url, "baseline": want["tenders"], "problems": []}
        p = portals.get(url)
        if not p:
            res["problems"].append("portal missing")
            out.append(res)
            continue
        ui = call(api, f"/_ui/portals/{p['id']}")
        runs_ = ui["runs"] or [{}]
        # a refresh still running is not a verdict: judge the latest FINISHED run (unless we started one and waited)
        run = runs_[0] if (refresh and url in started) else next((r for r in runs_ if r.get("finishedAt")), runs_[0])
        portal = ui["portal"]
        st = (run.get("detail") or {}).get("stats") or {}
        res.update(state=portal["state"], tenders=portal["tenderCount"], run=run.get("result"),
                   stop=st.get("stopReason"))
        if refresh and url in started and run.get("id") != started[url]:
            res["problems"].append("refresh did not finish in time")
        if portal["state"] != "serving":
            res["problems"].append(f"state {portal['state']}: {portal['reason'][:160]}")
        if run.get("result") != "ok":
            res["problems"].append(f"last run {run.get('result')}: {run.get('summary', '')[:160]}")
        if st.get("capped") and not want.get("allowPartial"):   # Patterno rule A: partial may serve, flagged
            res["problems"].append("read capped (partial listing)")
        n, b = portal["tenderCount"], want["tenders"]
        tol = float(want.get("tolerance", TOLERANCE))      # a change feed in backfill grows on every read
        if b and abs(n - b) / b > tol:
            res["problems"].append(f"count {n} vs baseline {b} (> {tol:.0%} change)")
        tenders = call(api, f"/portals/{p['id']}/tenders")
        for f in ("id", "url", "title"):
            k = sum(1 for t in tenders if not t.get(f))
            if k:
                res["problems"].append(f"{k} tender(s) without {f}")
        late = sum(1 for t in tenders if t.get("status") == "open" and t.get("submissionDeadline")
                   and parse_ts(t["submissionDeadline"]) <= now)
        if late:
            res["problems"].append(f"{late} 'open' tender(s) past their deadline")
        parts = sum(1 for t in tenders for d in t.get("documents") or [] if d.get("fileName") in OOXML_PARTS)
        if parts:
            res["problems"].append(f"{parts} document entries are Office package parts")
        res["documents"] = sum(len(t.get("documents") or []) for t in tenders)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8080")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--no-refresh", action="store_true")
    ap.add_argument("--wait", type=int, default=1200)
    a = ap.parse_args()
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    results = check(a.api, base, not a.no_refresh, a.wait)
    bad = [r for r in results if r["problems"]]
    for r in results:
        mark = "FAIL" if r["problems"] else "PASS"
        print(f"{mark}  {r['url'][:58]:<58} {r.get('tenders', '-'):>6} (baseline {r['baseline']}) "
              f"docs={r.get('documents', '-')} stop={r.get('stop')}")
        for pr in r["problems"]:
            print(f"        - {pr}")
    print(f"\n{len(results) - len(bad)}/{len(results)} portals pass")
    if a.update and not bad:
        BASELINE.write_text(json.dumps({r["url"]: {"tenders": r["tenders"],
                                                   **({"allowPartial": True} if base[r["url"]].get("allowPartial") else {})}
                                         for r in results}, indent=1) + "\n", encoding="utf-8")
        print("baseline updated")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
