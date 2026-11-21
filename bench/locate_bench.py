"""Locate-walk experiment: run the REAL onboard.locate_list from each portal's homepage and compare where it lands
with the known open-tender list (manual check + teammates' sheet). Isolated store; no portal state is touched.
Usage (inside a watchtower container): python /tmp/locate_bench.py /tmp/portals.json /tmp/out.jsonl [timeout_s]
Each result is appended to the output file as one JSON line as soon as that portal finishes (a hung portal cannot
hide the others); a portal that takes longer than timeout_s (default 240) is recorded as outcome "timeout"."""
import asyncio
import faulthandler
import json
import sqlite3
import sys
import tempfile
import time
from urllib.parse import urlsplit

from watchtower import onboard
from watchtower.config import Settings
from watchtower.http import BudgetExceeded, DocumentStore, HostRateLimiter, PoliteClient
from watchtower.llm import LLMRouter
from watchtower.pipeline import AttachFailed, JobContext, PortalBlocked
from watchtower.store import Store


def matches(found: str | None, truths: list[str]) -> bool:
    if not found:
        return False
    f = urlsplit(found)
    for t in truths:
        u = urlsplit(t)
        if (f.hostname or "").removeprefix("www.") != (u.hostname or "").removeprefix("www."):
            continue
        tp, fp = u.path.rstrip("/"), f.path.rstrip("/")
        key = [kv for kv in u.query.split("&") if kv and not kv.startswith(("searchType", "lang"))][:1]
        if tp and (tp in fp or (fp and fp in tp and len(fp) > 1)):
            if not key or key[0].split("=")[0] in f.query:
                return True
        # a list addressed by its query on the site root ('/?page=Search&AllCons'): same root path + the query key
        if not tp and not fp and key and key[0] in f.query.split("&"):
            return True
    return False


def llm_usage(tmp: str, name: str) -> dict:
    db = sqlite3.connect(f"{tmp}/watchtower.db")
    try:
        row = db.execute("select count(*), coalesce(sum(usd),0), coalesce(sum(tokens_in),0) from llm_calls "
                         "where portal_id=?", ("bench_" + name,)).fetchone()
    finally:
        db.close()
    return {"llmCalls": row[0], "llmUsd": round(row[1], 5), "tokensIn": row[2]}


async def one(p, settings, store, limiter, docs, llm, sem, tmp, timeout, out_path, lock):
    async with sem:
        t0 = time.monotonic()
        client = PoliteClient(limiter, settings.user_agent, settings.http_timeout, 150)
        ctx = JobContext("bench_" + p["name"], p["start"], "bench_" + p["name"], client, docs, llm, store)
        ctx.mode = "attach"
        out = {"name": p["name"], "start": p["start"], "truth": p["truth"]}
        try:
            found = await asyncio.wait_for(onboard.locate_list(ctx, p["start"]), timeout)
            api = found.get("api")
            out.update(outcome="found", found=found["url"], api=api[0] if api else None)
            out["correct"] = matches(found["url"], p["truth"]) or (api is not None and matches(api[0], p["truth"]))
            if found.get("candidates"):
                out["candidates"] = found["candidates"]
                out["tier"] = found.get("tier")
                out["evidence"] = found.get("evidence")
            walk = found.get("walk")
            if walk is not None:
                out["used"] = dict(walk.used)
                out["trace"] = walk.trace[:40]
        except asyncio.TimeoutError:
            out.update(outcome="timeout", error=f"no answer within {timeout}s")
        except PortalBlocked as e:
            out.update(outcome="blocked", error=str(e)[:300])
        except AttachFailed as e:
            out.update(outcome="dead_end", error=str(e)[:300])
        except BudgetExceeded as e:
            out.update(outcome="budget", error=str(e)[:200])
        except Exception as e:
            out.update(outcome="error", error=f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            await client.aclose()
        out["seconds"] = round(time.monotonic() - t0, 1)
        out["requests"] = client.requests
        out["log"] = [l[:400] for l in ctx.log if "request types" not in l][:80]
        out.update(llm_usage(tmp, p["name"]))
        async with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
        print(f"{out['name']:<22} {out['outcome']:<9} correct={out.get('correct')!s:<5} calls={out['llmCalls']} "
              f"${out['llmUsd']:.4f} {out['seconds']}s req={out['requests']} -> "
              f"{out.get('api') or out.get('found') or out.get('error', '')}"[:230], flush=True)
        return out


async def main():
    portals = json.load(open(sys.argv[1], encoding="utf-8"))
    out_path = sys.argv[2]
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 240.0
    open(out_path, "w").close()
    # A synchronous stall cannot be cancelled by asyncio: past the whole run's worst case, dump every stack and exit.
    # Results already written stay in the file; a missing portal is a hang.
    faulthandler.dump_traceback_later(timeout * (len(portals) // 5 + 1) + 180, exit=True)
    tmp = tempfile.mkdtemp(prefix="locbench-")
    settings = Settings(data_dir=tmp)
    store = Store(tmp)
    limiter = HostRateLimiter(settings.min_request_interval)
    docs = DocumentStore(store, settings.max_document_bytes, settings.max_zip_total_bytes)
    llm = LLMRouter(settings, store)
    sem = asyncio.Semaphore(5)
    lock = asyncio.Lock()
    try:
        results = await asyncio.gather(*(one(p, settings, store, limiter, docs, llm, sem, tmp, timeout, out_path, lock)
                                         for p in portals))
    finally:
        await llm.aclose()
    ok = sum(1 for r in results if r.get("correct"))
    print(f"TOTAL correct {ok}/{len(results)}; outcomes "
          f"{ {o: sum(1 for r in results if r['outcome'] == o) for o in sorted({r['outcome'] for r in results})} }; "
          f"${sum(r['llmUsd'] for r in results):.4f}; {sum(r['seconds'] for r in results):.0f}s total", flush=True)


asyncio.run(main())
