"""Documents check, 20 Nov (TenderNed): .docx/.xlsx are ZIP containers and were unpacked into their XML parts
(28 originals missing, 2,643 extra entries); files repeated across a procedure's notices were listed twice."""
from __future__ import annotations

import asyncio
import io
import zipfile

import httpx

from tests.test_contract import setup  # noqa: F401  (fixture)
from tests.test_open_filter import HOST, _ctx, _iso, spec_for, TODAY
from watchtower import onboard, runner
from watchtower.http import DocumentStore, HostRateLimiter, PoliteClient


def _zip(files: dict[str, bytes]) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
    return b.getvalue()


DOCX = _zip({"[Content_Types].xml": b"<Types/>", "word/document.xml": b"<w:document/>", "_rels/.rels": b"<r/>"})
ODT = _zip({"mimetype": b"application/vnd.oasis.opendocument.text", "content.xml": b"<c/>"})
DOSSIER = _zip({"RC.pdf": b"%PDF-1.7 rc", "CCTP.docx": DOCX, "annexes.zip": _zip({"plan.pdf": b"%PDF plan"})})


def _fetch(client, files: dict[str, tuple[bytes, str]]):
    docs = client.app.state.manager.docs

    def handler(req):
        body, name = files[req.url.path]
        return httpx.Response(200, content=body, headers={"content-disposition": f'attachment; filename="{name}"'})

    async def go():
        pc = PoliteClient(HostRateLimiter(0), "tests", 5, transport=httpx.MockTransport(handler))
        try:
            return {p: await docs.fetch(pc, f"{HOST}{p}") for p in files}
        finally:
            await pc.aclose()
    return asyncio.run(go())


def test_office_files_are_delivered_whole_and_dossiers_unpacked(setup):
    client, _ = setup
    got = _fetch(client, {"/a": (DOCX, "Bestek.docx"), "/b": (ODT, "Nota.odt"), "/c": (DOCX, "noext"),
                          "/d": (DOSSIER, "DCE.zip")})
    assert [e.fileName for e in got["/a"]] == ["Bestek.docx"] and got["/a"][0].sizeBytes == len(DOCX)
    assert [e.fileName for e in got["/b"]] == ["Nota.odt"]
    assert len(got["/c"]) == 1                      # recognised by content even without the extension
    names = sorted(e.fileName for e in got["/d"])
    assert names == ["CCTP.docx", "RC.pdf", "plan.pdf"]   # the dossier is unpacked; the .docx inside stays whole


def test_is_plain_archive():
    assert DocumentStore.is_plain_archive(io.BytesIO(DOSSIER), "DCE.zip")
    assert not DocumentStore.is_plain_archive(io.BytesIO(DOCX), "x.zip")
    assert not DocumentStore.is_plain_archive(io.BytesIO(b"%PDF"), "x.pdf")


def test_same_file_on_two_notices_is_listed_once(setup):
    client, _ = setup
    rows = [{"publicatieId": "9002", "kenmerk": "P1", "naam": "B (rect)", "koper": "K",
             "publicatieDatum": TODAY.isoformat(), "sluitingsDatum": _iso(30)},
            {"publicatieId": "9001", "kenmerk": "P1", "naam": "B", "koper": "K",
             "publicatieDatum": TODAY.isoformat(), "sluitingsDatum": _iso(30)}]

    def handler(req):
        path = req.url.path
        if path.endswith("/documenten"):
            n = path.split("/")[-2]
            return httpx.Response(200, json={"documenten": [
                {"links": {"download": {"href": f"/papi/publicaties/{n}/documenten/1/content"}}}]})
        if path.endswith("/content"):           # same bytes behind both notices' links
            return httpx.Response(200, content=b"%PDF same", headers={"content-type": "application/pdf"})
        page = int(req.url.params.get("page", "0"))
        return httpx.Response(200, json={"content": rows if page == 0 else []})
    spec = spec_for()
    spec["list"].pop("total")
    spec["fields"]["id"] = {"path": "kenmerk"}

    async def go():
        ctx, pc = _ctx(client, handler)
        try:
            first = await runner.read(ctx, spec, {}, runner.Limits(fetch_docs=False))
            await onboard._discover_docs_safely(ctx, spec, first.tenders)
            return await runner.read(ctx, spec, {}, runner.Limits())
        finally:
            await pc.aclose()
    [t] = asyncio.run(go()).tenders
    assert len(t["documents"]) == 1
