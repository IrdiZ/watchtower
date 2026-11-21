"""Regression tests for incomplete reads, concurrent repair, and document boundaries."""

import asyncio
import hashlib
from datetime import datetime, timezone

import httpx
import pytest

from tests.test_contract import setup, tender, wait_runs_done, wait_state
from watchtower import timeutil
from watchtower.http import HostRateLimiter, PoliteClient


def test_partial_refresh_leaves_unseen_tender_open_until_complete_read(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://paged.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    tenders_url = f"/portals/{pid}/tenders"
    before = {item["id"]: item for item in client.get(tenders_url).json()}

    fake.tenders = [tender(i) for i in range(4)]
    fake.complete = False
    assert client.post(f"/portals/{pid}/refresh").status_code == 202
    wait_runs_done(client, pid)

    partial = {item["id"]: item for item in client.get(tenders_url).json()}
    assert partial["T-4"] == before["T-4"]

    fake.complete = True
    for _ in range(2):      # a tender closes after missing from two complete reads in a row
        assert client.post(f"/portals/{pid}/refresh").status_code == 202
        wait_runs_done(client, pid)

    complete = {item["id"]: item for item in client.get(tenders_url).json()}
    assert complete["T-4"]["status"] == "closed"


def test_refresh_during_repair_returns_active_repair_run(setup):
    client, fake = setup
    pid = client.post("/portals", json={"url": "https://repairing.example"}).json()["id"]
    wait_state(client, pid, {"serving"})

    fake.read_ok = False
    release_repair = asyncio.Event()
    original_repair = fake.repair

    async def paused_repair(ctx, old_spec, failures):
        await release_repair.wait()
        return await original_repair(ctx, old_spec, failures)

    fake.repair = paused_repair
    broken_read = client.post(f"/portals/{pid}/refresh")
    try:
        wait_state(client, pid, {"healing"})
        runs = client.get(f"/portals/{pid}/runs").json()
        active = [run for run in runs if run["finishedAt"] is None]
        assert len(active) == 1
        assert active[0]["kind"] == "repair"
        assert active[0]["result"] is None
        assert active[0]["id"] != broken_read.json()["runId"]

        for _ in range(2):
            response = client.post(f"/portals/{pid}/refresh")
            assert response.status_code == 202
            assert response.json() == {"runId": active[0]["id"]}
        assert client.get(f"/portals/{pid}/runs").json() == runs
    finally:
        client.portal.call(release_repair.set)

    wait_state(client, pid, {"serving"})
    assert fake.repairs == 1


def test_deadline_crossing_changes_status_without_a_refresh(setup, monkeypatch):
    client, fake = setup
    now = datetime(2026, 11, 20, 8, 59, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(timeutil, "now", lambda: now)
    deadline = "2026-11-20T10:00:00+01:00"
    fake.tenders = [
        tender(0, submissionDeadline=deadline),
        tender(1, submissionDeadline=deadline, status="withdrawn"),
    ]
    pid = client.post("/portals", json={"url": "https://deadline.example"}).json()["id"]
    wait_state(client, pid, {"serving"})
    tenders_url = f"/portals/{pid}/tenders"
    before = {item["id"]: item for item in client.get(tenders_url).json()}
    assert before["T-0"]["status"] == "open"
    assert before["T-1"]["status"] == "withdrawn"

    just_before = now.isoformat()
    now = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)
    changed = client.get(tenders_url, params={"changedSince": just_before}).json()
    assert [(item["id"], item["status"]) for item in changed] == [("T-0", "closed")]
    assert timeutil.parse(changed[0]["lastChangedAt"]) == now
    assert client.get(tenders_url, params={"changedSince": deadline}).json() == []
    after = {item["id"]: item for item in client.get(tenders_url).json()}
    assert after["T-1"] == before["T-1"]
    assert fake.reads == 1


class DownloadStream(httpx.AsyncByteStream):
    def __init__(self, chunks, *, interrupted=False):
        self.chunks = chunks
        self.interrupted = interrupted
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.interrupted:
            raise httpx.ReadError("connection lost during download")

    async def aclose(self):
        self.closed = True


async def fetch_document(docs, stream):
    def handle_request(request):
        return httpx.Response(200, stream=stream)

    client = PoliteClient(
        HostRateLimiter(0), "Watchtower tests", 5,
        transport=httpx.MockTransport(handle_request),
    )
    try:
        return await docs.fetch(client, "https://files.example/notice.pdf")
    finally:
        await client.aclose()


@pytest.mark.parametrize("extra_bytes", [b"", b"x"], ids=["at-cap", "over-cap"])
def test_document_without_content_length_respects_byte_cap(setup, monkeypatch, extra_bytes):
    client, _ = setup
    docs = client.app.state.manager.docs
    chunk = b"x" * 65_536
    monkeypatch.setattr(docs, "max_bytes", len(chunk))
    stream = DownloadStream([chunk, extra_bytes])
    entries = asyncio.run(fetch_document(docs, stream))

    assert stream.closed
    digest = hashlib.sha256(chunk).hexdigest()
    if extra_bytes:
        assert entries == []
        assert client.get(f"/files/{digest}").status_code == 404
    else:
        assert [entry.as_dict() for entry in entries] == [{
            "fileName": "notice.pdf",
            "sizeBytes": len(chunk),
            "sha256": digest,
            "path": f"/files/{digest}",
        }]
        result = client.get(entries[0].path)
        assert result.status_code == 200
        assert result.content == chunk


def test_interrupted_download_discards_partial_file(setup, tmp_path):
    client, _ = setup
    docs = client.app.state.manager.docs
    stream = DownloadStream([b"x" * 65_536], interrupted=True)

    with pytest.raises(httpx.ReadError, match="connection lost"):
        asyncio.run(fetch_document(docs, stream))

    assert stream.closed
    assert not list((tmp_path / "files").iterdir()), "download left partial bytes on disk"

