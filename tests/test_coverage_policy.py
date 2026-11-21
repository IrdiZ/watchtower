"""Patterno decisions, 20 Nov: (A) a working integration with incomplete coverage serves, marked 'Coverage: partial';
(B) tenders whose documents sit behind a login are delivered without documents, and the portal says so plainly.
Plus: a row whose link the learned selector misses (a tender hosted on another platform) keeps its own link."""
from selectolax.lexbor import LexborHTMLParser as HTMLParser

from watchtower import runner
from watchtower.config import Settings
from watchtower.pipeline import ReadResult, Verdict
from watchtower.validate import decide


def _res(**stats):
    base = {"capped": False, "stopReason": "empty page", "rows": 10}
    base.update(stats)
    return ReadResult([{"id": "1"}], stats.pop("complete", True), base)


def test_capped_listing_serves_by_default_marked_partial():
    assert Settings().serve_capped is True
    d = decide(Verdict(ok=True), ReadResult([{"id": "1"}], False, {"capped": True, "stopReason": "portal refuses pages beyond 100"}),
               mode="attach", first=True, serve_capped=Settings().serve_capped)
    assert d.commit and d.state == "serving"
    assert any(n.startswith("Coverage: partial") for n in d.notes) and not d.close_missing


def test_complete_read_says_so():
    d = decide(Verdict(ok=True), _res(), mode="read", first=False)
    assert "Coverage: complete." in d.notes


def test_login_gated_documents_are_named_plainly():
    d = decide(Verdict(ok=True), _res(docBlocked=12, docBlockReason="access denied (HTTP 403)", docsFetched=0),
               mode="read", first=False)
    assert d.commit
    assert any(n.startswith("Documents: not available from this portal (access denied") for n in d.notes)
    d2 = decide(Verdict(ok=True), _res(docBlocked=3, docBlockReason="login page", docsFetched=40),
                mode="read", first=False)
    assert any(n.startswith("Documents: partial - 3 documents are blocked") for n in d2.notes)


def test_row_link_fallback():
    row = HTMLParser('<div class="row"><a href="javascript:;//x">Panier</a>'
                     '<a href="https://other.example/consultation/99">Travaux de voirie lot 2</a>'
                     '<a href="mailto:a@b.c">contact</a></div>').css_first("div.row")
    assert runner._row_link(row, "99", "Travaux de voirie lot 2") == "https://other.example/consultation/99"
    two = HTMLParser('<div class="row"><a href="/a">Aide</a><a href="/b">Plan du site</a></div>').css_first("div.row")
    assert runner._row_link(two, "7", "Something else") is None          # several unrelated links: no guessing
    one = HTMLParser('<div class="row"><a href="/t/7">voir</a></div>').css_first("div.row")
    assert runner._row_link(one, None, None) == "/t/7"


def test_small_minority_without_link_is_left_out_and_flagged():
    """maximilien: 13 of 402 rows link out only through a script. Deliver the 389, flag partial, close nothing."""
    d = decide(Verdict(ok=True), _res(droppedIncomplete=13), mode="read", first=False)
    assert d.commit and not d.close_missing
    assert any("13 listed tenders could not be read completely" in n for n in d.notes)
    assert runner.INCOMPLETE_SHARE == 0.05
