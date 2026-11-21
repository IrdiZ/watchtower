"""Small units: JS-shell detection, evidence-based blocks, JSON candidate scoring, document templates."""
from watchtower.onboard import _evidence_of_block, _js_shell, _score_json


def test_js_shell():
    assert _js_shell("<html><head><title>Aankondigingen</title></head><body><app-root></app-root></body></html>")
    assert not _js_shell("<body><a href=/a>A</a><a href=/b>B</a><a href=/c>Aktuelle Ausschreibungen</a>"
                         "<p>Willkommen beim Vergabeportal, hier finden Sie alle Ausschreibungen.</p></body>")


def test_block_needs_evidence():
    assert _evidence_of_block("<body><app-root></app-root></body>", 200) is None
    assert "login" in _evidence_of_block("<form><input type=password></form>", 200)
    assert _evidence_of_block("", 403)
    long_page = "<body><form><input type=password></form>" + "<p>tender text</p>" * 300 + "</body>"
    assert _evidence_of_block(long_page, 200) is None  # login box in the header of a public list


def test_score_json():
    tenders = {"content": [{"publicatieId": 1, "aanbestedingNaam": "x", "sluitingsDatum": "2026"}] * 3}
    assert _score_json(tenders) > _score_json({"menu": [{"label": "a"}, {"label": "b"}]})
    assert _score_json({"ok": True}) == 0
