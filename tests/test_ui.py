"""UI wiring that doesn't require loading a model.

The full request flow (start -> poll -> test) is exercised end-to-end over HTTP
against a tiny local model in the offline smoke test; here we cover the pure
logic: presets, input validation, and the served page.
"""

from abliterate import ui


def test_presets_present():
    assert set(ui.PRESETS) == {"quick", "balanced", "thorough"}
    for preset in ui.PRESETS.values():
        assert preset["n_trials"] >= 1
        assert "n_eval_harmful" in preset and "n_eval_harmless" in preset


def test_index_html_is_self_contained():
    html = ui.INDEX_HTML
    assert "<!doctype html>" in html.lower()
    assert "/api/start" in html and "/api/status" in html and "/api/test" in html
    assert "Remove refusals" in html
    # No external asset references (must work offline / no CSP surprises).
    assert "http://" not in html.replace("http://127.0.0.1", "").replace("http://{host}", "")
    assert "https://" not in html


def test_start_job_requires_model():
    ok, msg = ui._start_job({"model": "   "})
    assert ok is False
    assert "model" in msg.lower()
    assert ui.JOB.state != "running"


def test_test_prompt_requires_finished_job():
    result = ui._test_prompt("hello")
    assert "error" in result
