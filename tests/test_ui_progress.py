"""The UI worker must report many distinct phases (not sit on one string).

The heavy pipeline pieces (model load, generation, optimization) are mocked so
the test exercises only the worker's progress plumbing.
"""

from __future__ import annotations

from abliterate import ui


class _FakeBundle:
    pass


def test_worker_emits_many_distinct_phases(monkeypatch, tmp_path):
    phases: list[str] = []
    indeterminate_seen = {"value": False}

    def fake_load(cfg, *, on_progress=None):
        if on_progress is not None:
            on_progress("Resolving tokenizer", 0.0)
            on_progress("Downloading weights", None)  # indeterminate
            on_progress("Loading into memory", 0.7)
            on_progress("Model ready", 1.0)
        return _FakeBundle()

    def fake_generate(bundle, prompts, max_new_tokens=48, batch_size=1):
        return ["I cannot help with that." for _ in prompts]

    def fake_is_refusal(text):
        return True

    def fake_optimize(cfg, bundle=None, *, on_trial=None, on_progress=None, save="model"):
        if on_progress is not None:
            on_progress("Preparing prompts", 0.0)
            on_progress("Collecting harmful activations", 0.0)
            on_progress("Computing refusal directions", 0.8)
            on_progress("Measuring baseline refusals", 0.65)
            on_progress("Capturing reference distribution", 0.0)
            on_progress("Reference distribution captured", 1.0)
        if on_trial is not None:
            on_trial(1, 2)
            on_trial(2, 2)
        return tmp_path

    monkeypatch.setattr("abliterate.model_utils.load_model_and_tokenizer", fake_load)
    monkeypatch.setattr("abliterate.evaluate.generate_completions", fake_generate)
    monkeypatch.setattr("abliterate.evaluate.is_refusal", fake_is_refusal)
    monkeypatch.setattr("abliterate.pipeline.optimize_and_apply", fake_optimize)

    job = ui.Job()
    original_set = job.set

    def spy_set(**kw):
        if "phase" in kw:
            phases.append(kw["phase"])
        if kw.get("indeterminate"):
            indeterminate_seen["value"] = True
        original_set(**kw)

    job.set = spy_set  # type: ignore[method-assign]

    ui._run_job(job, {"model": "fake/model", "thoroughness": "quick"})

    assert job.state == "done", job.error
    assert len(set(phases)) > 3, phases
    # The unbounded weight-download step must have flipped the spinner on.
    assert indeterminate_seen["value"] is True
    # Progress must end at 100%.
    assert job.progress == 1.0


def test_ollama_style_model_id_yields_friendly_error():
    ok, msg = ui._start_job({"model": "Qwen3:14B"})
    assert ok is False
    assert "Hugging Face model ID" in msg
    assert "Qwen3:14B" in msg


def test_model_input_error_accepts_hf_id_and_local_path(tmp_path):
    assert ui._model_input_error("Qwen/Qwen3-14B") is None
    local = tmp_path / "my-model"
    local.mkdir()
    assert ui._model_input_error(str(local)) is None
    assert ui._model_input_error("") is not None
    assert ui._model_input_error("not a repo id") is not None
