"""The auto ("set and forget") planner must decide from facts, not guesses.

Each test pins one decision the novice flow depends on:

* the data split always leaves evaluation prompts OUT of the direction fit,
* the trial count bends to the measured generation speed, not the preset,
* precision matches the checkpoint size against free VRAM -- and never picks
  4-bit when a LoRA merge still has to happen (quantized weights cannot merge).
"""
from __future__ import annotations

import pytest

from abliterate import autosetup
from abliterate.autosetup import (
    Calibration,
    apply_precision,
    model_disk_bytes,
    plan_split,
    scale_trials,
)
from abliterate.config import Config, DataConfig, ModelConfig, OptimizeConfig


# --------------------------------------------------------------------------- #
# Data plan
# --------------------------------------------------------------------------- #
def test_eval_is_always_held_out_of_the_direction_fit():
    """The exact shape from the warning log: 160 prompts, want 160 dir + 48 eval.

    The naive split would exhaust the hold-out. The plan must shrink the
    direction fit instead so evaluation scores unseen prompts.
    """
    n_dir, n_eval, _ = plan_split(160, want_dir=160, want_eval=48)
    assert n_dir == 120
    assert n_eval == 40
    assert n_dir + n_eval <= 160, "plan overcommits the file"


def test_split_never_dips_below_a_usable_evaluation_set():
    """Tiny files keep at least _MIN_EVAL_PROMPTS prompts for scoring."""
    n_dir, n_eval, _ = plan_split(30, want_dir=64, want_eval=32)
    assert n_eval >= autosetup._MIN_EVAL_PROMPTS
    assert n_dir + n_eval <= 30


def test_small_files_note_that_a_bigger_file_would_help():
    n_dir, _n_eval, notes = plan_split(30, want_dir=64, want_eval=32)
    assert n_dir < autosetup._MIN_DIRECTION_PROMPTS
    assert notes and "larger prompt file" in notes[0]


def test_data_plan_sizes_both_sides_and_never_overcommits(tmp_path, monkeypatch):
    """End to end on config objects: 160/156 available, thorough targets."""
    _make_prompt_file(tmp_path, "harmful.txt", 160)
    _make_prompt_file(tmp_path, "harmless.txt", 156)
    cfg = _config(tmp_path, n_samples=160, n_eval_harmful=48, n_eval_harmless=48)

    autosetup.apply_data_plan(cfg)

    assert cfg.data.n_samples == 117          # min(120, 117)
    assert cfg.optimize.n_eval_harmful == 40  # capped from 48
    assert cfg.optimize.n_eval_harmless == 39
    assert cfg.data.n_samples + cfg.optimize.n_eval_harmful <= 160
    assert cfg.data.n_samples + cfg.optimize.n_eval_harmless <= 156


def test_data_plan_upgrades_a_too_small_file_to_the_builtin_dataset(tmp_path, monkeypatch):
    """A file too small to split at all should not silently self-score."""
    _make_prompt_file(tmp_path, "harmful.txt", 10)
    _make_prompt_file(tmp_path, "harmless.txt", 10)
    cfg = _config(tmp_path)

    calls = []

    def fake_upgrade(source, default_file, builtin):
        calls.append(builtin)
        return builtin, f"switched to {builtin!r}"

    monkeypatch.setattr(autosetup, "_maybe_upgrade_source", fake_upgrade)
    # apply_data_plan reads counts through load_prompts; stub the sources to
    # point at names the fake upgrade returned, with believable counts.
    monkeypatch.setattr(
        "abliterate.data.load_prompts",
        lambda source, *, default_file: [f"p{i}" for i in range(520)],
    )

    autosetup.apply_data_plan(cfg)

    assert calls == ["advbench", "harmless_alpaca"]


# --------------------------------------------------------------------------- #
# Token budget + trials
# --------------------------------------------------------------------------- #
def test_calibration_doubles_until_truncation_drops(monkeypatch):
    """A model that truncates everything at 48 tokens gets a bigger budget."""
    rates = {48: 1.0, 96: 1.0, 192: 0.0}
    seen = []

    def fake_generate(bundle, prompts, *, max_new_tokens, batch_size):
        seen.append(max_new_tokens)
        return ["x" * 10] * len(prompts), [rates[max_new_tokens]] * len(prompts)

    monkeypatch.setattr(
        "abliterate.evaluate.generate_completions_with_stats", fake_generate)

    calib = autosetup.calibrate_token_budget(object(), [f"p{i}" for i in range(4)])

    assert seen == [48, 96, 192]
    assert calib.budget == 192
    assert calib.truncated_rate == 0.0


def test_calibration_stops_immediately_for_a_well_behaved_model(monkeypatch):
    def fake_generate(bundle, prompts, *, max_new_tokens, batch_size):
        return ["done"] * len(prompts), [False] * len(prompts)

    monkeypatch.setattr(
        "abliterate.evaluate.generate_completions_with_stats", fake_generate)

    calib = autosetup.calibrate_token_budget(object(), [f"p{i}" for i in range(4)])

    assert calib.budget == autosetup._TOKEN_BUDGET_START


def test_speed_is_only_trusted_from_a_truncation_dominated_probe():
    """A probe whose completions finish early under-reports generated tokens.

    Trusting it would flatter the machine and over-fit the trial count.
    """
    finished = Calibration(budget=48, truncated_rate=0.0, seconds=1.0, n_prompts=6)
    truncated = Calibration(budget=48, truncated_rate=1.0, seconds=6.0, n_prompts=6)
    assert finished.tokens_per_second == 0.0
    assert truncated.tokens_per_second == pytest.approx(48.0)


def test_trials_bend_to_the_measured_speed_and_wall_clock_budget():
    # 48 tokens/s measured; 40 eval prompts x 96-token budget x 1.6 overhead
    # = ~64 s per trial. A 20-minute quick budget fits ~18 trials.
    calib = Calibration(budget=96, truncated_rate=1.0, seconds=6.0, n_prompts=6)
    fitted, notes = scale_trials(150, calib, n_eval=40, time_budget_minutes=20)
    assert fitted == 18
    assert notes and "trials 150 -> 18" in notes[0]


def test_trials_keep_a_floor_on_very_slow_machines():
    calib = Calibration(budget=512, truncated_rate=1.0, seconds=600.0, n_prompts=6)
    fitted, notes = scale_trials(150, calib, n_eval=48, time_budget_minutes=20)
    assert fitted == autosetup._MIN_TRIALS
    assert any("floor" in n for n in notes)


def test_trials_untouched_without_a_trusted_measurement():
    calib = Calibration(budget=48, truncated_rate=0.0, seconds=1.0, n_prompts=6)
    fitted, notes = scale_trials(60, calib, n_eval=32, time_budget_minutes=20)
    assert fitted == 60
    assert notes == []


def test_auto_fit_runtime_is_idempotent(monkeypatch, tmp_path):
    """UI calls it before its demos, the pipeline again later: one probe only."""
    _make_prompt_file(tmp_path, "harmful.txt", 160)
    _make_prompt_file(tmp_path, "harmless.txt", 156)
    cfg = _config(tmp_path, n_samples=160, n_eval_harmful=48, n_eval_harmless=48,
                  n_trials=150)

    calls = []
    monkeypatch.setattr(
        autosetup, "calibrate_token_budget",
        lambda bundle, prompts, **kw: (
            calls.append(1),
            Calibration(budget=96, truncated_rate=1.0, seconds=6.0, n_prompts=6),
        )[1],
    )

    first = autosetup.auto_fit_runtime(cfg, object(), level="quick")
    second = autosetup.auto_fit_runtime(cfg, object(), level="thorough")

    assert len(calls) == 1, "the token probe ran twice"
    assert first is second
    assert cfg.optimize.max_new_tokens == 96
    assert cfg.optimize.n_trials < 150, "quick's 20-minute budget trims the preset"


# --------------------------------------------------------------------------- #
# Precision
# --------------------------------------------------------------------------- #
def test_full_precision_when_the_model_fits_in_vram(monkeypatch):
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 24 * 2**30)
    cfg = ModelConfig(path="m")

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "bfloat16-gpu"
    assert cfg.dtype == "bfloat16" and not cfg.load_in_4bit


def test_4bit_when_the_model_does_not_fit_and_bitsandbytes_exists(monkeypatch):
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 8 * 2**30)
    monkeypatch.setattr(autosetup.importlib.util, "find_spec",
                        lambda name: object() if name == "bitsandbytes" else None)
    cfg = ModelConfig(path="m")

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "4bit-gpu"
    assert cfg.load_in_4bit


def test_lora_merge_never_gets_4bit(monkeypatch):
    """Quantized weights cannot be merged; the planner must not pick it."""
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 8 * 2**30)
    monkeypatch.setattr(autosetup.importlib.util, "find_spec",
                        lambda name: object() if name == "bitsandbytes" else None)
    cfg = ModelConfig(path="m", lora_adapter="reforge-adapter")

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "bf16-cpu-offload"
    assert not cfg.load_in_4bit
    assert any("slow" in n for n in report.notes)


def test_cpu_only_machine_gets_an_honest_slow_note(monkeypatch):
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: None)
    cfg = ModelConfig(path="m")

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "cpu-only"
    assert any("slow" in n for n in report.notes)


def test_unsizable_models_keep_the_defaults(monkeypatch):
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: None)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 24 * 2**30)
    cfg = ModelConfig(path="Qwen/Qwen2.5-7B-Instruct")

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "default"
    assert cfg.dtype == "bfloat16" and not cfg.load_in_4bit


def test_explicit_quantization_wins_over_auto(monkeypatch):
    monkeypatch.setattr(autosetup, "model_disk_bytes", lambda path: 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 24 * 2**30)
    cfg = ModelConfig(path="m", load_in_4bit=True)

    report = apply_precision(cfg)

    assert report.decisions["precision"] == "user-set"
    assert cfg.load_in_4bit


def test_precision_decision_is_cached_on_the_config(monkeypatch):
    """UI plans before load, the pipeline re-reads the decision after."""
    calls = []
    monkeypatch.setattr(
        autosetup, "model_disk_bytes",
        lambda path: calls.append(path) or 15 * 2**30)
    monkeypatch.setattr(autosetup, "free_cuda_vram", lambda: 8 * 2**30)
    monkeypatch.setattr(autosetup.importlib.util, "find_spec",
                        lambda name: object() if name == "bitsandbytes" else None)
    cfg = ModelConfig(path="m")

    first = apply_precision(cfg)
    second = apply_precision(cfg)

    assert len(calls) == 1, "the disk scan ran twice"
    assert second.decisions == first.decisions


def test_disk_bytes_reads_local_weight_files(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "model-00002.safetensors").write_bytes(b"x" * 5)
    assert model_disk_bytes(str(tmp_path)) == 15
    assert model_disk_bytes("Qwen/Qwen2.5-7B-Instruct") is None
    assert model_disk_bytes(str(tmp_path / "nope")) is None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_prompt_file(tmp_path, name, count):
    path = tmp_path / name
    path.write_text("\n".join(f"prompt {name} {i}" for i in range(count)) + "\n",
                    encoding="utf-8")
    return path


def _config(tmp_path, *, n_samples=128, n_eval_harmful=32, n_eval_harmless=32,
            n_trials=100):
    return Config(
        model=ModelConfig(path="unused"),
        data=DataConfig(harmful=str(tmp_path / "harmful.txt"),
                        harmless=str(tmp_path / "harmless.txt"),
                        n_samples=n_samples, seed=0),
        optimize=OptimizeConfig(n_eval_harmful=n_eval_harmful,
                                n_eval_harmless=n_eval_harmless,
                                n_trials=n_trials, seed=0),
    )
