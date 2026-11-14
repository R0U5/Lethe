"""Automatic run setup: the "set and forget" planner.

A novice user picks a model, a thoroughness level, and presses go. Every
setting that used to require judgment is decided here from facts about the
machine and the checkpoint, and each decision is logged in plain language so
a power user can see -- and then override -- exactly what the auto path chose.

The decisions, in the order the pipeline needs them:

1. ``apply_precision`` -- before the model loads. Sizes the checkpoint on
   disk against free VRAM: full bfloat16 when it fits, 4-bit when it does not
   and bitsandbytes is installed, and an honest "this will offload to CPU and
   run slowly" note when neither applies. Never picks 4-bit for a run that
   still has to merge a LoRA adapter (quantized weights cannot be merged).

2. ``auto_fit_runtime`` -- right after the model loads, before any scoring:
   * Sizes the data plan so evaluation prompts are always held OUT of the
     direction fit. Small prompt files used to exhaust the hold-out, which
     made the search grade its own training data; the split is capped so that
     cannot happen, and the source is upgraded to a built-in dataset only when
     the local file is too small to split at all.
   * Calibrates ``max_new_tokens`` by generating from the model itself and
     counting truncations. A budget shorter than the model's own refusals hid
     them from the scorer entirely, making a broken search look successful.
   * Fits the trial count to a wall-clock budget for the chosen thoroughness
     level, using the generation speed measured during calibration.

Everything here is idempotent: decisions are cached on the config object, so
calling from both a UI and the pipeline is safe and never re-runs a probe.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .config import Config, ModelConfig

logger = logging.getLogger("abliterate")

# Wall-clock budgets per thoroughness level (minutes). The auto path keeps a
# level's trial target when the machine is fast enough and trims the count when
# the measured generation speed says the run would otherwise take hours.
_LEVEL_TIME_BUDGETS = {"quick": 20, "balanced": 60, "thorough": 180}

_MIN_EVAL_PROMPTS = 8        # below this a refusal/KL score is mostly noise
_MIN_DIRECTION_PROMPTS = 32  # below this the contrastive fit is unreliable
_TOKEN_BUDGET_START = 48     # matches the old 'thorough' preset
_MAX_TOKEN_BUDGET = 512      # calibration never sets a longer budget
_TRUNCATION_TARGET = 0.25    # same threshold evaluate.py warns at
_PROBE_PROMPTS = 6           # prompts used for the truncation probe
_MIN_TRIALS = 12             # fewer trials than this is not a search
_TRIAL_OVERHEAD = 1.6        # KL forward passes + hook cost on top of generation


@dataclass
class AutoReport:
    """What the auto path decided, and why, in human-readable form."""

    decisions: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 1. Precision: fit the checkpoint to the machine
# --------------------------------------------------------------------------- #
def model_disk_bytes(path: str) -> Optional[int]:
    """Total size of the weight files in a local checkpoint folder.

    Returns ``None`` when ``path`` is not a local folder (a HuggingFace id) or
    contains no recognizable weight files, so callers can fall back to the
    configured defaults instead of guessing.
    """
    p = Path(path)
    if not p.is_dir():
        return None
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        for f in p.glob(pattern):
            try:
                total += f.stat().st_size
            except OSError:  # pragma: no cover - file vanished mid-scan
                pass
    return total or None


def free_cuda_vram() -> Optional[int]:
    """Free VRAM in bytes, or ``None`` when there is no usable CUDA device."""
    try:
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:  # pragma: no cover - driver quirks
        return None


def _gb(n: Optional[int]) -> str:
    return "unknown size" if n is None else f"{n / 2**30:.1f} GB"


def apply_precision(cfg: "ModelConfig") -> AutoReport:
    """Choose dtype/quantization for this machine; mutate ``cfg`` in place.

    Explicit user choices win: when 4/8-bit loading was requested directly the
    planner leaves the config alone. The decision is cached on
    ``cfg._auto_decisions`` so repeated calls (UI then pipeline) are no-ops and
    ``auto_fit_runtime`` can fold the decision into the run report.
    """
    cached = getattr(cfg, "_auto_decisions", None)
    if cached is not None:
        report = AutoReport()
        report.decisions.update(cached)
        return report

    report = AutoReport()

    def _finish(key: str) -> AutoReport:
        report.decisions["precision"] = key
        cfg._auto_decisions = report.decisions  # noqa: SLF001 - deliberate cache
        cfg._auto_notes = list(report.notes)    # noqa: SLF001 - for auto_fit_runtime
        return report

    if cfg.load_in_4bit or cfg.load_in_8bit:
        report.notes.append("precision was set explicitly; leaving it alone")
        return _finish("user-set")

    device_map = str(cfg.device_map or "").strip().lower()
    if device_map in ("", "none", "cpu"):
        report.notes.append("loading on the CPU as requested (bfloat16)")
        return _finish("cpu-forced")

    size = model_disk_bytes(cfg.path)
    if size is None:
        report.notes.append(
            "model size unknown (not a local folder); keeping bfloat16 with "
            "device_map=auto")
        return _finish("default")

    vram = free_cuda_vram()
    if vram is None:
        report.notes.append(
            f"no CUDA device detected; running on CPU ({_gb(size)} of weights). "
            "Generation will be slow.")
        return _finish("cpu-only")

    if size * 1.3 <= vram:
        cfg.dtype = "bfloat16"
        cfg.load_in_4bit = False
        cfg.load_in_8bit = False
        report.notes.append(
            f"model is {_gb(size)} and {_gb(vram)} VRAM is free -> loading "
            "full bfloat16 onto the GPU")
        return _finish("bfloat16-gpu")

    # Doesn't fit whole: prefer 4-bit, unless a LoRA merge forbids it.
    if cfg.lora_adapter:
        report.notes.append(
            f"model is {_gb(size)} but only {_gb(vram)} VRAM is free, and "
            "merging a LoRA adapter needs full-precision weights -> parts will "
            "offload to CPU. This run will be slow.")
        return _finish("bf16-cpu-offload")

    if importlib.util.find_spec("bitsandbytes") is not None:
        cfg.load_in_4bit = True
        cfg.load_in_8bit = False
        report.notes.append(
            f"model is {_gb(size)} but only {_gb(vram)} VRAM is free -> loading "
            "4-bit so it fits on the GPU; the output will be a runtime bundle "
            "(quantized weights cannot be edited in place)")
        return _finish("4bit-gpu")

    report.notes.append(
        f"model is {_gb(size)} but only {_gb(vram)} VRAM is free, and "
        "bitsandbytes is not installed -> parts will offload to CPU. This run "
        "will be slow; `pip install bitsandbytes` enables automatic 4-bit.")
    return _finish("bf16-cpu-offload")


# --------------------------------------------------------------------------- #
# 2a. Data plan: always keep evaluation prompts out of the direction fit
# --------------------------------------------------------------------------- #
def plan_split(avail: int, want_dir: int, want_eval: int) -> tuple[int, int, list[str]]:
    """Split ``avail`` prompts into direction-fit and held-out evaluation parts.

    The evaluation part always comes off the top first: it is better to fit the
    direction on fewer prompts than to score the search on prompts the
    direction was fitted to.
    """
    notes: list[str] = []
    n_eval = min(want_eval, max(_MIN_EVAL_PROMPTS, avail // 4))
    n_dir = min(want_dir, avail - n_eval)
    if n_dir < _MIN_DIRECTION_PROMPTS:
        notes.append(
            f"only {avail} prompt(s) available, so the direction fit gets "
            f"{n_dir} (aimed for {want_dir}) and evaluation keeps {n_eval} "
            "held out; a larger prompt file would improve both")
    return max(n_dir, 0), n_eval, notes


def _maybe_upgrade_source(
    source: Optional[str], default_file: str, builtin: str
) -> tuple[Optional[str], Optional[str]]:
    """When a local prompt file is too small to split, try a built-in dataset.

    Only fires on genuinely tiny files (fewer prompts than one direction fit +
    one evaluation set needs); normal files keep working offline as before.
    Download failures degrade to the local file with a note, never an error.
    """
    from .data import load_prompts

    prompts = load_prompts(source, default_file=default_file)
    if len(prompts) >= _MIN_DIRECTION_PROMPTS + _MIN_EVAL_PROMPTS:
        return source, None
    try:
        upgraded = load_prompts(builtin, default_file=default_file)
    except Exception as exc:
        return source, (
            f"only {len(prompts)} prompt(s) available and the built-in "
            f"{builtin!r} dataset could not be loaded "
            f"({exc.__class__.__name__}); continuing with the local file"
        )
    return builtin, (
        f"local file has only {len(prompts)} prompt(s); switching to the "
        f"built-in {builtin!r} dataset ({len(upgraded)} prompts)"
    )


def apply_data_plan(cfg: "Config") -> AutoReport:
    """Size direction/evaluation split from the prompt sources' real contents."""
    from .data import load_prompts

    report = AutoReport()

    for side, attr, default_file, builtin in (
        ("harmful", "harmful", "harmful.txt", "advbench"),
        ("harmless", "harmless", "harmless.txt", "harmless_alpaca"),
    ):
        resolved, note = _maybe_upgrade_source(
            getattr(cfg.data, attr), default_file, builtin)
        if note:
            report.notes.append(note)
        if resolved != getattr(cfg.data, attr):
            setattr(cfg.data, attr, resolved)

    n_harmful = len(load_prompts(cfg.data.harmful, default_file="harmful.txt"))
    n_harmless = len(load_prompts(cfg.data.harmless, default_file="harmless.txt"))

    dir_h, eval_h, _ = plan_split(n_harmful, cfg.data.n_samples, cfg.optimize.n_eval_harmful)
    dir_s, eval_s, _ = plan_split(n_harmless, cfg.data.n_samples, cfg.optimize.n_eval_harmless)

    cfg.optimize.n_eval_harmful = eval_h
    cfg.optimize.n_eval_harmless = eval_s
    # One ``n_samples`` drives both sides; keep them balanced at the smaller
    # split so neither side's direction fit leans on prompts the other side
    # leaves out (and so the hold-out still covers the eval targets).
    cfg.data.n_samples = max(min(dir_h, dir_s), 1)

    report.decisions["data_plan"] = {
        "harmful_available": n_harmful,
        "harmless_available": n_harmless,
        "direction_prompts": cfg.data.n_samples,
        "eval_harmful": eval_h,
        "eval_harmless": eval_s,
    }
    report.notes.append(
        f"prompt split: {cfg.data.n_samples} per side to fit the refusal "
        f"direction, {eval_h} harmful + {eval_s} harmless held out for "
        "scoring (evaluation never reuses direction prompts)")
    return report


# --------------------------------------------------------------------------- #
# 2b. Token budget: measure the model, don't guess
# --------------------------------------------------------------------------- #
@dataclass
class Calibration:
    """Result of the truncation probe: budget, rate, and measured speed."""

    budget: int
    truncated_rate: float
    seconds: float
    n_prompts: int

    @property
    def tokens_per_second(self) -> float:
        """Generation speed from the probe, or 0 when it cannot be trusted.

        The probe only knows how many tokens the budget *allowed*; when most
        completions finished early the generated count is far lower, so the
        estimate would flatter the machine. Only a probe dominated by
        truncations (where completions really did run the full budget) yields
        a believable rate.
        """
        if self.seconds <= 0 or self.truncated_rate < 0.5:
            return 0.0
        return self.n_prompts * self.budget / self.seconds


def calibrate_token_budget(
    bundle,
    prompts: list[str],
    *,
    start: int = _TOKEN_BUDGET_START,
    cap: int = _MAX_TOKEN_BUDGET,
    target: float = _TRUNCATION_TARGET,
) -> Calibration:
    """Find the smallest standard budget where the model finishes speaking.

    Generates greedily from ``prompts`` (chat-templated, exactly like scoring)
    and doubles the budget while too many completions are cut off. Truncated
    completions are the ones that hide refusals, so this keeps the scorer's
    input honest regardless of how chatty the checkpoint is.
    """
    from .evaluate import generate_completions_with_stats

    probe = list(prompts)[:_PROBE_PROMPTS]
    if not probe:
        return Calibration(start, 0.0, 0.0, 0)
    budget = start
    while True:
        t0 = time.perf_counter()
        _, truncated = generate_completions_with_stats(
            bundle, probe, max_new_tokens=budget, batch_size=len(probe))
        seconds = time.perf_counter() - t0
        rate = sum(truncated) / len(truncated)
        if rate <= target or budget >= cap:
            return Calibration(budget, rate, seconds, len(probe))
        budget = min(cap, budget * 2)


def scale_trials(
    target: int,
    calib: Optional[Calibration],
    n_eval: int,
    time_budget_minutes: float,
) -> tuple[int, list[str]]:
    """Fit the trial target to the measured speed and a wall-clock budget."""
    notes: list[str] = []
    tps = calib.tokens_per_second if calib is not None else 0.0
    if tps <= 0 or n_eval <= 0:
        return target, notes
    seconds_per_trial = n_eval * calib.budget / tps * _TRIAL_OVERHEAD
    fit = int(time_budget_minutes * 60 / max(seconds_per_trial, 1.0))
    fitted = min(target, max(_MIN_TRIALS, fit))
    if fitted < target:
        notes.append(
            f"trials {target} -> {fitted}: at the measured speed "
            f"(~{tps:.0f} tokens/s) {target} trials would take about "
            f"{target * seconds_per_trial / 60:.0f} min; keeping the search "
            f"under ~{time_budget_minutes:.0f} min")
    if fit < _MIN_TRIALS <= target:
        notes.append(
            "this machine is very slow for this model; the search keeps a "
            f"floor of {_MIN_TRIALS} trials "
            f"(~{_MIN_TRIALS * seconds_per_trial / 60:.0f} min)")
    return fitted, notes


# --------------------------------------------------------------------------- #
# 2c. The one entry point the pipeline calls after the model loads
# --------------------------------------------------------------------------- #
def auto_fit_runtime(
    cfg: "Config",
    bundle,
    *,
    level: str = "balanced",
    time_budget_minutes: Optional[float] = None,
) -> AutoReport:
    """Data plan + token budget + trial count, once, after the model loads.

    Idempotent by caching the full report on ``cfg.optimize._auto_report``:
    a UI may call this before its demo generations and the pipeline again
    later -- the second call returns the same decisions without re-probing.
    """
    o = cfg.optimize
    cached = getattr(o, "_auto_report", None)
    if cached is not None:
        return cached

    report = AutoReport()
    report.decisions.update(getattr(cfg.model, "_auto_decisions", {}))
    report.notes.extend(getattr(cfg.model, "_auto_notes", []))

    # 1. Data plan first: it decides which prompts everything else uses.
    data_report = apply_data_plan(cfg)
    report.decisions.update(data_report.decisions)
    report.notes.extend(data_report.notes)

    # 2. Token budget from a live probe. The probe reuses direction-set
    # prompts on purpose: it measures truncation and speed only, it never
    # scores anything, so there is nothing to hold out from it.
    from .data import load_prompts, sample_prompts

    pool = load_prompts(cfg.data.harmful, default_file="harmful.txt")
    calib = calibrate_token_budget(bundle, sample_prompts(pool, _PROBE_PROMPTS, 7))
    o._auto_calibration = calib  # noqa: SLF001 - deliberate cache
    o.max_new_tokens = calib.budget
    report.decisions["max_new_tokens"] = calib.budget
    if calib.truncated_rate > _TRUNCATION_TARGET:
        report.notes.append(
            f"generation budget raised to {calib.budget} tokens: the probe "
            f"was {100 * calib.truncated_rate:.0f}% truncated at "
            f"{_TOKEN_BUDGET_START}, which hides refusals from the scorer")
    else:
        report.notes.append(
            f"generation budget set to {calib.budget} tokens (probe "
            "completions finish on their own)")

    # 3. Trials: fit the level's target to the measured speed.
    minutes = (time_budget_minutes if time_budget_minutes is not None
               else _LEVEL_TIME_BUDGETS.get(level, 60))
    fitted, trial_notes = scale_trials(o.n_trials, calib, max(o.n_eval_harmful, 1), minutes)
    report.notes.extend(trial_notes)
    if fitted != o.n_trials:
        o.n_startup_trials = min(o.n_startup_trials, max(4, fitted // 3))
        o.n_trials = fitted
    report.decisions["n_trials"] = o.n_trials
    if calib.tokens_per_second > 0:
        report.notes.append(f"measured generation speed ~{calib.tokens_per_second:.0f} tokens/s")

    o._auto_report = report  # noqa: SLF001 - deliberate cache
    return report
