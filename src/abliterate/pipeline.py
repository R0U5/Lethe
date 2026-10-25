"""End-to-end orchestration of the abliteration workflow.

Stages (each usable on its own from the CLI):

    directions : load model + prompts -> per-layer candidate directions
    select     : pick the direction index (fractional / explicit / auto-by-eval)
    apply      : orthogonalize weights and save the abliterated model
    evaluate   : measure refusal rate, with or without ablation
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

from .ablation import (
    AblationParams,
    apply_weighted_ablation,
    orthogonalize_model,
)
from .activations import collect_mean_activations
from .bundle import build_bundle
from .config import Config
from .data import load_prompts, sample_prompts
from .directions import RefusalDirections, compute_refusal_directions
from .evaluate import evaluate_refusal
from .metrics import kl_divergence, next_token_logprobs
from .model_utils import (
    ModelBundle,
    is_quantized,
    load_model_and_tokenizer,
    save_model_and_tokenizer,
)
from .optimize import optimize_ablation
from .progress import ProgressCallback, emit

logger = logging.getLogger("abliterate")


def load_contrastive_prompts(cfg: Config) -> tuple[list[str], list[str]]:
    harmful = sample_prompts(
        load_prompts(cfg.data.harmful, default_file="harmful.txt"),
        cfg.data.n_samples, cfg.data.seed,
    )
    harmless = sample_prompts(
        load_prompts(cfg.data.harmless, default_file="harmless.txt"),
        cfg.data.n_samples, cfg.data.seed,
    )
    return harmful, harmless


def build_directions(
    cfg: Config,
    bundle: Optional[ModelBundle] = None,
    *,
    prompts: Optional[tuple[list[str], list[str]]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[ModelBundle, RefusalDirections]:
    """Load the model (if needed) and compute per-layer candidate directions.

    ``prompts`` may be a preloaded ``(harmful, harmless)`` pair to avoid
    re-reading and re-sampling the prompt sources. ``on_progress(phase,
    fraction)`` reports each sub-step (activation collection, direction
    computation) so a UI can advance during this otherwise-silent stage.
    """
    if bundle is None:
        bundle = load_model_and_tokenizer(cfg.model, on_progress=on_progress)
    harmful, harmless = prompts if prompts is not None else load_contrastive_prompts(cfg)

    emit(on_progress, "Collecting harmful activations", 0.0)
    mean_harmful = collect_mean_activations(bundle, harmful, cfg.ablation, desc="harmful")
    emit(on_progress, "Collecting harmless activations", 0.4)
    mean_harmless = collect_mean_activations(bundle, harmless, cfg.ablation, desc="harmless")

    emit(on_progress, "Computing refusal directions", 0.8)
    directions = compute_refusal_directions(
        mean_harmful, mean_harmless,
        model_path=cfg.model.path,
        n_harmful=len(harmful), n_harmless=len(harmless),
    )
    emit(on_progress, "Directions ready", 1.0)
    return bundle, directions


def select_direction_index(
    bundle: ModelBundle,
    directions: RefusalDirections,
    cfg: Config,
    *,
    method: str = "config",
    val_prompts: Optional[list[str]] = None,
    band: tuple[float, float] = (0.3, 0.9),
    eval_max_new_tokens: int = 32,
) -> tuple[int, dict]:
    """Resolve which candidate direction to use.

    ``method="config"`` uses explicit/fractional selection from the config.
    ``method="auto"`` sweeps a depth band and picks the direction that most
    reduces refusal rate on ``val_prompts`` (falls back to config on error).
    """
    if method != "auto":
        return directions.select_index(cfg.ablation), {}

    if not val_prompts:
        raise ValueError("auto selection requires validation prompts")

    n = directions.num_candidates
    lo, hi = int(band[0] * n), int(band[1] * n)
    candidates = list(range(max(1, lo), max(2, hi)))

    def scorer(_idx: int, direction: torch.Tensor) -> float:
        res = evaluate_refusal(
            bundle, val_prompts, direction=direction,
            model_cfg=cfg.model, max_new_tokens=eval_max_new_tokens,
        )
        return -res["refusal_rate"]  # higher (less refusal) is better

    best, scores = directions.select_by_score(
        scorer, candidate_indices=candidates, higher_is_better=True
    )
    logger.info("auto-selected direction index %d", best)
    return best, {str(k): round(-v, 4) for k, v in scores.items()}


def apply_and_save(
    bundle: ModelBundle,
    direction: torch.Tensor,
    cfg: Config,
    *,
    direction_index: Optional[int] = None,
    selection_scores: Optional[dict] = None,
) -> Path:
    """Orthogonalize the model in place and save it + tokenizer + manifest."""
    modified = orthogonalize_model(
        bundle.model, direction,
        model_cfg=cfg.model, ablation_cfg=cfg.ablation,
        hidden_size=bundle.hidden_size,
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("saving abliterated model to %s", out)
    save_model_and_tokenizer(bundle, out)

    manifest = {
        "source_model": cfg.model.path,
        "lora_adapter": cfg.model.lora_adapter,
        "hidden_size": bundle.hidden_size,
        "num_layers": bundle.num_layers,
        "direction_index": direction_index,
        "matrices_modified": modified,
        "ablation": {
            "ablate_embed": cfg.ablation.ablate_embed,
            "ablate_attn": cfg.ablation.ablate_attn,
            "ablate_mlp": cfg.ablation.ablate_mlp,
            "position": cfg.ablation.position,
        },
        "selection_scores": selection_scores or {},
    }
    (out / "abliteration_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def _sampled(source, default_file, n, seed):
    return sample_prompts(load_prompts(source, default_file=default_file), n, seed)


def _sampled_holdout(source, default_file, n_dir, n_eval, seed):
    """Sample direction prompts and evaluation prompts from DISJOINT parts.

    Sampling both with different seeds is not enough. ``sample_prompts`` returns the
    whole list once ``n`` covers it, so a prompt file no larger than ``n_dir`` yields
    evaluation prompts that overlap the very prompts the refusal direction was fitted
    to: the search then scores itself on its own training data and reports refusal
    rates it cannot reproduce on unseen prompts. Draw the evaluation prompts from the
    complement instead, and say so loudly when the file is too small to hold any out.
    """
    prompts = load_prompts(source, default_file=default_file)
    dir_prompts = sample_prompts(prompts, n_dir, seed)

    # Remove the sampled direction prompts multiset-wise, so duplicate lines in the
    # file are handled correctly and evaluation never reuses a direction prompt.
    remaining_counts = Counter(dir_prompts)
    remaining: list[str] = []
    for prompt in prompts:
        if remaining_counts[prompt] > 0:
            remaining_counts[prompt] -= 1
        else:
            remaining.append(prompt)

    if len(remaining) < n_eval:
        logger.warning(
            "hold-out exhausted: %d prompt(s) left after sampling %d for the refusal "
            "direction, but %d requested for evaluation -- the evaluation set will "
            "reuse direction prompts, so the search grades its own training data. "
            "Give a prompt file with more than %d entries to score on unseen prompts.",
            len(remaining), n_dir, n_eval, n_dir + n_eval,
        )
    eval_prompts = sample_prompts(remaining, n_eval, seed + 1)
    if not eval_prompts:
        eval_prompts = sample_prompts(prompts, n_eval, seed + 1)
    return dir_prompts, eval_prompts


def optimize_and_apply(
    cfg: Config,
    bundle: Optional[ModelBundle] = None,
    *,
    on_trial=None,
    on_progress: Optional[ProgressCallback] = None,
    save: str = "model",
) -> Path:
    """Optimized abliteration: search params (refusals + KL), apply best, save.

    Evaluation prompts are drawn from the complement of the direction prompts, so the
    search is scored on prompts the refusal direction was NOT fitted to. When the
    prompt file is too small to hold anything out, a warning says so rather than
    reporting a silently self-scored number. ``on_trial(done, total)`` is
    forwarded to the optimizer for progress reporting. ``on_progress(phase,
    fraction)`` reports the pre-trial warm-up (prompt prep, direction
    computation, baseline eval, reference capture) as a fraction of that stage.

    ``save`` selects the output:
      * ``"model"``   -- bake into weights, save a standalone HF model (default);
      * ``"bundle"``  -- save only the tiny runtime bundle (required for
        quantized models, which cannot be edited in place);
      * ``"both"``    -- save the model and a bundle beside it.

    In every case the in-memory model ends up behaving abliterated (via baked
    weights or attached hooks), so callers can generate from it afterwards.
    Returns the primary output path.
    """
    if bundle is None:
        bundle = load_model_and_tokenizer(cfg.model)

    quantized = is_quantized(bundle.model)
    if quantized and save != "bundle":
        logger.info("model is quantized; forcing save=bundle (weights can't be edited)")
        save = "bundle"

    ocfg = cfg.optimize

    # Warm-up progress is reported as a fraction of the whole pre-trial stage:
    # prompt prep + directions occupy [0, 0.6], baseline eval 0.65, and the
    # reference-distribution capture [0.7, 1.0].
    def _dir_progress(phase: str, frac: Optional[float]) -> None:
        emit(on_progress, phase, None if frac is None else 0.6 * frac)

    def _opt_progress(phase: str, frac: Optional[float]) -> None:
        emit(on_progress, phase, None if frac is None else 0.7 + 0.3 * frac)

    emit(on_progress, "Preparing prompts", 0.0)
    harmful_dir, harmful_eval = _sampled_holdout(
        cfg.data.harmful, "harmful.txt", cfg.data.n_samples, ocfg.n_eval_harmful, cfg.data.seed,
    )
    harmless_dir, harmless_eval = _sampled_holdout(
        cfg.data.harmless, "harmless.txt", cfg.data.n_samples, ocfg.n_eval_harmless, cfg.data.seed,
    )

    bundle, directions = build_directions(
        cfg, bundle=bundle, prompts=(harmful_dir, harmless_dir), on_progress=_dir_progress,
    )

    emit(on_progress, "Measuring baseline refusals", 0.65)
    before = evaluate_refusal(bundle, harmful_eval, max_new_tokens=ocfg.max_new_tokens)
    logger.info("refusal rate before optimization: %.1f%%", 100 * before["refusal_rate"])

    best_params, history = optimize_ablation(
        bundle, directions, ocfg, harmful_eval, harmless_eval,
        model_cfg=cfg.model, on_trial=on_trial, on_progress=_opt_progress,
    )
    best_result = min(history, key=lambda r: r.cost)

    metrics = {
        "refusal_rate_before": round(before["refusal_rate"], 4),
        "refusal_rate_after": round(best_result.refusal_rate, 4),
        "degenerate_rate_after": round(getattr(best_result, "degenerate_rate", 0.0), 4),
        "non_compliant_rate_after": round(getattr(best_result, "non_compliant_rate", 0.0), 4),
        "kl_divergence": round(best_result.kl, 4),
        "cost": round(best_result.cost, 4),
        "n_trials": len(history),
    }
    abl_bundle = build_bundle(
        directions.directions, best_params,
        hidden_size=bundle.hidden_size, num_layers=bundle.num_layers,
        source_model=cfg.model.path, metrics=metrics, model_cfg=cfg.model,
    )

    out = Path(cfg.output_dir)
    if save in ("model", "both"):
        modified = apply_weighted_ablation(
            bundle.model, directions.directions, best_params,
            model_cfg=cfg.model, hidden_size=bundle.hidden_size,
        )
        out.mkdir(parents=True, exist_ok=True)
        logger.info("saving optimized abliterated model to %s", out)
        save_model_and_tokenizer(bundle, out)
        manifest = {
            "source_model": cfg.model.path,
            "lora_adapter": cfg.model.lora_adapter,
            "method": "optimized-abliteration",
            "hidden_size": bundle.hidden_size,
            "num_layers": bundle.num_layers,
            "matrices_modified": modified,
            "best_params": best_params.to_dict(),
            "metrics": metrics,
            "optimize": {"n_trials": ocfg.n_trials, "kl_weight": ocfg.kl_weight},
        }
        (out / "abliteration_manifest.json").write_text(json.dumps(manifest, indent=2))
        primary = out
        if save == "both":
            abl_bundle.save(out / "abliteration_bundle")
    else:  # bundle only
        abl_bundle.save(out)
        # Make the in-memory model behave abliterated (hooks, since weights stay put).
        abl_bundle.attach(bundle.model, model_cfg=cfg.model)
        primary = out

    return primary


def run(cfg: Config, *, select: str = "config", evaluate: bool = True,
        save: str = "model") -> Path:
    """Full pipeline: directions -> select -> apply -> save (+ optional eval).

    ``select="optimize"`` runs the automated search instead (Heretic-style).
    """
    if select == "optimize":
        return optimize_and_apply(cfg, save=save)

    harmful, harmless = load_contrastive_prompts(cfg)
    bundle, directions = build_directions(cfg, prompts=(harmful, harmless))

    val_prompts = harmful[: min(16, len(harmful))]

    index, scores = select_direction_index(
        bundle, directions, cfg, method=select, val_prompts=val_prompts,
    )
    direction = directions.get(index)
    logger.info("using direction index %d (raw norm %.4f)", index, directions.raw_norms[index])

    if evaluate:
        before = evaluate_refusal(bundle, val_prompts, max_new_tokens=48)
        logger.info("refusal rate before: %.1f%%", 100 * before["refusal_rate"])

    out = apply_and_save(
        bundle, direction, cfg,
        direction_index=index, selection_scores=scores,
    )

    if evaluate:
        after = evaluate_refusal(bundle, val_prompts, max_new_tokens=48)
        logger.info("refusal rate after:  %.1f%%", 100 * after["refusal_rate"])

    return out
