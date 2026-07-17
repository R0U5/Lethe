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
from pathlib import Path
from typing import Optional

import torch

from .ablation import (
    AblationParams,
    apply_weighted_ablation,
    orthogonalize_model,
)
from .activations import collect_mean_activations
from .config import Config
from .data import load_prompts, sample_prompts
from .directions import RefusalDirections, compute_refusal_directions
from .evaluate import evaluate_refusal
from .metrics import kl_divergence, next_token_logprobs
from .model_utils import ModelBundle, load_model_and_tokenizer
from .optimize import optimize_ablation

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
) -> tuple[ModelBundle, RefusalDirections]:
    """Load the model (if needed) and compute per-layer candidate directions.

    ``prompts`` may be a preloaded ``(harmful, harmless)`` pair to avoid
    re-reading and re-sampling the prompt sources.
    """
    if bundle is None:
        bundle = load_model_and_tokenizer(cfg.model)
    harmful, harmless = prompts if prompts is not None else load_contrastive_prompts(cfg)

    mean_harmful = collect_mean_activations(bundle, harmful, cfg.ablation, desc="harmful")
    mean_harmless = collect_mean_activations(bundle, harmless, cfg.ablation, desc="harmless")

    directions = compute_refusal_directions(
        mean_harmful, mean_harmless,
        model_path=cfg.model.path,
        n_harmful=len(harmful), n_harmless=len(harmless),
    )
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
    bundle.model.save_pretrained(out)
    bundle.tokenizer.save_pretrained(out)

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


def optimize_and_apply(cfg: Config, bundle: Optional[ModelBundle] = None) -> Path:
    """Optimized abliteration: search params (refusals + KL), apply best, save.

    Direction-computation and evaluation prompts are sampled with different seeds
    so the search is scored on held-out prompts.
    """
    if bundle is None:
        bundle = load_model_and_tokenizer(cfg.model)

    ocfg = cfg.optimize
    harmful_dir = _sampled(cfg.data.harmful, "harmful.txt", cfg.data.n_samples, cfg.data.seed)
    harmless_dir = _sampled(cfg.data.harmless, "harmless.txt", cfg.data.n_samples, cfg.data.seed)
    harmful_eval = _sampled(cfg.data.harmful, "harmful.txt", ocfg.n_eval_harmful, cfg.data.seed + 1)
    harmless_eval = _sampled(cfg.data.harmless, "harmless.txt", ocfg.n_eval_harmless, cfg.data.seed + 1)

    bundle, directions = build_directions(cfg, bundle=bundle, prompts=(harmful_dir, harmless_dir))

    # Baseline refusal rate for the manifest.
    before = evaluate_refusal(bundle, harmful_eval, max_new_tokens=ocfg.max_new_tokens)
    logger.info("refusal rate before optimization: %.1f%%", 100 * before["refusal_rate"])

    best_params, history = optimize_ablation(
        bundle, directions, ocfg, harmful_eval, harmless_eval, model_cfg=cfg.model,
    )
    best_result = min(history, key=lambda r: r.cost)

    modified = apply_weighted_ablation(
        bundle.model, directions.directions, best_params,
        model_cfg=cfg.model, hidden_size=bundle.hidden_size,
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("saving optimized abliterated model to %s", out)
    bundle.model.save_pretrained(out)
    bundle.tokenizer.save_pretrained(out)

    manifest = {
        "source_model": cfg.model.path,
        "lora_adapter": cfg.model.lora_adapter,
        "method": "optimized-abliteration",
        "hidden_size": bundle.hidden_size,
        "num_layers": bundle.num_layers,
        "matrices_modified": modified,
        "best_params": best_params.to_dict(),
        "metrics": {
            "refusal_rate_before": round(before["refusal_rate"], 4),
            "refusal_rate_after": round(best_result.refusal_rate, 4),
            "kl_divergence": round(best_result.kl, 4),
            "cost": round(best_result.cost, 4),
            "n_trials": len(history),
        },
        "optimize": {
            "n_trials": ocfg.n_trials, "kl_weight": ocfg.kl_weight,
            "n_eval_harmful": ocfg.n_eval_harmful, "n_eval_harmless": ocfg.n_eval_harmless,
        },
    }
    (out / "abliteration_manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def run(cfg: Config, *, select: str = "config", evaluate: bool = True) -> Path:
    """Full pipeline: directions -> select -> apply -> save (+ optional eval).

    ``select="optimize"`` runs the automated search instead (Heretic-style).
    """
    if select == "optimize":
        return optimize_and_apply(cfg)

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
