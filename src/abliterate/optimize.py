"""Automated abliteration: search ablation parameters to co-minimize refusals
and KL divergence.

This is the difference between guessing a layer and getting a genuinely good
model. We treat abliteration as black-box optimization over:

    * the direction (a fixed hidden-state index, or per-layer directions), and
    * a per-layer strength *kernel* for the attention-out and MLP-down
      projections separately,

scoring each candidate by ``cost = refusal_rate + kl_weight * KL`` -- refusals
measured by generating on held-out harmful prompts, KL measured against the
original model on held-out harmless prompts. Optuna's TPE sampler drives the
search when installed; otherwise a seeded random search over the same space is
used (same objective, fewer samples needed to justify the dependency).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from .ablation import AblationParams, LayerWeightKernel, WeightedAblationHooks
from .config import ModelConfig, OptimizeConfig
from .directions import RefusalDirections
from .evaluate import generate_completions, refusal_rate
from .metrics import kl_divergence, next_token_logprobs
from .model_utils import ModelBundle

logger = logging.getLogger("abliterate")


@dataclass
class TrialResult:
    params: AblationParams
    cost: float
    refusal_rate: float
    kl: float


class _Suggester(Protocol):
    def suggest_float(self, name: str, low: float, high: float) -> float: ...
    def suggest_int(self, name: str, low: int, high: int) -> int: ...
    def suggest_categorical(self, name: str, choices: list): ...


class _RandomSuggester:
    """Minimal Optuna-trial-compatible sampler for the no-Optuna fallback."""

    def __init__(self, rng: random.Random):
        self._rng = rng

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self._rng.randint(low, high)

    def suggest_categorical(self, name: str, choices: list):
        return self._rng.choice(choices)


def _sample_kernel(s: _Suggester, prefix: str, n_layers: int, cfg: OptimizeConfig) -> LayerWeightKernel:
    return LayerWeightKernel(
        max_weight=s.suggest_float(f"{prefix}_max_weight", 0.0, cfg.max_weight_hi),
        max_weight_position=s.suggest_float(f"{prefix}_max_pos", 0.0, float(max(n_layers - 1, 1))),
        min_weight=s.suggest_float(f"{prefix}_min_weight", 0.0, cfg.max_weight_hi),
        min_weight_distance=s.suggest_float(f"{prefix}_min_dist", 1.0, float(max(n_layers, 2))),
    )


def sample_params(
    s: _Suggester, n_layers: int, n_candidates: int, cfg: OptimizeConfig
) -> AblationParams:
    """Draw one AblationParams from the search space using ``s``'s suggestions."""
    per_layer = s.suggest_categorical("direction_mode", ["fixed", "per_layer"]) == "per_layer"
    direction_index = None
    if not per_layer:
        lo = max(1, int(cfg.dir_band[0] * n_candidates))
        hi = max(lo, int(cfg.dir_band[1] * n_candidates) - 1)
        direction_index = s.suggest_int("direction_index", lo, hi)

    embed_strength = 0.0
    ablate_embed = False
    if cfg.optimize_embed:
        embed_strength = s.suggest_float("embed_strength", 0.0, cfg.max_weight_hi)
        ablate_embed = embed_strength > 0.0

    return AblationParams(
        direction_index=direction_index,
        per_layer_direction=per_layer,
        attn_kernel=_sample_kernel(s, "attn", n_layers, cfg),
        mlp_kernel=_sample_kernel(s, "mlp", n_layers, cfg),
        ablate_embed=ablate_embed,
        embed_strength=embed_strength,
    )


def optimize_ablation(
    bundle: ModelBundle,
    directions: RefusalDirections,
    cfg: OptimizeConfig,
    harmful_eval: list[str],
    harmless_eval: list[str],
    *,
    model_cfg: Optional[ModelConfig] = None,
    on_trial: Optional[Callable[[int, int], None]] = None,
) -> tuple[AblationParams, list[TrialResult]]:
    """Search for the best ablation params. Returns ``(best_params, history)``.

    ``on_trial(completed, total)`` is called after each evaluated trial (for UI
    progress reporting).
    """
    n_layers = bundle.num_layers
    n_candidates = directions.num_candidates
    dirs = directions.directions

    # Reference distribution from the ORIGINAL (un-ablated) model, captured once.
    logger.info("capturing reference distribution on %d harmless prompts", len(harmless_eval))
    reference = next_token_logprobs(
        bundle, harmless_eval, batch_size=max(1, cfg.n_eval_harmless // 4)
    )

    history: list[TrialResult] = []

    def evaluate_params(params: AblationParams) -> TrialResult:
        with WeightedAblationHooks(
            bundle.model, dirs, params, model_cfg=model_cfg, hidden_size=bundle.hidden_size
        ):
            completions = generate_completions(
                bundle, harmful_eval,
                max_new_tokens=cfg.max_new_tokens,
                batch_size=max(1, cfg.n_eval_harmful // 4),
            )
            rr = refusal_rate(completions)
            current = next_token_logprobs(
                bundle, harmless_eval, batch_size=max(1, cfg.n_eval_harmless // 4)
            )
        kl = kl_divergence(reference, current)
        cost = rr + cfg.kl_weight * kl
        result = TrialResult(params=params, cost=cost, refusal_rate=rr, kl=kl)
        history.append(result)
        logger.info(
            "trial %d/%d: cost=%.4f (refusal=%.1f%%, KL=%.4f)",
            len(history), cfg.n_trials, cost, 100 * rr, kl,
        )
        if on_trial is not None:
            on_trial(len(history), cfg.n_trials)
        return result

    best = _run_optuna(evaluate_params, n_layers, n_candidates, cfg)
    if best is None:
        best = _run_random(evaluate_params, n_layers, n_candidates, cfg)

    best_result = min(history, key=lambda r: r.cost)
    logger.info(
        "best: cost=%.4f (refusal=%.1f%%, KL=%.4f) after %d trials",
        best_result.cost, 100 * best_result.refusal_rate, best_result.kl, len(history),
    )
    return best_result.params, history


def _run_optuna(evaluate_params, n_layers, n_candidates, cfg: OptimizeConfig):
    """Run TPE search if Optuna is available; return the best params or None."""
    try:
        import optuna
    except ImportError:
        logger.info("optuna not installed; falling back to random search "
                    "(`pip install optuna` for TPE-guided optimization)")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=cfg.n_startup_trials, seed=cfg.seed
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = sample_params(trial, n_layers, n_candidates, cfg)
        result = evaluate_params(params)
        trial.set_user_attr("refusal_rate", result.refusal_rate)
        trial.set_user_attr("kl", result.kl)
        return result.cost

    study.optimize(objective, n_trials=cfg.n_trials)
    # Replay the winning trial's suggestions to rebuild the params object.
    return sample_params(
        optuna.trial.FixedTrial(study.best_trial.params), n_layers, n_candidates, cfg
    )


def _run_random(evaluate_params, n_layers, n_candidates, cfg: OptimizeConfig) -> AblationParams:
    rng = random.Random(cfg.seed)
    best_params: Optional[AblationParams] = None
    best_cost = float("inf")
    for _ in range(cfg.n_trials):
        params = sample_params(_RandomSuggester(rng), n_layers, n_candidates, cfg)
        result = evaluate_params(params)
        if result.cost < best_cost:
            best_cost, best_params = result.cost, params
    assert best_params is not None
    return best_params
