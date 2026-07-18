"""Model-agnostic abliteration of refusal directions from transformer LMs.

Public API:
    from abliterate import (
        Config,
        load_model_and_tokenizer,
        collect_mean_activations,
        compute_refusal_directions,
        RefusalDirections,
        orthogonalize_model,
        AblationHooks,
    )
"""

from .config import AblationConfig, Config, DataConfig, ModelConfig, OptimizeConfig
from .model_utils import load_model_and_tokenizer
from .activations import collect_mean_activations
from .directions import RefusalDirections, compute_refusal_directions
from .ablation import (
    AblationHooks,
    AblationParams,
    LayerWeightKernel,
    WeightedAblationHooks,
    apply_weighted_ablation,
    orthogonalize_model,
)
from .bundle import AblationBundle, build_bundle, load_abliterated
from .metrics import kl_divergence, next_token_logprobs
from .optimize import optimize_ablation

__all__ = [
    "AblationConfig",
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimizeConfig",
    "load_model_and_tokenizer",
    "collect_mean_activations",
    "RefusalDirections",
    "compute_refusal_directions",
    "orthogonalize_model",
    "AblationHooks",
    "AblationParams",
    "LayerWeightKernel",
    "WeightedAblationHooks",
    "apply_weighted_ablation",
    "optimize_ablation",
    "kl_divergence",
    "next_token_logprobs",
    "AblationBundle",
    "build_bundle",
    "load_abliterated",
]

__version__ = "0.1.0"
