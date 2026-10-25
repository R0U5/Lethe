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
from .model_utils import (
    load_model_and_tokenizer,
    make_generation_config_saveable,
    save_model_and_tokenizer,
)
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
from .evaluate import (
    CAUTION_MARKERS,
    REFUSAL_MARKERS,
    evaluate_refusal,
    generate_completions,
    has_caution,
    is_refusal,
    refusal_rate,
    refusal_stats,
    set_caution_markers,
    set_refusal_markers,
)
from .optimize import optimize_ablation

__all__ = [
    "AblationConfig",
    "Config",
    "DataConfig",
    "ModelConfig",
    "OptimizeConfig",
    "load_model_and_tokenizer",
    "make_generation_config_saveable",
    "save_model_and_tokenizer",
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
    "evaluate_refusal",
    "generate_completions",
    "is_refusal",
    "has_caution",
    "refusal_rate",
    "refusal_stats",
    "set_refusal_markers",
    "set_caution_markers",
    "REFUSAL_MARKERS",
    "CAUTION_MARKERS",
    "AblationBundle",
    "build_bundle",
    "load_abliterated",
]

__version__ = "0.1.0"
