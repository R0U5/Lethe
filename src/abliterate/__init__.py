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

from .config import AblationConfig, Config, DataConfig, ModelConfig
from .model_utils import load_model_and_tokenizer
from .activations import collect_mean_activations
from .directions import RefusalDirections, compute_refusal_directions
from .ablation import AblationHooks, orthogonalize_model

__all__ = [
    "AblationConfig",
    "Config",
    "DataConfig",
    "ModelConfig",
    "load_model_and_tokenizer",
    "collect_mean_activations",
    "RefusalDirections",
    "compute_refusal_directions",
    "orthogonalize_model",
    "AblationHooks",
]

__version__ = "0.1.0"
