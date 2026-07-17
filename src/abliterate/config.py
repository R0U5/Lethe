"""Configuration objects for the abliteration pipeline.

Configs can be built in code or loaded from YAML. Every field has a sensible
default except the model source, so a minimal config is just::

    model:
      path: Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """Where the weights come from and how to load them.

    ``path`` is any HuggingFace model id or local directory. It may be a full
    model, or a base model that a LoRA/PEFT adapter (``lora_adapter``) is merged
    onto -- the latter is how reforge fine-tunes are consumed, but any PEFT
    adapter works.
    """

    path: str
    lora_adapter: Optional[str] = None
    dtype: str = "bfloat16"          # float32 | float16 | bfloat16
    device_map: str = "auto"         # passed to from_pretrained
    trust_remote_code: bool = False
    tokenizer_path: Optional[str] = None  # defaults to `path`

    # Optional overrides for architectures whose module names are not in the
    # built-in registry. See model_utils.ModulePatterns.
    attn_out_names: Optional[list[str]] = None
    mlp_out_names: Optional[list[str]] = None
    decoder_layers_path: Optional[str] = None


@dataclass
class DataConfig:
    """Contrastive prompt sources."""

    harmful: Optional[str] = None    # file path, or a known dataset name
    harmless: Optional[str] = None
    n_samples: int = 128             # prompts sampled per side
    seed: int = 42


@dataclass
class AblationConfig:
    """How activations are collected and how the direction is chosen/applied."""

    position: int = -1               # token index for activation extraction
    batch_size: int = 16
    max_length: int = 512
    # Direction selection: pick the layer explicitly, or by fractional depth.
    layer_index: Optional[int] = None
    layer_fraction: float = 0.6
    # Which residual-writing sites to orthogonalize when applying to weights.
    ablate_embed: bool = True
    ablate_attn: bool = True
    ablate_mlp: bool = True


@dataclass
class OptimizeConfig:
    """Automated abliteration search (TPE/Optuna, co-minimizing refusals + KL)."""

    n_trials: int = 100
    n_startup_trials: int = 30       # random exploration before TPE kicks in
    n_eval_harmful: int = 32         # held-out prompts scored for refusals
    n_eval_harmless: int = 32        # held-out prompts scored for KL divergence
    max_new_tokens: int = 32         # generation length during refusal scoring
    kl_weight: float = 1.0           # cost = refusal_rate + kl_weight * KL
    seed: int = 0
    # Search-space bounds.
    max_weight_hi: float = 1.5       # upper bound on per-component ablation strength
    dir_band: tuple[float, float] = (0.3, 0.95)  # fractional-depth band for fixed index
    optimize_embed: bool = False     # also search an embedding ablation strength


@dataclass
class Config:
    model: ModelConfig
    data: DataConfig = field(default_factory=DataConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    optimize: OptimizeConfig = field(default_factory=OptimizeConfig)
    output_dir: str = "output"

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        raw = dict(raw)
        model_raw = raw.get("model")
        if not model_raw or "path" not in model_raw:
            raise ValueError("config must contain `model.path`")
        model = _build(ModelConfig, model_raw)
        data = _build(DataConfig, raw.get("data", {}))
        ablation = _build(AblationConfig, raw.get("ablation", {}))
        optimize = _build(OptimizeConfig, raw.get("optimize", {}))
        return Config(
            model=model,
            data=data,
            ablation=ablation,
            optimize=optimize,
            output_dir=raw.get("output_dir", "output"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, raw: dict[str, Any]):
    """Construct a dataclass, ignoring unknown keys with a clear error."""
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)
