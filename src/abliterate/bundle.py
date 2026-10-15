"""Runtime abliteration bundles.

A *bundle* is a tiny, portable artifact -- the refusal direction(s) plus the
weighted-ablation parameters -- that reproduces an abliteration at inference time
via forward hooks, without ever rewriting the model's weights. It is the
LoRA-adapter equivalent of an abliteration:

* It applies on top of the **original** weights, so it works on quantized models
  (4/8-bit bitsandbytes) where the packed weights cannot be edited in place.
* It is a few KB -- ship it alongside the base model instead of a full copy.
* It is reversible: detach the hooks and the model is exactly as it was.

On disk a bundle is a small directory::

    my-abliteration/
      bundle.json            # params + metadata (human-readable)
      directions.safetensors # the direction tensor

Apply it in Python::

    from abliterate.bundle import AblationBundle, load_abliterated
    model, tokenizer, _ = load_abliterated("Qwen/Qwen2.5-1.5B-Instruct",
                                           "my-abliteration", load_in_4bit=True)
    # `model` now generates as if abliterated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

from .ablation import (
    AblationParams,
    WeightedAblationHooks,
    apply_weighted_ablation,
)
from .config import ModelConfig
from .model_utils import ModelBundle

logger = logging.getLogger("abliterate")

_BUNDLE_VERSION = 1
_JSON_NAME = "bundle.json"
_TENSOR_NAME = "directions.safetensors"


@dataclass
class AblationBundle:
    """Everything needed to reproduce an abliteration at load time."""

    directions: torch.Tensor            # [num_hidden_states, hidden]
    params: AblationParams
    hidden_size: int
    num_layers: int
    source_model: str = ""
    metrics: dict = field(default_factory=dict)
    # Architecture overrides so hooks find the right modules without a config.
    model_overrides: dict = field(default_factory=dict)
    version: int = _BUNDLE_VERSION

    # --- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the bundle to a directory. Returns the directory path."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        save_file({"directions": self.directions.detach().cpu().contiguous().float()},
                  str(out / _TENSOR_NAME))
        meta = {
            "version": self.version,
            "kind": "abliteration-bundle",
            "source_model": self.source_model,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "params": self.params.to_dict(),
            "metrics": self.metrics,
            "model_overrides": self.model_overrides,
        }
        (out / _JSON_NAME).write_text(json.dumps(meta, indent=2))
        logger.info("saved abliteration bundle to %s", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "AblationBundle":
        src = Path(path)
        meta = json.loads((src / _JSON_NAME).read_text())
        directions = load_file(str(src / _TENSOR_NAME))["directions"]
        return cls(
            directions=directions,
            params=AblationParams.from_dict(meta.get("params", {})),
            hidden_size=int(meta.get("hidden_size", directions.shape[-1])),
            num_layers=int(meta.get("num_layers", 0)),
            source_model=meta.get("source_model", ""),
            metrics=meta.get("metrics", {}),
            model_overrides=meta.get("model_overrides", {}),
            version=int(meta.get("version", _BUNDLE_VERSION)),
        )

    # --- application --------------------------------------------------------
    def _model_cfg(self, model_cfg: Optional[ModelConfig]) -> Optional[ModelConfig]:
        """Prefer an explicit config; otherwise synthesize one from overrides."""
        if model_cfg is not None:
            return model_cfg
        if not self.model_overrides:
            return None
        return ModelConfig(path="", **{
            k: v for k, v in self.model_overrides.items()
            if k in ("attn_out_names", "mlp_out_names", "decoder_layers_path")
        })

    def hooks(self, model, *, model_cfg: Optional[ModelConfig] = None) -> WeightedAblationHooks:
        """Return a (not-yet-entered) hook context: ``with bundle.hooks(m): ...``."""
        return WeightedAblationHooks(
            model, self.directions, self.params,
            model_cfg=self._model_cfg(model_cfg), hidden_size=self.hidden_size,
        )

    def attach(self, model, *, model_cfg: Optional[ModelConfig] = None) -> WeightedAblationHooks:
        """Attach the hooks *persistently* and return them (call ``.remove()`` to undo).

        This is how you serve a quantized model as abliterated: load the model
        however you like, attach the bundle, then generate normally.
        """
        return self.hooks(model, model_cfg=model_cfg).__enter__()

    def apply_permanently(self, model, *, model_cfg: Optional[ModelConfig] = None) -> int:
        """Bake the bundle into full-precision weights (not for quantized models)."""
        return apply_weighted_ablation(
            model, self.directions, self.params,
            model_cfg=self._model_cfg(model_cfg), hidden_size=self.hidden_size,
        )


def build_bundle(
    directions: torch.Tensor,
    params: AblationParams,
    *,
    hidden_size: int,
    num_layers: int,
    source_model: str = "",
    metrics: Optional[dict] = None,
    model_cfg: Optional[ModelConfig] = None,
) -> AblationBundle:
    """Assemble a bundle, capturing architecture overrides from ``model_cfg``."""
    overrides = {}
    if model_cfg is not None:
        for key in ("attn_out_names", "mlp_out_names", "decoder_layers_path"):
            val = getattr(model_cfg, key, None)
            if val:
                overrides[key] = val
    return AblationBundle(
        directions=directions.detach().cpu(),
        params=params,
        hidden_size=hidden_size,
        num_layers=num_layers,
        source_model=source_model,
        metrics=metrics or {},
        model_overrides=overrides,
    )


def load_abliterated(
    model_path: str,
    bundle_path: str | Path,
    *,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = False,
) -> tuple[ModelBundle, AblationBundle, WeightedAblationHooks]:
    """Load a base model (optionally quantized) with a bundle attached.

    Returns ``(model_bundle, ablation_bundle, hooks)``. The model generates as if
    abliterated; ``hooks.remove()`` restores it. Loading in 4/8-bit requires a
    CUDA GPU and bitsandbytes, and is the intended way to run large models with a
    bundle without the memory of a full-precision copy.
    """
    from .model_utils import load_model_and_tokenizer

    ablation = AblationBundle.load(bundle_path)
    mcfg = ModelConfig(
        path=model_path, dtype=dtype, device_map=device_map,
        load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
        attn_out_names=ablation.model_overrides.get("attn_out_names"),
        mlp_out_names=ablation.model_overrides.get("mlp_out_names"),
        decoder_layers_path=ablation.model_overrides.get("decoder_layers_path"),
    )
    bundle = load_model_and_tokenizer(mcfg)
    hooks = ablation.attach(bundle.model, model_cfg=mcfg)
    return bundle, ablation, hooks
