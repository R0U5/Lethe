"""Applying a refusal direction to a model.

Two mechanisms, same effect -- remove the refusal direction ``r`` from
everything written into the residual stream:

* **Weight orthogonalization** (permanent): rewrite each residual-writing weight
  matrix ``W`` so its output can never have a component along ``r``. For a matrix
  whose output axis is the hidden dimension, ``W' = W - r (rᵀW)``; for one whose
  input-shaped rows live in hidden space (embeddings, gpt2 Conv1D), the
  projection is taken on the other axis. The result is a normal model you can
  save and serve anywhere.

* **Inference-time hooks** (reversible): forward hooks subtract ``(x·r) r`` from
  the embedding output and every decoder layer's output. Nothing on disk
  changes; used for fast experimentation and layer selection.

The math is applied in float32 and cast back to the weight's dtype.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from .config import AblationConfig, ModelConfig
from .model_utils import (
    get_decoder_layers,
    get_embedding_module,
    is_quantized,
    iter_residual_write_modules,
)

logger = logging.getLogger("abliterate")


# --------------------------------------------------------------------------- #
# Core projection math
# --------------------------------------------------------------------------- #
def _reject_quantized(model: nn.Module) -> None:
    if is_quantized(model):
        raise ValueError(
            "cannot orthogonalize a quantized model's packed weights. Save a "
            "runtime bundle instead (AblationBundle / `--save bundle`) and apply "
            "it with hooks at load time, or load in full precision to bake it in."
        )


def _hidden_axis_for(module: nn.Module) -> int:
    """Which weight axis is the hidden (residual) dimension for this module."""
    if isinstance(module, nn.Linear):
        return 0                     # weight [out=hidden, in]
    if isinstance(module, nn.Embedding):
        return 1                     # weight [vocab, hidden]
    if type(module).__name__ == "Conv1D":
        return 1                     # weight [in, out=hidden]
    raise TypeError(f"don't know how to orthogonalize {type(module).__name__}")


def orthogonalize_weight_(
    weight: torch.Tensor,
    direction: torch.Tensor,
    hidden_axis: int,
    strength: float = 1.0,
) -> None:
    """In-place: remove ``strength`` x the ``direction`` component from ``weight``.

    ``strength == 1.0`` is exact orthogonalization; ``0 < strength < 1`` is the
    partial/regularized ablation that preserves more task-relevant information
    (complete removal tends to over-damage the model). ``strength`` may exceed 1
    to over-suppress.
    """
    if strength == 0.0:
        return
    orig_dtype = weight.dtype
    w = weight.data.to(torch.float32)
    d = direction.to(torch.float32).to(w.device)
    d = d / d.norm().clamp_min(1e-8)
    if hidden_axis == 0:
        # columns of W are hidden-space vectors: W - d (dᵀ W)
        proj = torch.outer(d, d @ w)
    else:
        # rows of W are hidden-space vectors: W - (W d) dᵀ
        proj = torch.outer(w @ d, d)
    w -= strength * proj
    weight.data.copy_(w.to(orig_dtype))


def project_out_activation(
    x: torch.Tensor, direction: torch.Tensor, strength: float = 1.0
) -> torch.Tensor:
    """Return ``x`` with ``strength`` x its ``direction`` component removed."""
    if strength == 0.0:
        return x
    d = direction.to(x.dtype).to(x.device)
    d = d / d.norm().clamp_min(1e-8)
    proj = (x @ d).unsqueeze(-1) * d
    return x - strength * proj


# --------------------------------------------------------------------------- #
# Permanent weight orthogonalization
# --------------------------------------------------------------------------- #
@torch.no_grad()
def orthogonalize_model(
    model: nn.Module,
    direction: torch.Tensor,
    *,
    model_cfg: Optional[ModelConfig] = None,
    ablation_cfg: Optional[AblationConfig] = None,
    hidden_size: Optional[int] = None,
) -> int:
    """Bake the refusal direction out of a model's residual-writing weights.

    Returns the number of weight matrices modified. Which sites are touched
    (embedding / attention-out / MLP-down) is controlled by ``ablation_cfg``.
    """
    _reject_quantized(model)
    ablation_cfg = ablation_cfg or AblationConfig()
    if hidden_size is None:
        hidden_size = get_embedding_module(model).weight.shape[-1]

    direction = direction.detach().to(torch.float32).flatten()
    if direction.numel() != hidden_size:
        raise ValueError(
            f"direction has {direction.numel()} dims, expected hidden_size {hidden_size}"
        )

    modified = 0
    if ablation_cfg.ablate_embed:
        emb = get_embedding_module(model)
        orthogonalize_weight_(emb.weight, direction, _hidden_axis_for(emb))
        modified += 1

    layers = get_decoder_layers(model, model_cfg)
    for layer in layers:
        for role, module in iter_residual_write_modules(
            layer,
            hidden_size,
            cfg=model_cfg,
            include_attn=ablation_cfg.ablate_attn,
            include_mlp=ablation_cfg.ablate_mlp,
        ):
            orthogonalize_weight_(module.weight, direction, _hidden_axis_for(module))
            modified += 1

    logger.info("orthogonalized %d weight matrices", modified)
    if modified <= 1:
        logger.warning(
            "only %d matrix modified -- residual-write projections may not have "
            "matched this architecture; check attn_out_names/mlp_out_names",
            modified,
        )
    return modified


# --------------------------------------------------------------------------- #
# Inference-time ablation hooks
# --------------------------------------------------------------------------- #
class AblationHooks:
    """Context manager that projects a direction out of the residual stream.

    Registers forward hooks on the embedding and every decoder layer; removing
    them (on ``__exit__`` or ``remove()``) fully restores the model. The hook's
    return value replaces each layer's output, so the ablated residual stream is
    what propagates forward and what generation sees.

    Observability caveat: on transformers >= 5, ``output_hidden_states`` is
    captured by the library's own (prepended) forward hooks that fire before
    these, so the intermediate ``hidden_states`` snapshots reflect the
    *pre-ablation* values even though the forward computation is ablated. To
    observe the effect, read the final ``last_hidden_state`` or generate.
    """

    def __init__(
        self,
        model: nn.Module,
        direction: torch.Tensor,
        *,
        model_cfg: Optional[ModelConfig] = None,
        hook_embed: bool = True,
    ):
        self.model = model
        self.direction = direction.detach().flatten()
        self.model_cfg = model_cfg
        self.hook_embed = hook_embed
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "AblationHooks":
        d = self.direction

        def layer_hook(module, inputs, output):
            if isinstance(output, tuple):
                return (project_out_activation(output[0], d), *output[1:])
            return project_out_activation(output, d)

        def embed_hook(module, inputs, output):
            return project_out_activation(output, d)

        if self.hook_embed:
            emb = get_embedding_module(self.model)
            self._handles.append(emb.register_forward_hook(embed_hook))

        for layer in get_decoder_layers(self.model, self.model_cfg):
            self._handles.append(layer.register_forward_hook(layer_hook))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Weighted / optimized ablation
# --------------------------------------------------------------------------- #
@dataclass
class LayerWeightKernel:
    """Per-layer ablation strength as a tent function over decoder depth.

    Strength equals ``max_weight`` at layer ``max_weight_position`` and falls
    linearly to ``min_weight`` over ``min_weight_distance`` layers, staying at
    ``min_weight`` beyond that. The default (all-1.0) kernel reproduces uniform
    full ablation.
    """

    max_weight: float = 1.0
    max_weight_position: float = 0.0
    min_weight: float = 1.0
    min_weight_distance: float = 1.0

    def weight_at(self, layer_index: int) -> float:
        if self.min_weight_distance <= 0:
            return self.max_weight
        frac = min(abs(layer_index - self.max_weight_position) / self.min_weight_distance, 1.0)
        return self.max_weight + (self.min_weight - self.max_weight) * frac

    @classmethod
    def from_dict(cls, d: dict) -> "LayerWeightKernel":
        return cls(**{k: d[k] for k in vars(cls()) if k in d})


@dataclass
class AblationParams:
    """A full ablation specification the optimizer searches over.

    * direction: either a fixed hidden-state ``direction_index`` used at every
      layer, or ``per_layer_direction`` (each decoder layer uses its own
      difference-of-means direction).
    * separate ``attn_kernel`` / ``mlp_kernel`` because MLP-down interventions
      damage the model more than attention-out ones and want their own strength.
    """

    direction_index: Optional[int] = None
    per_layer_direction: bool = False
    attn_kernel: LayerWeightKernel = field(default_factory=LayerWeightKernel)
    mlp_kernel: LayerWeightKernel = field(default_factory=LayerWeightKernel)
    ablate_embed: bool = False
    embed_strength: float = 1.0

    def to_dict(self) -> dict:
        return {
            "direction_index": self.direction_index,
            "per_layer_direction": self.per_layer_direction,
            "attn_kernel": vars(self.attn_kernel),
            "mlp_kernel": vars(self.mlp_kernel),
            "ablate_embed": self.ablate_embed,
            "embed_strength": self.embed_strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AblationParams":
        return cls(
            direction_index=d.get("direction_index"),
            per_layer_direction=bool(d.get("per_layer_direction", False)),
            attn_kernel=LayerWeightKernel.from_dict(d.get("attn_kernel", {})),
            mlp_kernel=LayerWeightKernel.from_dict(d.get("mlp_kernel", {})),
            ablate_embed=bool(d.get("ablate_embed", False)),
            embed_strength=float(d.get("embed_strength", 1.0)),
        )


def _direction_for_layer(
    directions: torch.Tensor, layer_index: int, params: AblationParams
) -> torch.Tensor:
    """Pick the direction row for a decoder layer given the params.

    ``directions`` is ``[num_hidden_states, hidden]`` (row 0 = embedding, row i =
    output of decoder layer i-1). Per-layer mode maps decoder layer ``l`` to its
    output residual (row ``l+1``).
    """
    n = directions.shape[0]
    if params.per_layer_direction:
        idx = min(layer_index + 1, n - 1)
    else:
        idx = params.direction_index if params.direction_index is not None else n - 1
        idx = idx if idx >= 0 else n + idx
    return directions[idx]


def _hook_module_output(module: nn.Module, direction: torch.Tensor, strength: float):
    """Register a forward hook projecting the module's output; return handle."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (project_out_activation(output[0], direction, strength), *output[1:])
        return project_out_activation(output, direction, strength)

    return module.register_forward_hook(hook)


class WeightedAblationHooks:
    """Reversible weighted ablation: the hook-based twin of ``apply_weighted_ablation``.

    Hooks the attention-out and MLP-down projection *modules* in each layer (so
    attention and MLP get independent strengths) plus, optionally, the embedding.
    Projecting a module's output is exactly equivalent to orthogonalizing that
    module's weight, but reversible -- which is what makes optimizer trials cheap.
    """

    def __init__(
        self,
        model: nn.Module,
        directions: torch.Tensor,
        params: AblationParams,
        *,
        model_cfg: Optional[ModelConfig] = None,
        hidden_size: Optional[int] = None,
    ):
        self.model = model
        self.directions = directions.detach()
        self.params = params
        self.model_cfg = model_cfg
        self.hidden_size = hidden_size or get_embedding_module(model).weight.shape[-1]
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "WeightedAblationHooks":
        p = self.params
        for l, layer in enumerate(get_decoder_layers(self.model, self.model_cfg)):
            direction = _direction_for_layer(self.directions, l, p)
            if direction.norm() == 0:
                continue
            w_attn = p.attn_kernel.weight_at(l)
            w_mlp = p.mlp_kernel.weight_at(l)
            for role, module in iter_residual_write_modules(
                layer, self.hidden_size, cfg=self.model_cfg
            ):
                strength = w_attn if role == "attn_out" else w_mlp
                if strength != 0.0:
                    self._handles.append(_hook_module_output(module, direction, strength))
        if p.ablate_embed and p.embed_strength != 0.0:
            d0 = _direction_for_layer(self.directions, 0, p)
            emb = get_embedding_module(self.model)
            self._handles.append(_hook_module_output(emb, d0, p.embed_strength))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@torch.no_grad()
def apply_weighted_ablation(
    model: nn.Module,
    directions: torch.Tensor,
    params: AblationParams,
    *,
    model_cfg: Optional[ModelConfig] = None,
    hidden_size: Optional[int] = None,
) -> int:
    """Permanently bake a weighted ablation into the model's weights.

    Applies, per decoder layer, ``params.attn_kernel`` strength to the
    attention-out projection and ``params.mlp_kernel`` strength to the MLP-down
    projection, using the fixed or per-layer direction. Returns the number of
    weight matrices modified.
    """
    _reject_quantized(model)
    if hidden_size is None:
        hidden_size = get_embedding_module(model).weight.shape[-1]

    modified = 0
    for l, layer in enumerate(get_decoder_layers(model, model_cfg)):
        direction = _direction_for_layer(directions, l, params).flatten()
        if direction.norm() == 0:
            continue
        w_attn = params.attn_kernel.weight_at(l)
        w_mlp = params.mlp_kernel.weight_at(l)
        for role, module in iter_residual_write_modules(layer, hidden_size, cfg=model_cfg):
            strength = w_attn if role == "attn_out" else w_mlp
            if strength != 0.0:
                orthogonalize_weight_(
                    module.weight, direction, _hidden_axis_for(module), strength=strength
                )
                modified += 1

    if params.ablate_embed and params.embed_strength != 0.0:
        d0 = _direction_for_layer(directions, 0, params).flatten()
        emb = get_embedding_module(model)
        orthogonalize_weight_(
            emb.weight, d0, _hidden_axis_for(emb), strength=params.embed_strength
        )
        modified += 1

    logger.info("applied weighted ablation to %d weight matrices", modified)
    return modified
