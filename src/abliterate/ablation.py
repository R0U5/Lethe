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
from typing import Optional

import torch
from torch import nn

from .config import AblationConfig, ModelConfig
from .model_utils import (
    get_decoder_layers,
    get_embedding_module,
    iter_residual_write_modules,
)

logger = logging.getLogger("abliterate")


# --------------------------------------------------------------------------- #
# Core projection math
# --------------------------------------------------------------------------- #
def _hidden_axis_for(module: nn.Module) -> int:
    """Which weight axis is the hidden (residual) dimension for this module."""
    if isinstance(module, nn.Linear):
        return 0                     # weight [out=hidden, in]
    if isinstance(module, nn.Embedding):
        return 1                     # weight [vocab, hidden]
    if type(module).__name__ == "Conv1D":
        return 1                     # weight [in, out=hidden]
    raise TypeError(f"don't know how to orthogonalize {type(module).__name__}")


def orthogonalize_weight_(weight: torch.Tensor, direction: torch.Tensor, hidden_axis: int) -> None:
    """In-place: remove the ``direction`` component from ``weight``'s hidden axis."""
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
    w -= proj
    weight.data.copy_(w.to(orig_dtype))


def project_out_activation(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Return ``x`` with its ``direction`` component removed along the last axis."""
    d = direction.to(x.dtype).to(x.device)
    d = d / d.norm().clamp_min(1e-8)
    proj = (x @ d).unsqueeze(-1) * d
    return x - proj


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
