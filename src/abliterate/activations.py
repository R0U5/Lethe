"""Collecting residual-stream activations for contrastive prompt sets.

We run the model over a batch of prompts with ``output_hidden_states=True`` and
read the residual stream at a single token position (default: the last real
token). Rather than store every activation, we accumulate a running mean per
hidden-state index, which is all the difference-of-means direction needs and
keeps memory flat regardless of sample count.

``hidden_states`` from a HF forward is a tuple of length ``num_layers + 1``:
index 0 is the embedding output, index ``i`` is the output of decoder layer
``i - 1``. We keep that indexing so a "direction for layer i" refers to the
residual stream entering layer ``i``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tqdm import tqdm

from .chat import format_batch
from .config import AblationConfig
from .model_utils import ModelBundle, get_base_model

logger = logging.getLogger("abliterate")


@torch.no_grad()
def collect_mean_activations(
    bundle: ModelBundle,
    prompts: list[str],
    cfg: AblationConfig,
    *,
    system: Optional[str] = None,
    desc: str = "activations",
) -> torch.Tensor:
    """Return mean residual activations, shape ``[num_hidden_states, hidden]``.

    The mean is taken over ``prompts`` at token position ``cfg.position``.
    """
    tokenizer = bundle.tokenizer
    # Run the base transformer only: we need hidden states, not vocab logits.
    base = get_base_model(bundle.model)
    device = _model_device(base)

    running_sum: Optional[torch.Tensor] = None
    count = 0

    for start in tqdm(range(0, len(prompts), cfg.batch_size), desc=desc):
        batch = prompts[start : start + cfg.batch_size]
        rendered = format_batch(tokenizer, batch, system=system)
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
        ).to(device)

        out = base(**enc, output_hidden_states=True, use_cache=False)
        # tuple[L+1] of [B, S, H]; pick the target position before stacking so we
        # never materialize the full [L+1, B, S, H] tensor.
        picked = torch.stack(
            [h[:, cfg.position, :] for h in out.hidden_states], dim=0
        )                                                # [L+1, B, H]
        batch_sum = picked.sum(dim=1).to(torch.float32)  # [L+1, H]

        if running_sum is None:
            running_sum = torch.zeros_like(batch_sum)
        running_sum += batch_sum
        count += len(batch)

    if running_sum is None or count == 0:
        raise ValueError("no prompts provided to collect_mean_activations")

    return (running_sum / count).cpu()


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - models always have params
        return torch.device("cpu")
