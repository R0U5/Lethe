"""Damage metrics for abliteration.

The single most important signal separating a good abliteration from a wrecked
model is how far the ablated model's output distribution drifts from the
original on *benign* inputs. We quantify that with KL divergence over the
next-token distribution at the end of harmless prompts.

Usage: capture ``next_token_logprobs`` on the original model once, then compare
against the ablated model (weights or hooks) each time via ``kl_divergence``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tqdm import tqdm

from .chat import format_batch
from .model_utils import ModelBundle

logger = logging.getLogger("abliterate")


@torch.no_grad()
def next_token_logprobs(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
    system: Optional[str] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Return last-token log-probabilities, shape ``[N, vocab]`` on CPU (fp32).

    With left padding the last position is the last real token, so this is the
    model's next-token distribution for each prompt. Kept on CPU so a reference
    copy can be cached cheaply across many optimizer trials.
    """
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = next(model.parameters()).device

    rows: list[torch.Tensor] = []
    iterator = range(0, len(prompts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="logprobs")
    for start in iterator:
        batch = prompts[start : start + batch_size]
        rendered = format_batch(tokenizer, batch, system=system)
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=not bool(getattr(tokenizer, "chat_template", None)),
        ).to(device)
        logits = model(**enc, use_cache=False).logits[:, -1, :]  # [B, V]
        rows.append(torch.log_softmax(logits.float(), dim=-1).cpu())
    return torch.cat(rows, dim=0)


def kl_divergence(reference: torch.Tensor, current: torch.Tensor) -> float:
    """Mean ``KL(reference || current)`` over rows of two log-prob tensors."""
    if reference.shape != current.shape:
        raise ValueError(
            f"logprob shape mismatch: {tuple(reference.shape)} vs {tuple(current.shape)}"
        )
    kl = (reference.exp() * (reference - current)).sum(dim=-1)  # [N]
    return float(kl.clamp_min(0.0).mean())
