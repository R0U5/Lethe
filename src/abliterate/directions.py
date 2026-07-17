"""Computing, storing, and selecting refusal directions.

The refusal direction at a hidden-state index is the (normalized) difference of
mean activations between refusal-eliciting and benign prompts::

    d_i = normalize(mean_harmful_i - mean_harmless_i)

We keep one candidate per hidden-state index and pick one to apply. Selection
defaults to a fractional depth (mid-to-late layers carry the cleanest refusal
signal), or an explicit index, or -- with a scorer -- the candidate that best
suppresses refusals on held-out prompts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .config import AblationConfig

logger = logging.getLogger("abliterate")


@dataclass
class RefusalDirections:
    """Per-hidden-state candidate directions and provenance metadata.

    ``directions`` has shape ``[num_hidden_states, hidden]``; row ``i`` is the
    unit direction for the residual stream at hidden-state index ``i`` (index 0
    is the embedding output).
    """

    directions: torch.Tensor          # [L+1, H], each row unit-norm (or zero)
    raw_norms: torch.Tensor           # [L+1] pre-normalization magnitudes
    model_path: str = ""
    n_harmful: int = 0
    n_harmless: int = 0

    @property
    def num_candidates(self) -> int:
        return self.directions.shape[0]

    def get(self, index: int) -> torch.Tensor:
        return self.directions[index]

    def select_index(self, cfg: AblationConfig) -> int:
        """Resolve the layer index to use from config (explicit or fractional)."""
        n = self.num_candidates
        if cfg.layer_index is not None:
            idx = cfg.layer_index if cfg.layer_index >= 0 else n + cfg.layer_index
            if not 0 <= idx < n:
                raise IndexError(f"layer_index {cfg.layer_index} out of range [0,{n})")
            return idx
        frac = min(max(cfg.layer_fraction, 0.0), 1.0)
        return max(1, min(n - 1, round(frac * (n - 1))))

    def select_by_score(
        self,
        scorer: Callable[[int, torch.Tensor], float],
        *,
        candidate_indices: Optional[list[int]] = None,
        higher_is_better: bool = True,
    ) -> tuple[int, dict[int, float]]:
        """Choose the candidate that optimizes ``scorer(index, direction)``.

        Returns the winning index and the full score map. Zero-norm candidates
        are skipped.
        """
        if candidate_indices is None:
            candidate_indices = list(range(1, self.num_candidates))
        scores: dict[int, float] = {}
        for idx in candidate_indices:
            if self.raw_norms[idx] <= 0:
                continue
            scores[idx] = scorer(idx, self.directions[idx])
        if not scores:
            raise ValueError("no scorable candidate directions")
        best = (max if higher_is_better else min)(scores, key=scores.get)
        return best, scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "directions": self.directions,
                "raw_norms": self.raw_norms,
                "model_path": self.model_path,
                "n_harmful": self.n_harmful,
                "n_harmless": self.n_harmless,
            },
            path,
        )
        # Human-readable sidecar for quick inspection.
        meta = {
            "num_candidates": self.num_candidates,
            "hidden_size": int(self.directions.shape[1]),
            "model_path": self.model_path,
            "n_harmful": self.n_harmful,
            "n_harmless": self.n_harmless,
            "raw_norms": [round(float(x), 5) for x in self.raw_norms.tolist()],
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        logger.info("saved %d candidate directions to %s", self.num_candidates, path)

    @staticmethod
    def load(path: str | Path) -> "RefusalDirections":
        blob = torch.load(path, map_location="cpu")
        return RefusalDirections(
            directions=blob["directions"],
            raw_norms=blob["raw_norms"],
            model_path=blob.get("model_path", ""),
            n_harmful=blob.get("n_harmful", 0),
            n_harmless=blob.get("n_harmless", 0),
        )


def compute_refusal_directions(
    mean_harmful: torch.Tensor,
    mean_harmless: torch.Tensor,
    *,
    model_path: str = "",
    n_harmful: int = 0,
    n_harmless: int = 0,
    eps: float = 1e-8,
) -> RefusalDirections:
    """Difference-of-means directions from paired per-layer mean activations.

    Both inputs are ``[num_hidden_states, hidden]``. Rows with near-zero
    magnitude are left as zero vectors (they carry no usable signal and must not
    be normalized).
    """
    if mean_harmful.shape != mean_harmless.shape:
        raise ValueError(
            f"shape mismatch: {tuple(mean_harmful.shape)} vs "
            f"{tuple(mean_harmless.shape)}"
        )
    diff = (mean_harmful - mean_harmless).to(torch.float32)
    norms = diff.norm(dim=-1)                                # [L+1]
    safe = norms.clamp_min(eps).unsqueeze(-1)               # [L+1, 1]
    unit = diff / safe
    # Zero out rows whose raw norm is essentially zero.
    mask = (norms > eps).unsqueeze(-1)
    unit = torch.where(mask, unit, torch.zeros_like(unit))
    return RefusalDirections(
        directions=unit,
        raw_norms=norms,
        model_path=model_path,
        n_harmful=n_harmful,
        n_harmless=n_harmless,
    )
