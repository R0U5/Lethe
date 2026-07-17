"""Loading contrastive prompt sets.

A "source" is one of:
    * a path to a newline-delimited text file ('#' comments and blanks ignored),
    * a built-in name ("advbench", "alpaca") loaded via the `datasets` library, or
    * ``None``, which falls back to the packaged sample files under data/.
"""

from __future__ import annotations

import logging
import random
from importlib import resources
from pathlib import Path
from typing import Optional

logger = logging.getLogger("abliterate")

# Built-in dataset recipes: name -> (hf_repo, split, column, config_name)
_KNOWN = {
    "advbench": ("walledai/AdvBench", "train", "prompt", None),
    "alpaca": ("tatsu-lab/alpaca", "train", "instruction", None),
    # The contrastive pair popularized by mlabonne / used by Heretic.
    "harmful_behaviors": ("mlabonne/harmful_behaviors", "train", "text", None),
    "harmless_alpaca": ("mlabonne/harmless_alpaca", "train", "text", None),
}


def load_prompts(source: Optional[str], *, default_file: str) -> list[str]:
    """Return a list of prompt strings from a source (see module docstring)."""
    if source is None:
        return _read_text_file(_packaged_data_path(default_file))

    key = source.lower()
    if key in _KNOWN:
        return _load_known_dataset(key)

    path = Path(source)
    if path.exists():
        return _read_text_file(path)

    raise FileNotFoundError(
        f"prompt source {source!r} is neither a file nor a known dataset "
        f"({sorted(_KNOWN)})"
    )


def sample_prompts(prompts: list[str], n: int, seed: int) -> list[str]:
    """Deterministically sample up to ``n`` prompts without replacement."""
    if n >= len(prompts):
        return list(prompts)
    rng = random.Random(seed)
    return rng.sample(prompts, n)


def _read_text_file(path: Path) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    if not lines:
        raise ValueError(f"no usable prompts found in {path}")
    logger.info("loaded %d prompts from %s", len(lines), path)
    return lines


def _packaged_data_path(filename: str) -> Path:
    """Locate a file shipped in the repo-level data/ directory.

    Works both from a source checkout and, best-effort, an installed package.
    """
    # Source checkout: <repo>/data/<filename> (this file is src/abliterate/).
    candidate = Path(__file__).resolve().parents[2] / "data" / filename
    if candidate.exists():
        return candidate
    # Installed layout fallback.
    try:
        res = resources.files("abliterate").joinpath("data", filename)
        if res.is_file():
            return Path(str(res))
    except (ModuleNotFoundError, AttributeError):
        pass
    raise FileNotFoundError(
        f"packaged sample data {filename!r} not found; pass an explicit source"
    )


def _load_known_dataset(key: str) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"loading {key!r} needs the datasets library; "
            f"`pip install datasets` or pass a file path instead"
        ) from exc

    repo, split, column, config_name = _KNOWN[key]
    logger.info("loading %s from %s[%s]", key, repo, split)
    ds = load_dataset(repo, config_name, split=split) if config_name else load_dataset(repo, split=split)

    # Column names vary between mirrors; fall back to the first text-like column.
    if column not in ds.column_names:
        candidates = [c for c in ("text", "prompt", "instruction", "goal", "behavior")
                      if c in ds.column_names]
        if not candidates:
            raise ValueError(
                f"dataset {key!r} has columns {ds.column_names}; none look like prompts"
            )
        column = candidates[0]

    prompts = [str(row[column]) for row in ds if row.get(column)]
    if not prompts:
        raise ValueError(f"no prompts extracted from dataset {key!r}")
    return prompts
