"""The refusal direction must not be fitted on the prompts used to score it.

Regression: sampling direction and evaluation prompts from the same source with
different seeds looks like a hold-out but is not one -- ``sample_prompts`` returns
the whole list once ``n`` covers it, so a prompt file no larger than ``n_dir``
produced evaluation prompts that were a subset of the direction prompts. The 7B
search then reported 4.2% refusals on prompts it was fitted to while the same model
refused 30% of unseen prompts when served.
"""
from __future__ import annotations

import logging

from abliterate.pipeline import _sampled_holdout


def _write(tmp_path, name, prompts):
    path = tmp_path / name
    path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return path


def test_eval_prompts_are_disjoint_from_direction_prompts(tmp_path):
    """Given a file bigger than n_dir + n_eval, nothing may be reused."""
    prompts = [f"prompt number {i}" for i in range(200)]
    src = _write(tmp_path, "harmful.txt", prompts)

    dir_prompts, eval_prompts = _sampled_holdout(src, "harmful.txt", 128, 48, seed=0)

    assert len(dir_prompts) == 128
    assert len(eval_prompts) == 48
    assert not (set(dir_prompts) & set(eval_prompts)), "evaluation reused direction prompts"
    assert set(dir_prompts) | set(eval_prompts) <= set(prompts)


def test_full_file_coverage_no_longer_reuses_prompts(tmp_path):
    """The exact shape that broke: 128 prompts, n_samples=128.

    Before the fix the evaluation set was a subset of the direction set because the
    sampler returned every prompt. With a file large enough to hold out, it must not.
    """
    prompts = [f"prompt number {i}" for i in range(148)]
    src = _write(tmp_path, "harmful.txt", prompts)

    dir_prompts, eval_prompts = _sampled_holdout(src, "harmful.txt", 128, 20, seed=0)

    assert len(dir_prompts) == 128
    assert len(eval_prompts) == 20
    assert not (set(dir_prompts) & set(eval_prompts))


def test_small_file_warns_instead_of_reporting_a_silent_self_score(tmp_path, caplog):
    """A file too small to hold out must warn, not pretend the split happened."""
    prompts = [f"prompt number {i}" for i in range(128)]
    src = _write(tmp_path, "harmful.txt", prompts)

    with caplog.at_level(logging.WARNING, logger="abliterate.pipeline"):
        dir_prompts, eval_prompts = _sampled_holdout(src, "harmful.txt", 128, 48, seed=0)

    assert len(dir_prompts) == 128
    assert "hold-out exhausted" in caplog.text
    # and it still returns a usable evaluation set rather than crashing
    assert eval_prompts


def test_duplicates_in_the_source_are_not_handed_to_both_sides(tmp_path):
    """Duplicate lines must be consumed multiset-wise, not by set membership."""
    prompts = ["same prompt"] * 10 + [f"unique {i}" for i in range(10)]
    src = _write(tmp_path, "harmful.txt", prompts)

    dir_prompts, eval_prompts = _sampled_holdout(src, "harmful.txt", 10, 5, seed=0)

    assert dir_prompts.count("same prompt") + eval_prompts.count("same prompt") <= 10


def test_classic_path_validates_on_held_out_prompts(tmp_path, monkeypatch):
    """The classic path scored itself on harmful[:16] -- its own direction prompts.

    Regression for the non-search path: the prompts that pick the direction and
    produce the reported refusal rate must come from outside the direction set.
    """
    from abliterate.config import Config, DataConfig, ModelConfig
    from abliterate.pipeline import _classic_val_prompts, _sampled_holdout

    prompts = [f"prompt number {i}" for i in range(148)]
    src = tmp_path / "harmful.txt"
    src.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    cfg = Config(model=ModelConfig(path="unused"),
                 data=DataConfig(harmful=str(src), n_samples=128, seed=0))

    val = _classic_val_prompts(cfg)
    assert val, "classic path must still produce validation prompts"
    assert len(val) == 16

    dir_prompts, _ = _sampled_holdout(str(src), "harmful.txt", 128, 16, seed=0)
    overlap = set(val) & set(dir_prompts)
    assert not overlap, f"classic validation reused direction prompts: {sorted(overlap)[:3]}"


def test_classic_and_search_paths_agree_on_the_split(tmp_path):
    """Both paths must draw their held-out prompts from the same complement."""
    from abliterate.pipeline import _classic_val_prompts, _sampled_holdout
    from abliterate.config import Config, DataConfig, ModelConfig

    prompts = [f"p{i}" for i in range(200)]
    src = tmp_path / "harmful.txt"
    src.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    cfg = Config(model=ModelConfig(path="unused"),
                 data=DataConfig(harmful=str(src), n_samples=100, seed=7))

    classic = set(_classic_val_prompts(cfg))
    search_dir, search_eval = _sampled_holdout(str(src), "harmful.txt", 100, 16, seed=7)

    assert not classic & set(search_dir)
    assert not set(search_eval) & set(search_dir)
