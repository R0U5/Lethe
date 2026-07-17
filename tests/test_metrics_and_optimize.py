"""KL divergence metric and optimizer search-space sampling."""

import torch

from abliterate.ablation import AblationParams
from abliterate.config import OptimizeConfig
from abliterate.metrics import kl_divergence
from abliterate.optimize import _RandomSuggester, sample_params
import random


def _logprobs(logits):
    return torch.log_softmax(logits, dim=-1)


def test_kl_zero_for_identical_distributions():
    lp = _logprobs(torch.randn(6, 20))
    assert abs(kl_divergence(lp, lp)) < 1e-6


def test_kl_positive_for_different_distributions():
    a = _logprobs(torch.randn(6, 20))
    b = _logprobs(torch.randn(6, 20))
    assert kl_divergence(a, b) > 0.0


def test_kl_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        kl_divergence(torch.randn(3, 20), torch.randn(3, 21))


def test_sample_params_within_bounds():
    cfg = OptimizeConfig(max_weight_hi=1.5, dir_band=(0.3, 0.95))
    n_layers, n_candidates = 12, 13
    s = _RandomSuggester(random.Random(0))
    for _ in range(50):
        p = sample_params(s, n_layers, n_candidates, cfg)
        assert isinstance(p, AblationParams)
        for kernel in (p.attn_kernel, p.mlp_kernel):
            assert 0.0 <= kernel.max_weight <= 1.5
            assert 0.0 <= kernel.min_weight <= 1.5
            assert 0.0 <= kernel.max_weight_position <= n_layers - 1
            assert 1.0 <= kernel.min_weight_distance <= n_layers
        if not p.per_layer_direction:
            lo = max(1, int(0.3 * n_candidates))
            hi = max(lo, int(0.95 * n_candidates) - 1)
            assert lo <= p.direction_index <= hi


def test_sample_params_is_seed_deterministic():
    cfg = OptimizeConfig()
    p1 = sample_params(_RandomSuggester(random.Random(42)), 12, 13, cfg)
    p2 = sample_params(_RandomSuggester(random.Random(42)), 12, 13, cfg)
    assert p1.to_dict() == p2.to_dict()
