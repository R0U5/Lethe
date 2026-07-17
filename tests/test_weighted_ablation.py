"""Weight kernel, partial-strength ablation, and weighted application."""

from types import SimpleNamespace

import torch
from torch import nn

from abliterate.ablation import (
    AblationParams,
    LayerWeightKernel,
    WeightedAblationHooks,
    apply_weighted_ablation,
    orthogonalize_weight_,
    project_out_activation,
)
from test_discovery import ToyModel


def _unit(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


# --- LayerWeightKernel --------------------------------------------------------
def test_kernel_defaults_are_uniform_one():
    k = LayerWeightKernel()
    assert all(k.weight_at(i) == 1.0 for i in range(10))


def test_kernel_tent_shape():
    k = LayerWeightKernel(max_weight=1.0, max_weight_position=5, min_weight=0.2, min_weight_distance=5)
    assert k.weight_at(5) == 1.0                      # peak
    assert abs(k.weight_at(10) - 0.2) < 1e-9          # at distance == min_dist
    assert abs(k.weight_at(0) - 0.2) < 1e-9           # clamped beyond min_dist
    assert abs(k.weight_at(7) - (1.0 - 0.8 * 2 / 5)) < 1e-9  # linear interpolation


def test_kernel_zero_distance_is_flat_max():
    k = LayerWeightKernel(max_weight=0.7, min_weight=0.1, min_weight_distance=0)
    assert k.weight_at(3) == 0.7


# --- partial strength ---------------------------------------------------------
def test_partial_weight_ablation_leaves_fraction():
    hidden, d_in = 24, 32
    layer = nn.Linear(d_in, hidden, bias=False)
    direction = _unit(hidden)
    x = torch.randn(8, d_in)

    before = (layer(x) @ direction)
    orthogonalize_weight_(layer.weight, direction, hidden_axis=0, strength=0.5)
    after = (layer(x) @ direction)
    # strength 0.5 removes half the component along the direction.
    assert torch.allclose(after, 0.5 * before, atol=1e-4)


def test_partial_activation_ablation_leaves_fraction():
    hidden = 16
    direction = _unit(hidden, seed=1)
    x = torch.randn(4, 5, hidden)
    y = project_out_activation(x, direction, strength=0.25)
    assert torch.allclose(y @ direction, 0.75 * (x @ direction), atol=1e-5)


def test_strength_zero_is_noop():
    hidden = 8
    w = nn.Linear(8, hidden, bias=False)
    ref = w.weight.detach().clone()
    orthogonalize_weight_(w.weight, _unit(hidden), hidden_axis=0, strength=0.0)
    assert torch.equal(w.weight, ref)


# --- weighted application on a toy model -------------------------------------
def test_apply_weighted_ablation_counts_and_scales():
    torch.manual_seed(0)
    model = ToyModel(h=16, n=3)
    directions = torch.stack([_unit(16, seed=i) for i in range(4)])  # [L+1, H]
    params = AblationParams(
        direction_index=2,
        attn_kernel=LayerWeightKernel(0.5, 0, 0.5, 1),   # constant 0.5
        mlp_kernel=LayerWeightKernel(0.5, 0, 0.5, 1),
    )

    block = model.model.layers[0]
    x = torch.randn(5, 16)
    before = block.self_attn.o_proj(x) @ directions[2]

    modified = apply_weighted_ablation(model, directions, params, hidden_size=16)
    assert modified == 6  # 3 layers * (o_proj + down_proj), embed off

    after = block.self_attn.o_proj(x) @ directions[2]
    assert torch.allclose(after, 0.5 * before, atol=1e-3)


def test_weighted_hooks_register_expected_handles():
    model = ToyModel(h=16, n=3)
    directions = torch.stack([_unit(16, seed=i) for i in range(4)])
    params = AblationParams(direction_index=2)
    with WeightedAblationHooks(model, directions, params, hidden_size=16) as h:
        assert len(h._handles) == 6  # o_proj + down_proj per layer
    assert len(h._handles) == 0      # cleaned up on exit
