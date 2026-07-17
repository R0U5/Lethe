"""Projection math: after orthogonalization, outputs carry no refusal component."""

import torch
from torch import nn

from abliterate.ablation import (
    orthogonalize_weight_,
    project_out_activation,
    _hidden_axis_for,
)


def _unit(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, generator=g)
    return v / v.norm()


def test_linear_output_orthogonal_to_direction():
    hidden, d_in = 32, 48
    layer = nn.Linear(d_in, hidden, bias=False)
    direction = _unit(hidden)

    orthogonalize_weight_(layer.weight, direction, _hidden_axis_for(layer))

    x = torch.randn(16, d_in)
    out = layer(x)                       # [16, hidden]
    comp = out @ direction               # component along refusal direction
    assert comp.abs().max() < 1e-4, comp.abs().max()


def test_conv1d_output_orthogonal_to_direction():
    from transformers.pytorch_utils import Conv1D

    d_in, hidden = 40, 32
    conv = Conv1D(hidden, d_in)          # weight shape (d_in, hidden)
    direction = _unit(hidden, seed=1)

    orthogonalize_weight_(conv.weight, direction, _hidden_axis_for(conv))

    x = torch.randn(8, d_in)
    out = conv(x)                        # [8, hidden]
    comp = out @ direction
    assert comp.abs().max() < 1e-4


def test_embedding_rows_orthogonal_to_direction():
    vocab, hidden = 50, 32
    emb = nn.Embedding(vocab, hidden)
    direction = _unit(hidden, seed=2)

    orthogonalize_weight_(emb.weight, direction, _hidden_axis_for(emb))

    comp = emb.weight.data @ direction   # [vocab]
    assert comp.abs().max() < 1e-4


def test_project_out_activation_removes_component():
    hidden = 32
    direction = _unit(hidden, seed=3)
    x = torch.randn(4, 7, hidden)

    y = project_out_activation(x, direction)
    comp = y @ direction
    assert comp.abs().max() < 1e-5
    # Component orthogonal to the direction is preserved.
    orth = x - (x @ direction).unsqueeze(-1) * direction
    assert torch.allclose(y, orth, atol=1e-5)
