"""Architecture-agnostic discovery + end-to-end orthogonalization on a toy model.

Builds a minimal llama-shaped module tree (no weights downloaded) to verify that
decoder layers, residual-writing projections, and the embedding are located and
orthogonalized correctly.
"""

from types import SimpleNamespace

import torch
from torch import nn

from abliterate.ablation import orthogonalize_model
from abliterate.config import AblationConfig
from abliterate.model_utils import (
    get_decoder_layers,
    iter_residual_write_modules,
)


class Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.q_proj = nn.Linear(h, h, bias=False)   # out=hidden but NOT residual-write
        self.o_proj = nn.Linear(h, h, bias=False)   # residual-write (attn_out)


class MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.up_proj = nn.Linear(h, inter, bias=False)     # out=inter, skipped
        self.down_proj = nn.Linear(inter, h, bias=False)   # residual-write (mlp_out)


class Block(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.self_attn = Attn(h)
        self.mlp = MLP(h, inter)


class Inner(nn.Module):
    def __init__(self, h, inter, n, vocab):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, h)
        self.layers = nn.ModuleList([Block(h, inter) for _ in range(n)])


class ToyModel(nn.Module):
    def __init__(self, h=16, inter=32, n=3, vocab=40):
        super().__init__()
        self.model = Inner(h, inter, n, vocab)
        self.config = SimpleNamespace(hidden_size=h)
        self._h = h

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_get_decoder_layers_finds_module_list():
    model = ToyModel(n=3)
    layers = get_decoder_layers(model)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 3


def test_iter_residual_write_modules_selects_o_and_down_proj():
    model = ToyModel()
    block = model.model.layers[0]
    found = list(iter_residual_write_modules(block, hidden_size=model._h))
    roles = sorted(role for role, _ in found)
    assert roles == ["attn_out", "mlp_out"]

    modules = {role: mod for role, mod in found}
    assert modules["attn_out"] is block.self_attn.o_proj
    assert modules["mlp_out"] is block.mlp.down_proj


def test_orthogonalize_model_modifies_expected_count_and_removes_component():
    torch.manual_seed(0)
    model = ToyModel(h=16, n=3)
    direction = torch.randn(16)
    direction = direction / direction.norm()

    modified = orthogonalize_model(
        model, direction,
        ablation_cfg=AblationConfig(), hidden_size=16,
    )
    # embed (1) + per layer {o_proj, down_proj} (2) * 3 layers = 7
    assert modified == 7

    # Each residual-writing projection's output is now orthogonal to direction.
    block = model.model.layers[0]
    x = torch.randn(5, 16)
    assert (block.self_attn.o_proj(x) @ direction).abs().max() < 1e-3
    xi = torch.randn(5, 32)
    assert (block.mlp.down_proj(xi) @ direction).abs().max() < 1e-3
    assert (model.model.embed_tokens.weight.data @ direction).abs().max() < 1e-3
