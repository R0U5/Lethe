"""Runtime ablation bundle: round-trip, permanent application, hook attachment."""

import torch

from abliterate.ablation import AblationParams, LayerWeightKernel
from abliterate.bundle import AblationBundle
from test_discovery import ToyModel


def _dirs(n=4, h=16):
    g = torch.Generator().manual_seed(0)
    d = torch.randn(n, h, generator=g)
    return d / d.norm(dim=-1, keepdim=True)


def test_bundle_roundtrip(tmp_path):
    dirs = _dirs()
    params = AblationParams(
        direction_index=2,
        attn_kernel=LayerWeightKernel(0.5, 1, 0.3, 2),
        mlp_kernel=LayerWeightKernel(0.7, 0, 0.7, 1),
    )
    b = AblationBundle(directions=dirs, params=params, hidden_size=16, num_layers=3,
                       source_model="acme/model", metrics={"kl_divergence": 0.1},
                       model_overrides={"attn_out_names": ["o_proj"]})
    path = b.save(tmp_path / "bundle")
    assert (path / "bundle.json").exists()
    assert (path / "directions.safetensors").exists()

    loaded = AblationBundle.load(path)
    assert torch.allclose(loaded.directions, dirs, atol=1e-6)
    assert loaded.params.to_dict() == params.to_dict()
    assert loaded.source_model == "acme/model"
    assert loaded.metrics["kl_divergence"] == 0.1
    assert loaded.model_overrides["attn_out_names"] == ["o_proj"]


def test_params_from_dict_roundtrip():
    params = AblationParams(
        direction_index=3, per_layer_direction=True,
        attn_kernel=LayerWeightKernel(1.2, 4, 0.1, 3),
        mlp_kernel=LayerWeightKernel(0.9, 2, 0.4, 5),
        ablate_embed=True, embed_strength=0.5,
    )
    assert AblationParams.from_dict(params.to_dict()).to_dict() == params.to_dict()


def test_bundle_apply_permanently_matches_strength():
    model = ToyModel(h=16, n=3)
    dirs = _dirs()
    params = AblationParams(
        direction_index=2,
        attn_kernel=LayerWeightKernel(0.5, 0, 0.5, 1),
        mlp_kernel=LayerWeightKernel(0.5, 0, 0.5, 1),
    )
    b = AblationBundle(directions=dirs, params=params, hidden_size=16, num_layers=3)

    block = model.model.layers[0]
    x = torch.randn(5, 16)
    before = block.self_attn.o_proj(x) @ dirs[2]
    modified = b.apply_permanently(model)
    assert modified == 6
    after = block.self_attn.o_proj(x) @ dirs[2]
    assert torch.allclose(after, 0.5 * before, atol=1e-3)


def test_bundle_hooks_attach_and_detach():
    model = ToyModel(h=16, n=3)
    b = AblationBundle(directions=_dirs(), params=AblationParams(direction_index=2),
                       hidden_size=16, num_layers=3)
    with b.hooks(model) as h:
        assert len(h._handles) == 6
    assert len(h._handles) == 0
