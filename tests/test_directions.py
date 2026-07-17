"""Direction computation, selection, and serialization."""

import torch

from abliterate.config import AblationConfig
from abliterate.directions import RefusalDirections, compute_refusal_directions


def test_direction_is_unit_and_points_from_harmless_to_harmful():
    hidden = 8
    mean_harmful = torch.zeros(3, hidden)
    mean_harmless = torch.zeros(3, hidden)
    mean_harmful[1, 0] = 5.0            # layer 1 differs only along axis 0

    dirs = compute_refusal_directions(mean_harmful, mean_harmless)
    row = dirs.get(1)
    assert torch.allclose(row.norm(), torch.tensor(1.0), atol=1e-6)
    assert row[0] > 0.99               # points along the difference


def test_zero_difference_rows_are_zero():
    hidden = 8
    means = torch.randn(4, hidden)
    dirs = compute_refusal_directions(means, means)  # identical -> zero diff
    assert torch.count_nonzero(dirs.directions) == 0
    assert dirs.raw_norms.max() < 1e-6


def test_select_index_fractional_and_explicit():
    hidden = 4
    dirs = compute_refusal_directions(torch.randn(11, hidden), torch.zeros(11, hidden))

    # 11 candidates (indices 0..10); fraction 0.6 -> round(0.6*10)=6
    assert dirs.select_index(AblationConfig(layer_fraction=0.6)) == 6
    # explicit positive
    assert dirs.select_index(AblationConfig(layer_index=3)) == 3
    # explicit negative indexes from the end
    assert dirs.select_index(AblationConfig(layer_index=-1)) == 10


def test_save_and_load_roundtrip(tmp_path):
    hidden = 6
    dirs = compute_refusal_directions(
        torch.randn(5, hidden), torch.zeros(5, hidden),
        model_path="unit/test", n_harmful=10, n_harmless=12,
    )
    path = tmp_path / "d.pt"
    dirs.save(path)
    assert (tmp_path / "d.json").exists()

    loaded = RefusalDirections.load(path)
    assert torch.allclose(loaded.directions, dirs.directions)
    assert loaded.model_path == "unit/test"
    assert loaded.n_harmful == 10 and loaded.n_harmless == 12


def test_select_by_score_picks_best():
    hidden = 4
    dirs = compute_refusal_directions(torch.randn(6, hidden), torch.zeros(6, hidden))
    # scorer rewards index 4 the most
    best, scores = dirs.select_by_score(lambda i, d: 1.0 if i == 4 else 0.0)
    assert best == 4
    assert scores[4] == 1.0
