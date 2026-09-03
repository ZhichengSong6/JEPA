import torch

from train import _raw_reachability_loss


def _loss(x):
    return _raw_reachability_loss(
        emb=x,
        goal_index=5,
        min_pair_gap=1,
        max_pair_gap=4,
        margin_fraction=0.02,
        temperature_fraction=0.10,
        scale_eps=1e-6,
    )


def test_reachability_prefers_monotone_goal_progress():
    # One-dimensional synthetic trajectory that approaches the final goal.
    ordered = torch.tensor(
        [[[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]]]
    )
    reversed_prefix = torch.tensor(
        [[[1.0], [2.0], [3.0], [4.0], [5.0], [0.0]]]
    )
    good = _loss(ordered)
    bad = _loss(reversed_prefix)
    assert good["accuracy"].item() == 1.0
    assert bad["accuracy"].item() == 0.0
    assert good["loss"].item() < bad["loss"].item()


def test_reachability_loss_is_scale_normalized():
    ordered = torch.tensor(
        [[[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]]]
    )
    a = _loss(ordered)["loss"]
    b = _loss(10.0 * ordered)["loss"]
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
