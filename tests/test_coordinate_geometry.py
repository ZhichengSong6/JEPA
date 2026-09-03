import torch

from coordinate_geometry import SPDCoordinateAdapter


def test_spd_coordinate_adapter_starts_as_identity():
    adapter = SPDCoordinateAdapter(8)
    x = torch.randn(5, 8)
    y = adapter(x)
    assert torch.allclose(x, y, atol=1e-6, rtol=1e-6)


def test_spd_coordinate_adapter_roundtrip():
    torch.manual_seed(0)
    adapter = SPDCoordinateAdapter(8)
    with torch.no_grad():
        adapter.log_metric_raw.normal_(mean=0.0, std=0.03)
    adapter.eval()
    adapter.refresh_cache()
    x = torch.randn(17, 8)
    recovered = adapter.inverse(adapter(x))
    assert torch.allclose(x, recovered, atol=2e-5, rtol=2e-5)


def test_metric_is_positive_definite():
    torch.manual_seed(1)
    adapter = SPDCoordinateAdapter(8)
    with torch.no_grad():
        adapter.log_metric_raw.normal_(mean=0.0, std=0.2)
    a = adapter._fresh_matrix()
    metric = a.T @ a
    eig = torch.linalg.eigvalsh(metric)
    assert torch.all(eig > 0)
