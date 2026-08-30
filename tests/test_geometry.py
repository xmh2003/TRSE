import numpy as np

from src.trse.geometry import compute_riemannian_distances, response_values
from src.trse.basis import build_low_rank_basis, distances_from_basis


def test_responses_have_fixed_endpoints():
    for name in ("linear", "exponential", "rational", "power"):
        values = response_values(np.array([0.0, 1.0]), name)
        np.testing.assert_allclose(values, [0.0, 1.0], atol=1e-12)


def test_ranked_geometry_is_finite_and_positive():
    x = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    result = compute_riemannian_distances(x, edges, rank=2)
    assert np.all(np.isfinite(result.rho2))
    assert np.all(result.rho2 >= 0)
    assert np.all((result.weights > 0) & (result.weights <= 1))


def test_batched_basis_matches_direct_geometry():
    x = np.array([[1, 0], [0, 1], [1, 1], [1, -1]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3]], dtype=np.int64)
    direct = compute_riemannian_distances(x, edges, rank=2)
    basis = build_low_rank_basis(x, edges, rank=2, device="cpu", target_matrix_elements=100)
    batched = distances_from_basis(x, edges, basis)
    np.testing.assert_allclose(batched.rho2, direct.rho2, rtol=2e-5, atol=2e-5)
