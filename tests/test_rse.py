import numpy as np

from src.trse.rse import entropy_1d, entropy_2d, graph_degrees, merge_reduction, optimize_height2_rse


def test_incremental_merge_matches_direct_objective():
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    weights = np.array([0.8, 0.3, 0.7, 0.2], dtype=np.float64)
    degree = graph_degrees(4, edges, weights)
    before = entropy_2d(degree, edges, weights, np.arange(4))
    after = entropy_2d(degree, edges, weights, np.array([0, 0, 2, 3]))
    expected = merge_reduction(degree[0], degree[0], degree[1], degree[1], 0.8, degree.sum())
    np.testing.assert_allclose(before - after, expected, atol=1e-12)


def test_positive_only_and_deterministic_partition():
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    weights = np.ones(4, dtype=np.float64)
    first = optimize_height2_rse(4, edges, weights, target_k=2)
    second = optimize_height2_rse(4, edges, weights, target_k=2)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    assert np.all(first.merge_history[:, 3] > 0)
    assert np.unique(first.assignment).size == first.achieved_k
    assert entropy_1d(graph_degrees(4, edges, weights)) >= first.h2


def test_near_greedy_is_seed_deterministic_and_same_k():
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [0, 4], [1, 4]], dtype=np.int64)
    weights = np.array([1.0, 0.8, 0.7, 0.9, 0.4, 0.3])
    first = optimize_height2_rse(5, edges, weights, 3, temperature=0.2, random_seed=7)
    second = optimize_height2_rse(5, edges, weights, 3, temperature=0.2, random_seed=7)
    np.testing.assert_array_equal(first.assignment, second.assignment)
    assert first.achieved_k == second.achieved_k == 3


def test_modularity_priority_preserves_default_and_positive_rse_merges():
    edges = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5], [1, 4]],
        dtype=np.int64,
    )
    weights = np.array([1.0, 0.8, 0.7, 0.9, 0.6, 0.4, 0.3])
    default = optimize_height2_rse(6, edges, weights, target_k=3)
    explicit_zero = optimize_height2_rse(
        6,
        edges,
        weights,
        target_k=3,
        modularity_priority_alpha=0.0,
    )
    prioritized = optimize_height2_rse(
        6,
        edges,
        weights,
        target_k=3,
        modularity_priority_alpha=2.0,
    )
    modularity_only = optimize_height2_rse(
        6,
        edges,
        weights,
        target_k=3,
        modularity_priority_only=True,
    )
    np.testing.assert_array_equal(default.assignment, explicit_zero.assignment)
    assert prioritized.achieved_k == 3
    assert np.all(prioritized.merge_history[:, 3] > 0.0)
    assert modularity_only.achieved_k == 3
    assert np.all(modularity_only.merge_history[:, 3] > 0.0)
