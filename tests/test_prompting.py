import numpy as np

from src.prompting.prompts import (
    PROMPT_PROTOCOL,
    build_lp_prompt,
    build_nc_prompt,
    token_representatives,
)


def fixture_context():
    texts = ["zero text", "one text", "two text", "three text"]
    assignment = np.array([0, 0, 1, 1], dtype=np.int64)
    degree = np.array([1.0, 4.0, 3.0, 2.0])
    representatives = token_representatives(assignment, degree, count=2)
    adjacency = {
        0: [(2, 0.9), (1, 0.8)],
        1: [(0, 0.8)],
        2: [(0, 0.9), (3, 0.7)],
        3: [(2, 0.7)],
    }
    return texts, assignment, degree, representatives, adjacency


def test_nc_prompt_has_ordered_three_level_context():
    texts, assignment, degree, representatives, adjacency = fixture_context()
    prompt = build_nc_prompt(
        0,
        ["class a", "class b"],
        texts,
        assignment,
        representatives,
        adjacency,
        neighbor_cap=2,
        degree=degree,
    )
    assert f"Protocol: {PROMPT_PROTOCOL}" in prompt
    assert prompt.index("LEVEL 1") < prompt.index("LEVEL 2") < prompt.index("LEVEL 3")
    assert "Representative 1 | node=1 | riemannian_degree=4" in prompt
    assert "Target node=0 | assigned_macro_token=0" in prompt
    assert "Neighbor 1 | node=2 | reconstructed_edge_weight=0.9" in prompt
    assert 'Return exactly: {"label": "<one candidate label>"}' in prompt


def test_lp_prompt_hides_candidate_edge_from_micro_context():
    texts, assignment, degree, representatives, adjacency = fixture_context()
    prompt = build_lp_prompt(
        0,
        2,
        texts,
        assignment,
        representatives,
        adjacency,
        neighbor_cap=2,
        degree=degree,
    )
    micro = prompt.split("[LEVEL 3 - MICROSCOPIC LOCAL CONTEXT]", maxsplit=1)[1]
    assert "candidate edge itself is hidden" in prompt
    assert "reconstructed_edge_weight=0.9" not in micro
    assert "Neighbor 1 | node=1 | reconstructed_edge_weight=0.8" in micro
    assert "Neighbor 1 | node=3 | reconstructed_edge_weight=0.7" in micro
