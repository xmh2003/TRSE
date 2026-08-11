from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROMPT_PROTOCOL = "TRSE-III-C-hierarchical-v1"
SYSTEM_PROMPT = (
    "You are a zero-shot graph reasoning model. Use only the supplied "
    "hierarchical tree context. Reason internally from macro community to "
    "target anchor to micro neighborhood, but return only the requested JSON "
    "object without explanation or markdown."
)


def load_texts(path: Path, expected_nodes: int) -> list[str]:
    texts = [""] * expected_nodes
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            texts[int(record["node_id"])] = str(record["text"])
    return texts


def weighted_degrees(num_nodes: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, edges[:, 0], weights)
    np.add.at(degree, edges[:, 1], weights)
    return degree


def adjacency_records(num_nodes: int, edges: np.ndarray, weights: np.ndarray, needed_nodes: np.ndarray) -> dict[int, list[tuple[int, float]]]:
    needed_nodes = np.unique(np.asarray(needed_nodes, dtype=np.int64))
    wanted = np.zeros(num_nodes, dtype=bool)
    wanted[needed_nodes] = True
    adjacency = {int(node): [] for node in needed_nodes}
    left_ids = np.flatnonzero(wanted[edges[:, 0]])
    right_ids = np.flatnonzero(wanted[edges[:, 1]])
    for edge_id in left_ids:
        left, right = edges[edge_id]
        adjacency[int(left)].append((int(right), float(weights[edge_id])))
    for edge_id in right_ids:
        left, right = edges[edge_id]
        adjacency[int(right)].append((int(left), float(weights[edge_id])))
    for values in adjacency.values():
        values.sort(key=lambda item: (-item[1], item[0]))
    return adjacency


def token_representatives(assignment: np.ndarray, degree: np.ndarray, count: int) -> dict[int, list[int]]:
    nodes = np.arange(assignment.size, dtype=np.int64)
    order = np.lexsort((nodes, -degree, assignment))
    result: dict[int, list[int]] = {}
    for node in nodes[order]:
        token = int(assignment[node])
        selected = result.setdefault(token, [])
        if len(selected) < count:
            selected.append(int(node))
    return result


def _node_text(node: int, texts: list[str], max_chars: int) -> str:
    text = " ".join(texts[node].split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _macro_context(
    token: int,
    representatives: dict[int, list[int]],
    degree: np.ndarray,
    texts: list[str],
    max_chars: int,
) -> str:
    records = []
    for rank, representative in enumerate(representatives.get(token, []), start=1):
        records.append(
            f"  Representative {rank} | node={representative} | "
            f"riemannian_degree={degree[representative]:.6g}\n"
            f"  {_node_text(representative, texts, max_chars)}"
        )
    if not records:
        records.append("  No representative text is available.")
    return f"Macro-token {token} representatives (highest Riemannian degree first):\n" + "\n".join(records)


def _micro_context(
    node: int,
    adjacency: dict[int, list[tuple[int, float]]],
    texts: list[str],
    neighbor_cap: int,
    max_chars: int,
    excluded: set[int] | None = None,
) -> str:
    excluded = excluded or set()
    selected = [
        (neighbor, weight)
        for neighbor, weight in adjacency[node]
        if neighbor not in excluded
    ][:neighbor_cap]
    records = []
    for rank, (neighbor, weight) in enumerate(selected, start=1):
        records.append(
            f"  Neighbor {rank} | node={neighbor} | reconstructed_edge_weight={weight:.6g}\n"
            f"  {_node_text(neighbor, texts, max_chars)}"
        )
    if not records:
        records.append("  No eligible direct neighbor is available.")
    return "Direct neighbors (strongest reconstructed edges first):\n" + "\n".join(records)


def build_nc_prompt(
    node: int,
    label_names: list[str],
    texts: list[str],
    assignment: np.ndarray,
    representatives: dict[int, list[int]],
    adjacency: dict[int, list[tuple[int, float]]],
    neighbor_cap: int,
    degree: np.ndarray | None = None,
    text_char_limit: int = 1600,
) -> str:
    if degree is None:
        degree = np.zeros(assignment.size, dtype=np.float64)
    token = int(assignment[node])
    candidates = " | ".join(label_names)
    return (
        f"Protocol: {PROMPT_PROTOCOL}\n"
        "Task: zero-shot node classification\n"
        f"Candidate labels (choose exactly one): {candidates}\n\n"
        "[LEVEL 1 - MACROSCOPIC COMMUNITY CONTEXT]\n"
        f"{_macro_context(token, representatives, degree, texts, text_char_limit)}\n\n"
        "[LEVEL 2 - TARGET EGO-CENTRIC ANCHOR]\n"
        f"Target node={node} | assigned_macro_token={token}\n"
        f"{_node_text(node, texts, text_char_limit)}\n\n"
        "[LEVEL 3 - MICROSCOPIC LOCAL CONTEXT]\n"
        f"{_micro_context(node, adjacency, texts, neighbor_cap, text_char_limit)}\n\n"
        "Reasoning instruction: first infer the dominant macro-token theme; "
        "then classify the target from its own metadata; finally use direct "
        "neighbors to refine the decision and discount heterophilic outliers.\n"
        "Return exactly: {\"label\": \"<one candidate label>\"}"
    )


def build_lp_prompt(
    source: int,
    target: int,
    texts: list[str],
    assignment: np.ndarray,
    representatives: dict[int, list[int]],
    adjacency: dict[int, list[tuple[int, float]]],
    neighbor_cap: int,
    degree: np.ndarray | None = None,
    text_char_limit: int = 1600,
) -> str:
    if degree is None:
        degree = np.zeros(assignment.size, dtype=np.float64)
    source_token = int(assignment[source])
    target_token = int(assignment[target])
    if source_token == target_token:
        macro = _macro_context(
            source_token, representatives, degree, texts, text_char_limit
        )
    else:
        macro = (
            "Source endpoint community:\n"
            f"{_macro_context(source_token, representatives, degree, texts, text_char_limit)}\n"
            "Target endpoint community:\n"
            f"{_macro_context(target_token, representatives, degree, texts, text_char_limit)}"
        )
    return (
        f"Protocol: {PROMPT_PROTOCOL}\n"
        "Task: zero-shot link prediction\n"
        "Candidate labels (choose exactly one): linked | not linked\n"
        "The candidate edge itself is hidden from the local context.\n\n"
        "[LEVEL 1 - MACROSCOPIC COMMUNITY CONTEXT]\n"
        f"{macro}\n\n"
        "[LEVEL 2 - ENDPOINT EGO-CENTRIC ANCHORS]\n"
        f"Source node={source} | assigned_macro_token={source_token}\n"
        f"{_node_text(source, texts, text_char_limit)}\n"
        f"Target node={target} | assigned_macro_token={target_token}\n"
        f"{_node_text(target, texts, text_char_limit)}\n\n"
        "[LEVEL 3 - MICROSCOPIC LOCAL CONTEXT]\n"
        f"Source endpoint:\n{_micro_context(source, adjacency, texts, neighbor_cap, text_char_limit, {target})}\n"
        f"Target endpoint:\n{_micro_context(target, adjacency, texts, neighbor_cap, text_char_limit, {source})}\n\n"
        "Reasoning instruction: compare the endpoints' macro communities, "
        "their own semantics, and their remaining direct neighborhoods. "
        "Predict whether an edge is plausible without treating the hidden "
        "candidate edge as observed evidence.\n"
        "Return exactly: {\"label\": \"<linked|not linked>\"}"
    )


def count_positions(system: str, user: str, model: str) -> int:
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(system)) + len(encoding.encode(user))


def write_prompts(
    dataset: str,
    task: str,
    queries: pd.DataFrame,
    output_path: Path,
    texts: list[str],
    label_names: list[str],
    assignment: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
    representatives_count: int,
    neighbor_cap: int,
    model: str,
    degree: np.ndarray | None = None,
    adjacency: dict[int, list[tuple[int, float]]] | None = None,
    text_char_limit: int = 1600,
) -> pd.DataFrame:
    degree = weighted_degrees(assignment.size, edges, weights) if degree is None else degree
    if task == "nc":
        needed_nodes = queries["node_id"].to_numpy(dtype=np.int64)
    else:
        needed_nodes = np.concatenate((queries["source"].to_numpy(dtype=np.int64), queries["target"].to_numpy(dtype=np.int64)))
    adjacency = adjacency_records(assignment.size, edges, weights, needed_nodes) if adjacency is None else adjacency
    representatives = token_representatives(assignment, degree, representatives_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output_path.open("w", encoding="utf-8") as handle:
        for query in queries.to_dict("records"):
            if task == "nc":
                user = build_nc_prompt(
                    int(query["node_id"]), label_names, texts, assignment,
                    representatives, adjacency, neighbor_cap, degree,
                    text_char_limit,
                )
            else:
                user = build_lp_prompt(
                    int(query["source"]), int(query["target"]), texts,
                    assignment, representatives, adjacency, neighbor_cap,
                    degree, text_char_limit,
                )
            positions = count_positions(SYSTEM_PROMPT, user, model)
            record = {
                "dataset": dataset,
                "task": task,
                "query_id": query["query_id"],
                "protocol": PROMPT_PROTOCOL,
                "system": SYSTEM_PROMPT,
                "user": user,
                "text_tokens": positions,
                "graph_tokens": 0,
                "total_input_positions": positions,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.append({key: record[key] for key in ("dataset", "task", "query_id", "text_tokens", "graph_tokens", "total_input_positions")})
    return pd.DataFrame(rows)
