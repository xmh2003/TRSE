#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_graph
from src.llm.client import OpenAICompatibleChatClient
from src.llm.parsing import parse_prediction
from src.prompting.prompts import PROMPT_PROTOCOL, load_texts, write_prompts
from src.prompting.queries import write_query_manifests
from src.utils.io import load_default_config, sha256_file, write_json


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def bootstrap_accuracy(
    values: np.ndarray,
    seed: int,
    resamples: int,
) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1000):
        count = min(1000, resamples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        estimates[start:start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def summarize_raw(
    records: list[dict],
    expected_queries: int,
    seed: int,
    resamples: int,
) -> dict:
    api_success = [record for record in records if record["status"] != "API_ERROR"]
    correct = np.asarray(
        [bool(record.get("correct", False)) for record in api_success],
        dtype=np.float64,
    )
    ci_low, ci_high = bootstrap_accuracy(correct, seed, resamples)
    parsed = [record for record in api_success if record["status"] == "OK"]
    usage = [
        int(record["api_prompt_tokens"])
        for record in api_success
        if record.get("api_prompt_tokens") is not None
    ]
    latencies = [float(record["latency_ms"]) for record in api_success]
    return {
        "dataset": records[0]["dataset"] if records else None,
        "task": records[0]["task"] if records else None,
        "model": records[0]["model"] if records else None,
        "protocol": PROMPT_PROTOCOL,
        "expected_queries": expected_queries,
        "attempted_queries": len(records),
        "api_success_queries": len(api_success),
        "parsed_queries": len(parsed),
        "accuracy": float(correct.mean()) if correct.size else None,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "parse_failure_rate": (
            1.0 - len(parsed) / len(api_success) if api_success else None
        ),
        "api_failure_rate": (
            sum(record["status"] == "API_ERROR" for record in records) / len(records)
            if records else None
        ),
        "mean_api_prompt_tokens": float(np.mean(usage)) if usage else None,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p50_latency_ms": float(np.quantile(latencies, 0.50)) if latencies else None,
        "p95_latency_ms": float(np.quantile(latencies, 0.95)) if latencies else None,
        "status": (
            "COMPLETE_REAL"
            if len(records) == expected_queries
            and all(record["status"] != "API_ERROR" for record in records)
            else (
                "COMPLETE_WITH_API_FAILURES"
                if len(records) == expected_queries
                else "PARTIAL_REAL"
            )
        ),
    }


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def safe_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(r"(?i)(api[_ -]?key[=: ]+)[^\s,;]+", r"\1<redacted>", message)
    return message[:2000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Cora"],
        choices=["Cora", "PubMed", "ogbn-arxiv", "ogbn-products"],
    )
    parser.add_argument("--tasks", nargs="+", default=["nc", "lp"], choices=["nc", "lp"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-name", default="hierarchical_tree_prompting")
    parser.add_argument("--model")
    parser.add_argument("--assignment-root", type=Path, default=Path("outputs/default"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hierarchical_reasoning"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()

    config = load_default_config()
    model = args.model or os.getenv("OPENAI_MODEL", config["llm"]["default_model"])
    output_root = resolve_path(args.output_root) / args.run_name
    assignment_root = resolve_path(args.assignment_root)
    output_root.mkdir(parents=True, exist_ok=True)
    api_available = bool(os.getenv("OPENAI_API_KEY"))
    api_blocked = not args.dry_run and not api_available
    client = None
    if not args.dry_run and api_available:
        client = OpenAICompatibleChatClient(
            model=model,
            temperature=float(config["llm"]["temperature"]),
            top_p=float(config["llm"]["top_p"]),
            max_output_tokens=int(config["llm"]["max_output_tokens"]),
            max_retries=int(config["llm"].get("max_retries", 4)),
            retry_base_seconds=float(config["llm"].get("retry_base_seconds", 1.0)),
        )

    run_sources = []
    summaries = []
    for dataset in args.datasets:
        graph = load_graph(dataset)
        data_dir = ROOT / "outputs" / "data" / dataset
        text_path = data_dir / "node_text.jsonl"
        label_path = data_dir / "label_names.json"
        assignment_path = assignment_root / dataset / "assignment.npy"
        weight_path = ROOT / "outputs" / "default" / dataset / "edge_weights.npy"
        for required in (text_path, assignment_path, weight_path):
            if not required.exists():
                raise FileNotFoundError(required)
        texts = load_texts(text_path, graph["x"].shape[0])
        if not all(texts):
            raise RuntimeError(f"node text coverage is incomplete for {dataset}")
        if label_path.exists():
            label_names = json.loads(label_path.read_text(encoding="utf-8"))
        else:
            label_names = [str(value) for value in sorted(np.unique(graph["y"]).tolist())]
        assignment = np.load(assignment_path)
        weights = np.load(weight_path, mmap_mode="r")
        if assignment.shape != (graph["x"].shape[0],):
            raise ValueError(f"assignment shape mismatch for {dataset}")

        query_dir = ROOT / "outputs" / "queries"
        query_paths = {
            task: query_dir / f"{dataset}_{task}.csv" for task in ("nc", "lp")
        }
        if not all(path.exists() for path in query_paths.values()):
            split = {
                key: graph[key] for key in ("train", "val", "test") if key in graph
            }
            nc_path, lp_path = write_query_manifests(
                dataset,
                graph["y"],
                graph["edges"],
                split,
                query_dir,
                nc_total=500,
                lp_total=500,
                seed=int(config["query_seed"]),
            )
            query_paths = {"nc": nc_path, "lp": lp_path}

        run_sources.append(
            {
                "dataset": dataset,
                "assignment": str(assignment_path.relative_to(ROOT)),
                "assignment_sha256": sha256_file(assignment_path),
                "edge_weights": str(weight_path.relative_to(ROOT)),
                "edge_weights_sha256": sha256_file(weight_path),
                "node_text": str(text_path.relative_to(ROOT)),
                "node_text_sha256": sha256_file(text_path),
            }
        )

        for task in args.tasks:
            queries = pd.read_csv(query_paths[task])
            if args.limit is not None:
                queries = queries.iloc[:args.limit].copy()
            prompt_path = output_root / "prompts" / f"{dataset}_{task}.jsonl"
            raw_path = output_root / "predictions" / f"{dataset}_{task}.jsonl"
            if args.force:
                raw_path.unlink(missing_ok=True)
            write_prompts(
                dataset,
                task,
                queries,
                prompt_path,
                texts,
                label_names,
                assignment,
                graph["edges"],
                weights,
                int(config["macro_representatives"]),
                int(config["micro_neighbor_cap"]),
                model,
                text_char_limit=int(config.get("node_text_char_limit", 1600)),
            )
            if args.dry_run:
                print(f"DRY_RUN {dataset} {task}: {len(queries)} prompts")
                continue
            if api_blocked:
                print(
                    f"BLOCKED_LLM {dataset} {task}: OPENAI_API_KEY is not set; "
                    "prompts were generated but no prediction was created",
                    flush=True,
                )
                continue

            assert client is not None
            prompts = {record["query_id"]: record for record in load_jsonl(prompt_path)}
            existing = load_jsonl(raw_path)
            completed = {record["query_id"] for record in existing}
            for query in queries.to_dict("records"):
                query_id = query["query_id"]
                if query_id in completed:
                    continue
                prompt = prompts[query_id]
                candidates = label_names if task == "nc" else ["linked", "not linked"]
                truth = (
                    label_names[int(query["label"])]
                    if task == "nc"
                    else ("linked" if int(query["label"]) == 1 else "not linked")
                )
                base_record = {
                    "dataset": dataset,
                    "task": task,
                    "query_id": query_id,
                    "model": model,
                    "protocol": PROMPT_PROTOCOL,
                    "ground_truth": truth,
                    "estimated_input_positions": prompt["total_input_positions"],
                }
                try:
                    result = client.complete(prompt["system"], prompt["user"])
                    prediction, parse_error = parse_prediction(result.text, candidates)
                    record = {
                        **base_record,
                        "status": "OK" if parse_error is None else "PARSE_FAILURE",
                        "prediction": prediction,
                        "correct": prediction == truth if prediction is not None else False,
                        "parse_error": parse_error,
                        "raw_response": result.text,
                        "response_id": result.response_id,
                        "api_prompt_tokens": result.prompt_tokens,
                        "api_completion_tokens": result.completion_tokens,
                        "api_total_tokens": result.total_tokens,
                        "latency_ms": result.latency_ms,
                        "api_attempts": result.attempts,
                    }
                except Exception as error:
                    record = {
                        **base_record,
                        "status": "API_ERROR",
                        "prediction": None,
                        "correct": False,
                        "error_type": type(error).__name__,
                        "error": safe_error(error),
                    }
                append_jsonl(raw_path, record)
                print(
                    f"{dataset} {task} {query_id} {record['status']} "
                    f"prediction={record.get('prediction')} truth={truth}",
                    flush=True,
                )
            records = load_jsonl(raw_path)
            expected_ids = set(queries["query_id"].tolist())
            records = [record for record in records if record["query_id"] in expected_ids]
            summary = summarize_raw(
                records,
                len(queries),
                int(config["seed"]),
                args.bootstrap_resamples,
            )
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

    run_config = {
        "protocol": PROMPT_PROTOCOL,
        "model": model,
        "temperature": config["llm"]["temperature"],
        "top_p": config["llm"]["top_p"],
        "max_output_tokens": config["llm"]["max_output_tokens"],
        "macro_representatives": config["macro_representatives"],
        "micro_neighbor_cap": config["micro_neighbor_cap"],
        "node_text_char_limit": config.get("node_text_char_limit", 1600),
        "lp_protocol": "controlled task extension of Section III-C",
        "lp_candidate_edge_hidden_from_micro_context": True,
        "lp_assignment_scope": (
            "precomputed TRSE assignment; it may encode the full graph, "
            "including a held-out query edge"
        ),
        "api_key_present": api_available,
        "api_base_url_present": bool(os.getenv("OPENAI_BASE_URL")),
        "dry_run": args.dry_run,
        "sources": run_sources,
    }
    write_json(output_root / "run_config.json", run_config)
    status = {
        "status": (
            "DRY_RUN"
            if args.dry_run
            else ("BLOCKED_LLM" if api_blocked else "COMPLETE_OR_PARTIAL_REAL")
        ),
        "reason": (
            "OPENAI_API_KEY is not set; no API request or prediction was created"
            if api_blocked
            else None
        ),
        "protocol": PROMPT_PROTOCOL,
        "model": model,
    }
    write_json(output_root / "status.json", status)
    if summaries:
        pd.DataFrame(summaries).to_csv(output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
