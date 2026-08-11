from __future__ import annotations

import json
import re


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def parse_prediction(raw: str, candidates: list[str]) -> tuple[str | None, str | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    value: object = text
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            value = decoded.get("label")
        else:
            value = decoded
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match:
            try:
                decoded = json.loads(match.group(0))
                value = decoded.get("label") if isinstance(decoded, dict) else decoded
            except json.JSONDecodeError:
                value = text
    if not isinstance(value, str):
        return None, "response does not contain a string label"
    normalized = normalize_label(value)
    lookup: dict[str, list[str]] = {}
    for candidate in candidates:
        lookup.setdefault(normalize_label(candidate), []).append(candidate)
    matches = lookup.get(normalized, [])
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, "normalized label is ambiguous"
    return None, f"label is not one of {candidates}"
