from src.llm.parsing import parse_prediction


def test_parse_prediction_accepts_json_and_normalized_candidate():
    candidates = ["Neural_Networks", "Case Based"]
    assert parse_prediction('{"label": "Neural_Networks"}', candidates) == (
        "Neural_Networks",
        None,
    )
    assert parse_prediction("```json\n{\"label\": \"case-based\"}\n```", candidates) == (
        "Case Based",
        None,
    )


def test_parse_prediction_rejects_unknown_or_missing_label():
    candidates = ["linked", "not linked"]
    prediction, error = parse_prediction('{"label": "maybe"}', candidates)
    assert prediction is None
    assert error is not None
    prediction, error = parse_prediction('{"answer": "linked"}', candidates)
    assert prediction is None
    assert error is not None
