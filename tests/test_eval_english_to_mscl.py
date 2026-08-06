"""Tests for Evaluation 1. Run: python tests/test_eval_english_to_mscl.py"""
from __future__ import annotations

import importlib.util
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl.eval_english_to_mscl import evaluate_predictions


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER_PATH = os.path.join(ROOT, "benchmarks", "build_eval1_pilot.py")
spec = importlib.util.spec_from_file_location("build_eval1_pilot", BUILDER_PATH)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def _gold_predictions(rows):
    return {row["id"]: {"id": row["id"], "formula": row["formula"]} for row in rows}


def test_pilot_covers_every_choice_kind_and_controls():
    rows = builder.build_records()
    kinds = {row["ambiguity_type"] for row in rows}
    assert len(rows) == 44
    assert kinds == {"unambiguous", "direction", "offset", "reference",
                     "unsupported_type", "scope"}
    assert sum(not row["ambiguous"] for row in rows) == 12


def test_gold_predictions_score_perfectly():
    rows = builder.build_records()
    report = evaluate_predictions(rows, _gold_predictions(rows))["metrics"]
    assert report["valid_mscl_rate"] == 1.0
    assert report["ambiguity_detection"]["f1"] == 1.0
    assert report["choice_option_set"]["f1"] == 1.0
    assert report["intended_option_coverage"] == 1.0
    assert report["unambiguous_accuracy"] == 1.0
    assert report["false_choice_rate_on_unambiguous"] == 0.0
    assert report["structural_mscl_accuracy"] == 1.0


def test_single_best_parse_misses_ambiguity():
    rows = builder.build_records()
    predictions = {}

    def collapse(node):
        if isinstance(node, dict) and node.get("node") == "choice":
            return collapse(node["options"][0]["formula"])
        if isinstance(node, dict) and node.get("node") in ("and", "or"):
            children = [collapse(child) for child in node["args"]]
            children = [child for child in children if child is not None]
            return children[0] if len(children) == 1 else {"node": node["node"], "args": children}
        if isinstance(node, dict) and node.get("node") == "not":
            return {"node": "not", "arg": collapse(node["arg"])}
        return node

    for row in rows:
        predictions[row["id"]] = {"id": row["id"], "formula": collapse(row["formula"])}
    metrics = evaluate_predictions(rows, predictions)["metrics"]
    assert metrics["choice_emission_recall"] == 0.0
    assert metrics["ambiguity_detection"]["fn"] == 32
    assert metrics["intended_option_coverage"] == 0.0


def test_partial_option_set_gets_partial_credit():
    row = next(row for row in builder.build_records() if row["id"] == "d003")
    prediction = {"id": row["id"], "formula": copy.deepcopy(row["formula"])}
    predicted_choice = next(arg for arg in prediction["formula"]["args"]
                            if arg.get("node") == "choice")
    predicted_choice["options"] = predicted_choice["options"][:2]
    metrics = evaluate_predictions([row], {row["id"]: prediction})["metrics"]
    assert metrics["ambiguity_detection"]["f1"] == 1.0
    assert metrics["choice_option_set"]["precision"] == 1.0
    assert metrics["choice_option_set"]["recall"] == 0.5
    assert metrics["structural_mscl_accuracy"] == 0.0


if __name__ == "__main__":
    import traceback
    functions = [value for name, value in sorted(globals().items())
                 if name.startswith("test_") and callable(value)]
    passed = 0
    for function in functions:
        try:
            function()
            passed += 1
            print(f"PASS {function.__name__}")
        except Exception as exc:
            print(f"FAIL {function.__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{passed}/{len(functions)} passed")
    raise SystemExit(0 if passed == len(functions) else 1)
