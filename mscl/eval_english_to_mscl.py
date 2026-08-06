"""Evaluation 1: English -> MSCL and CHOICE construction.

This module deliberately stops before dialogue resolution.  It measures whether a parser:

* detects that an instruction is ambiguous;
* emits the right number and kinds of CHOICE nodes;
* preserves the plausible option set (and the benchmark's hidden intended option);
* preserves the non-CHOICE MSCL skeleton; and
* avoids CHOICE nodes on unambiguous controls.

Priors, CHOICE-option order, and the exact positive value selected for a vague offset are not
graded.  They are parser preferences rather than user-visible logical structure.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .json_io import spec_from_json, spec_to_json
from .validate import validate


CHOICE_KINDS = ("direction", "offset", "reference", "scope", "unsupported_type")


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _normal_span(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def choice_nodes(formula: Mapping[str, Any]) -> List[dict]:
    """Return every CHOICE node, including choices nested inside options."""
    found: List[dict] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("node") == "choice":
                found.append(node)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(formula)
    return found


def has_choice(formula: Mapping[str, Any]) -> bool:
    return bool(choice_nodes(formula))


def _relation_const(node: Mapping[str, Any], *, offset_shape: bool) -> Any:
    value = node.get("const")
    # A missing binary offset has the same MSCL meaning as zero.
    if value is None:
        value = 0
    if offset_shape and isinstance(value, (int, float)) and value > 0:
        return "POSITIVE"
    return value


def formula_signature(node: Optional[Mapping[str, Any]], *, offset_shape: bool = False) -> str:
    """Canonical, order-insensitive structural signature for an MSCL JSON node."""
    if node is None:
        return "SKIP"
    kind = node.get("node")
    if kind in ("and", "or"):
        children = sorted(formula_signature(arg, offset_shape=offset_shape)
                          for arg in node.get("args", []))
        return f"{kind}(" + ",".join(children) + ")"
    if kind == "not":
        return f"not({formula_signature(node.get('arg'), offset_shape=offset_shape)})"
    if kind == "rel":
        args = ",".join(node.get("args", []))
        const = _relation_const(node, offset_shape=offset_shape)
        return f"rel:{node.get('name')}({args}|{const})"
    if kind == "type":
        return f"type:{node.get('obj')}={node.get('type')}"
    if kind == "property":
        return f"property:{node.get('obj')}={node.get('value')}"
    if kind == "default":
        return f"default:{node.get('obj')}"
    if kind == "choice":
        choice_kind = node.get("kind")
        option_shape = choice_kind == "offset"
        options = sorted(formula_signature(option.get("formula"), offset_shape=option_shape)
                         for option in node.get("options", []))
        emphasis = bool(node.get("emphasis", False)) if choice_kind == "offset" else False
        return f"choice:{choice_kind}:{int(emphasis)}(" + ",".join(options) + ")"
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def option_signatures(choice: Mapping[str, Any]) -> set[str]:
    offset_shape = choice.get("kind") == "offset"
    return {
        formula_signature(option.get("formula"), offset_shape=offset_shape)
        for option in choice.get("options", [])
    }


def _set_counts(gold: set[str], pred: set[str]) -> Tuple[int, int, int]:
    return len(gold & pred), len(pred - gold), len(gold - pred)


def _set_f1(gold: set[str], pred: set[str]) -> float:
    tp, fp, fn = _set_counts(gold, pred)
    return _prf(tp, fp, fn)["f1"]


def match_choices(gold_choices: Sequence[dict], pred_choices: Sequence[dict]) -> List[Tuple[int, int]]:
    """Greedily pair same-kind choices by option overlap, then normalized span agreement."""
    candidates: List[Tuple[float, int, int, int]] = []
    for gi, gold in enumerate(gold_choices):
        for pi, pred in enumerate(pred_choices):
            if gold.get("kind") != pred.get("kind"):
                continue
            score = _set_f1(option_signatures(gold), option_signatures(pred))
            span_equal = int(_normal_span(gold.get("span", "")) ==
                             _normal_span(pred.get("span", "")))
            candidates.append((score, span_equal, gi, pi))
    candidates.sort(reverse=True)
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for _, _, gi, pi in candidates:
        if gi not in used_gold and pi not in used_pred:
            used_gold.add(gi)
            used_pred.add(pi)
            pairs.append((gi, pi))
    return pairs


def _hard_leaves(formula: Mapping[str, Any]) -> set[str]:
    """Leaf signatures outside CHOICE nodes: the unambiguous MSCL skeleton."""
    leaves: set[str] = set()

    def visit(node: Any, inside_choice: bool = False) -> None:
        if not isinstance(node, dict):
            return
        kind = node.get("node")
        if kind == "choice":
            return
        if kind in ("rel", "type", "property", "default") and not inside_choice:
            leaves.add(formula_signature(node))
            return
        if kind in ("and", "or"):
            for child in node.get("args", []):
                visit(child, inside_choice)
        elif kind == "not":
            visit(node.get("arg"), inside_choice)

    visit(formula)
    return leaves


def structural_match(gold_formula: Mapping[str, Any], pred_formula: Mapping[str, Any]) -> bool:
    """Strict structural MSCL match, ignoring only non-semantic CHOICE metadata."""
    return formula_signature(gold_formula) == formula_signature(pred_formula)


def _normalize_prediction(example: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict:
    if "prediction" in prediction and isinstance(prediction["prediction"], dict):
        prediction = prediction["prediction"]
    if "formula" not in prediction:
        raise ValueError("prediction does not contain a formula")
    return {"objects": example["objects"], "formula": prediction["formula"]}


def _validate_prediction(example: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict:
    normalized = _normalize_prediction(example, prediction)
    spec = spec_from_json(normalized)
    validate(spec)
    return spec_to_json(spec)


@dataclass
class _KindCounts:
    gold: int = 0
    pred: int = 0
    matched: int = 0
    exact_option_sets: int = 0
    option_tp: int = 0
    option_fp: int = 0
    option_fn: int = 0
    intent_total: int = 0
    intent_covered: int = 0


@dataclass
class _Totals:
    examples: int = 0
    valid: int = 0
    parse_errors: int = 0
    amb_tp: int = 0
    amb_fp: int = 0
    amb_fn: int = 0
    amb_tn: int = 0
    gold_choices: int = 0
    pred_choices: int = 0
    matched_choices: int = 0
    option_tp: int = 0
    option_fp: int = 0
    option_fn: int = 0
    ambiguous_total: int = 0
    exact_choice_ambiguous: int = 0
    exact_choice_examples: int = 0
    structural_correct: int = 0
    unambiguous_total: int = 0
    unambiguous_structural_correct: int = 0
    false_choices_on_unambiguous: int = 0
    intent_total: int = 0
    intent_covered: int = 0
    hard_tp: int = 0
    hard_fp: int = 0
    hard_fn: int = 0
    kinds: Dict[str, _KindCounts] = field(
        default_factory=lambda: defaultdict(_KindCounts))


def evaluate_predictions(examples: Sequence[Mapping[str, Any]],
                         predictions: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Score predictions keyed by benchmark example id.

    Invalid or missing predictions count as parse errors and as false negatives for all gold
    structures.  The returned object is JSON serializable and includes per-example diagnostics.
    """
    totals = _Totals()
    rows: List[dict] = []

    for example in examples:
        totals.examples += 1
        example_id = example["id"]
        gold_formula = example["formula"]
        gold_choices = choice_nodes(gold_formula)
        gold_ambiguous = bool(gold_choices)
        if gold_ambiguous:
            totals.ambiguous_total += 1
        if not gold_ambiguous:
            totals.unambiguous_total += 1
        for choice in gold_choices:
            totals.kinds[choice["kind"]].gold += 1
        totals.gold_choices += len(gold_choices)

        raw_prediction = predictions.get(example_id)
        error: Optional[str] = None
        pred_formula: Mapping[str, Any] = {"node": "and", "args": []}
        if raw_prediction is None:
            error = "missing prediction"
        elif raw_prediction.get("error"):
            error = str(raw_prediction["error"])
        else:
            try:
                pred_formula = _validate_prediction(example, raw_prediction)["formula"]
                totals.valid += 1
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if error:
            totals.parse_errors += 1

        pred_choices = choice_nodes(pred_formula) if not error else []
        pred_ambiguous = bool(pred_choices)
        for choice in pred_choices:
            totals.kinds[choice["kind"]].pred += 1
        totals.pred_choices += len(pred_choices)

        if gold_ambiguous and pred_ambiguous:
            totals.amb_tp += 1
        elif not gold_ambiguous and pred_ambiguous:
            totals.amb_fp += 1
        elif gold_ambiguous and not pred_ambiguous:
            totals.amb_fn += 1
        else:
            totals.amb_tn += 1

        if not gold_ambiguous:
            totals.false_choices_on_unambiguous += len(pred_choices)

        pairs = match_choices(gold_choices, pred_choices)
        totals.matched_choices += len(pairs)
        pair_by_gold = {gi: pi for gi, pi in pairs}
        exact_options = True
        for gi, gold_choice in enumerate(gold_choices):
            kind_counts = totals.kinds[gold_choice["kind"]]
            if gi not in pair_by_gold:
                exact_options = False
                missing = len(option_signatures(gold_choice))
                totals.option_fn += missing
                kind_counts.option_fn += missing
                continue
            pred_choice = pred_choices[pair_by_gold[gi]]
            kind_counts.matched += 1
            gold_options = option_signatures(gold_choice)
            pred_options = option_signatures(pred_choice)
            tp, fp, fn = _set_counts(gold_options, pred_options)
            totals.option_tp += tp
            totals.option_fp += fp
            totals.option_fn += fn
            kind_counts.option_tp += tp
            kind_counts.option_fp += fp
            kind_counts.option_fn += fn
            if gold_options == pred_options:
                kind_counts.exact_option_sets += 1
            else:
                exact_options = False

        matched_pred = {pi for _, pi in pairs}
        for pi, pred_choice in enumerate(pred_choices):
            if pi not in matched_pred:
                extra = len(option_signatures(pred_choice))
                totals.option_fp += extra
                totals.kinds[pred_choice["kind"]].option_fp += extra
                exact_options = False

        if len(gold_choices) != len(pred_choices):
            exact_options = False
        if exact_options and not error:
            totals.exact_choice_examples += 1
            if gold_ambiguous:
                totals.exact_choice_ambiguous += 1

        hard_gold = _hard_leaves(gold_formula)
        hard_pred = _hard_leaves(pred_formula) if not error else set()
        hard_tp, hard_fp, hard_fn = _set_counts(hard_gold, hard_pred)
        totals.hard_tp += hard_tp
        totals.hard_fp += hard_fp
        totals.hard_fn += hard_fn

        structural_ok = False if error else structural_match(gold_formula, pred_formula)
        if structural_ok:
            totals.structural_correct += 1
            if not gold_ambiguous:
                totals.unambiguous_structural_correct += 1

        example_intent_total = 0
        example_intent_covered = 0
        for annotation in example.get("intended_options", []):
            kind = annotation["kind"]
            signature = formula_signature(
                annotation.get("formula"), offset_shape=(kind == "offset"))
            covered = any(
                choice.get("kind") == kind and signature in option_signatures(choice)
                for choice in pred_choices
            )
            totals.intent_total += 1
            totals.kinds[kind].intent_total += 1
            example_intent_total += 1
            if covered:
                totals.intent_covered += 1
                totals.kinds[kind].intent_covered += 1
                example_intent_covered += 1

        rows.append({
            "id": example_id,
            "ambiguity_type": example.get("ambiguity_type", "unambiguous"),
            "gold_ambiguous": gold_ambiguous,
            "pred_ambiguous": pred_ambiguous,
            "valid": error is None,
            "error": error,
            "gold_choice_kinds": [c.get("kind") for c in gold_choices],
            "pred_choice_kinds": [c.get("kind") for c in pred_choices],
            "choice_kind_matches": len(pairs),
            "exact_option_sets": exact_options,
            "intent_coverage": _safe_div(example_intent_covered, example_intent_total),
            "hard_skeleton_f1": _prf(hard_tp, hard_fp, hard_fn)["f1"],
            "structural_mscl_correct": structural_ok,
        })

    ambiguity = _prf(totals.amb_tp, totals.amb_fp, totals.amb_fn)
    choice_nodes_metric = _prf(
        totals.matched_choices,
        totals.pred_choices - totals.matched_choices,
        totals.gold_choices - totals.matched_choices,
    )
    option_metric = _prf(totals.option_tp, totals.option_fp, totals.option_fn)
    hard_metric = _prf(totals.hard_tp, totals.hard_fp, totals.hard_fn)

    by_kind: Dict[str, dict] = {}
    all_kinds = sorted(set(CHOICE_KINDS) | set(totals.kinds))
    for kind in all_kinds:
        counts = totals.kinds[kind]
        node_metric = _prf(counts.matched, counts.pred - counts.matched,
                           counts.gold - counts.matched)
        by_kind[kind] = {
            "gold_choices": counts.gold,
            "pred_choices": counts.pred,
            "choice_detection": node_metric,
            "exact_option_set_accuracy": _safe_div(counts.exact_option_sets, counts.gold),
            "option_set": _prf(counts.option_tp, counts.option_fp, counts.option_fn),
            "intended_option_coverage": _safe_div(counts.intent_covered, counts.intent_total),
            "intended_options": counts.intent_total,
        }

    metrics = {
        "n_examples": totals.examples,
        "valid_mscl_rate": _safe_div(totals.valid, totals.examples),
        "parse_errors": totals.parse_errors,
        "ambiguity_detection": ambiguity,
        "choice_emission_recall": ambiguity["recall"],
        "choice_node_detection": choice_nodes_metric,
        "choice_option_set": option_metric,
        "exact_choice_construction_rate": _safe_div(
            totals.exact_choice_ambiguous, totals.ambiguous_total),
        "correct_choice_behavior_rate": _safe_div(
            totals.exact_choice_examples, totals.examples),
        "intended_option_coverage": _safe_div(totals.intent_covered, totals.intent_total),
        "intended_options": totals.intent_total,
        "structural_mscl_accuracy": _safe_div(totals.structural_correct, totals.examples),
        "hard_skeleton": hard_metric,
        "unambiguous_accuracy": _safe_div(
            totals.unambiguous_structural_correct, totals.unambiguous_total),
        "false_choice_rate_on_unambiguous": _safe_div(
            totals.amb_fp, totals.unambiguous_total),
        "potential_false_questions_per_unambiguous_prompt": _safe_div(
            totals.false_choices_on_unambiguous, totals.unambiguous_total),
        "by_choice_kind": by_kind,
    }
    return {"metrics": metrics, "per_example": rows}


def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def predictions_by_id(records: Iterable[Mapping[str, Any]]) -> Dict[str, dict]:
    output: Dict[str, dict] = {}
    for record in records:
        example_id = record.get("id") or record.get("example_id")
        if not example_id:
            raise ValueError("every prediction record must contain id or example_id")
        if example_id in output:
            raise ValueError(f"duplicate prediction id: {example_id}")
        output[str(example_id)] = dict(record)
    return output
