"""Run Evaluation 1: English -> MSCL and CHOICE construction.

Examples:

  # Verify the scorer against gold predictions (must produce 100% structural accuracy).
  python -m examples.evaluate_english_to_mscl --backend gold

  # Run the existing non-neural stub as a low baseline.
  python -m examples.evaluate_english_to_mscl --backend stub

  # Run Qwen on a GPU and save reusable predictions.
  python -m examples.evaluate_english_to_mscl --backend local --quantize 4bit

  # Re-score already generated predictions without reloading the model.
  python -m examples.evaluate_english_to_mscl --predictions results/eval1/predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from mscl.eval_english_to_mscl import (evaluate_predictions, load_jsonl,
                                       predictions_by_id)
from mscl.json_io import spec_to_json
from mscl.local_debug import parse_with_repair
from mscl.parser import LocalBackend, StubBackend


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "eval1_pilot.jsonl"


def _run_backend(examples, backend_name, args):
    if backend_name == "gold":
        return {
            row["id"]: {"id": row["id"], "prediction": {
                "objects": row["objects"], "formula": row["formula"]}}
            for row in examples
        }
    if backend_name == "stub":
        backend = StubBackend()
    elif backend_name == "local":
        quantize = None if args.quantize == "none" else args.quantize
        backend = LocalBackend(
            args.model,
            quantize=quantize,
            adapter_path=args.adapter,
            max_tokens=args.max_tokens,
        )
    else:
        raise ValueError(f"unknown backend: {backend_name}")

    predictions: Dict[str, dict] = {}
    for index, row in enumerate(examples, 1):
        print(f"[{index}/{len(examples)}] {row['id']}  {row['english']}")
        try:
            spec, timing = parse_with_repair(
                row["english"], row["objects"], backend,
                retries=args.retries,
                adaptive_examples=not args.no_adaptive_examples,
                return_timing=True,
            )
            predictions[row["id"]] = {
                "id": row["id"],
                "prediction": spec_to_json(spec),
                "timing": asdict(timing),
            }
        except Exception as exc:
            predictions[row["id"]] = {
                "id": row["id"],
                "error": f"{type(exc).__name__}: {exc}",
            }
    return predictions


def _pct(value):
    return f"{100.0 * value:.1f}%"


def _print_summary(metrics):
    ambiguity = metrics["ambiguity_detection"]
    options = metrics["choice_option_set"]
    print("\n=== EVALUATION 1: ENGLISH -> MSCL / CHOICE ===")
    print(f"examples:                         {metrics['n_examples']}")
    print(f"valid MSCL:                       {_pct(metrics['valid_mscl_rate'])}")
    print(f"ambiguity precision / recall / F1:{_pct(ambiguity['precision'])} / "
          f"{_pct(ambiguity['recall'])} / {_pct(ambiguity['f1'])}")
    print(f"CHOICE emission recall:           {_pct(metrics['choice_emission_recall'])}")
    print(f"option precision / recall / F1:   {_pct(options['precision'])} / "
          f"{_pct(options['recall'])} / {_pct(options['f1'])}")
    print(f"intended-option coverage:         {_pct(metrics['intended_option_coverage'])}")
    print(f"unambiguous structural accuracy:  {_pct(metrics['unambiguous_accuracy'])}")
    print(f"false-CHOICE rate (unambiguous):  {_pct(metrics['false_choice_rate_on_unambiguous'])}")
    print(f"overall structural MSCL accuracy: {_pct(metrics['structural_mscl_accuracy'])}")
    print("\nBy CHOICE kind:")
    for kind, row in metrics["by_choice_kind"].items():
        detection = row["choice_detection"]
        option_set = row["option_set"]
        print(f"  {kind:18s} n={row['gold_choices']:2d}  detect-F1={_pct(detection['f1'])}  "
              f"option-F1={_pct(option_set['f1'])}  "
              f"intent={_pct(row['intended_option_coverage'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--backend", choices=("gold", "stub", "local"), default="gold")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--quantize", choices=("none", "4bit", "8bit"), default="4bit")
    parser.add_argument("--adapter")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-adaptive-examples", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "eval1")
    args = parser.parse_args()

    examples = load_jsonl(str(args.benchmark))
    if args.limit is not None:
        examples = examples[:args.limit]
    if args.predictions:
        predictions = predictions_by_id(load_jsonl(str(args.predictions)))
    else:
        predictions = _run_backend(examples, args.backend, args)

    report = evaluate_predictions(examples, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    prediction_path.write_text(
        "\n".join(json.dumps(predictions[key], sort_keys=True) for key in predictions) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report["metrics"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "per_example.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in report["per_example"]) + "\n",
        encoding="utf-8",
    )
    _print_summary(report["metrics"])
    print(f"\nsaved predictions: {prediction_path}")
    print(f"saved metrics:     {args.output_dir / 'metrics.json'}")
    print(f"saved diagnostics: {args.output_dir / 'per_example.jsonl'}")


if __name__ == "__main__":
    main()

