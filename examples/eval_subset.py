"""Diagnostic subset eval: accuracy + CHOICE-emission tally + miss dumps.

Run AFTER `backend` and `test` exist in your notebook (or run standalone; it will
generate the test set itself). This answers the question the plain accuracy number
can't: is the model EMITTING CHOICE on ambiguous input at all?

Usage in notebook:
    from examples.eval_subset import run_diagnostic
    run_diagnostic(backend, test, n=12)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl.evaluate import exact_match, semantic_equiv, is_correct
from mscl.local_debug import parse_with_repair
from mscl.json_io import spec_to_json


def _has_choice(formula_json) -> bool:
    return '"choice"' in json.dumps(formula_json)


def run_diagnostic(backend, test_samples, n=12, dump_misses=True,
                   adaptive_examples=True):
    """test_samples: list of datagen Sample objects OR dicts with english/objects/formula."""
    def _get(s, k):
        return getattr(s, k, None) if not isinstance(s, dict) else s.get(k)

    subset = test_samples[:n]
    stats = {"exact": 0, "semantic": 0, "miss": 0, "error": 0}
    ambig_total = ambig_choice_emitted = 0
    unamb_total = unamb_correct = 0
    ambig_correct = 0
    timing_totals = {"inference": 0.0, "postprocess": 0.0, "validation": 0.0,
                     "scoring": 0.0, "wall": 0.0, "attempts": 0,
                     "prompt_tokens": 0, "prompt_tokens_known": True}

    def _record_timing(pt, score_s, wall_s):
        if pt is not None:
            timing_totals["inference"] += pt.inference_s
            timing_totals["postprocess"] += pt.postprocess_s
            timing_totals["validation"] += pt.validation_s
            timing_totals["attempts"] += pt.attempts
            if pt.prompt_tokens_sent is None:
                timing_totals["prompt_tokens_known"] = False
            else:
                timing_totals["prompt_tokens"] += pt.prompt_tokens_sent
        timing_totals["scoring"] += score_s
        timing_totals["wall"] += wall_s

    def _timing_text(pt, score_s, wall_s):
        if pt is None:
            return f"total={wall_s:.1f}s"
        return (f"infer={pt.inference_s:.1f}s  check={pt.checking_s:.3f}s "
                f"score={score_s:.3f}s  total={wall_s:.1f}s  tries={pt.attempts}")

    for i, s in enumerate(subset, 1):
        english = _get(s, "english")
        objects = _get(s, "objects_json") or _get(s, "objects")
        gold_formula = (_get(s, "gold_json") or {}).get("formula") if _get(s, "gold_json") else _get(s, "formula")
        gold = {"objects": objects, "formula": gold_formula}
        is_ambig = bool(_get(s, "ambiguous"))

        t0 = time.perf_counter()
        try:
            pred, pt = parse_with_repair(english, objects, backend, retries=1,
                                         adaptive_examples=adaptive_examples,
                                         return_timing=True)
        except Exception as e:
            wall_s = time.perf_counter() - t0
            pt = getattr(e, "parse_timing", None)
            _record_timing(pt, 0.0, wall_s)
            stats["error"] += 1
            print(f"[{i}/{n}] ERROR {type(e).__name__:15s} :: {english[:60]}")
            print(f"      TIME: {_timing_text(pt, 0.0, wall_s)}")
            print(f"      ERROR MSG: {e}")
            if is_ambig: ambig_total += 1
            else: unamb_total += 1
            continue

        pred_json = spec_to_json(pred)["formula"]
        pred_has_choice = _has_choice(pred_json)
        gold_has_choice = _has_choice(gold_formula)

        score_t0 = time.perf_counter()
        if exact_match(pred, gold):
            stats["exact"] += 1; tag = "EXACT"; ok = True
        elif is_correct(pred, gold):    # semantic (unambiguous) or structural (CHOICE)
            stats["semantic"] += 1; tag = "correct"; ok = True
        else:
            stats["miss"] += 1; tag = "miss"; ok = False
        score_s = time.perf_counter() - score_t0
        wall_s = time.perf_counter() - t0
        _record_timing(pt, score_s, wall_s)

        if is_ambig:
            ambig_total += 1
            if pred_has_choice: ambig_choice_emitted += 1
            if ok: ambig_correct += 1
        else:
            unamb_total += 1
            if ok: unamb_correct += 1

        marker = ""
        if is_ambig:
            marker = f"  [ambig; CHOICE emitted: {'YES' if pred_has_choice else 'NO'}]"
        print(f"[{i}/{n}] {tag:9s} :: {english[:55]}{marker}")
        print(f"      TIME: {_timing_text(pt, score_s, wall_s)}")

        if not ok and dump_misses:
            print(f"      GOLD: {json.dumps(gold_formula)[:250]}")
            print(f"      PRED: {json.dumps(pred_json)[:250]}")

    print("\n===== DIAGNOSTIC SUMMARY =====")
    total = len(subset)
    print(f"overall: exact {stats['exact']}/{total}, +semantic -> "
          f"{stats['exact']+stats['semantic']}/{total}, miss {stats['miss']}, err {stats['error']}")
    if unamb_total:
        print(f"UNAMBIGUOUS accuracy: {unamb_correct}/{unamb_total} "
              f"({unamb_correct/unamb_total:.0%})  <- few-shot ceiling on the easy half")
    if ambig_total:
        print(f"AMBIGUOUS accuracy:   {ambig_correct}/{ambig_total} "
              f"({ambig_correct/ambig_total:.0%})")
        print(f"CHOICE emission rate: {ambig_choice_emitted}/{ambig_total} "
              f"({ambig_choice_emitted/ambig_total:.0%})  <- THE fine-tune decision number")
        print("   rule of thumb: <50% emission => fine-tune; >80% => stay few-shot")
    denom = max(1, total)
    checking = timing_totals["postprocess"] + timing_totals["validation"]
    print("\n===== TIMING SUMMARY =====")
    print(f"average per sample: inference={timing_totals['inference']/denom:.2f}s  "
          f"checking={checking/denom:.4f}s  scoring={timing_totals['scoring']/denom:.4f}s  "
          f"wall={timing_totals['wall']/denom:.2f}s")
    print(f"totals: inference={timing_totals['inference']:.1f}s  "
          f"postprocess={timing_totals['postprocess']:.4f}s  "
          f"validation={timing_totals['validation']:.4f}s  "
          f"scoring={timing_totals['scoring']:.4f}s  attempts={timing_totals['attempts']}")
    if timing_totals["prompt_tokens_known"]:
        print(f"average prompt tokens sent={timing_totals['prompt_tokens']/denom:.0f}")
    stats["timing"] = timing_totals
    return stats


if __name__ == "__main__":
    print("This module is meant to be imported in your notebook after `backend` exists:")
    print("  from examples.eval_subset import run_diagnostic")
    print("  run_diagnostic(backend, test, n=12)")
