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

from mscl.evaluate import exact_match, semantic_equiv
from mscl.local_debug import parse_with_repair
from mscl.json_io import spec_to_json


def _has_choice(formula_json) -> bool:
    return '"choice"' in json.dumps(formula_json)


def run_diagnostic(backend, test_samples, n=12, dump_misses=True, log_file="run_diag_subset.txt"):
    """test_samples: list of datagen Sample objects OR dicts with english/objects/formula."""
    
    # Open the file and define a helper function to print to both console and file
    with open(log_file, "w", encoding="utf-8") as f_out:
        
        def log_print(*args, **kwargs):
            # Print to console
            print(*args, **kwargs)
            # Print to file
            print(*args, file=f_out, **kwargs)

        def _get(s, k):
            return getattr(s, k, None) if not isinstance(s, dict) else s.get(k)

        subset = test_samples[:n]
        stats = {"exact": 0, "semantic": 0, "miss": 0, "error": 0}
        ambig_total = ambig_choice_emitted = 0
        unamb_total = unamb_correct = 0
        ambig_correct = 0

        for i, s in enumerate(subset, 1):
            english = _get(s, "english")
            objects = _get(s, "objects_json") or _get(s, "objects")
            gold_formula = (_get(s, "gold_json") or {}).get("formula") if _get(s, "gold_json") else _get(s, "formula")
            gold = {"objects": objects, "formula": gold_formula}
            is_ambig = bool(_get(s, "ambiguous"))

            t0 = time.time()
            try:
                pred = parse_with_repair(english, objects, backend, retries=1)
            except Exception as e:
                stats["error"] += 1
                log_print(f"[{i}/{n}] ERROR {type(e).__name__:15s} {time.time()-t0:5.1f}s :: {english[:60]}")
                log_print(f"      ERROR MSG: {e}")
                if is_ambig: ambig_total += 1
                else: unamb_total += 1
                continue

            pred_json = spec_to_json(pred)["formula"]
            pred_has_choice = _has_choice(pred_json)
            gold_has_choice = _has_choice(gold_formula)

            if exact_match(pred, gold):
                stats["exact"] += 1; tag = "EXACT"; ok = True
            elif semantic_equiv(pred, gold):
                stats["semantic"] += 1; tag = "semantic"; ok = True
            else:
                stats["miss"] += 1; tag = "miss"; ok = False

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
            log_print(f"[{i}/{n}] {tag:9s} {time.time()-t0:5.1f}s :: {english[:55]}{marker}")

            if not ok and dump_misses:
                log_print(f"      GOLD: {json.dumps(gold_formula)[:250]}")
                log_print(f"      PRED: {json.dumps(pred_json)[:250]}")

        log_print("\n===== DIAGNOSTIC SUMMARY =====")
        total = len(subset)
        log_print(f"overall: exact {stats['exact']}/{total}, +semantic -> "
              f"{stats['exact']+stats['semantic']}/{total}, miss {stats['miss']}, err {stats['error']}")
        if unamb_total:
            log_print(f"UNAMBIGUOUS accuracy: {unamb_correct}/{unamb_total} "
                  f"({unamb_correct/unamb_total:.0%})  <- few-shot ceiling on the easy half")
        if ambig_total:
            log_print(f"AMBIGUOUS accuracy:   {ambig_correct}/{ambig_total} "
                  f"({ambig_correct/ambig_total:.0%})")
            log_print(f"CHOICE emission rate: {ambig_choice_emitted}/{ambig_total} "
                  f"({ambig_choice_emitted/ambig_total:.0%})  <- THE fine-tune decision number")
            log_print("   rule of thumb: <50% emission => fine-tune; >80% => stay few-shot")
            
        return stats


if __name__ == "__main__":
    print("This module is meant to be imported in your notebook after `backend` exists:")
    print("  from examples.eval_subset import run_diagnostic")
    print("  run_diagnostic(backend, test, n=12)")
