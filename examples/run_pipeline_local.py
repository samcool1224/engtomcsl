import sys, os, json, time
sys.path.insert(0, os.getcwd())          # so `import mscl` works from /content/engtomcsl

from mscl import datagen, evaluate
from mscl.parser import LocalBackend, StubBackend, build_prompt, parse
from mscl.datagen import generate_dataset, to_jsonl

OUT = os.getcwd()

# ---------------------------------------------------------------------------
# 1) data
# ---------------------------------------------------------------------------
print("1) generating synthetic dataset (offline)...")
train = generate_dataset(n=120, ambiguous_frac=0.5, seed=1)
test  = generate_dataset(n=60,  ambiguous_frac=0.5, seed=99)
print(f"   train={len(train)}  test={len(test)}  "
      f"(ambiguous: train={sum(s.ambiguous for s in train)}, test={sum(s.ambiguous for s in test)})")

test_dicts = [{"english": s.english, "objects": s.objects_json,
               "formula": s.gold_json["formula"], "ambiguous": s.ambiguous,
               "meta": s.meta} for s in test]

# ---------------------------------------------------------------------------
# 2) HARNESS SANITY (gold stub) — instant, must be ~100%. Proves eval is correct.
# ---------------------------------------------------------------------------
print("\n2) harness sanity check with gold stub (instant, expect ~100%):")
gold_stub = StubBackend(gold={s.english: s.gold_json["formula"] for s in test})
print("   ", evaluate.parse_accuracy(test_dicts, gold_stub).summary())

# ---------------------------------------------------------------------------
# 3) load the REAL model ONCE
# ---------------------------------------------------------------------------
print("\n3) loading Qwen2.5-7B-Instruct (one time)...")
t0 = time.time()
backend = LocalBackend("Qwen/Qwen2.5-7B-Instruct", quantize="4bit")
print(f"   loaded in {time.time()-t0:.0f}s; outlines api = {backend._api}")

# ---------------------------------------------------------------------------
# 3a) SMOKE TEST the real model on ONE example first (catches API errors fast)
# ---------------------------------------------------------------------------
print("\n3a) single-example smoke test (fast, verifies the model call works):")
ex = test[0]
t0 = time.time()
try:
    spec = parse(ex.english, ex.objects_json, backend=backend)
    print(f"   OK in {time.time()-t0:.1f}s")
    print("   EN:  ", ex.english)
    from mscl.json_io import spec_to_json
    print("   PRED:", json.dumps(spec_to_json(spec)["formula"])[:220], "...")
except Exception as e:
    import traceback; traceback.print_exc()
    print("   >>> model-call failed; fix this before running the full eval.")
    raise

# ---------------------------------------------------------------------------
# 3b) SMALL real-model accuracy (e.g. first 8) with a progress print.
#     Bump SUBSET to len(test) for the full (slow) run once this looks right.
# ---------------------------------------------------------------------------
SUBSET = 8                                   # <-- raise to 60 for the full run
print(f"\n3b) real-model parse accuracy on first {SUBSET} (each ~5-30s):")
subset = test_dicts[:SUBSET]
# lightweight progress: evaluate one at a time
exact = sem = err = 0
from mscl.evaluate import exact_match, is_correct
for i, s in enumerate(subset, 1):
    t0 = time.time()
    try:
        pred = parse(s["english"], s["objects"], backend=backend)
        gold = {"objects": s["objects"], "formula": s["formula"]}
        if exact_match(pred, gold): exact += 1; sem += 1; tag = "EXACT"
        elif is_correct(pred, gold): sem += 1; tag = "correct"
        else: tag = "miss"
    except Exception as e:
        err += 1; tag = f"ERROR {type(e).__name__}"
    print(f"   [{i}/{SUBSET}] {tag:12s} {time.time()-t0:5.1f}s  ::  {s['english'][:55]}")
print(f"   -> exact {exact}/{SUBSET}, semantic {sem}/{SUBSET}, errors {err}")

# ---------------------------------------------------------------------------
# 4) dialogue-policy eval — pure logic, no model, fast. Runs on gold ambiguous specs.
# ---------------------------------------------------------------------------
print("\n4) dialogue-policy evaluation vs baselines (fast, no model):")
print("   ", evaluate.policy_eval(test_dicts, budget=3).summary())

# ---------------------------------------------------------------------------
# 5) save data
# ---------------------------------------------------------------------------
open(os.path.join(OUT, "train.jsonl"), "w").write(to_jsonl(train))
open(os.path.join(OUT, "test.jsonl"), "w").write(to_jsonl(test))
print(f"\n5) wrote train.jsonl ({len(train)}) and test.jsonl ({len(test)}).")
print("\nDONE. Raise SUBSET to 60 in 3b for the full real-model number once it looks right.")
