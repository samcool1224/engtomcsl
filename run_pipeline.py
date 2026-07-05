"""End-to-end harness for the English->MSCL system (Piece 1-4 + eval), GPU-free.

What it does:
  1. Generates a synthetic dataset (50% ambiguous) with the template verbalizer.
  2. Builds a StubBackend that returns the gold formula (so we can exercise the FULL
     pipeline + eval without a model). This isolates: does the harness work end to end?
  3. Runs parse-accuracy, round-trip, and the dialogue-policy evaluation vs baselines.
  4. Writes train/eval JSONL you can later use for few-shot exemplars or fine-tuning.

To use the REAL model on your GPU, replace StubBackend(gold=...) with
LocalBackend("Qwen/Qwen2.5-7B-Instruct") and re-run parse_accuracy — nothing else changes.

Run: python examples/run_pipeline.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import datagen, evaluate
from mscl.parser import StubBackend, build_prompt
from mscl.datagen import generate_dataset, to_jsonl

OUT = os.path.dirname(os.path.abspath(__file__))

print("1) generating synthetic dataset (this runs entirely offline)...")
train = generate_dataset(n=120, ambiguous_frac=0.5, seed=1)
test  = generate_dataset(n=60,  ambiguous_frac=0.5, seed=99)
print(f"   train={len(train)}  test={len(test)}  "
      f"(ambiguous: train={sum(s.ambiguous for s in train)}, test={sum(s.ambiguous for s in test)})")

# show a couple of generated examples
print("\n   sample UNAMBIGUOUS:")
u = next(s for s in train if not s.ambiguous)
print("     EN:", u.english)
print("     GOLD:", json.dumps(u.gold_json["formula"])[:160], "...")
print("   sample AMBIGUOUS:")
a = next(s for s in train if s.ambiguous)
print("     EN:", a.english, "  [kind:", a.meta.get("kind"), "]")
print("     GOLD:", json.dumps(a.gold_json["formula"])[:200], "...")

# build a gold-driven stub backend (maps english -> gold formula) to test the pipeline
gold_map = {s.english: s.gold_json["formula"] for s in test}
backend = StubBackend(gold=gold_map)

print("\n2) prompt harness sanity check:")
ex = test[0]
prompt = build_prompt(ex.english, ex.objects_json, k_shot=3)
print(f"   prompt length: {len(prompt)} chars; ends with 'OUTPUT:' -> {prompt.rstrip().endswith('OUTPUT:')}")

# eval expects dict samples
test_dicts = [{"english": s.english, "objects": s.objects_json,
               "formula": s.gold_json["formula"], "ambiguous": s.ambiguous,
               "meta": s.meta} for s in test]

print("\n3a) parse accuracy (gold-stub backend -> should be ~100%, validating the harness):")
rep = evaluate.parse_accuracy(test_dicts, backend)
print("   ", rep.summary())

print("\n3b) parse accuracy with a NON-gold stub (trivial guesser -> low, sanity floor):")
rep2 = evaluate.parse_accuracy(test_dicts, StubBackend(gold={}))
print("   ", rep2.summary())

print("\n3c) dialogue-policy evaluation vs baselines (on ambiguous gold):")
prep = evaluate.policy_eval(test_dicts, budget=3)
print("   ", prep.summary())
print("   interpretation: policy should ask FEWER questions than ASK_ALL while keeping")
print("   correctness at/above ASK_NONE — the value-of-information win.")

# write datasets
open(os.path.join(OUT, "train.jsonl"), "w").write(to_jsonl(train))
open(os.path.join(OUT, "test.jsonl"), "w").write(to_jsonl(test))
print(f"\n4) wrote train.jsonl ({len(train)}) and test.jsonl ({len(test)}).")
print("\nDONE. To run the real model: swap StubBackend for LocalBackend on your GPU.")
