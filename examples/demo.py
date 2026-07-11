"""End-to-end demo of the MSCL language layer.

For each seed pair:
  load JSON -> validate -> (if CHOICE) run dialogue policy with a simulated user
  -> assert resolved -> render to SPRING string -> show round-trip.

Run: python examples/demo.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (spec_from_json, validate, to_spring, from_spring, resolve,
                  resolve_ask_all, resolve_ask_none, assert_resolved, json_schema)
import copy

HERE = os.path.dirname(os.path.abspath(__file__))
pairs = json.load(open(os.path.join(HERE, "seed_pairs.json")))["pairs"]


def simulated_user(q):
    """Pretends to be the user: picks the most sensible option for the demo.
    For 'lamp', choose 'potted plant' (idx 0) and direction 'cright' (idx 1)."""
    print(f"   ?  [{q.kind}] '{q.span}'  ->  options: {q.options_text}")
    pick = 0
    if q.kind == "direction":
        pick = 1  # cright
    print(f"      user picks: {q.options_text[pick]}")
    return pick


for p in pairs:
    print("=" * 78)
    print(f"[{p['id']}]")
    print(f"  EN: {p['english']}")
    spec = spec_from_json(p)
    warnings = validate(spec)
    if warnings:
        print("  warnings:", warnings)

    has_choice = "choice" in json.dumps(p["formula"])
    if has_choice:
        print("  -- ambiguous: running dialogue policy --")
        # policy
        spec_policy = spec_from_json(p)
        _, log = resolve(spec_policy, oracle=simulated_user, budget=3)
        print(f"  policy asked {len(log.asked)} question(s); "
              f"auto-resolved {len(log.auto_resolved)}; "
              f"low-confidence {len(log.low_confidence)}")
        # baselines
        spec_all = spec_from_json(p)
        _, n_all = resolve_ask_all(spec_all, oracle=lambda q: 0)
        spec_none = spec_from_json(p)
        resolve_ask_none(spec_none)
        print(f"  baseline ASK_ALL asked {n_all}; ASK_NONE asked 0")
        spec = spec_policy

    assert_resolved(spec)
    s = to_spring(spec.formula)
    print(f"  SPRING: {s}")
    # round-trip
    assert to_spring(from_spring(s)) == s
    print("  round-trip: OK")

print("=" * 78)
sch = json_schema()
print(f"JSON schema: {len(json.dumps(sch))} bytes, "
      f"{len(sch['$defs']['node']['oneOf'])} node kinds, "
      f"28 relations enumerated.")
print("All demos completed.")
