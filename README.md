# MSCL — the complete language layer (ready for an English→MSCL model)

This package is the **target-language layer** for the English→logic system. It implements
MSCL v1 (Metric Spatial Constraint Logic) and the MSCL-SPRING profile, end to end, with
the **CHOICE underspecification construct** and the **E1 interactive resolution policy**.
An English→MSCL model sits *on top* of this and emits the JSON described below; everything
downstream (validation, ambiguity resolution, rendering to SPRING, feasibility) is here and
tested.

## What's implemented (and tested — `python tests/test_core.py`, 25/25)

| File | Spec section | What it does |
|---|---|---|
| `mscl/ast.py` | §3–§6 | Internal AST: objects, normalized atoms, 28 named relations, boolean, typing, `CHOICE` w/ provenance |
| `mscl/relations.py` | §4–§5 | The 28 relations + desugaring to Layer-1 linear atoms (verified vs Table 1) |
| `mscl/profile.py` | §9 | MSCL-SPRING profile: 17 types, per-mille range, `default()` macro |
| `mscl/json_io.py` | §6 | LLM-facing **JSON AST** ⇄ internal AST, + the **JSON Schema** for constrained decoding |
| `mscl/render.py` | §9.1 | Lossless **MSCL ⇄ SPRING-string** rendering (Lark parser for the reverse) |
| `mscl/feasibility.py` | §7 | Exact `model_check`; **sound bounds-propagation feasibility** (exact SAT → SampleSearch) |
| `mscl/dialogue.py` | §6.3, E1 | **The resolution policy**: prune → silent-collapse → rank/ask under budget → close. Plus ASK_ALL/ASK_NONE baselines |
| `mscl/validate.py` | §9 | Well-formedness + profile conformance; `assert_resolved` (no CHOICE in final spec) |
| `mscl/mscl_spring.schema.json` | — | The JSON Schema artifact (for Outlines / xgrammar / structured-output APIs) |
| `mscl/mscl_spring.gbnf` | — | GBNF grammar (for llama.cpp token-level constrained decoding) |
| `examples/seed_pairs.json` | §10 | Gold English→MSCL pairs from SPRING's appendix (few-shot + seed eval) |
| `examples/demo.py` | — | Full pipeline over the gold pairs |

## The contract the English→MSCL model fills

The model receives:
1. the **English utterance**, and
2. the **detected-object table** (from the perception module): `[{id,status,type,box}...]`.

It emits **one JSON object** conforming to `mscl_spring.schema.json`:
```json
{"objects":[...], "formula": <node>}
```
where ambiguity is expressed with `choice` nodes (see `seed_pairs.json` → `ambiguous_lamp_by_couch`).
Grammar-constrained decoding (`mscl_spring.gbnf` for local models, or the JSON schema for
structured-output) guarantees the output is syntactically valid — no hallucinated predicates.

Then this layer does the rest:
```
spec = spec_from_json(model_output)
validate(spec)                         # well-formed + profile-conformant
resolve(spec, oracle=user, budget=3)   # E1: ask only when it changes the outcome
assert_resolved(spec)                  # CHOICE-free
spring_string = to_spring(spec.formula)  # feed to (reimplemented) SampleSearch + VEG
```

## The research contribution lives in `dialogue.py`

The policy's objective is **ask a clarifying question only when the answer changes the
outcome**, under a question budget:
1. **Prune** CHOICE options that are infeasible given the hard constraints (feasibility oracle).
   A CHOICE that collapses to one option resolves silently.
2. **Silent-collapse** offset CHOICEs (no emphasis, not degenerate) to `c=0`.
3. **Rank & ask** the rest, structural (`unsupported_type`,`reference`) before geometric
   (`scope`,`direction`,`offset`), re-pruning after each answer, until budget spent.
4. **Close** leftovers by max prior, flagged low-confidence.

Evaluable against baselines `ASK_ALL`, `ASK_NONE`, `ASK_RANDOM`. In the demo, the policy
asks 2 questions where ASK_ALL asks 3 (offset auto-resolves; one direction option is pruned
as infeasible once the type is fixed).

## What is deliberately NOT in this layer (next artifacts)

- **SampleSearch** (the exact sampler/solver). `feasibility.feasible()` is a *sound* pruning
  oracle — enough for the dialogue policy — but exact satisfiability + preference-weighted
  sampling is the next major build (spec §7 bridge).
- **The English→MSCL model itself** (the thing that consumes this contract).
- **Open-vocabulary detector** (E5) and **soft/weighted atoms** (E2) — reserved hooks.

## Honest status notes
- Desugaring of all 28 relations is verified against Table 1 in tests.
- `coffee table` in the appendix is mapped to the in-vocab `dining table` (no `coffee table`
  type in SPRING's 17); flagged as a profile warning, not silently dropped.
- `default()` per-mille bounds approximate SPRING's 256–512px-on-512 convention; confirm
  against the reimplemented pipeline once SampleSearch exists.
