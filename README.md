# MSCL — English spatial logic plus exact layout generation

This package is the **target-language layer** for the English→logic system. It implements
MSCL v1 (Metric Spatial Constraint Logic) and the MSCL-SPRING profile, end to end, with
the **CHOICE underspecification construct** and the **E1 interactive resolution policy**.
An English→MSCL model sits *on top* of this and emits the JSON described below; everything
downstream (validation, ambiguity resolution, exact layout solving, and verification) is here
and tested.

## From-scratch English to image

The complete prototype path is now:

`English -> automatic object table -> MSCL -> CHOICE resolution -> exact SampleSearch -> GLIGEN image`

Install the optional image stage after the core requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-image.txt
```

For a first Colab run, execute `examples/run_english_to_image_colab.py`. It loads the
quantized English parser once, keeps grounded diffusion in low-VRAM CPU-offload mode,
prints separate parser/layout/image timings, saves `english_to_image_result.png`, and
displays the image in the notebook. The first run downloads the GLIGEN checkpoint and is
therefore much slower than subsequent runs.

The from-scratch inventory currently recognizes the 17 SPRING types, common aliases,
the trained color/material properties, and explicit counts from one through four. GLIGEN is a
research checkpoint; keep its safety checker enabled and review its model license and
limitations before deployment.

## What's implemented

| File | Spec section | What it does |
|---|---|---|
| `mscl/ast.py` | §3–§6 | Internal AST: objects, normalized atoms, 28 named relations, boolean, typing, `CHOICE` w/ provenance |
| `mscl/relations.py` | §4–§5 | The 28 relations + desugaring to Layer-1 linear atoms (verified vs Table 1) |
| `mscl/profile.py` | §9 | MSCL-SPRING profile: 17 types, per-mille range, `default()` macro |
| `mscl/json_io.py` | §6 | LLM-facing **JSON AST** ⇄ internal AST, + the **JSON Schema** for constrained decoding |
| `mscl/render.py` | §9.1 | Lossless **MSCL ⇄ SPRING-string** rendering (Lark parser for the reverse) |
| `mscl/feasibility.py` | §7 | Exact `model_check`; **sound bounds-propagation feasibility** (exact SAT → SampleSearch) |
| `mscl/z3_backend.py` | §7 | Exact MSCL→Z3 compilation, incremental SAT checks, model extraction, unsat cores, independent verification |
| `mscl/samplesearch.py` | §7 | Complete preference-guided SampleSearch with infeasible-proposal zeroing, renormalization, backtracking, seeds, traces, and timings |
| `mscl/layout_eval.py` | — | Validity, diversity, runtime, typical-size, center-spread, area, and overlap distribution metrics |
| `mscl/layout_viz.py` | — | Dependency-free SVG rendering for one layout or a multi-sample grid |
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
result = SampleSearch().sample(spec, seed=7)
layout = result.layout                 # exact x,y,w,h boxes for every object
```

Install and test the core system:

```bash
python -m pip install -r requirements.txt
python tests/test_core.py
python tests/test_samplesearch.py
python -m examples.run_samplesearch
python -m examples.evaluate_samplesearch --count 40
```

The sampler reports solver, preference, final-verification, and total time separately through
`result.stats`. Its default `GeometricPreference` is an inspectable baseline; implement the
`PreferenceModel` interface to plug in `core.pt` or a modern learned layout prior.

`evaluate_samplesearch` compares uniform and geometric guides, prints a metrics table, and writes
`samplesearch_layouts.svg`. In Colab:

```python
from IPython.display import SVG, display
display(SVG(filename="samplesearch_layouts.svg"))
```

The geometric guide uses a single global Gaussian/uniform mixture. This matters: mixing uniform
exploration independently at every binary decision creates unintended heavy tails and implausibly
wide or tall furniture. The current values are a calibrated baseline, not a learned aesthetic
model.

New synthetic datasets use the Z3 backend as their final satisfiability filter. Existing JSONL
files are not silently rewritten; regenerate them when you are ready to replace those artifacts.

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

## What remains

- **A learned SampleSearch preference adapter.** The exact sampler is complete; its current guide
  is a transparent geometric baseline rather than SPRING's old RNN checkpoint.
- **Open-vocabulary detector** (E5) and **soft/weighted atoms** (E2) — reserved hooks.
- **Visual Element Generator integration** — consume the valid boxes to produce the final image.

## Honest status notes
- Desugaring of all 28 relations is verified against Table 1 in tests.
- `coffee table` in the appendix is mapped to the in-vocab `dining table` (no `coffee table`
  type in SPRING's 17); flagged as a profile warning, not silently dropped.
- `default()` per-mille bounds approximate SPRING's 256–512px-on-512 convention; calibrate them
  against the selected image generator.
