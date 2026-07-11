# Diagnostic review and remediation

## Bottom line

The 8/20 user-facing score was not a single model-capacity failure. It combined a weaker-than-
intended diagnostic prompt, contradictory exemplars, unlearnable/ambiguous gold data, predictable
small-model output errors, and one cleanup bug. Those issues are now corrected. A GPU rerun is still
required to measure the new real-model score; the offline gold-stub path is 400/400 and all 25 tests
pass.

## What each miss revealed

| Diagnostic cases | Failure pattern | Root cause | Implemented fix |
|---|---|---|---|
| 1, 9 | `well` became two hard relations, or a malformed offset `CHOICE` | Offset exemplar was present, but the repair diagnostic used only 3 shots and bypassed normalization | Unified the parse/repair paths; 10 coverage exemplars; deterministic `well/far/way` → two-option offset `CHOICE`; canonical span |
| 10, 14 | `near` / `next to` collapsed to one direction | The diagnostic's 3-shot prompt omitted the direction exemplar | Default prompt now includes every ambiguity kind; vague direction is deterministically preserved as a four-reading `CHOICE` |
| 12, 18, 19 | Duplicate references were ANDed together, omitted, or labeled as direction | Reference exemplar was omitted by the diagnostic; no guardrail checked duplicate candidates | Duplicate matching detected objects now produce a reference `CHOICE` over candidate ids |
| 2, 15 | Unsupported object handling was fragile | The model had to emit both unsupported-type and direction choices correctly | A null-typed new object now gets a deterministic nearest-type + SKIP `CHOICE`; vague direction remains separate |
| 4 | `shorter than a chair` became unary `shorter_value(...,0)` | Binary-object vs. unary-number distinction was underdemonstrated | Prompt rule added; atomic controlled templates are normalized deterministically |
| 11, 17 | Plain `above/right` became `cabove/cright` | A load-bearing exemplar contradicted the language definition by mapping plain words to complete relations | Corrected exemplar; explicit rule: only `completely/fully` selects the complete form; numeric offsets do not change it |
| 3 | Object ids and relation directions drifted in a long instruction | Long repeated noun phrases exceeded the few-shot binding demonstration; the old unambiguous generator also allowed duplicate types | Unambiguous data now uses unique types; added a multi-object/id-binding exemplar; expanded training data to 2,000 pairs |
| 16 | `cright` emitted with unary arity | Gold used a hidden random threshold (518) absent from the English; schema cannot express relation-specific arity | “Half” now means an explicit 500 threshold; deterministic unary normalization; arity remains validated |
| Several | Extra `default()` on existing objects, missing new-object boilerplate, changed object table | The model was allowed to regenerate facts already supplied as input | Input object table is authoritative; boilerplate is rebuilt exactly from it |
| Cleanup path | A one-item `and` could acquire a dictionary instead of an args list | Bug in `dedupe_relations()` | The whole `and` now collapses to its single child, preserving schema validity |

## Data and evaluation corrections

- `examples/train.jsonl`: 2,000 examples, 1,076 ambiguous.
- `examples/test.jsonl`: 400 held-out examples, 239 ambiguous.
- No duplicate object types remain in the *unambiguous* stream.
- No random, linguistically hidden image-part thresholds remain.
- `choice_structural_match` now requires the full direction/reference reading set instead of
  accepting one overlapping option. Vague offset magnitude remains intentionally ungraded.
- Gold-stub user-facing correctness: **400/400**.
- Dialogue policy on the corrected held-out set: **211/239 (88%) at 1.10 questions**, vs.
  ASK_NONE **136/239 (57%)** and ASK_ALL **239/239 (100%) at 1.20 questions**.

## Recommended GPU decision sequence

1. Rerun `run_diagnostic(backend, test, n=20)` with the updated code. Do not compare a new score
   against the old 8/20 without noting that the benchmark labels were corrected.
2. If the corrected unambiguous score is below 80%, inspect long multi-object binding errors first.
3. If final-system `CHOICE` emission is below 80%, distinguish raw-model emission from deterministic
   recovery before deciding to fine-tune.
4. Run the bundled QLoRA training only after the new few-shot result. The 2,000-pair set is now large
   enough for a meaningful first adapter run.
5. Report metrics by category and on paraphrased or human-written English. Template-only accuracy is
   a unit benchmark, not evidence of general natural-language performance.

## Commands

```bash
python tests/test_core.py

# GPU environment
python -c "from examples.eval_subset import run_diagnostic; print('import ok')"
python examples/finetune_lora.py   # only if the corrected few-shot gate still fails
```
