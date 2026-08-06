# Evaluation 1: English to MSCL and `CHOICE`

This evaluation intentionally stops before `CHOICE` resolution, SampleSearch, and image
generation. Its research question is:

> Given an English instruction and the authoritative scene-object table, does the parser produce
> the correct MSCL structure and preserve ambiguity as the correct `CHOICE` option set?

## Pilot data

`eval1_pilot.jsonl` contains 44 deterministic development examples:

| Category | Count |
|---|---:|
| Unambiguous controls | 12 |
| Direction `CHOICE` | 8 |
| Offset `CHOICE` | 6 |
| Reference `CHOICE` | 6 |
| Unsupported-type `CHOICE` | 6 |
| Scope `CHOICE` | 6 |

Every ambiguous record stores the full plausible option set and one hidden intended option. The
parser is not expected to identify that hidden intention without clarification; it is expected to
keep that option available. Thus, intended-option coverage measures whether later resolution still
has a chance to succeed.

This is a **development pilot**, not a paper test set. It validates the scorer, exposes current
failure modes, and fixes the record format. Do not tune the parser and then report this set as
held-out evidence.

## Metrics

- Valid MSCL rate
- Ambiguity-detection precision, recall, and F1 at the instruction level
- `CHOICE`-emission recall
- `CHOICE`-node detection precision, recall, and F1 by kind
- Option-set precision, recall, and F1
- Exact `CHOICE`-construction rate
- Hidden intended-option coverage
- Non-`CHOICE` MSCL skeleton precision, recall, and F1
- Structural MSCL accuracy
- Unambiguous structural accuracy
- False-`CHOICE` rate and potential false questions on unambiguous prompts

The scorer ignores `CHOICE` prior values, option ordering, superficial span punctuation, and the
exact positive distance chosen for vague offsets. It does not ignore relation direction,
reference IDs, scope structure, supported object types, or missing/extra alternatives.

## Run

```bash
python benchmarks/build_eval1_pilot.py
python -m examples.evaluate_english_to_mscl --backend gold
python -m examples.evaluate_english_to_mscl --backend stub
python -m examples.evaluate_english_to_mscl --backend local --quantize 4bit
```

The local run saves `predictions.jsonl`, `metrics.json`, and `per_example.jsonl` under
`results/eval1/`. Predictions can be re-scored without loading the model:

```bash
python -m examples.evaluate_english_to_mscl \
  --predictions results/eval1/predictions.jsonl
```

## Data expansion after the pilot

1. Add roughly 100 independently human-authored prompts without showing writers the templates.
2. Have two annotators label ambiguity, plausible interpretations, and the intended option; use a
   third adjudicator for disagreement.
3. Expand controlled coverage to at least 500 examples, balanced by ambiguity kind and with
   multi-`CHOICE` compositions.
4. Split by template/paraphrase family, not by random rows.
5. Freeze the final test set before changing prompts, exemplars, post-processing, or model weights.

