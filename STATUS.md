# English→MSCL system — status & next steps

## Where the project stands

The **entire English→MSCL system is built and runs end-to-end**, with the single exception
of the real model call (which needs your GPU). Everything around it — data, prompting,
parsing interface, ambiguity resolution, and evaluation — is implemented and tested.

```
mscl/                      THE LANGUAGE LAYER (done earlier)
  ast, relations, profile, json_io, render, feasibility, validate, dialogue
  mscl_spring.schema.json  ← JSON schema for constrained decoding
  mscl_spring.gbnf         ← GBNF grammar for llama.cpp

mscl/datagen.py            Piece 4 — synthetic data generator (templates + paraphrase hook,
                           ambiguous + unambiguous, satisfiability-guarded)
mscl/parser.py             Pieces 1+2 — prompt harness + parse(); StubBackend (no GPU) and
                           LocalBackend (Outlines + HF, your GPU)
mscl/evaluate.py           Piece 3 — parse accuracy (exact + semantic), round-trip,
                           dialogue-policy eval vs ASK_ALL/ASK_NONE
examples/run_pipeline.py   full harness: generate → parse → evaluate (runs offline)
examples/train.jsonl       120 generated pairs (65 ambiguous)
examples/test.jsonl        60 generated pairs (35 ambiguous)
tests/test_core.py         16/16 passing
```

## What runs in this environment (verified)

- Synthetic dataset generation (unambiguous + all four ambiguity kinds).
- Full pipeline on a **StubBackend** (gold-driven) → 100% parse accuracy = harness is correct.
- Sanity floor with a trivial guesser → 0% exact / 15% semantic = metric discriminates.
- **Dialogue-policy evaluation vs baselines** (the headline result):
  policy ≈ 86% correct at ~1.1 questions, between ASK_NONE (60%, 0 Q) and ASK_ALL (100%, 1.2 Q).

## The ONE thing you run on your GPU

Replace the stub with the local model — nothing else changes:

```python
from mscl.parser import LocalBackend
from mscl import evaluate
import json

backend = LocalBackend("Qwen/Qwen2.5-7B-Instruct")   # grammar-constrained JSON decoding
test = [json.loads(l) for l in open("examples/test.jsonl")]
print(evaluate.parse_accuracy(test, backend).summary())
```

Setup on your machine: `pip install outlines transformers torch accelerate`. The grammar
constraint (`mscl_spring.schema.json`) guarantees syntactically valid output, so the only
question the eval answers is *semantic* accuracy — does the model pick the right relations,
bind the right objects, and (critically) **emit CHOICE when the English is ambiguous**.

## Decision gate after the first GPU run

- **If few-shot semantic accuracy is good** (say ≥80% on unambiguous, and it emits CHOICE on
  most ambiguous cases): you're essentially done with the parser — move to SampleSearch +
  the image demo.
- **If it's poor** (likely on the CHOICE cases — small models under-emit ambiguity): turn on
  fine-tuning. `train.jsonl` is already in the exact format. The LoRA script is the next
  artifact I can write; you run it. Template English can be paraphrased first with the same
  local model (the `paraphrase` hook in `datagen.generate_dataset`) to de-robotify the data.

## Remaining work to a finished, publishable result

1. **GPU run #1** (you): few-shot `parse_accuracy` + `policy_eval` with the real model. ← next
2. **Paraphrase pass** (you, optional): run the local model over template English to improve
   naturalness; regenerate train/test.
3. **Fine-tune** (conditional): LoRA on `train.jsonl` if few-shot under-performs. Script TBD.
4. **Scale the eval**: bump dataset to a few thousand; report parse accuracy by ambiguity
   kind, and the full questions-vs-accuracy curve (sweep `budget` 0→5) — that curve is the
   paper figure.
5. **Human study** (ISEF gold): real users vs. the policy on a handful of scenes — do people
   reach their intended layout in fewer interactions than ASK_ALL / a plain text-to-image tool.
6. **SampleSearch + VEG** (separate track): turn resolved specs into actual guaranteed images
   — the visual demo that wins the booth.

## Honest caveats baked into the code
- `feasibility.feasible()` is a *sound pruning* oracle, not a complete SAT solver; exact
  satisfiability is delegated to SampleSearch (next track). The dialogue policy only needs
  soundness, so this is fine — but don't report it as the solver.
- `default()` per-mille bounds were retuned to furniture-realistic values (the SPRING-paper
  512px convention was too large to satisfy directional constraints); reconcile with the
  reimplemented VEG later.
- The simulated user in `policy_eval` deviates from the prior 35% of the time by design — that
  is what makes asking valuable and ASK_NONE imperfect. State this assumption in the paper and
  replace it with real human answers in the study.
```
