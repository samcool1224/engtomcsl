# CLAUDE.md — Project Bible: English→MSCL (a controllable front-end for SPRING)

> Read this first. It gives you (a fresh model or collaborator) everything needed to work on
> this project productively without re-deriving context. It is **forward-looking**: it describes
> what exists, why, and where it's going — not the debugging history of how it got here.
> Assumes you're comfortable with ML/LLMs but new to *this* project. For line-level internals,
> read `PROJECT_GUIDE.md` and the code; this file is the map.

---

## 0. TL;DR (30 seconds)

We are building the **natural-language front door to SPRING**, a system from the paper
*"Integrating Symbolic Reasoning into Neural Generative Models for Design Generation"* (Jacobson
& Xue). SPRING generates interior-design images that are **provably guaranteed** to satisfy
spatial rules — but only if you write those rules in formal logic. Our project lets a user type
plain English instead, and — the novel part — **handles ambiguity explicitly**: when English is
underspecified, the system either resolves it automatically or asks a targeted clarifying
question, *only when the answer would change the result*.

The core artifact is **MSCL** (Metric Spatial Constraint Logic): a clean, typed, metric spatial
logic that (1) generalizes SPRING's constraint language, (2) can carry ambiguity as first-class
`CHOICE` nodes, and (3) maps losslessly back to SPRING. A local LLM parses English → MSCL; a
resolution policy clears the `CHOICE` nodes; the result is fed to SPRING's machinery.

**Status:** the logic layer, ambiguity machinery, data generator, model interface, and evaluation
are built and tested (25/25). The English→MSCL model runs on a quantized local LLM (few-shot).
Remaining: push parser accuracy, reimplement SampleSearch (the exact solver), and hook up image
generation.

---

## 1. Research framing (why this matters, how to pitch it)

### 1.1 The gap
- **Neural generators** (Stable Diffusion, etc.) make appealing images but *cannot be trusted to
  obey precise instructions* — ask for "microwave right of the oven" and you often get the wrong
  count, color, or placement.
- **Symbolic/constraint methods** obey rules exactly but can't perceive images or judge aesthetics.
- **SPRING** fused them: a neural net proposes object placements; a symbolic sampler
  (SampleSearch) filters out any placement that violates the user's constraints, *guaranteeing*
  satisfaction. But SPRING's interface is propositional logic — unusable by normal people.

### 1.2 Our contribution
A **general English→(metric spatial logic)** system, with SPRING as the flagship application:
1. **MSCL**, a metric + typed extension of classical qualitative spatial calculi (Rectangle
   Algebra, Cardinal Direction Calculus, size calculi). The skeleton is borrowed and cited; the
   metric offsets, absolute-value constraints, typing, and the `CHOICE` construct are new as a
   *parsing target*.
2. **The `CHOICE` construct** — ambiguity as first-class data, distinct from logical `or`. `or`
   means "either is fine"; `CHOICE` means "the user meant one thing and we're unsure which."
3. **The resolution / dialogue policy** — the headline algorithmic contribution: clear ambiguity
   by asking the user a clarifying question *only when it changes the outcome*, otherwise resolve
   automatically. Evaluated against ASK_ALL / ASK_NONE / ASK_RANDOM baselines.

### 1.3 Honest novelty boundaries (state these; don't overclaim)
- Spatial logic, English→logic parsing, and qualitative spatial reasoning **already exist** and
  must be cited. Existing English→spatial-logic work targets *qualitative* relations only.
- Defensible new surface: the **metric+typed target**, the **`CHOICE` construct**, and the
  **ask-only-when-it-matters resolution policy** with its evaluation.
- **Open-vocabulary generation** (placing a brand-new object type like "lamp") is *not* solved —
  the preference model has no learned prior for unseen types. We handle it by asking or skipping.
  Open-vocabulary *reference* (constraining against an unusual detected object) is a planned,
  cheap perception upgrade.

### 1.4 Two goals, sometimes in tension
- **ISEF** (the near-term target): rewards a *working, impactful, demoable system* and a human
  study. Favors polish, a live demo, and a clear "why this matters" story.
- **A publishable paper** (NeurIPS/AAAI-caliber): rewards novelty + rigor (the resolution-policy
  algorithm, ablations, the questions-vs-accuracy curve).
  If forced to choose where to spend effort: ISEF wants the demo + human study polished; the
  paper wants the algorithm + evaluation rigorous.

### 1.5 Advisor guidance (paraphrased)
- **Prof. Yexiang Xue** (SPRING co-author, constraint-reasoning + ML) encouraged finding a
  strong, self-chosen extension to SPRING; the neuro-symbolic/constraint-reasoning angle is his
  wheelhouse. Pitches should lead with one crisp framing, not a feature list.
- **Max Jacobson** (SPRING first author) advised: (a) the original SPRING code is old/messy —
  treat it as reference and **reimplement the core (especially SampleSearch) cleanly**; (b) frame
  the project as **language-to-logic in general**, with SPRING as a motivating application/testbed
  — that gives a bigger related-work base and real baselines, and is a stronger story than a
  SPRING-only tool.

---

## 2. The overall pipeline

```
  ENGLISH  ("put a blue microwave right of the oven")
     │
     ▼   English→MSCL model  (local LLM, few-shot, JSON-schema-constrained decoding)   [GPU]
  MSCL JSON  (may contain CHOICE nodes for ambiguity)
     │
     ▼   resolution / dialogue policy  (asks the user only when it changes the outcome)
  resolved MSCL  (CHOICE-free)
     │
     ▼   render to SPRING string  (lossless)
  SPRING logic
     │
     ▼   SampleSearch (exact solver) + Visual Element Generator   [NOT YET REBUILT]
  IMAGE  (guaranteed to satisfy the constraints)
```

Everything from "MSCL JSON" down to "SPRING logic" is **built and tested**. The top box (the LLM)
**runs** (few-shot, quantized). The bottom box (SampleSearch + image gen) is the **main remaining
build**.

---

## 3. MSCL in one page

- **Objects are boxes**: four per-mille integers `x` (left), `y` (top, grows *downward*), `w`, `h`.
  Derived edges `R=x+w`, `B=y+h`.
- **One atomic form**: every spatial relation is a linear (in)equality over box terms, e.g.
  "microwave completely right of oven" ≡ `x_microwave ≥ R_oven`. This is why one small solver/
  checker handles the whole language.
- **28 named relations** (friendly aliases over atoms), in 4 families + `_value` (absolute) forms:
  directional partial (`above/below/left/right`), directional complete (`cabove/cbelow/cleft/
  cright`), size (`wider/narrower/taller/shorter`), alignment (`xeq/yeq/weq/heq`).
- **Types & properties**: `type(o,"microwave")` (must be one of the 17 SPRING types),
  `property(o,"blue")`, and `default(o)` (a macro for sensible size/in-frame bounds).
- **Boolean**: `and`, `or`, `not`.
- **`CHOICE`** (the novel node): kinds `direction`, `offset`, `reference`, `scope`,
  `unsupported_type`; each with weighted option-formulas and provenance (`span`, `kind`).
- **Profiles**: MSCL-SPRING is one profile (17 types, `[0,1000]` range). A second profile later
  would prove generality without touching the parser.

The **17 legal types**: chair, couch, potted plant, bed, mirror, dining table, window, desk,
toilet, door, TV, microwave, oven, toaster, sink, refrigerator, blender.

---

## 4. Repository map (what each file is)

```
mscl/mscl/
  ast.py          Internal data structures (objects, atoms, relations, boolean, CHOICE).
  relations.py    The 28 relations + desugaring to normalized linear atoms (verified vs paper Table 1).
  profile.py      MSCL-SPRING profile: 17 types, per-mille range, default() macro, nearest_types().
  json_io.py      LLM-facing JSON <-> AST; the JSON Schema for constrained decoding; dedupe_relations().
  render.py       MSCL <-> SPRING-string (lossless; Lark grammar for the reverse).
  feasibility.py  model_check (exact) + feasible() (SOUND bounds-propagation; NOT a full solver).
  validate.py     Well-formedness + profile conformance; assert_resolved (no CHOICE in final spec).
  dialogue.py     THE resolution policy (prune -> silent-collapse -> rank/ask under budget -> close)
                  + ASK_ALL / ASK_NONE baselines.
  datagen.py      Synthetic (english, objects, gold-formula) generator; unambiguous + all CHOICE kinds.
  parser.py       Prompt harness + backends. SYSTEM prompt, _exemplars (coverage-guaranteed few-shot),
                  StubBackend (no GPU), LocalBackend (quantized local LLM + Outlines), parse().
  evaluate.py     exact_match, semantic_equiv (unambiguous), choice_structural_match + is_correct
                  (CHOICE-aware), parse_accuracy, policy_eval.
  local_debug.py  time_constrained_call, parse_with_repair (retry-on-validation-error).
  mscl_spring.schema.json / .gbnf   Machine-readable grammars for constrained decoding.
examples/
  seed_pairs.json     Few-shot exemplars (paper appendix + hand-built ambiguous ones). COVERAGE MATTERS.
  train.jsonl/test.jsonl   Generated data.
  run_pipeline.py          Full offline pipeline on the stub (no GPU).
  run_pipeline_local.py    Real-model pipeline (GPU): loads once, smoke-tests, subset, then full.
  eval_subset.py           Diagnostic: accuracy + CHOICE-emission tally + miss dumps.
  finetune_lora.py         QLoRA script (only if few-shot underperforms; not yet needed).
  demo.py                  Walks the appendix examples end to end.
tests/test_core.py    25 tests (logic, round-trips, policy, prompt coverage, parser normalization).
PROJECT_GUIDE.md      The long, from-scratch explanation of everything. Read after this file.
```

---

## 5. Current state (what works, what's measured)

**Runs and passes with no GPU (reproducible, fixed seeds):**
- 25/25 unit tests.
- Data generation (unambiguous + all 4 ambiguity kinds; satisfiability-checked).
- Full pipeline on the stub backend; the dialogue policy vs baselines.
- Dialogue policy: reaches most of ASK_ALL's accuracy at fewer questions than ASK_ALL, well above
  ASK_NONE. (Exact numbers depend on the simulated-user model; see caveat 8.4.)

**Runs on GPU (few-shot, quantized 4-bit Qwen2.5-7B-Instruct):**
- Unambiguous parsing: solid (majority correct).
- Ambiguous parsing / `CHOICE` emission: the hard part. Improving via prompt + exemplar coverage.
  Grade ambiguous output with `is_correct` / `choice_structural_match`, NOT exact match — priors
  and option ordering are not user-visible and must not be graded.

**Key design choices that affect how you should work:**
- The model uses **JSON-schema-constrained decoding** (Outlines ≥1.3, `from_transformers` +
  `Generator(model, JsonSchema(...))`, built ONCE). The `type` field is **enum-constrained to the
  17 types**, so illegal types are impossible to generate.
- The model is loaded **4-bit** (`LocalBackend(..., quantize="4bit")`) to fit a T4/Colab.
- **Few-shot exemplar coverage is load-bearing**: small models imitate examples far more than
  prose rules. `_exemplars` guarantees every CHOICE kind + a `_value`-relation example appears.
  If you add failure modes, add a demonstrating exemplar, not just a prose rule.

---

## 6. How to run

**Offline (any machine):**
```bash
cd mscl
python tests/test_core.py          # 25/25
python examples/run_pipeline.py    # data + stub pipeline + policy eval
```

**GPU (real model):**
```bash
pip install outlines transformers torch accelerate lark bitsandbytes
# in a notebook / script:
from mscl.parser import LocalBackend
backend = LocalBackend("Qwen/Qwen2.5-7B-Instruct", quantize="4bit")
from examples.eval_subset import run_diagnostic
run_diagnostic(backend, test, n=15)   # accuracy + CHOICE-emission tally + miss dumps
```
Notes: first generation is slow (one-time warmup); ~40–75s/call at 4-bit on a T4. Use
`parse_with_repair` (in `local_debug.py`) to retry once on a validation error.

---

## 7. Roadmap (priority order)

1. **Push English→MSCL accuracy** (current focus). Lever order: (a) few-shot exemplar coverage +
   prompt rules for each observed failure pattern; (b) grade with `is_correct`, not exact match;
   (c) only if `CHOICE` emission stays low after (a), run the QLoRA fine-tune (`finetune_lora.py`,
   data already formatted). Generate more training data first (`--n_extra 1500`).
2. **Reimplement SampleSearch** — the exact, complete, preference-weighted solver. `feasible()` is
   only a sound pruning oracle; SampleSearch is what actually produces guaranteed layouts. Max's
   advice: understand SampleSearch deeply and the rest follows.
3. **Hook up image generation** (Visual Element Generator) — modernize the backbone; this is the
   demo that wins an ISEF booth.
4. **Scale evaluation** — thousands of samples; accuracy by ambiguity kind; the full
   questions-vs-accuracy curve as the budget sweeps 0→5 (the key paper figure).
5. **Human study** — real users vs. the policy and vs. a plain text-to-image tool; replaces the
   simulated user with ground truth.
6. **Second MSCL profile** (optional) — substantiate the "general, not SPRING-only" claim.

---

## 8. Known limitations to respect (forward-looking; don't reintroduce)

- **`feasibility.feasible()` is sound-for-pruning, not a complete solver.** Never describe it as
  the solver; exact satisfiability is SampleSearch's job (roadmap item 2).
- **Grade `CHOICE` output structurally** (`is_correct`), not by exact match — priors/option-order
  are not user-visible correctness.
- **`default()` size bounds are furniture-realistic, not the paper's literal 256–512px/512** (which
  were too large for objects to satisfy directional constraints). Reconcile with the real VEG later.
- **The policy evaluation's simulated user deviates from the prior a fixed fraction of the time** —
  that assumption is what makes asking valuable; replace it with real humans in the study, and
  state it in any writeup.
- **Open-vocabulary generation is not solved** — unseen types are handled by asking/skipping.

---

## 9. Working conventions (how to collaborate on this)

- **Add an exemplar, not just a rule**, when fixing a model failure mode. Prose rules alone move a
  few-shot model much less than a demonstrated example.
- **Every relation desugars to one linear atom** — keep that invariant; it's what keeps the solver
  and checker simple.
- **Never let a `CHOICE` reach SPRING** — `assert_resolved` enforces this.
- **Keep MSCL⇄SPRING lossless** — anything the parser emits (once resolved) must run SPRING unchanged.
- **When you touch scoring, sanity-check against the stub backend first** (should be ~100%): it
  isolates "is the harness correct?" from "is the model good?"
- Tests must stay green (`python tests/test_core.py`). Add a test when you add a behavior.
```
