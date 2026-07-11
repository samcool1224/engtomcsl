# The English→MSCL Project — Definitive Guide

*A single document explaining the whole system: the idea, the theory, every piece of code,
what has been run, what remains, and every caveat worth knowing. Written to be readable by
someone who has only skimmed the SPRING paper. It starts gently and gets progressively more
technical — nothing technical is used before it is explained.*

---

# PART A — The Big Picture (start here)

## A.1 What problem are we solving?

SPRING (the paper this project extends) does one impressive thing: it generates interior-design
images that are **guaranteed** to obey spatial rules the user gives. If you tell it "put the
microwave to the right of the oven," the microwave *will* be to the right of the oven — not
"usually," but always. It achieves this by combining a neural network (which knows what things
should look like) with a symbolic reasoner (which enforces the rules exactly).

But SPRING has a wall between it and ordinary people: **you have to speak its language, and
its language is formal logic.** To ask for that microwave, you must write:

```
type(m,"microwave") ∧ property(m,"blue") ∧ cright(m, oven)
```

Nobody outside a research lab will do that. This project builds the missing front door:
**a system that takes plain English and turns it into SPRING's formal logic** — and, crucially,
handles the fact that English is *ambiguous* in ways formal logic is not.

## A.2 The one hard idea: ambiguity

"Put a lamp by the couch" is a perfectly normal sentence, but it hides three problems a logic
engine can't ignore:

1. **"lamp"** isn't one of the 17 object types SPRING knows. What do we do?
2. **"by"** — by *which side*? Left? Right? Above? The logic needs a specific direction.
3. **"by"** — *how far*? Touching? A little gap? A big gap?

A naive system silently guesses and often guesses wrong. A rigid system rejects the sentence.
Our system does neither: it **represents the ambiguity explicitly**, then decides — using a
smart policy — whether to just resolve it automatically or to ask you a quick clarifying
question. And it only asks when the answer would actually change the result. That "ask only
when it matters" behavior is the **core research contribution** of this part of the project.

## A.3 The three-layer mental model

Picture the whole system as three stacked layers:

```
   ENGLISH  ("put a blue microwave right of the oven")
      │
      ▼   [ the English→MSCL model — a local LLM ]   ← the part that needs a GPU
   MSCL    (a clean, machine-checkable logic; may contain "CHOICE" nodes for ambiguity)
      │
      ▼   [ the resolution policy — asks you questions only when needed ]
   SPRING logic  (the exact string SPRING already understands)
      │
      ▼   [ SampleSearch + image generator — SPRING's existing machinery ]
   IMAGE   (guaranteed to obey the rules)
```

**MSCL** (Metric Spatial Constraint Logic) is our invention: a middle language that is more
general than SPRING's, has a clean mathematical footing, and — unlike SPRING's language — can
*carry ambiguity*. SPRING's language is then just one "profile" (one configuration) of MSCL.

Everything in this repository builds the **middle two layers**: the target logic, the ambiguity
machinery, the data to train the model, the model interface, and the evaluation. The top layer
(the actual LLM generation) is the only thing that needs a GPU and is the only thing not yet run.

## A.4 Why invent MSCL instead of using something off the shelf?

There is an existing academic field of "spatial logics" (called qualitative spatial reasoning:
Rectangle Algebra, Cardinal Direction Calculus, RCC-8, etc.). People have even built systems
that translate English into those. So why not use them?

Because those logics are **purely qualitative** — they can say "A is left of B" but *not* "A is
left of B by at least 300 units," and they can't carry object types, colors, or absolute
positions ("in the right half of the image"). SPRING needs all of that. MSCL is deliberately
positioned as a **metric (measurement-carrying), typed extension** of those known logics — so
it inherits their academic grounding and their baselines, while adding exactly the expressive
power SPRING requires, *plus* the ambiguity construct nobody else has. This is the honest
novelty story: skeleton borrowed and cited, the metric/typed/ambiguity layer is new.

---

# PART B — The Logic (MSCL), Explained From Scratch

## B.1 How positions are described

Every object is a **box** (a rectangle). A box is fully described by four numbers:

- `x` — how far the **left edge** is from the left of the image
- `y` — how far the **top edge** is from the top of the image (note: **y grows downward**, as in
  image coordinates — so "higher up" means a *smaller* y)
- `w` — width
- `h` — height

All four are in **per-mille**: thousandths of the image's width/height. So `x = 500` means the
left edge is halfway across the image. This makes everything resolution-independent.

Two more useful quantities are derived from those four:

- `R` (right edge) `= x + w`
- `B` (bottom edge) `= y + h`

That's the entire geometry. Six terms per object — `x, y, w, h, R, B` — and everything else is
built from them.

## B.2 The single atomic building block

Here is the key simplification that makes the whole system clean. **Every spatial rule in MSCL,
no matter how it's named, is really just a linear inequality over those box terms.** For example:

- "microwave completely right of oven" really means: `x_microwave ≥ R_oven`
  (the microwave's left edge is at or past the oven's right edge)
- "chair above table by at least 100" really means: `y_chair ≤ y_table − 100`

So under the hood there is *one* kind of atom:

```
(sum of coefficient × box-term)   OP   constant
```

where `OP` is one of `< ≤ = ≥ >`. That's it. Every named relation is just a friendly alias for
one of these. This is why a single small piece of code can check, prune, and reason about the
*entire* language — there is really only one thing to handle.

## B.3 The 28 named relations (the friendly aliases)

MSCL gives 28 human-readable names, in four families. You do **not** need to memorize these; the
point is that each one desugars to exactly one atom from B.2.

**Family 1 — directional, partial overlap** (things can still overlap a bit):
`above, below, left, right` — each takes an optional distance `c`.
Example: `right(a, b, c)` → `x_a ≥ x_b + c`.

**Family 2 — directional, complete/no-overlap** (the whole object is clear of the other):
`cabove, cbelow, cleft, cright`.
Example: `cright(a, b, c)` → `x_a ≥ R_b + c` (a's left edge past b's right edge).

**Family 3 — size comparisons:** `wider, narrower, taller, shorter`.
Example: `wider(a, b, c)` → `w_a ≥ w_b + c`.

**Family 4 — alignment/equality:** `xeq, yeq, weq, heq` (share an x, y, width, or height).

**Plus the `_value` variants** of most of the above (12 of them): instead of comparing to another
object, compare to an absolute number. Example: `right_value(a, 500)` → `x_a > 500` ("a is in the
right half"). These have **no equivalent in the older qualitative logics** — they're part of what
makes MSCL more expressive (and part of why off-the-shelf English→logic tools can't target SPRING).

`16 binary + 12 value = 28 relations.` All verified against SPRING's Table 1 in the test suite.

## B.4 Building whole specifications: types, properties, boolean logic

On top of the spatial relations, MSCL adds:

- `type(o, "microwave")` — object o is a microwave (must be one of SPRING's 17 types)
- `property(o, "blue")` — object o is blue
- `default(o)` — a shorthand macro that expands to sensible size/in-frame bounds for o
- the connectives `∧` (and), `∨` (or), `¬` (not) to combine everything

So a full spec is a **tree**: connectives are branches, relations/types are leaves. This mirrors
exactly how SPRING's own paper (Appendix A) represents specs internally, which is what lets us map
back to SPRING losslessly.

## B.5 The star of the show: the `CHOICE` node

This is MSCL's genuinely new construct. A `CHOICE` node represents **a point where the English was
ambiguous**. It looks like this (in plain terms):

> CHOICE, kind = "direction", from the phrase "by the couch":
> &nbsp;&nbsp;• option A (prior 0.4): lamp is `cleft` of couch
> &nbsp;&nbsp;• option B (prior 0.4): lamp is `cright` of couch
> &nbsp;&nbsp;• option C (prior 0.2): lamp is `cabove` couch

Each option is a possible reading, with a **prior** (how likely the parser thinks that reading is).
The node also records its **provenance**: which words caused it (`span`) and what *kind* of
ambiguity it is (`direction`, `offset`, `reference`, `scope`, or `unsupported_type`).

**Why not just use "or"?** Because `∨` (or) means "the user genuinely wants *either* — both are
fine." `CHOICE` means "the user meant *one specific thing* and we don't yet know which." Those are
different situations that call for different handling: an `∨` you leave alone; a `CHOICE` you try
to *resolve* (by reasoning, or by asking). Keeping them distinct is what makes the resolution
policy possible, and is itself a small conceptual contribution.

The five kinds of CHOICE:
- **direction** — "by/near the X" gave no specific side
- **offset** — a direction but no magnitude ("well to the left" → how far?)
- **reference** — "the oven" but there are two ovens; which one?
- **scope** — "a chair and a couch left of the TV" — does "left of the TV" bind both or just one?
- **unsupported_type** — an object word outside SPRING's 17 types ("lamp")

## B.6 Meaning (semantics), briefly

A **layout** assigns each object concrete numbers `(x, y, w, h)`. A formula is *satisfied* by a
layout if all its atoms come out true (with the usual and/or/not rules). A formula is
**satisfiable** if *some* layout satisfies it. Because every atom is a linear integer inequality,
checking satisfiability is a well-defined (and decidable) math problem — and it is exactly the
problem SPRING's SampleSearch solves when it looks for a valid arrangement.

---

# PART C — The System, Module by Module

The code lives in a Python package `mscl/`. Here is every module, what it does, and how it
connects. (File paths are under `mscl/mscl/` unless noted.)

## C.1 `ast.py` — the internal data structures
Defines the Python objects that represent everything above: `Obj` (an object with id/type/box),
`Atom` (the normalized linear inequality from B.2), `Relation` (a named relation from B.3),
`TypePred`/`PropertyPred`/`Default`, the boolean nodes `And`/`Or`/`Not`, and `Choice`/`Option`
(from B.5). A whole spec is a `Spec` = object list + formula tree. This is the "vocabulary" every
other module speaks in.

## C.2 `relations.py` — turning names into math
Contains the tables that map each of the 28 names to its atom (B.2). The function `desugar(rel)`
does the conversion. This is where the correctness against SPRING's Table 1 lives — e.g. it knows
`cright(a,b,c)` becomes `x_a − x_b − w_b ≥ c`. Every truth condition from the paper is encoded and
tested here.

## C.3 `profile.py` — the SPRING-specific settings
A "profile" is a configuration of MSCL for a specific domain. `MSCL-SPRING` fixes: the 17 legal
types, the per-mille range `[0,1000]`, and the `default()` macro's expansion (size/in-frame
bounds). It also has `nearest_types(word)`, a helper that suggests in-vocabulary types for an
unsupported word ("lamp" → "potted plant", "mirror"). **Caveat flagged here and revisited in
Part F:** the default size bounds had to be retuned away from SPRING's literal numbers.

## C.4 `json_io.py` — the format the model reads/writes + the schema
The English→MSCL model doesn't emit Python objects; it emits **JSON**. This module converts
between that JSON and the internal AST (`spec_from_json`, `spec_to_json`), and — importantly —
produces the **JSON Schema** (`json_schema()`) that constrains the model's output. Every node has
a `"node"` tag (`"rel"`, `"and"`, `"choice"`, …) so the schema can enforce validity.

## C.5 `render.py` — the bridge back to SPRING
`to_spring(formula)` prints an MSCL formula as the exact string SPRING expects. `from_spring(s)`
parses a SPRING string back into MSCL (using a small formal grammar via the `lark` library). This
round-trip is **lossless**: anything our system produces (once ambiguity is resolved) runs on the
existing SPRING pipeline unchanged.

## C.6 `feasibility.py` — the lightweight reasoner
Two services:
1. `model_check(formula, layout)` — is a *specific* layout valid? Exact and trivial.
2. `feasible(atoms, domains)` — *could* a conjunction of atoms be satisfiable, given each
   variable's allowed range? This uses **bounds propagation** (repeatedly tightening each
   variable's min/max until nothing changes or a variable's range becomes empty).

**Crucial honesty point (expanded in Part F):** `feasible` is **sound but not complete**. If it
says "infeasible," that is definitely true. If it says "maybe feasible," it might still be
infeasible in tangled cases. That's fine, because the resolution policy only needs the sound
direction (it uses feasibility to *prune impossible options*, and pruning a truly-impossible
option is always safe). Exact, complete satisfiability is deliberately left to SampleSearch.

## C.7 `validate.py` — catching malformed specs
Checks well-formedness (correct number of arguments per relation, `_value` relations actually have
a value, types are in vocabulary, CHOICE priors roughly sum to 1, etc.). `assert_resolved(spec)`
enforces that a spec is CHOICE-free before it's allowed to become a SPRING string — you can't send
an unresolved ambiguity to SPRING.

## C.8 `dialogue.py` — the resolution policy (the research core)
This is the most important module and gets its own deep-dive in Part D.

## C.9 `datagen.py` — making training/testing data
Because SPRING's paper only gives ~5 example (English, logic) pairs, we **generate our own** by
the thousands. This module samples random MSCL specs — both unambiguous and deliberately
ambiguous (all five CHOICE kinds) — verbalizes them to English with templates, and checks each is
satisfiable before keeping it. It has a `paraphrase` hook so you can later rewrite the robotic
template English with the local model. Deep-dive in Part E.

## C.10 `parser.py` — prompt + the actual model call
Builds the few-shot prompt (system instructions + coverage exemplars + the new input) and
defines the **backends**:
- `StubBackend` — no GPU; returns gold answers (to test the pipeline) or a trivial guess (to
  establish a floor). This is what let the entire system be validated without a GPU.
- `LocalBackend` — the real thing: loads a local model (e.g. Qwen2.5-7B-Instruct) and uses
  schema-constrained decoding so output is always valid JSON. Auto-detects the Outlines library
  version and applies the model's chat template.

## C.11 `evaluate.py` — measuring success
Three evaluations, deep-dived in Part D.3. In short: parse accuracy (did the model produce the
right logic?), round-trip consistency (self-supervised accuracy at scale), and the dialogue-policy
evaluation (does asking-only-when-needed beat the baselines?).

---

# PART D — The Algorithms In Depth

## D.1 Desugaring (names → atoms), worked example

Take `cright(m, oven, 100)` ("microwave completely right of oven, by ≥100").
- The rule from SPRING's Table 1: *m's left edge is at least 100 past oven's right edge.*
- In terms: `x_m ≥ R_oven + 100`.
- Expand `R_oven = x_oven + w_oven`: `x_m ≥ x_oven + w_oven + 100`.
- Move everything to one side (the normalized atom form): `x_m − x_oven − w_oven ≥ 100`.
- As coefficients: `{ +1·x_m, −1·x_oven, −1·w_oven }  ≥  100`.

Every one of the 28 relations reduces this way. The test `test_desugar_cright_matches_table1`
checks exactly this.

## D.2 Feasibility by bounds propagation, worked intuition

Suppose we require `left_value(a, 100)` (a's left edge `< 100`) **and** `right_value(a, 800)`
(a's left edge `> 800`). Each variable starts with range `[0, 1000]`.
- First constraint tightens `x_a` to `[0, 99]`.
- Second constraint tightens `x_a` to `[801, 1000]`.
- Those can't both hold — the range is empty → **infeasible.** Detected instantly.

Now the incompleteness: with several variables tangled across many atoms, bounds propagation can
sometimes *fail to notice* an emptiness that a full solver would. It never wrongly reports empty,
so using it only to *prune impossible options* is safe. This is the deliberate soundness/complete-
ness trade that keeps the policy fast without a full solver.

## D.3 The resolution/dialogue policy — the heart of the contribution

**Goal:** given a spec containing CHOICE nodes, produce a single CHOICE-free spec, asking the user
as *few* clarifying questions as possible — ideally only when the answer would change the outcome.

The policy runs in four stages:

**Stage 1 — PRUNE.** For each CHOICE, drop any option that is infeasible given the other (hard)
constraints, using `feasible`. If a CHOICE collapses to a single surviving option, it's resolved
**silently** (no question). *Example:* if the couch is at the far-left edge, the "put the lamp to
the *left* of the couch" option is infeasible (no room), so it's dropped automatically.

**Stage 2 — SILENT DEFAULTS.** An `offset` CHOICE ("how far?") with no emphasis word collapses
silently to "no gap" (`c=0`). We don't pester the user about exact distances unless they said
something like "*well* to the left," which sets an emphasis flag.

**Stage 3 — RANK & ASK.** Whatever CHOICEs remain genuinely change the outcome. Ask about them,
in a deliberate order: **structural first** (`unsupported_type`, `reference` — these decide *what*
and *which object*), **then geometric** (`scope`, `direction`, `offset`). Why this order? Because
resolving *what object* something is can make some geometric options infeasible — so answering the
structural question first lets Stage-1-style pruning clean up the geometry for free. After each
answer, we **re-prune**. Asking continues until a **question budget** (default 3) is spent.

**Stage 4 — CLOSE.** Any CHOICE still open when the budget runs out is resolved by its highest
prior and flagged **low-confidence**, so nothing is left dangling.

Finally, the tree is rewritten to physically remove the resolved CHOICE nodes, leaving a clean
CHOICE-free spec ready for SPRING.

**Baselines it's measured against:**
- `ASK_NONE` — never ask, always take the most likely reading. (0 questions, but wrong whenever
  the user's intent isn't the most likely reading.)
- `ASK_ALL` — confirm every CHOICE. (Always correct, but maximally annoying.)
- `ASK_RANDOM` — ask about a random subset (a sanity middle-point).

The whole point is to land near ASK_ALL's accuracy at far fewer questions than ASK_ALL.

## D.4 How we measure "correct" without real users (and the assumption inside it)

To evaluate at scale we simulate a user. But here's a subtlety we got wrong once and then fixed:
if the simulated user's "true intent" were *always* the highest-prior option, then `ASK_NONE`
(which always takes the highest prior) would be trivially perfect and asking could never help —
a meaningless test. So the simulated user is modeled to **deviate from the prior about 35% of the
time**. That 35% is exactly the situation where a clarifying question earns its keep. **This is an
assumption, and in the real ISEF human study it must be replaced with actual human answers.** It's
called out again in Part F.

---

# PART E — The Data Generator In Depth

## E.1 Why synthetic data
The model that turns English into MSCL needs examples to learn from (for few-shot prompting now,
possibly fine-tuning later). SPRING's appendix gives about five. We need thousands, spanning every
relation and every ambiguity kind. So we generate them.

## E.2 How a sample is made
1. **Sample objects** — pick 1–3 new objects (random types, sometimes a color) and 0–2 existing
   objects (with random boxes, as if detected by the perception module).
2. **Sample constraints** — add a handful of random relations among them.
3. **Verbalize** — turn each relation into an English fragment via a template
   (`cright` → "completely to the right of"), and stitch into a sentence.
4. **Satisfiability-check** — throw the spec at `feasible`; discard if impossible. This guarantees
   the dataset never contains unsatisfiable unambiguous specs.

## E.3 Manufacturing ambiguity on purpose
Four generators intentionally produce sentences whose gold logic contains a CHOICE:
- **direction:** "put X *by* the Y" → a direction CHOICE over the four sides.
- **offset:** "put X *well* to the left of Y" → an offset CHOICE (with emphasis set).
- **reference:** two existing objects of the same type, then "the Y" → a reference CHOICE.
- **unsupported_type:** an out-of-vocab word ("lamp", "rug", "clock") → an unsupported_type CHOICE
  plus a direction CHOICE.

Ambiguous specs are also feasibility-guarded (at least one option combination must work).

## E.4 The paraphrase hook (why the English looks robotic for now)
Template English reads stiffly ("Add a brown door, with a brown door in the top part of the
image."). That's intentional for v1 — it's deterministic and free. The `generate_dataset(...,
paraphrase=fn)` hook lets you later pass a function that rewrites each sentence with your local
model to sound natural, **without changing the gold logic**. You run that step on your GPU when
ready.

---

# PART F — Honest Caveats, Known Issues, and What a Reviewer Would Attack

This section is deliberately adversarial. If you present this project, expect these.

## F.1 The feasibility checker is not a real solver
`feasible()` is sound-for-pruning only (see D.2). **Do not describe it as the solver or claim it
decides satisfiability in general.** The real, complete solver is SampleSearch, which is *not yet
reimplemented* — it's the next major build. A reviewer who reads `feasibility.py` and thinks it's
the constraint engine will (correctly) object; pre-empt this by stating the division of labor.

## F.2 The `default()` size bounds were retuned, and are provisional
SPRING's paper uses object size bounds of 256–512px on a 512px canvas — i.e. objects can be
50–100% of the whole image. At that size, two objects literally cannot satisfy "completely left
of" each other (there isn't room), which made most generated ambiguous specs infeasible. We
retuned the bounds to furniture-realistic values (roughly 15–45% of the image). **This is a
deviation from the paper and must be reconciled** once the real image generator is back in the
loop. It's a reasonable engineering fix, but it is a fix, and you should own it.

## F.3 The "35% deviation" evaluation assumption
As explained in D.4, the simulated user disagrees with the prior 35% of the time by construction.
The dialogue-policy numbers (policy 86% correct, ASK_NONE 60%, ASK_ALL 100%) **depend on that
number.** Change it and the gaps move. This is fine for a proof-of-concept and a sanity check, but
the headline human-facing claim must come from a **real human study**, not this simulation. State
the assumption explicitly in any writeup.

## F.4 Semantic-equivalence checking is probabilistic
`semantic_equiv` decides whether two formulas mean the same thing by testing agreement on ~200
random layouts. If they ever disagree, they're definitely different; if they always agree, they're
*probably* equivalent. It can, in principle, miss a difference that only shows up on rare layouts.
For CHOICE-bearing formulas it falls back to exact structural match. Adequate for evaluation, not a
theorem.

## F.5 The model layer is unrun and may underperform on ambiguity
Everything except the actual LLM generation has been executed and passes. The open empirical
question is whether a 7B local model, few-shot, will **reliably emit CHOICE** on ambiguous input.
Small models tend to over-commit (guess a single reading) rather than express uncertainty. If that
happens, the fix is fine-tuning on the generated data (the pipeline is ready; the LoRA script is
not yet written). Don't claim the parser works until the GPU run shows it does.

## F.6 Type coverage is a hard ceiling, half-solved
Referring to an unusual *existing* object can be handled by an open-vocabulary detector (a planned
perception upgrade). But *generating* a brand-new out-of-vocab object (a "lamp") is **not** solved
— the preference model has no idea where lamps go — so v1 simply asks or skips. Be honest that
open-vocabulary *generation* is future work, not a current capability.

## F.7 Novelty must be stated precisely
The spatial-logic skeleton is borrowed from existing qualitative calculi and must be cited as such.
The defensible new contributions are: the metric+typed target as a parsing destination, the
`CHOICE` construct distinct from `∨`, and the ask-only-when-it-matters resolution policy with its
evaluation. Overclaiming "we invented spatial logic / English-to-logic" would be wrong and easily
rebutted.

---

# PART G — What Has Been Run vs. What Remains

## G.1 Run and verified (no GPU needed, reproducible with fixed seeds)
- **Unit tests:** 25/25 pass (`python tests/test_core.py`). Covers desugaring correctness vs
  Table 1, JSON round-trip, SPRING-string round-trip, feasibility soundness, and the full
  dialogue policy (auto-resolve vs ask, ordering, budget, tree rewrite).
- **Data generation:** 2,000 train + 400 test samples generated, ~half ambiguous, all
  satisfiability-checked. Written to `train.jsonl` / `test.jsonl`.
- **Full pipeline on the stub backend** (`examples/run_pipeline.py`), with these current results:
  - Harness sanity (gold stub): **user-facing correctness 400/400 (100%)**.
  - Trivial-guesser floor: **exact 0/60 (0%), semantic 9/60 (15%)** — proves the metric
    discriminates.
  - Dialogue policy vs baselines (on 239 ambiguous cases):
    **policy 211/239 correct (88%) at 1.10 questions**; **ASK_ALL 239/239 (100%) at 1.20 questions**;
    **ASK_NONE 136/239 (57%) at 0 questions.**
    Reading: the policy recovers most of ASK_ALL's accuracy while asking slightly fewer questions,
    and its misses are cases where the answer didn't change the feasible outcome.

## G.2 Not yet run (needs your GPU)
- **The real English→MSCL model.** Swap `StubBackend` for `LocalBackend` and run
  `examples/run_pipeline_local.py`. This is the single missing measurement. Recommended: smoke-test
  on ONE example (catches library-API issues fast), then a subset of 8 (each ~5–30s), then the full
  60 only once it looks right.
- **Paraphrase pass** (optional): use the local model to naturalize template English, regenerate.

## G.3 Not yet built (next artifacts, in rough priority)
1. **Fine-tuning (LoRA) script** — *only if* few-shot underperforms on CHOICE emission. Data is
   already in the right format.
2. **SampleSearch (the real, complete solver)** — upgrades `feasible` to exact satisfiability +
   preference-weighted sampling; this is what actually produces guaranteed layouts. Max's advice:
   understand SampleSearch deeply and the rest follows.
3. **Visual Element Generator hookup** — turn resolved specs into actual images (the demo that
   wins an ISEF booth). Modernize the backbone here.
4. **Scaled evaluation** — thousands of samples, accuracy broken down by ambiguity kind, and the
   full questions-vs-accuracy curve as you sweep the budget from 0 to 5 (this curve is your key
   figure).
5. **Human study** — real users vs. the policy and vs. a plain text-to-image tool; replaces the
   35%-deviation simulation with ground truth.

---

# PART H — How To Run It Yourself

## H.1 Offline (any machine, no GPU)
```bash
cd mscl
python tests/test_core.py        # expect 25/25
python examples/run_pipeline.py  # generates data, runs stub pipeline + policy eval
python examples/demo.py          # walks the appendix examples end to end
```

## H.2 With the real model (GPU)
```bash
pip install outlines transformers torch accelerate lark
# then, in examples/run_pipeline_local.py:
#   - it loads the model ONCE
#   - smoke-tests on 1 example (fix any Outlines API error here)
#   - runs a subset of 8 with per-example progress
#   - raise SUBSET to 60 for the full number
python examples/run_pipeline_local.py
```
If the single-example smoke test errors, capture the traceback and your `outlines.__version__`
— the generation-call signature is the one part that varies by library version.

## H.3 File map
```
mscl/
  mscl/
    ast.py          data structures (Part C.1)
    relations.py    28 names → atoms (C.2, D.1)
    profile.py      SPRING settings, default() (C.3, F.2)
    json_io.py      JSON ⇄ AST + JSON Schema (C.4)
    render.py       MSCL ⇄ SPRING string (C.5)
    feasibility.py  model_check + bounds propagation (C.6, D.2, F.1)
    validate.py     well-formedness + profile checks (C.7)
    dialogue.py     THE resolution policy (C.8, D.3)
    datagen.py      synthetic data (C.9, Part E)
    parser.py       prompt + Stub/Local backends (C.10, H.2)
    evaluate.py     the three evaluations (C.11, D.4)
    mscl_spring.schema.json   schema for constrained decoding
    mscl_spring.gbnf          grammar for llama.cpp decoding
  examples/
    seed_pairs.json           gold pairs from SPRING's appendix
    demo.py                   appendix walk-through
    run_pipeline.py           full stub pipeline (offline)
    run_pipeline_local.py     real-model pipeline (GPU)
    train.jsonl / test.jsonl  generated data
  tests/test_core.py          25 tests
  README.md / STATUS.md / this document
```

---

# PART I — One-Paragraph Summary (for when someone asks "what is this?")

This project builds the missing front door to SPRING: a system that turns ordinary English design
instructions into the exact formal logic SPRING needs, while handling the ambiguity that English
has and logic doesn't. It introduces MSCL, a clean metric-and-typed spatial logic that generalizes
SPRING's language and, uniquely, can carry ambiguity as first-class `CHOICE` nodes; and a
resolution policy that clears that ambiguity by asking the user a clarifying question *only when
the answer would change the result*, otherwise resolving it automatically. The logic layer, the
ambiguity machinery, a synthetic-data generator, the model interface, and a full evaluation are
built and tested end-to-end (25/25 tests; policy reaches 88% of intended readings at ~1.1 questions
versus 57% for never-asking and 100%-but-costly for always-asking). The one remaining measurement
is running the real local language model on a GPU; the remaining builds are the exact SampleSearch
solver and the image generator that together turn resolved specs into guaranteed images.
