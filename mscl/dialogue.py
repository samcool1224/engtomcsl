"""MSCL v1 — E1: the interactive resolution / dialogue policy over CHOICE nodes.

This is the core research contribution: turn ambiguity handling into an evaluable
algorithm whose objective is *ask the user a clarifying question only when the answer
changes the outcome*, under a question budget.

Pipeline per utterance:
  1. PRUNE      : drop CHOICE options that are infeasible given the hard (non-CHOICE)
                  constraints. (Uses the feasibility oracle; exact SAT -> SampleSearch.)
                  A CHOICE that collapses to one feasible option is resolved silently.
  2. SILENT     : offset-kind CHOICEs with no emphasis and no degeneracy collapse to
                  their default (c=0) option with no question.
  3. RANK & ASK : remaining "outcome-affecting" CHOICEs are ordered
                  structural (unsupported_type, reference) before geometric
                  (scope, direction, offset), and asked until the budget is spent.
                  After each answer we re-PRUNE (an identity answer can make geometric
                  options infeasible).
  4. CLOSE      : any CHOICE left when the budget is exhausted is resolved by max prior
                  and flagged low-confidence.

Baselines to compare against (for the paper): ASK_ALL (confirm every CHOICE),
ASK_NONE (always max-prior, never ask), ASK_RANDOM (ask k random CHOICEs).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from .ast import (Spec, Obj, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Option, Formula)
from .feasibility import feasible, collect_atoms, init_domains, Layout
from .relations import desugar

# Order in which kinds are asked (lower = asked first). Structural before geometric.
_KIND_PRIORITY = {"unsupported_type": 0, "reference": 1, "scope": 2,
                  "direction": 3, "offset": 4}


@dataclass
class Question:
    kind: str
    span: str
    options_text: List[str]
    choice_ref: Choice

@dataclass
class ResolutionLog:
    asked: List[Question] = field(default_factory=list)
    auto_resolved: List[Tuple[str, str]] = field(default_factory=list)   # (kind, why)
    low_confidence: List[str] = field(default_factory=list)


# An "oracle" answers a question by returning the index of the chosen option.
# In deployment this is the user; in evaluation it's a simulated user / gold label.
Oracle = Callable[[Question], int]


def _hard_atoms(spec: Spec, exclude: Choice):
    """Collect atoms from the non-CHOICE skeleton (treat every *other* CHOICE as absent
    by taking its max-prior option as a provisional assumption)."""
    def skeleton(f: Formula) -> Optional[Formula]:
        if isinstance(f, Choice):
            if f is exclude:
                return None
            # provisional: assume current best option
            best = max(f.options, key=lambda o: o.prior)
            return best.formula  # may be None (SKIP)
        if isinstance(f, And):
            kids = [skeleton(a) for a in f.args]
            kids = [k for k in kids if k is not None]
            return And(kids) if kids else None
        return f
    sk = skeleton(spec.formula)
    if sk is None:
        return []
    atoms = collect_atoms(sk)
    return atoms if atoms is not None else []   # if non-conjunctive, skip pruning (sound: no prune)


def _option_feasible(spec: Spec, ch: Choice, opt: Option, existing: Layout) -> bool:
    if opt.skip or opt.formula is None:
        return True   # SKIP is always "feasible" (drops the object)
    base = _hard_atoms(spec, exclude=ch)
    extra = collect_atoms(opt.formula)
    if extra is None:
        return True   # non-conjunctive option: don't claim infeasible (sound)
    obj_ids = [o.id for o in spec.objects]
    dom = init_domains(obj_ids, existing)
    return feasible(base + extra, dom)


def _all_choices(f: Formula) -> List[Choice]:
    out: List[Choice] = []
    def rec(g: Formula):
        if isinstance(g, Choice):
            out.append(g)
            for o in g.options:
                if o.formula:
                    rec(o.formula)
        elif isinstance(g, And): [rec(a) for a in g.args]
        elif isinstance(g, Or): [rec(a) for a in g.args]
        elif isinstance(g, Not): rec(g.arg)
    rec(f)
    return out


def _collapse(ch: Choice, chosen: Option):
    """Mutate a Choice into a resolved single option by tagging it. We represent
    resolution by reducing options to the single chosen one (formula may be None=SKIP)."""
    ch.options = [chosen]


def _degenerate_offset(spec, ch, existing) -> bool:
    """An offset CHOICE is 'degenerate' (worth asking) if the no-offset option is
    feasible but pins objects essentially on top of each other — a hook for the later
    calibration layer (E2). v1 stub: never degenerate unless emphasis flagged."""
    return False


def _rewrite_resolved(f: Formula) -> Optional[Formula]:
    """Replace any single-option Choice with its chosen formula (or None for SKIP).
    Returns the rewritten formula, or None if it collapses away entirely."""
    if isinstance(f, Choice):
        if len(f.options) == 1:
            opt = f.options[0]
            if opt.formula is None:        # SKIP -> drop
                return None
            return _rewrite_resolved(opt.formula)
        # still unresolved: rewrite inside its options but keep the Choice
        f.options = [Option(o.prior,
                            (_rewrite_resolved(o.formula) if o.formula else None),
                            o.skip) for o in f.options]
        return f
    if isinstance(f, And):
        kids = [_rewrite_resolved(a) for a in f.args]
        kids = [k for k in kids if k is not None]
        if not kids:
            return None
        return And(kids) if len(kids) > 1 else kids[0]
    if isinstance(f, Or):
        kids = [_rewrite_resolved(a) for a in f.args]
        kids = [k for k in kids if k is not None]
        if not kids:
            return None
        return Or(kids) if len(kids) > 1 else kids[0]
    if isinstance(f, Not):
        inner = _rewrite_resolved(f.arg)
        return Not(inner) if inner is not None else None
    return f


def resolve(spec: Spec, oracle: Oracle, budget: int = 3,
            existing: Optional[Layout] = None) -> Tuple[Spec, ResolutionLog]:
    """Run the dialogue policy. Mutates spec's CHOICE nodes into resolved single options.
    Returns (spec, log)."""
    log = ResolutionLog()
    existing = existing or {o.id: o.box for o in spec.objects if o.box}

    # work queue of unresolved CHOICEs (top-level discovery; nested ones surface after parent resolves)
    def unresolved() -> List[Choice]:
        return [c for c in _all_choices(spec.formula) if len(c.options) > 1]

    # ---- step 1+2 loop: prune & silent-collapse until stable ----
    progressed = True
    guard = 0
    while progressed and guard < 10_000:
        progressed = False
        guard += 1
        for ch in unresolved():
            feas = [o for o in ch.options if _option_feasible(spec, ch, o, existing)]
            # only shrink (and flag progress) when SOME but not all options drop out
            if 0 < len(feas) < len(ch.options):
                ch.options = feas
                progressed = True
            if len(ch.options) == 1:
                log.auto_resolved.append((ch.kind, "single feasible option"))
                continue
            if ch.kind == "offset" and not ch.emphasis and not _degenerate_offset(spec, ch, existing):
                default = min(ch.options, key=lambda o: abs(_offset_magnitude(o)))
                _collapse(ch, default)
                log.auto_resolved.append((ch.kind, "offset default c=0, no emphasis"))
                progressed = True

    # ---- step 3: rank remaining outcome-affecting CHOICEs and ask ----
    asked = 0
    pending = sorted(unresolved(), key=lambda c: _KIND_PRIORITY.get(c.kind, 9))
    while pending and asked < budget:
        ch = pending[0]
        q = Question(kind=ch.kind, span=ch.span,
                     options_text=[_describe(o) for o in ch.options], choice_ref=ch)
        idx = oracle(q)
        idx = max(0, min(idx, len(ch.options) - 1))
        _collapse(ch, ch.options[idx])
        log.asked.append(q)
        asked += 1
        # re-prune: an identity/scope answer can make geometric options infeasible
        progressed = True
        guard = 0
        while progressed and guard < 10_000:
            progressed = False
            guard += 1
            for c2 in unresolved():
                feas = [o for o in c2.options if _option_feasible(spec, c2, o, existing)]
                if 0 < len(feas) < len(c2.options):
                    c2.options = feas
                    progressed = True
                if len(c2.options) == 1:
                    log.auto_resolved.append((c2.kind, "single feasible after answer"))
        pending = sorted(unresolved(), key=lambda c: _KIND_PRIORITY.get(c.kind, 9))

    # ---- step 4: close out remaining by max prior, flag low confidence ----
    for ch in unresolved():
        best = max(ch.options, key=lambda o: o.prior)
        _collapse(ch, best)
        log.low_confidence.append(f"{ch.kind}@'{ch.span}' resolved by prior (budget spent)")

    # collapse all single-option CHOICE nodes out of the tree
    rewritten = _rewrite_resolved(spec.formula)
    spec.formula = rewritten if rewritten is not None else And([])
    return spec, log


def _offset_magnitude(opt: Option) -> int:
    """Heuristic magnitude of an offset option (0 for no-offset/SKIP)."""
    if not opt.formula:
        return 0
    if isinstance(opt.formula, Relation) and opt.formula.const:
        return abs(opt.formula.const)
    return 0


def _describe(opt: Option) -> str:
    if opt.skip or opt.formula is None:
        return "skip this object"
    f = opt.formula
    if isinstance(f, Relation):
        return f"{f.name}({', '.join(f.args)}" + (f", {f.const})" if f.const is not None else ")")
    if isinstance(f, TypePred):
        return f'treat as "{f.type}"'
    return str(type(f).__name__)


# ---------------------------------------------------------------------------
# Baseline policies (for evaluation)
# ---------------------------------------------------------------------------
def resolve_ask_none(spec: Spec) -> Spec:
    for ch in _all_choices(spec.formula):
        if len(ch.options) > 1:
            _collapse(ch, max(ch.options, key=lambda o: o.prior))
    r = _rewrite_resolved(spec.formula)
    spec.formula = r if r is not None else And([])
    return spec

def resolve_ask_all(spec: Spec, oracle: Oracle) -> Tuple[Spec, int]:
    n = 0
    for ch in _all_choices(spec.formula):
        if len(ch.options) > 1:
            q = Question(ch.kind, ch.span, [_describe(o) for o in ch.options], ch)
            idx = oracle(q); n += 1
            _collapse(ch, ch.options[max(0, min(idx, len(ch.options)-1))])
    r = _rewrite_resolved(spec.formula)
    spec.formula = r if r is not None else And([])
    return spec, n
