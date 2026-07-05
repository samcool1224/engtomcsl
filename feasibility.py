"""MSCL v1 — feasibility & model checking.

Two services:
  1. model_check(spec, M)      : exact M |= phi for a concrete layout M (trivial, exact).
  2. feasible(atoms, domains)  : SOUND bounds-propagation feasibility for a CONJUNCTION
                                 of Layer-1 atoms. If it returns False, the conjunction is
                                 truly UNSAT. If True, it's "not provably UNSAT" (may still
                                 be UNSAT for tangled cases). Exact SAT is delegated to
                                 SampleSearch (spec §7) — this is the lightweight oracle the
                                 dialogue policy needs to prune CHOICE options.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from .ast import (Spec, Atom, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Formula)
from .relations import desugar
from .profile import COORD_MIN, COORD_MAX, expand_default

Var = Tuple[str, str]              # (obj_id, prim)
Domain = Dict[Var, Tuple[int, int]]
Layout = Dict[str, Tuple[int, int, int, int]]   # obj_id -> (x,y,w,h)


# ---------------------------------------------------------------------------
# 1. exact model checking
# ---------------------------------------------------------------------------
def _val(M: Layout, o: str, p: str) -> int:
    x, y, w, h = M[o]
    return {"x": x, "y": y, "w": w, "h": h}[p]

def _atom_holds(a: Atom, M: Layout) -> bool:
    lhs = sum(c * _val(M, o, p) for c, o, p in a.terms)
    return {"<": lhs < a.const, "<=": lhs <= a.const, "=": lhs == a.const,
            ">=": lhs >= a.const, ">": lhs > a.const}[a.op]

def model_check(f: Formula, M: Layout, backbone: str = "stable_diffusion") -> bool:
    if isinstance(f, Atom):
        return _atom_holds(f, M)
    if isinstance(f, Relation):
        return _atom_holds(desugar(f), M)
    if isinstance(f, (TypePred, PropertyPred)):
        return True            # typing checked separately against object table
    if isinstance(f, Default):
        return model_check(expand_default(f, backbone), M)
    if isinstance(f, Not):
        return not model_check(f.arg, M)
    if isinstance(f, And):
        return all(model_check(a, M) for a in f.args)
    if isinstance(f, Or):
        return any(model_check(a, M) for a in f.args)
    if isinstance(f, Choice):
        return any(model_check(o.formula, M) for o in f.options if o.formula)
    raise TypeError(type(f))


# ---------------------------------------------------------------------------
# 2. bounds-propagation feasibility for a conjunction of atoms
# ---------------------------------------------------------------------------
def _strict_to_nonstrict(a: Atom) -> Atom:
    """Integers: a<b <=> a<=b-1 ; a>b <=> a>=b+1. Normalize to <=, >=, =."""
    if a.op == "<":
        return Atom(a.terms, "<=", a.const - 1)
    if a.op == ">":
        return Atom(a.terms, ">=", a.const + 1)
    return a

def init_domains(obj_ids: List[str], existing: Optional[Layout] = None) -> Domain:
    dom: Domain = {}
    for o in obj_ids:
        if existing and o in existing:
            x, y, w, h = existing[o]
            dom[(o, "x")] = (x, x); dom[(o, "y")] = (y, y)
            dom[(o, "w")] = (w, w); dom[(o, "h")] = (h, h)
        else:
            for p in ("x", "y"):
                dom[(o, p)] = (COORD_MIN, COORD_MAX)
            for p in ("w", "h"):
                dom[(o, p)] = (1, COORD_MAX)
    return dom

def feasible(atoms: List[Atom], domains: Domain, max_iter: int = 1000) -> bool:
    """Sound (one-directional) feasibility via interval bound propagation."""
    dom = dict(domains)
    atoms = [_strict_to_nonstrict(a) for a in atoms]
    changed = True
    it = 0
    while changed and it < max_iter:
        changed = False
        it += 1
        for a in atoms:
            # For each variable v in the atom, isolate it and tighten its bound.
            for (cv, ov, pv) in a.terms:
                if cv == 0:
                    continue
                key = (ov, pv)
                if key not in dom:
                    dom[key] = (COORD_MIN, COORD_MAX)
                # sum of other terms' interval
                lo_other = hi_other = 0
                ok = True
                for (c, o, p) in a.terms:
                    if (o, p) == key and c == cv:
                        # handle the (possibly repeated) target term once
                        continue
                    dl, dh = dom.get((o, p), (COORD_MIN, COORD_MAX))
                    if c >= 0:
                        lo_other += c * dl; hi_other += c * dh
                    else:
                        lo_other += c * dh; hi_other += c * dl
                # constraint: cv*v  OP  const - other
                # => v OP' (const - other)/cv
                cur_lo, cur_hi = dom[key]
                if a.op in ("<=", "="):
                    # cv*v <= const - lo_other  (use tightest rhs)
                    rhs = a.const - lo_other
                    bound = _div_bound(rhs, cv, upper=True)
                    if cv > 0:
                        cur_hi = min(cur_hi, bound)
                    else:
                        cur_lo = max(cur_lo, bound)
                if a.op in (">=", "="):
                    rhs = a.const - hi_other
                    bound = _div_bound(rhs, cv, upper=False)
                    if cv > 0:
                        cur_lo = max(cur_lo, bound)
                    else:
                        cur_hi = min(cur_hi, bound)
                if (cur_lo, cur_hi) != dom[key]:
                    dom[key] = (cur_lo, cur_hi)
                    changed = True
                if cur_lo > cur_hi:
                    return False
    return all(lo <= hi for lo, hi in dom.values())

def _div_bound(rhs: int, coeff: int, upper: bool) -> int:
    """Integer bound for v from coeff*v <= rhs (upper) or coeff*v >= rhs (lower),
    accounting for sign of coeff. Returns a value to min/max against current bound."""
    import math
    if coeff > 0:
        return math.floor(rhs / coeff) if upper else math.ceil(rhs / coeff)
    else:
        # dividing flips direction; caller assigns to the correct side
        return math.ceil(rhs / coeff) if upper else math.floor(rhs / coeff)


def collect_atoms(f: Formula, backbone: str = "stable_diffusion") -> Optional[List[Atom]]:
    """Flatten a CHOICE-free, OR-free conjunction into a list of atoms.
    Returns None if the formula contains OR/NOT/CHOICE (not a pure conjunction)."""
    out: List[Atom] = []
    def rec(g: Formula) -> bool:
        if isinstance(g, Atom):
            out.append(g); return True
        if isinstance(g, Relation):
            out.append(desugar(g)); return True
        if isinstance(g, (TypePred, PropertyPred)):
            return True
        if isinstance(g, Default):
            return rec(expand_default(g, backbone))
        if isinstance(g, And):
            return all(rec(a) for a in g.args)
        return False   # Or/Not/Choice -> not a pure conjunction
    return out if rec(f) else None
