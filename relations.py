"""MSCL v1 — the 28 named relations and their desugaring to Layer-1 atoms.

Each relation is defined by an edge-term inequality from the spec (Table 1).
We expand edge terms R=x+w, B=y+h into primitive variables and emit a normalized
Atom:  sum coeff*var  OP  const.

Verified against SPRING Table 1 truth conditions (y grows downward => 'above' is smaller y).
"""
from __future__ import annotations
from typing import Dict, Optional
from .ast import Atom, Relation

# Edge term -> linear combination of primitives, as {(prim): coeff}
_EDGE = {
    "X": {"x": 1},
    "Y": {"y": 1},
    "W": {"w": 1},
    "H": {"h": 1},
    "R": {"x": 1, "w": 1},   # right  = x + w
    "B": {"y": 1, "h": 1},   # bottom = y + h
}

# Binary relations:  edge(o1)  OP  edge(o2) + sign*c
#   (lhs_edge, op, rhs_edge, c_sign)
# c defaults to 0 when omitted.
_BINARY = {
    # directional, partial overlap
    "above":   ("Y", "<=", "Y", -1),   # Y1 <= Y2 - c
    "below":   ("Y", ">=", "Y", +1),   # Y1 >= Y2 + c
    "left":    ("X", "<=", "X", -1),
    "right":   ("X", ">=", "X", +1),
    # directional, complete / non-overlap
    "cabove":  ("B", "<=", "Y", -1),   # B1 <= Y2 - c
    "cbelow":  ("Y", ">=", "B", +1),   # Y1 >= B2 + c
    "cleft":   ("R", "<=", "X", -1),   # R1 <= X2 - c
    "cright":  ("X", ">=", "R", +1),   # X1 >= R2 + c
    # size comparisons
    "wider":   ("W", ">=", "W", +1),
    "narrower":("W", "<=", "W", -1),
    "taller":  ("H", ">=", "H", +1),
    "shorter": ("H", "<=", "H", -1),
    # alignment / equality (no offset)
    "xeq":     ("X", "=",  "X", 0),
    "yeq":     ("Y", "=",  "Y", 0),
    "weq":     ("W", "=",  "W", 0),
    "heq":     ("H", "=",  "H", 0),
}

# Unary (_value) relations:  edge(o)  OP  c
_UNARY = {
    "above_value":   ("Y", "<"),
    "below_value":   ("Y", ">"),
    "left_value":    ("X", "<"),
    "right_value":   ("X", ">"),
    "wider_value":   ("W", ">"),
    "narrower_value":("W", "<"),
    "taller_value":  ("H", ">"),
    "shorter_value": ("H", "<"),
    "xeq_value":     ("X", "="),
    "yeq_value":     ("Y", "="),
    "weq_value":     ("W", "="),
    "heq_value":     ("H", "="),
}

ALL_RELATIONS = tuple(_BINARY) + tuple(_UNARY)
NO_OFFSET = {"xeq", "yeq", "weq", "heq"}  # binary relations that never take c


def arity(name: str) -> int:
    if name in _BINARY:
        return 2
    if name in _UNARY:
        return 1
    raise KeyError(f"unknown relation: {name}")


def desugar(rel: Relation) -> Atom:
    """Named relation -> normalized Atom (terms OP const)."""
    name = rel.name
    if name in _BINARY:
        if len(rel.args) != 2:
            raise ValueError(f"{name} needs 2 objects, got {rel.args}")
        o1, o2 = rel.args
        lhs_edge, op, rhs_edge, csign = _BINARY[name]
        c = 0 if (rel.const is None or name in NO_OFFSET) else int(rel.const)
        terms = []
        for prim, coeff in _EDGE[lhs_edge].items():
            terms.append((coeff, o1, prim))
        for prim, coeff in _EDGE[rhs_edge].items():
            terms.append((-coeff, o2, prim))
        # move +csign*c from rhs to const:  lhs - rhs  OP  csign*c
        const = csign * c
        return _merge(Atom(terms, op, const))
    if name in _UNARY:
        if len(rel.args) != 1:
            raise ValueError(f"{name} needs 1 object, got {rel.args}")
        if rel.const is None:
            raise ValueError(f"{name} requires a constant value")
        (o1,) = rel.args
        edge, op = _UNARY[name]
        terms = [(coeff, o1, prim) for prim, coeff in _EDGE[edge].items()]
        return Atom(terms, op, int(rel.const))
    raise KeyError(f"unknown relation: {name}")


def _merge(atom: Atom) -> Atom:
    """Combine like (obj,prim) terms (e.g. y appearing from B expansion)."""
    acc: Dict[tuple, int] = {}
    for coeff, o, p in atom.terms:
        acc[(o, p)] = acc.get((o, p), 0) + coeff
    terms = [(c, o, p) for (o, p), c in acc.items() if c != 0]
    return Atom(terms, atom.op, atom.const)
