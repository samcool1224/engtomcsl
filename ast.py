"""MSCL v1 — Abstract Syntax Tree.

Internal representation of MSCL formulas. Three layers, mirroring the spec:
  Layer 1 : Atom            — a normalized linear (in)equality over box-edge variables.
  Layer 2 : Relation        — one of the 28 named spatial relations (sugar over Atom).
  Layer 3 : boolean + typing + CHOICE (underspecification).

Object geometry uses four PRIMITIVE variables per object: x, y, w, h  (per-mille ints).
Derived edges expand at desugar time:  R = x + w,  B = y + h.
y increases downward (image coords): "above" => smaller y.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union

PRIMS = ("x", "y", "w", "h")          # primitive per-object variables
EDGES = ("X", "Y", "W", "H", "R", "B")  # edge terms usable in named-relation defs
OPS = ("<", "<=", "=", ">=", ">")
CHOICE_KINDS = ("direction", "offset", "reference", "scope", "unsupported_type")


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------
@dataclass
class Obj:
    """An object in the scene. `box` is set iff status == 'existing' (from perception)."""
    id: str
    status: str                      # 'existing' | 'new'
    type: Optional[str] = None       # an in-profile type, or None if unknown/unsupported
    properties: List[str] = field(default_factory=list)
    box: Optional[Tuple[int, int, int, int]] = None  # (x,y,w,h) for existing objects


# ---------------------------------------------------------------------------
# Layer 1 — normalized atom:  sum_i coeff_i * var_i   OP   const
#   var_i = (obj_id, prim)  with prim in PRIMS
# Strict ops are normalized away over integers in feasibility (a<b <=> a<=b-1).
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    terms: List[Tuple[int, str, str]]   # list of (coeff, obj_id, prim)
    op: str                             # one of OPS
    const: int

    def vars(self) -> List[Tuple[str, str]]:
        return [(o, p) for _, o, p in self.terms]


# ---------------------------------------------------------------------------
# Layer 2 — named relation (desugars to one Atom via relations.py)
# ---------------------------------------------------------------------------
@dataclass
class Relation:
    name: str                 # one of the 28 names
    args: List[str]           # object ids (1 for *_value, else 2)
    const: Optional[int] = None  # offset c; None => default 0 (or required for *_value)


# ---------------------------------------------------------------------------
# Layer 3 — typing predicates, boolean, macro, CHOICE
# ---------------------------------------------------------------------------
@dataclass
class TypePred:
    obj: str
    type: str

@dataclass
class PropertyPred:
    obj: str
    value: str

@dataclass
class Default:
    obj: str

@dataclass
class Not:
    arg: "Formula"

@dataclass
class And:
    args: List["Formula"]

@dataclass
class Or:
    args: List["Formula"]

@dataclass
class Option:
    prior: float
    formula: Optional["Formula"] = None   # None  <=>  SKIP option
    skip: bool = False

@dataclass
class Choice:
    """Underspecification node. Truth-functionally a disjunction over option formulas;
    priors + provenance (kind, span) are metadata consumed by the dialogue policy (E1)."""
    kind: str                 # one of CHOICE_KINDS
    span: str                 # source-text span that produced the ambiguity
    options: List[Option]
    emphasis: bool = False    # parser flag: e.g. "WELL to the left" (drives offset asking)

Formula = Union[Atom, Relation, TypePred, PropertyPred, Default, Not, And, Or, Choice]


@dataclass
class Spec:
    """A full parsed specification: the object table + the root formula."""
    objects: List[Obj]
    formula: Formula

    def obj(self, oid: str) -> Obj:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(f"unknown object id: {oid}")
