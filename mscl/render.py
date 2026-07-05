"""MSCL v1 — bidirectional rendering between internal AST and SPRING surface syntax.

SPRING string form (from the paper):  cright(m,o0) ^ type(m,"microwave") ^ ...
We use:  &  for ^ (and),  |  for v (or),  ~ for not, to keep it ASCII-parseable.
Rendering is total both ways; round-trip is identity on the SPRING profile (no CHOICE).
"""
from __future__ import annotations
from typing import List
from lark import Lark, Transformer, v_args
from .ast import (Spec, Obj, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Formula)

# ---------------------------------------------------------------------------
# AST -> SPRING string
# ---------------------------------------------------------------------------
def to_spring(f: Formula) -> str:
    if isinstance(f, Relation):
        args = list(f.args)
        if f.const is not None:
            args = args + [str(f.const)]
        return f"{f.name}({','.join(args)})"
    if isinstance(f, TypePred):
        return f'type({f.obj},"{f.type}")'
    if isinstance(f, PropertyPred):
        return f'property({f.obj},"{f.value}")'
    if isinstance(f, Default):
        return f"default({f.obj})"
    if isinstance(f, Not):
        return f"~({to_spring(f.arg)})"
    if isinstance(f, And):
        return " & ".join(f"({to_spring(a)})" for a in f.args)
    if isinstance(f, Or):
        return " | ".join(f"({to_spring(a)})" for a in f.args)
    if isinstance(f, Choice):
        raise ValueError("CHOICE present: resolve before rendering to SPRING (profile §9 forbids it)")
    raise TypeError(f"cannot render {type(f)}")


# ---------------------------------------------------------------------------
# SPRING string -> AST  (Lark grammar)
# ---------------------------------------------------------------------------
_GRAMMAR = r"""
?start: formula
?formula: or_expr
?or_expr: and_expr ("|" and_expr)*       -> or_
?and_expr: unary ("&" unary)*            -> and_
?unary: "~" "(" formula ")"              -> not_
      | "(" formula ")"
      | atom
atom: NAME "(" args ")"
args: arg ("," arg)*
?arg: SIGNED_INT                         -> int_arg
    | ESCAPED_STRING                     -> str_arg
    | NAME                               -> id_arg
NAME: /[A-Za-z_][A-Za-z0-9_]*/
%import common.SIGNED_INT
%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""

_parser = Lark(_GRAMMAR, parser="lalr")

@v_args(inline=True)
class _T(Transformer):
    def or_(self, *xs):
        xs = list(xs)
        return xs[0] if len(xs) == 1 else Or(xs)
    def and_(self, *xs):
        xs = list(xs)
        return xs[0] if len(xs) == 1 else And(xs)
    def not_(self, x):
        return Not(x)
    def int_arg(self, t):
        return ("int", int(t))
    def str_arg(self, t):
        return ("str", str(t)[1:-1])
    def id_arg(self, t):
        return ("id", str(t))
    def args(self, *xs):
        return list(xs)
    def atom(self, name, arglist):
        name = str(name)
        if name == "type":
            return TypePred(arglist[0][1], arglist[1][1])
        if name == "property":
            return PropertyPred(arglist[0][1], arglist[1][1])
        if name == "default":
            return Default(arglist[0][1])
        ids = [a[1] for a in arglist if a[0] == "id"]
        ints = [a[1] for a in arglist if a[0] == "int"]
        const = ints[0] if ints else None
        return Relation(name, ids, const)

def from_spring(s: str) -> Formula:
    return _T().transform(_parser.parse(s))
