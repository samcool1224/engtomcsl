"""Exact Z3 semantics for resolved MSCL specifications.

This module is deliberately smaller than SampleSearch.  It owns the correctness
boundary: translating every MSCL formula into integer arithmetic, fixing detected
objects to their observed boxes, enforcing valid in-frame geometry, and checking a
concrete layout independently of the sampler.

``z3-solver`` is imported lazily so the English-to-MSCL parser can still be used in
environments that have not installed the solver extra yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .ast import (And, Atom, Choice, Default, Formula, Not, Or, PropertyPred,
                  Relation, Spec, TypePred)
from .feasibility import Layout, model_check
from .profile import COORD_MAX, COORD_MIN, expand_default
from .relations import desugar
from .validate import validate

try:  # Keep the language/parser layer importable without the optional solver install.
    import z3  # type: ignore
except ImportError:  # pragma: no cover - exercised only in solver-free environments
    z3 = None


Variable = Tuple[str, str]


class SolverUnavailableError(RuntimeError):
    """Raised when layout solving is requested without ``z3-solver`` installed."""


class SolverUnknownError(RuntimeError):
    """Raised if Z3 cannot decide a query (MSCL's integer fragment should be decidable)."""


@dataclass(frozen=True)
class UnsatExplanation:
    """A compact unsatisfiable-core explanation from the exact solver."""

    labels: Tuple[str, ...]
    constraints: Tuple[str, ...]


def z3_available() -> bool:
    return z3 is not None


def _require_z3():
    if z3 is None:
        raise SolverUnavailableError(
            "SampleSearch requires z3-solver. Install it with "
            "`python -m pip install -r requirements.txt`."
        )
    return z3


class Z3Backend:
    """Compile one MSCL ``Spec`` into an incremental exact solver.

    The backend accepts Boolean MSCL formulas, including ``or`` and ``not``.  A
    ``Choice`` can be compiled as a disjunction for standalone satisfiability queries,
    but SampleSearch intentionally calls ``assert_resolved`` before constructing this
    backend: ambiguity resolution and spatial sampling remain separate stages.
    """

    def __init__(self, spec: Spec, *, backbone: str = "stable_diffusion",
                 enforce_in_frame: bool = True):
        self.z3 = _require_z3()
        validate(spec, allow_choice=True)
        self.spec = spec
        self.backbone = backbone
        self.enforce_in_frame = enforce_in_frame
        self.solver = self.z3.Solver()
        self.variables: Dict[Variable, Any] = {}
        self._labels: Dict[str, str] = {}
        self._label_counter = 0

        for oi, obj in enumerate(spec.objects):
            for prim in ("x", "y", "w", "h"):
                self.variables[(obj.id, prim)] = self.z3.Int(f"mscl_o{oi}_{prim}")

        self._add_object_constraints()
        parts: Iterable[Formula]
        parts = spec.formula.args if isinstance(spec.formula, And) else (spec.formula,)
        for i, part in enumerate(parts):
            self._track(self.formula_to_z3(part), f"formula[{i}]: {_describe(part)}")

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def formula_to_z3(self, formula: Formula):
        z = self.z3
        if isinstance(formula, Atom):
            return self._atom_to_z3(formula)
        if isinstance(formula, Relation):
            return self._atom_to_z3(desugar(formula))
        if isinstance(formula, (TypePred, PropertyPred)):
            # Profile/type conformance is handled by validate(); these predicates do
            # not constrain x/y/w/h.
            return z.BoolVal(True)
        if isinstance(formula, Default):
            return self.formula_to_z3(expand_default(formula, self.backbone))
        if isinstance(formula, Not):
            return z.Not(self.formula_to_z3(formula.arg))
        if isinstance(formula, And):
            return z.And(*(self.formula_to_z3(a) for a in formula.args))
        if isinstance(formula, Or):
            return z.Or(*(self.formula_to_z3(a) for a in formula.args))
        if isinstance(formula, Choice):
            branches = [z.BoolVal(True) if o.skip or o.formula is None
                        else self.formula_to_z3(o.formula)
                        for o in formula.options]
            return z.Or(*branches)
        raise TypeError(f"unsupported MSCL node: {type(formula).__name__}")

    def _atom_to_z3(self, atom: Atom):
        lhs = sum((coeff * self.variables[(obj, prim)]
                   for coeff, obj, prim in atom.terms), self.z3.IntVal(0))
        rhs = self.z3.IntVal(atom.const)
        if atom.op == "<":
            return lhs < rhs
        if atom.op == "<=":
            return lhs <= rhs
        if atom.op == "=":
            return lhs == rhs
        if atom.op == ">=":
            return lhs >= rhs
        if atom.op == ">":
            return lhs > rhs
        raise ValueError(f"unknown atom operator: {atom.op}")

    def _add_object_constraints(self) -> None:
        for obj in self.spec.objects:
            x = self.variables[(obj.id, "x")]
            y = self.variables[(obj.id, "y")]
            w = self.variables[(obj.id, "w")]
            h = self.variables[(obj.id, "h")]
            geometry = [x >= COORD_MIN, y >= COORD_MIN, w >= 1, h >= 1,
                        x <= COORD_MAX, y <= COORD_MAX,
                        w <= COORD_MAX, h <= COORD_MAX]
            if self.enforce_in_frame:
                geometry.extend((x + w <= COORD_MAX, y + h <= COORD_MAX))
            self._track(self.z3.And(*geometry), f"geometry({obj.id})")

            if obj.status == "existing":
                if obj.box is None:
                    raise ValueError(f"existing object {obj.id!r} has no observed box")
                if len(obj.box) != 4:
                    raise ValueError(f"box for {obj.id!r} must contain x,y,w,h")
                fixed = self.z3.And(*(self.variables[(obj.id, prim)] == int(value)
                                      for prim, value in zip(("x", "y", "w", "h"), obj.box)))
                self._track(fixed, f"observed_box({obj.id})={tuple(obj.box)}")

    def _track(self, expression, description: str) -> None:
        name = f"mscl_track_{self._label_counter}"
        self._label_counter += 1
        label = self.z3.Bool(name)
        self._labels[name] = description
        self.solver.assert_and_track(expression, label)

    # ------------------------------------------------------------------
    # Exact queries and model extraction
    # ------------------------------------------------------------------
    def check(self, extra: Optional[Iterable[Any]] = None):
        if extra is None:
            result = self.solver.check()
        else:
            self.solver.push()
            self.solver.add(*list(extra))
            result = self.solver.check()
            self.solver.pop()
        if result == self.z3.unknown:
            raise SolverUnknownError(self.solver.reason_unknown())
        return result

    def is_satisfiable(self, extra: Optional[Iterable[Any]] = None) -> bool:
        return self.check(extra) == self.z3.sat

    def interval_constraint(self, variable: Variable, lower: int, upper: int):
        v = self.variables[variable]
        return self.z3.And(v >= int(lower), v <= int(upper))

    def layout_from_model(self, model: Optional[Any] = None) -> Layout:
        if model is None:
            result = self.check()
            if result != self.z3.sat:
                raise ValueError("cannot extract a layout from an unsatisfiable specification")
            model = self.solver.model()
        layout: Layout = {}
        for obj in self.spec.objects:
            values = []
            for prim in ("x", "y", "w", "h"):
                value = model.eval(self.variables[(obj.id, prim)], model_completion=True)
                values.append(value.as_long())
            layout[obj.id] = tuple(values)  # type: ignore[assignment]
        return layout

    def explain_unsat(self) -> UnsatExplanation:
        if self.check() == self.z3.sat:
            return UnsatExplanation((), ())
        names = tuple(str(label) for label in self.solver.unsat_core())
        return UnsatExplanation(names, tuple(self._labels.get(n, n) for n in names))

    # ------------------------------------------------------------------
    # Independent correctness check
    # ------------------------------------------------------------------
    def verify_layout(self, layout: Mapping[str, Tuple[int, int, int, int]]) -> bool:
        if any(obj.id not in layout for obj in self.spec.objects):
            return False
        for obj in self.spec.objects:
            box = layout[obj.id]
            if len(box) != 4 or any(not isinstance(v, int) for v in box):
                return False
            x, y, w, h = box
            if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
                return False
            if not (1 <= w <= COORD_MAX and 1 <= h <= COORD_MAX):
                return False
            if self.enforce_in_frame and (x + w > COORD_MAX or y + h > COORD_MAX):
                return False
            if obj.status == "existing" and tuple(box) != tuple(obj.box or ()):
                return False
        return model_check(self.spec.formula, dict(layout), self.backbone)


def _describe(formula: Formula) -> str:
    if isinstance(formula, Relation):
        suffix = "" if formula.const is None else f", {formula.const}"
        return f"{formula.name}({', '.join(formula.args)}{suffix})"
    if isinstance(formula, Default):
        return f"default({formula.obj})"
    if isinstance(formula, TypePred):
        return f"type({formula.obj}, {formula.type})"
    if isinstance(formula, PropertyPred):
        return f"property({formula.obj}, {formula.value})"
    return type(formula).__name__.lower()
