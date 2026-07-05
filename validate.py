"""MSCL v1 — validation: well-formedness + MSCL-SPRING profile conformance."""
from __future__ import annotations
from typing import List
from .ast import (Spec, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Formula, CHOICE_KINDS)
from .relations import ALL_RELATIONS, arity, NO_OFFSET, _UNARY
from .profile import is_supported_type


class ValidationError(Exception):
    pass


def validate(spec: Spec, *, profile_spring: bool = True,
             allow_choice: bool = True) -> List[str]:
    """Returns a list of warnings; raises ValidationError on hard violations."""
    warnings: List[str] = []
    ids = {o.id for o in spec.objects}
    if len(ids) != len(spec.objects):
        raise ValidationError("duplicate object ids")

    def check(f: Formula):
        if isinstance(f, Relation):
            if f.name not in ALL_RELATIONS:
                raise ValidationError(f"unknown relation {f.name}")
            if len(f.args) != arity(f.name):
                raise ValidationError(f"{f.name} arity: expected {arity(f.name)}, got {f.args}")
            for a in f.args:
                if a not in ids:
                    raise ValidationError(f"{f.name} references unknown object {a}")
            if f.name in _UNARY and f.const is None:
                raise ValidationError(f"{f.name} requires a constant value")
            if f.name in NO_OFFSET and f.const is not None:
                warnings.append(f"{f.name} ignores offset {f.const}")
        elif isinstance(f, TypePred):
            if f.obj not in ids:
                raise ValidationError(f"type() references unknown object {f.obj}")
            if profile_spring and not is_supported_type(f.type):
                warnings.append(f'type "{f.type}" not in SPRING vocabulary')
        elif isinstance(f, PropertyPred):
            if f.obj not in ids:
                raise ValidationError(f"property() references unknown object {f.obj}")
        elif isinstance(f, Default):
            if f.obj not in ids:
                raise ValidationError(f"default() references unknown object {f.obj}")
        elif isinstance(f, Not):
            check(f.arg)
        elif isinstance(f, (And, Or)):
            if not f.args:
                raise ValidationError("empty and/or")
            for a in f.args:
                check(a)
        elif isinstance(f, Choice):
            if not allow_choice:
                raise ValidationError("CHOICE present where forbidden (final SPRING spec)")
            if f.kind not in CHOICE_KINDS:
                raise ValidationError(f"bad CHOICE kind {f.kind}")
            if len(f.options) < 2:
                raise ValidationError("CHOICE needs >= 2 options")
            ptot = sum(o.prior for o in f.options)
            if abs(ptot - 1.0) > 1e-6:
                warnings.append(f"CHOICE@'{f.span}' priors sum to {ptot:.3f}, not 1.0")
            for o in f.options:
                if o.formula is not None:
                    check(o.formula)
        else:
            raise ValidationError(f"unknown node type {type(f)}")

    check(spec.formula)
    return warnings


def assert_resolved(spec: Spec) -> None:
    """Hard check that a spec is CHOICE-free (ready to render to SPRING)."""
    validate(spec, profile_spring=True, allow_choice=False)
