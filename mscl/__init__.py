"""MSCL v1 — Metric Spatial Constraint Logic. Public API."""
from .ast import (Spec, Obj, Atom, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Option, CHOICE_KINDS)
from .relations import desugar, ALL_RELATIONS, arity
from .json_io import spec_from_json, spec_to_json, node_to_json, json_schema
from .render import to_spring, from_spring
from .feasibility import model_check, feasible, collect_atoms, init_domains
from .z3_backend import (Z3Backend, UnsatExplanation, SolverUnavailableError,
                         SolverUnknownError, z3_available)
from .samplesearch import (SampleSearch, SampleResult, SearchStats, SearchEvent,
                           PreferenceModel, UniformPreference, GeometricPreference,
                           UnsatError, generate_layout)
from .validate import validate, assert_resolved, ValidationError
from .dialogue import resolve, resolve_ask_none, resolve_ask_all, Question, ResolutionLog
from . import profile
from . import datagen
from .parser import build_prompt, parse, StubBackend, LocalBackend
from . import evaluate

__all__ = [
    "Spec", "Obj", "Atom", "Relation", "TypePred", "PropertyPred", "Default",
    "Not", "And", "Or", "Choice", "Option", "CHOICE_KINDS",
    "desugar", "ALL_RELATIONS", "arity",
    "spec_from_json", "spec_to_json", "node_to_json", "json_schema",
    "to_spring", "from_spring",
    "model_check", "feasible", "collect_atoms", "init_domains",
    "Z3Backend", "UnsatExplanation", "SolverUnavailableError", "SolverUnknownError",
    "z3_available", "SampleSearch", "SampleResult", "SearchStats", "SearchEvent",
    "PreferenceModel", "UniformPreference", "GeometricPreference", "UnsatError",
    "generate_layout",
    "validate", "assert_resolved", "ValidationError",
    "resolve", "resolve_ask_none", "resolve_ask_all", "Question", "ResolutionLog",
    "profile",
]
