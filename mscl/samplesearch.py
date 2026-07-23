"""Preference-guided, exact SampleSearch for MSCL layouts.

SPRING's useful idea is retained while its preliminary interval checker is replaced
with exact incremental SMT queries.  At every binary interval decision:

1. a preference model supplies the original branch weights;
2. Z3 assigns weight zero to branches with no valid completion;
3. the remaining weights are renormalized and sampled; and
4. depth-first search keeps the alternative branch for complete backtracking.

The result is always independently model-checked before it is returned.  The default
guide is a transparent geometric prior; it can later be replaced with an adapter around
SPRING's ``core.pt`` or another learned layout model without changing the solver.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import secrets
import time
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .ast import And, Not, Obj, Or, Relation, Spec
from .feasibility import Layout
from .profile import COORD_MAX, COORD_MIN
from .validate import assert_resolved
from .z3_backend import UnsatExplanation, Variable, Z3Backend


Interval = Tuple[int, int]


class UnsatError(ValueError):
    """Raised when a resolved MSCL specification has no valid layout."""

    def __init__(self, explanation: UnsatExplanation):
        self.explanation = explanation
        details = "; ".join(explanation.constraints) or "no satisfiable model"
        super().__init__(f"MSCL specification is unsatisfiable: {details}")


class PreferenceModel(Protocol):
    """Interface for a neural or heuristic spatial preference distribution."""

    def branch_weights(self, *, spec: Spec, variable: Variable,
                       branches: Sequence[Interval],
                       assigned: Mapping[Variable, int]) -> Sequence[float]:
        """Return one non-negative, unnormalized weight per candidate branch."""


class UniformPreference:
    """Uniform over integer values before symbolic pruning."""

    def branch_weights(self, *, spec: Spec, variable: Variable,
                       branches: Sequence[Interval],
                       assigned: Mapping[Variable, int]) -> Sequence[float]:
        return [float(hi - lo + 1) for lo, hi in branches]


# Transparent baseline priors, not hard constraints.  They only influence which of
# several valid layouts is sampled.  A learned provider can replace them wholesale.
TYPICAL_SIZE = {
    "chair": (190, 260), "couch": (390, 250), "potted plant": (180, 310),
    "bed": (410, 310), "mirror": (190, 290), "dining table": (390, 240),
    "window": (310, 270), "desk": (350, 220), "toilet": (180, 240),
    "door": (190, 380), "TV": (270, 180), "microwave": (210, 160),
    "oven": (220, 310), "toaster": (180, 150), "sink": (290, 180),
    "refrigerator": (250, 380), "blender": (160, 190),
}
_ANCHORS = ((0.50, 0.50), (0.27, 0.34), (0.73, 0.66),
            (0.27, 0.72), (0.73, 0.28), (0.50, 0.20), (0.50, 0.80))


class GeometricPreference:
    """A fast, inspectable baseline prior for plausible box size and placement.

    ``exploration`` defines one global uniform-mixture component over the complete
    coordinate domain. This preserves diversity without re-injecting tail probability
    at every binary decision.
    """

    def __init__(self, *, exploration: float = 0.03,
                 position_sigma: float = 170.0, size_sigma: float = 45.0):
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be between 0 and 1")
        if position_sigma <= 0 or size_sigma <= 0:
            raise ValueError("preference sigmas must be positive")
        self.exploration = exploration
        self.position_sigma = position_sigma
        self.size_sigma = size_sigma

    def branch_weights(self, *, spec: Spec, variable: Variable,
                       branches: Sequence[Interval],
                       assigned: Mapping[Variable, int]) -> Sequence[float]:
        obj_id, prim = variable
        obj = spec.obj(obj_id)
        target, sigma = self._target(spec, obj, prim, assigned)
        domain_values = float(COORD_MAX - (COORD_MIN if prim in ("x", "y") else 1) + 1)
        weights = []
        for lo, hi in branches:
            gaussian = _normal_interval_mass(lo, hi, target, sigma)
            # This denominator must remain the full domain, not the current parent
            # interval. Using the parent interval compounds exploration at every bit
            # and creates an unintended heavy-tailed distribution.
            uniform = (hi - lo + 1) / domain_values
            weights.append((1.0 - self.exploration) * gaussian +
                           self.exploration * uniform)
        return weights

    def _target(self, spec: Spec, obj: Obj, prim: str,
                assigned: Mapping[Variable, int]) -> Tuple[float, float]:
        typical_w, typical_h = TYPICAL_SIZE.get(obj.type or "", (280, 260))
        if prim == "w":
            return float(typical_w), self.size_sigma
        if prim == "h":
            return float(typical_h), self.size_sigma

        relational_target = self._relational_target(
            spec, obj, prim, assigned, typical_w, typical_h
        )
        if relational_target is not None:
            extent = assigned.get(
                (obj.id, "w" if prim == "x" else "h"),
                typical_w if prim == "x" else typical_h,
            )
            target = max(0.0, min(COORD_MAX - extent, relational_target))
            return target, min(self.position_sigma, 40.0)

        new_objects = [o for o in spec.objects if o.status == "new"]
        index = next((i for i, o in enumerate(new_objects) if o.id == obj.id), 0)
        ax, ay = _ANCHORS[index % len(_ANCHORS)]
        if prim == "x":
            width = assigned.get((obj.id, "w"), typical_w)
            return max(0.0, min(COORD_MAX - width, ax * COORD_MAX - width / 2)), self.position_sigma
        height = assigned.get((obj.id, "h"), typical_h)
        return max(0.0, min(COORD_MAX - height, ay * COORD_MAX - height / 2)), self.position_sigma

    def _relational_target(self, spec: Spec, obj: Obj, prim: str,
                           assigned: Mapping[Variable, int],
                           typical_w: int, typical_h: int) -> Optional[float]:
        """Favor compact alignment for explicitly related objects.

        This is a soft preference only. Z3 still owns the exact relation, and a
        missing/not-yet-assigned reference falls back to the generic anchor.
        """
        width = assigned.get((obj.id, "w"), typical_w)
        height = assigned.get((obj.id, "h"), typical_h)
        gap = 35.0
        for relation in _positive_relations(spec.formula):
            if len(relation.args) != 2 or obj.id not in relation.args:
                continue
            first_id, second_id = relation.args
            other_id = second_id if obj.id == first_id else first_id
            other = _known_box(spec, other_id, assigned)
            if other is None:
                continue
            ox, oy, ow, oh = other
            obj_is_first = obj.id == first_id

            if relation.name in ("cleft", "cright"):
                if prim == "y":
                    return oy + oh - height
                if prim == "x":
                    if relation.name == "cleft":
                        return ox - gap - width if obj_is_first else ox + ow + gap
                    return ox + ow + gap if obj_is_first else ox - gap - width

            if relation.name in ("cabove", "cbelow"):
                if prim == "x":
                    return ox + ow / 2.0 - width / 2.0
                if prim == "y":
                    if relation.name == "cabove":
                        return oy - gap - height if obj_is_first else oy + oh + gap
                    return oy + oh + gap if obj_is_first else oy - gap - height

            if relation.name in ("left", "right") and prim == "y":
                return oy + oh - height
            if relation.name in ("above", "below") and prim == "x":
                return ox + ow / 2.0 - width / 2.0
        return None


@dataclass
class SearchEvent:
    variable: Variable
    interval: Interval
    branches: Tuple[Interval, ...]
    raw_weights: Tuple[float, ...]
    feasible: Tuple[bool, ...]
    chosen: Optional[Interval]


@dataclass
class SearchStats:
    seed: int
    sampled_variables: int = 0
    decisions: int = 0
    proposals: int = 0
    rejected_proposals: int = 0
    backtracks: int = 0
    solver_checks: int = 0
    solver_time_s: float = 0.0
    preference_time_s: float = 0.0
    verification_time_s: float = 0.0
    total_time_s: float = 0.0


@dataclass
class SampleResult:
    layout: Layout
    stats: SearchStats
    trace: List[SearchEvent] = field(default_factory=list)


class SampleSearch:
    """Generate exact, preference-weighted layouts for resolved MSCL specs."""

    def __init__(self, preference: Optional[PreferenceModel] = None, *,
                 backbone: str = "stable_diffusion",
                 variable_order: Sequence[str] = ("w", "h", "x", "y"),
                 enforce_in_frame: bool = True):
        if tuple(sorted(variable_order)) != ("h", "w", "x", "y"):
            raise ValueError("variable_order must contain x, y, w, h exactly once")
        self.preference = preference or GeometricPreference()
        self.backbone = backbone
        self.variable_order = tuple(variable_order)
        self.enforce_in_frame = enforce_in_frame

    def sample(self, spec: Spec, *, seed: Optional[int] = None) -> SampleResult:
        """Sample one valid layout, or raise ``UnsatError``.

        ``spec`` must already be CHOICE-free.  Supplying a seed makes both the layout
        and the complete search trace reproducible.
        """
        assert_resolved(spec)
        actual_seed = secrets.randbits(64) if seed is None else int(seed)
        rng = random.Random(actual_seed)
        stats = SearchStats(seed=actual_seed)
        trace: List[SearchEvent] = []
        started = time.perf_counter()
        backend = Z3Backend(spec, backbone=self.backbone,
                            enforce_in_frame=self.enforce_in_frame)

        if not self._check(backend, stats):
            stats.total_time_s = time.perf_counter() - started
            raise UnsatError(backend.explain_unsat())

        variables = [(obj.id, prim) for obj in spec.objects if obj.status == "new"
                     for prim in self.variable_order]
        stats.sampled_variables = len(variables)
        assigned: Dict[Variable, int] = {}

        layout = self._search_variable(backend, spec, variables, 0, assigned,
                                       rng, stats, trace)
        if layout is None:  # Defensive: exact interval splitting should only reach this on UNSAT.
            stats.total_time_s = time.perf_counter() - started
            raise UnsatError(backend.explain_unsat())

        verify_started = time.perf_counter()
        verified = backend.verify_layout(layout)
        stats.verification_time_s += time.perf_counter() - verify_started
        stats.total_time_s = time.perf_counter() - started
        if not verified:
            raise RuntimeError("SampleSearch produced a layout that failed independent verification")
        return SampleResult(layout, stats, trace)

    def sample_many(self, spec: Spec, count: int, *, seed: Optional[int] = None) -> List[SampleResult]:
        if count < 0:
            raise ValueError("count must be non-negative")
        root_seed = secrets.randbits(64) if seed is None else int(seed)
        rng = random.Random(root_seed)
        return [self.sample(spec, seed=rng.getrandbits(64)) for _ in range(count)]

    def _search_variable(self, backend: Z3Backend, spec: Spec,
                         variables: Sequence[Variable], index: int,
                         assigned: Dict[Variable, int], rng: random.Random,
                         stats: SearchStats, trace: List[SearchEvent]) -> Optional[Layout]:
        if index >= len(variables):
            if not self._check(backend, stats):
                return None
            return backend.layout_from_model(backend.solver.model())

        variable = variables[index]
        lower = COORD_MIN if variable[1] in ("x", "y") else 1
        return self._narrow(backend, spec, variables, index, variable,
                            (lower, COORD_MAX), assigned, rng, stats, trace)

    def _narrow(self, backend: Z3Backend, spec: Spec,
                variables: Sequence[Variable], index: int, variable: Variable,
                interval: Interval, assigned: Dict[Variable, int],
                rng: random.Random, stats: SearchStats,
                trace: List[SearchEvent]) -> Optional[Layout]:
        lo, hi = interval
        if lo == hi:
            backend.solver.push()
            backend.solver.add(backend.variables[variable] == lo)
            assigned[variable] = lo
            result = self._search_variable(backend, spec, variables, index + 1,
                                           assigned, rng, stats, trace)
            assigned.pop(variable, None)
            backend.solver.pop()
            return result

        mid = (lo + hi) // 2
        branches: Tuple[Interval, Interval] = ((lo, mid), (mid + 1, hi))

        pref_started = time.perf_counter()
        raw = tuple(float(w) for w in self.preference.branch_weights(
            spec=spec, variable=variable, branches=branches, assigned=assigned))
        stats.preference_time_s += time.perf_counter() - pref_started
        if len(raw) != len(branches):
            raise ValueError("preference model returned the wrong number of branch weights")
        if any(not math.isfinite(w) or w < 0 for w in raw):
            raise ValueError("preference weights must be finite and non-negative")

        feasible: List[bool] = []
        for branch in branches:
            stats.proposals += 1
            backend.solver.push()
            backend.solver.add(backend.interval_constraint(variable, *branch))
            ok = self._check(backend, stats)
            backend.solver.pop()
            feasible.append(ok)
            if not ok:
                stats.rejected_proposals += 1

        candidates = [i for i, ok in enumerate(feasible) if ok]
        if not candidates:
            trace.append(SearchEvent(variable, interval, branches, raw,
                                     tuple(feasible), None))
            return None

        order = _weighted_candidate_order(candidates, raw, branches, rng)
        chosen = branches[order[0]]
        trace.append(SearchEvent(variable, interval, branches, raw,
                                 tuple(feasible), chosen))
        stats.decisions += 1

        for attempt, branch_index in enumerate(order):
            branch = branches[branch_index]
            backend.solver.push()
            backend.solver.add(backend.interval_constraint(variable, *branch))
            result = self._narrow(backend, spec, variables, index, variable,
                                  branch, assigned, rng, stats, trace)
            backend.solver.pop()
            if result is not None:
                return result
            if attempt + 1 < len(order):
                stats.backtracks += 1
        return None

    @staticmethod
    def _check(backend: Z3Backend, stats: SearchStats) -> bool:
        started = time.perf_counter()
        result = backend.check()
        stats.solver_time_s += time.perf_counter() - started
        stats.solver_checks += 1
        return result == backend.z3.sat


def generate_layout(spec: Spec, *, seed: Optional[int] = None,
                    preference: Optional[PreferenceModel] = None,
                    backbone: str = "stable_diffusion") -> SampleResult:
    """Convenience wrapper for the common one-layout case."""
    return SampleSearch(preference, backbone=backbone).sample(spec, seed=seed)


def _normal_interval_mass(lo: int, hi: int, mean: float, sigma: float) -> float:
    scale = sigma * math.sqrt(2.0)
    upper = 0.5 * (1.0 + math.erf((hi + 0.5 - mean) / scale))
    lower = 0.5 * (1.0 + math.erf((lo - 0.5 - mean) / scale))
    return max(0.0, upper - lower)


def _positive_relations(formula):
    if isinstance(formula, Relation):
        yield formula
    elif isinstance(formula, And):
        for arg in formula.args:
            yield from _positive_relations(arg)
    elif isinstance(formula, (Not, Or)):
        return


def _known_box(spec: Spec, object_id: str,
               assigned: Mapping[Variable, int]) -> Optional[Tuple[int, int, int, int]]:
    obj = spec.obj(object_id)
    if obj.box is not None:
        return obj.box
    values = [assigned.get((object_id, prim)) for prim in ("x", "y", "w", "h")]
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values)


def _weighted_candidate_order(candidates: Sequence[int], weights: Sequence[float],
                              branches: Sequence[Interval],
                              rng: random.Random) -> List[int]:
    remaining = list(candidates)
    ordered: List[int] = []
    while remaining:
        positive_total = sum(weights[i] for i in remaining)
        if positive_total <= 0:
            # A malformed/overconfident guide must not remove feasible support.
            fallback = [branches[i][1] - branches[i][0] + 1 for i in remaining]
            total = float(sum(fallback))
            draw = rng.random() * total
            running = 0.0
            selected = remaining[-1]
            for i, mass in zip(remaining, fallback):
                running += mass
                if draw <= running:
                    selected = i
                    break
        else:
            draw = rng.random() * positive_total
            running = 0.0
            selected = remaining[-1]
            for i in remaining:
                running += weights[i]
                if draw <= running:
                    selected = i
                    break
        ordered.append(selected)
        remaining.remove(selected)
    return ordered
