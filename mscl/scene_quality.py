"""Aesthetic selection among layouts that already satisfy MSCL exactly.

Z3 decides whether a layout is valid.  These soft metrics decide which valid
layout is the most useful input to an image generator.  They never replace or
relax symbolic constraints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .ast import And, Formula, Not, Or, Relation, Spec
from .feasibility import Layout
from .samplesearch import SampleResult, TYPICAL_SIZE


Box = Tuple[int, int, int, int]
_FLOOR_OBJECTS = {
    "chair", "couch", "potted plant", "bed", "dining table", "desk",
    "toilet", "oven", "refrigerator",
}
_HORIZONTAL = {"left", "right", "cleft", "cright"}
_VERTICAL = {"above", "below", "cabove", "cbelow"}
_COMPLETE_HORIZONTAL = {"cleft", "cright"}
_COMPLETE_VERTICAL = {"cabove", "cbelow"}


@dataclass(frozen=True)
class SceneQuality:
    """Normalized soft costs; lower is better."""

    total: float
    typical_size: float
    scene_center: float
    primary_center: float
    edge_safety: float
    floor_alignment: float
    relation_composition: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LayoutSelection:
    candidate_count: int
    selected_index: int
    selected_quality: SceneQuality
    candidate_totals: Tuple[float, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "selected_index": self.selected_index,
            "selected_quality": self.selected_quality.as_dict(),
            "candidate_totals": list(self.candidate_totals),
        }


def score_scene_layout(spec: Spec, layout: Mapping[str, Box]) -> SceneQuality:
    """Score the image-composition quality of one already-valid layout."""
    objects = [obj for obj in spec.objects if obj.id in layout]
    if not objects:
        return SceneQuality(*(0.0 for _ in range(7)))

    size_costs: List[float] = []
    edge_costs: List[float] = []
    floor_bottoms: List[float] = []
    for obj in objects:
        x, y, width, height = layout[obj.id]
        target_w, target_h = TYPICAL_SIZE.get(obj.type or "", (280, 260))
        size_costs.append(
            (abs(math.log(max(width, 1) / target_w))
             + abs(math.log(max(height, 1) / target_h))) / 2.0
        )
        margins = (x, y, 1000 - (x + width), 1000 - (y + height))
        edge_costs.append(statistics.fmean(
            max(0.0, (45.0 - margin) / 45.0) ** 2 for margin in margins
        ))
        if obj.type in _FLOOR_OBJECTS:
            floor_bottoms.append(float(y + height))

    left = min(layout[obj.id][0] for obj in objects)
    top = min(layout[obj.id][1] for obj in objects)
    right = max(layout[obj.id][0] + layout[obj.id][2] for obj in objects)
    bottom = max(layout[obj.id][1] + layout[obj.id][3] for obj in objects)
    union_cx, union_cy = (left + right) / 2.0, (top + bottom) / 2.0
    scene_center = math.hypot(union_cx - 500.0, union_cy - 540.0) / 650.0

    primary = _primary_object(spec, objects)
    px, py, pw, ph = layout[primary.id]
    primary_center = math.hypot(
        px + pw / 2.0 - 500.0,
        py + ph / 2.0 - 520.0,
    ) / 650.0

    if len(floor_bottoms) >= 2:
        floor_alignment = statistics.pstdev(floor_bottoms) / 260.0
    else:
        floor_alignment = 0.0

    relation_costs = [
        _relation_composition_cost(relation, layout)
        for relation in _positive_relations(spec.formula)
        if all(obj_id in layout for obj_id in relation.args)
    ]
    relation_composition = (
        statistics.fmean(relation_costs) if relation_costs else 0.0
    )

    typical_size = statistics.fmean(size_costs)
    edge_safety = statistics.fmean(edge_costs)
    total = (
        1.20 * typical_size
        + 1.15 * scene_center
        + 1.10 * primary_center
        + 0.45 * edge_safety
        + 1.25 * floor_alignment
        + 1.75 * relation_composition
    )
    return SceneQuality(
        total=total,
        typical_size=typical_size,
        scene_center=scene_center,
        primary_center=primary_center,
        edge_safety=edge_safety,
        floor_alignment=floor_alignment,
        relation_composition=relation_composition,
    )


def select_best_layout(spec: Spec, candidates: Sequence[SampleResult]
                       ) -> Tuple[SampleResult, LayoutSelection]:
    """Return the lowest-cost candidate while preserving deterministic ties."""
    if not candidates:
        raise ValueError("at least one layout candidate is required")
    qualities = [score_scene_layout(spec, result.layout) for result in candidates]
    selected_index = min(range(len(candidates)), key=lambda index: qualities[index].total)
    selection = LayoutSelection(
        candidate_count=len(candidates),
        selected_index=selected_index,
        selected_quality=qualities[selected_index],
        candidate_totals=tuple(quality.total for quality in qualities),
    )
    return candidates[selected_index], selection


def _primary_object(spec: Spec, objects):
    priority = {
        "dining table": 0, "bed": 1, "couch": 2, "desk": 3,
        "refrigerator": 4, "chair": 5, "potted plant": 6,
    }
    order = {obj.id: index for index, obj in enumerate(spec.objects)}
    return min(objects, key=lambda obj: (priority.get(obj.type or "", 20),
                                         order[obj.id]))


def _positive_relations(formula: Formula) -> Iterable[Relation]:
    if isinstance(formula, Relation):
        yield formula
    elif isinstance(formula, And):
        for arg in formula.args:
            yield from _positive_relations(arg)
    elif isinstance(formula, (Not, Or)):
        # A relation under NOT or an unresolved alternative is not an aesthetic
        # instruction that can safely guide composition.
        return


def _relation_composition_cost(relation: Relation,
                               layout: Mapping[str, Box]) -> float:
    if len(relation.args) != 2:
        return 0.0
    first, second = (layout[obj_id] for obj_id in relation.args)
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    ar, ab = ax + aw, ay + ah
    br, bb = bx + bw, by + bh

    if relation.name in _HORIZONTAL:
        overlap = max(0, min(ab, bb) - max(ay, by))
        overlap_ratio = overlap / max(1.0, min(ah, bh))
        alignment_cost = 1.0 - min(1.0, overlap_ratio)
        gap_cost = 0.0
        if relation.name in _COMPLETE_HORIZONTAL:
            gap = (bx - ar) if relation.name == "cleft" else (ax - br)
            gap_cost = min(2.0, abs(gap - 35.0) / 180.0)
        return 0.65 * alignment_cost + 0.35 * gap_cost

    if relation.name in _VERTICAL:
        overlap = max(0, min(ar, br) - max(ax, bx))
        overlap_ratio = overlap / max(1.0, min(aw, bw))
        alignment_cost = 1.0 - min(1.0, overlap_ratio)
        gap_cost = 0.0
        if relation.name in _COMPLETE_VERTICAL:
            gap = (by - ab) if relation.name == "cabove" else (ay - bb)
            gap_cost = min(2.0, abs(gap - 35.0) / 180.0)
        return 0.65 * alignment_cost + 0.35 * gap_cost
    return 0.0
