"""Quantitative diagnostics for SampleSearch layout distributions.

These metrics do not pretend to replace human aesthetic judgments. They establish the
engineering basics first: exact validity, diversity, latency, typical-size fit, center
spread, and overlap behavior. A learned preference model should beat the transparent
baselines here and in a later human study.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import statistics
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ast import Spec
from .feasibility import Layout
from .samplesearch import (GeometricPreference, PreferenceModel, SampleResult,
                           SampleSearch, TYPICAL_SIZE)
from .z3_backend import Z3Backend


@dataclass(frozen=True)
class LayoutMetrics:
    valid: bool
    typical_size_error: float
    mean_area_fraction: float
    mean_center_offset: float
    mean_pairwise_iou: float


@dataclass
class EvaluationReport:
    label: str
    count: int
    valid_count: int
    unique_count: int
    validity_rate: float
    unique_rate: float
    mean_diversity: float
    mean_typical_size_error: float
    mean_area_fraction: float
    mean_center_offset: float
    mean_pairwise_iou: float
    mean_runtime_s: float
    p95_runtime_s: float
    max_runtime_s: float
    mean_solver_time_s: float
    mean_rejected_proposals: float
    total_backtracks: int
    results: List[SampleResult] = field(repr=False)
    layout_metrics: List[LayoutMetrics] = field(repr=False)

    def as_dict(self) -> Dict[str, object]:
        return {name: value for name, value in vars(self).items()
                if name not in ("results", "layout_metrics")}


def measure_layout(spec: Spec, layout: Mapping[str, Tuple[int, int, int, int]], *,
                   backend: Optional[Z3Backend] = None) -> LayoutMetrics:
    """Measure one concrete layout. Lower typical-size error is better."""
    checker = backend or Z3Backend(spec)
    valid = checker.verify_layout(layout)
    new_objects = [obj for obj in spec.objects if obj.status == "new" and obj.id in layout]

    size_errors = []
    areas = []
    center_offsets = []
    for obj in new_objects:
        x, y, w, h = layout[obj.id]
        typical_w, typical_h = TYPICAL_SIZE.get(obj.type or "", (280, 260))
        size_errors.append((abs(math.log(w / typical_w)) +
                            abs(math.log(h / typical_h))) / 2.0)
        areas.append((w * h) / 1_000_000.0)
        cx, cy = x + w / 2.0, y + h / 2.0
        center_offsets.append(math.hypot(cx - 500.0, cy - 500.0) /
                              math.hypot(500.0, 500.0))

    boxes = [layout[obj.id] for obj in spec.objects if obj.id in layout]
    ious = [_iou(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
    return LayoutMetrics(
        valid=valid,
        typical_size_error=_mean(size_errors),
        mean_area_fraction=_mean(areas),
        mean_center_offset=_mean(center_offsets),
        mean_pairwise_iou=_mean(ious),
    )


def evaluate_sampler(spec: Spec, sampler: SampleSearch, *, count: int = 50,
                     seed: int = 0, label: Optional[str] = None) -> EvaluationReport:
    """Sample and summarize one preference configuration."""
    if count <= 0:
        raise ValueError("count must be positive")
    results = sampler.sample_many(spec, count, seed=seed)
    backend = Z3Backend(spec, backbone=sampler.backbone,
                        enforce_in_frame=sampler.enforce_in_frame)
    metrics = [measure_layout(spec, result.layout, backend=backend) for result in results]
    signatures = {_signature(spec, result.layout) for result in results}
    runtimes = [result.stats.total_time_s for result in results]
    return EvaluationReport(
        label=label or type(sampler.preference).__name__,
        count=count,
        valid_count=sum(metric.valid for metric in metrics),
        unique_count=len(signatures),
        validity_rate=sum(metric.valid for metric in metrics) / count,
        unique_rate=len(signatures) / count,
        mean_diversity=_distribution_diversity(spec, results),
        mean_typical_size_error=_mean([m.typical_size_error for m in metrics]),
        mean_area_fraction=_mean([m.mean_area_fraction for m in metrics]),
        mean_center_offset=_mean([m.mean_center_offset for m in metrics]),
        mean_pairwise_iou=_mean([m.mean_pairwise_iou for m in metrics]),
        mean_runtime_s=_mean(runtimes),
        p95_runtime_s=_percentile(runtimes, 0.95),
        max_runtime_s=max(runtimes),
        mean_solver_time_s=_mean([r.stats.solver_time_s for r in results]),
        mean_rejected_proposals=_mean([float(r.stats.rejected_proposals) for r in results]),
        total_backtracks=sum(r.stats.backtracks for r in results),
        results=results,
        layout_metrics=metrics,
    )


def compare_preferences(spec: Spec, preferences: Mapping[str, PreferenceModel], *,
                        count: int = 50, seed: int = 0) -> List[EvaluationReport]:
    """Evaluate multiple guides using the same root seed and sample count."""
    return [evaluate_sampler(spec, SampleSearch(preference), count=count, seed=seed, label=label)
            for label, preference in preferences.items()]


def format_comparison_table(reports: Sequence[EvaluationReport]) -> str:
    """Return a console-friendly comparison table with explicit metric direction."""
    headers = ("guide", "valid", "unique", "size err↓", "diversity↑",
               "overlap", "mean ms", "p95 ms", "rejects")
    rows = [headers]
    for report in reports:
        rows.append((
            report.label,
            f"{report.valid_count}/{report.count}",
            f"{report.unique_count}/{report.count}",
            f"{report.mean_typical_size_error:.3f}",
            f"{report.mean_diversity:.3f}",
            f"{report.mean_pairwise_iou:.3f}",
            f"{report.mean_runtime_s * 1000:.1f}",
            f"{report.p95_runtime_s * 1000:.1f}",
            f"{report.mean_rejected_proposals:.1f}",
        ))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
        if row_index == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def _signature(spec: Spec, layout: Mapping[str, Tuple[int, int, int, int]]) -> Tuple[int, ...]:
    values = []
    for obj in spec.objects:
        if obj.status == "new" and obj.id in layout:
            values.extend(layout[obj.id])
    return tuple(values)


def _distribution_diversity(spec: Spec, results: Sequence[SampleResult]) -> float:
    vectors = [_signature(spec, result.layout) for result in results]
    if len(vectors) < 2 or not vectors[0]:
        return 0.0
    distances = []
    for i, first in enumerate(vectors):
        for second in vectors[i + 1:]:
            distances.append(sum(abs(a - b) / 1000.0 for a, b in zip(first, second)) /
                             len(first))
    return _mean(distances)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]
