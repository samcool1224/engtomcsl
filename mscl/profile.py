"""MSCL v1 — the MSCL-SPRING profile (spec §9).

A profile fixes: value range, type vocabulary, allowed constructs, and macro expansions.
"""
from __future__ import annotations
from typing import List
from .ast import And, Relation, Default, Formula

# 17 SPRING types
SPRING_TYPES = (
    "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
    "desk", "toilet", "door", "TV", "microwave", "oven", "toaster", "sink",
    "refrigerator", "blender",
)

# per-mille coordinate range
COORD_MIN, COORD_MAX = 0, 1000

# default() macro: VEG-backbone box bounds, in per-mille of the image.
# NOTE: SPRING used 256-512px on a 512 canvas (=500-1000 per-mille), but those bounds are
# so large that two objects often cannot satisfy a "completely left/above" constraint.
# We use furniture-realistic bounds; the exact numbers are profile-tunable and should be
# reconciled with the reimplemented VEG once SampleSearch exists.
DEFAULT_BACKBONES = {
    "stable_diffusion": (150, 450),    # per-mille w,h bounds (15%-45% of image)
    "glide": (120, 350),
}


def expand_default(d: Default, backbone: str = "stable_diffusion") -> Formula:
    lo, hi = DEFAULT_BACKBONES[backbone]
    o = d.obj
    return And([
        Relation("wider_value",   [o], lo),  # W > lo  (approx >=; SampleSearch uses ints)
        Relation("narrower_value",[o], hi),  # W < hi
        Relation("taller_value",  [o], lo),  # H > lo
        Relation("shorter_value", [o], hi),  # H < hi
        # in-frame bounds
        Relation("right_value",   [o], COORD_MIN),  # X > 0  (left edge in frame)
        Relation("below_value",   [o], COORD_MIN),  # Y > 0
    ])


def is_supported_type(t: str) -> bool:
    return t in SPRING_TYPES


def nearest_types(word: str, k: int = 2) -> List[str]:
    """Cheap lexical fallback for unsupported-type CHOICE options.
    (A real system would use embedding similarity; this is a deterministic stub.)"""
    word = word.lower()
    scored = []
    for t in SPRING_TYPES:
        # crude overlap score
        s = len(set(word) & set(t.lower()))
        scored.append((s, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:k]]
