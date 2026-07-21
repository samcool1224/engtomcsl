"""Deterministic object inventory extraction for from-scratch scene generation.

The semantic parser intentionally treats its object table as authoritative.  This module
builds that table directly from English for the supported SPRING vocabulary, removing the
last manual JSON input from the new-scene pipeline.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

from .profile import SPRING_TYPES


class ObjectPlanError(ValueError):
    """Raised when an instruction cannot be converted into a supported object inventory."""


OBJECT_PROPERTIES = (
    "red", "blue", "green", "black", "white", "brown", "wooden", "metal",
)

_ALIASES = {
    "plant": "potted plant",
    "potted plants": "potted plant",
    "chairs": "chair",
    "couches": "couch",
    "sofa": "couch",
    "sofas": "couch",
    "beds": "bed",
    "mirrors": "mirror",
    "dining tables": "dining table",
    "table": "dining table",
    "tables": "dining table",
    "windows": "window",
    "desks": "desk",
    "toilets": "toilet",
    "doors": "door",
    "television": "TV",
    "televisions": "TV",
    "tvs": "TV",
    "microwaves": "microwave",
    "ovens": "oven",
    "toasters": "toaster",
    "sinks": "sink",
    "fridge": "refrigerator",
    "fridges": "refrigerator",
    "refrigerators": "refrigerator",
    "blenders": "blender",
}

_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4}


def _aliases() -> Dict[str, str]:
    out = {typ.lower(): typ for typ in SPRING_TYPES}
    out.update(_ALIASES)
    return out


def _nonoverlapping_mentions(text: str) -> List[Tuple[int, int, str]]:
    candidates = []
    for alias, canonical in _aliases().items():
        for match in re.finditer(
                rf"(?<![a-z0-9-]){re.escape(alias)}(?![a-z0-9-])", text):
            candidates.append((match.start(), match.end(), canonical, len(alias)))
    # Prefer longer aliases at the same location so "potted plant" does not also become
    # a second object through its shorter "plant" alias.
    candidates.sort(key=lambda m: (m[0], -m[3], m[1]))
    kept = []
    occupied_until = -1
    for start, end, canonical, _ in candidates:
        if start < occupied_until:
            continue
        kept.append((start, end, canonical))
        occupied_until = end
    return kept


def _properties_before(text: str, start: int) -> List[str]:
    prefix = text[max(0, start - 48):start]
    words = re.findall(r"[a-z-]+", prefix)[-4:]
    return [p for p in OBJECT_PROPERTIES if p in words]


def _explicit_count_before(text: str, start: int) -> int:
    words = re.findall(r"[a-z0-9]+", text[max(0, start - 24):start])[-2:]
    if not words:
        return 1
    token = words[-1]
    if token.isdigit():
        return max(1, min(int(token), 4))
    return _COUNTS.get(token, 1)


def extract_new_objects(english: str) -> List[dict]:
    """Extract an all-new object table from an English instruction.

    The MVP supports the 17 SPRING object types, common singular/plural aliases, the eight
    trained color/material properties, and explicit counts from one through four.  Repeated
    mentions of a type are treated as references to the same planned object unless the first
    mention explicitly gives a count.
    """
    if not isinstance(english, str) or not english.strip():
        raise ObjectPlanError("English instruction must be a non-empty string")
    text = english.lower()
    mentions = _nonoverlapping_mentions(text)
    if not mentions:
        supported = ", ".join(SPRING_TYPES)
        raise ObjectPlanError(
            "No supported interior object was found. Supported types are: " + supported
        )

    first_by_type: Dict[str, Tuple[int, int]] = {}
    for start, end, canonical in mentions:
        first_by_type.setdefault(canonical, (start, end))

    ordered = sorted(first_by_type.items(), key=lambda item: item[1][0])
    objects = []
    for canonical, (start, _) in ordered:
        count = _explicit_count_before(text, start)
        properties = _properties_before(text, start)
        for _ in range(count):
            obj = {
                "id": f"o{len(objects)}",
                "status": "new",
                "type": canonical,
            }
            if properties:
                obj["properties"] = list(properties)
            objects.append(obj)
    return objects


def object_phrase(obj) -> str:
    """Return the localized text phrase supplied to the grounded image model."""
    properties: Sequence[str] = (
        obj.properties if hasattr(obj, "properties") else obj.get("properties", [])
    )
    typ = obj.type if hasattr(obj, "type") else obj.get("type")
    words = [str(p) for p in properties] + [str(typ)]
    return "a " + " ".join(word for word in words if word and word != "None")
