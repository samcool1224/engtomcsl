"""Deterministic normalization for model-produced MSCL JSON.

The LLM is responsible for semantic parsing, but a few parts of the input contract are
fully observable and should not be left to sampling: the supplied object table, boilerplate
for new objects, explicit ambiguity cues, and the small controlled language used by the
synthetic diagnostic.  This module repairs only those cases.  It never consults gold output.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Dict, List, Optional


_COMPLETE = {"left": "cleft", "right": "cright", "above": "cabove", "below": "cbelow"}


def _is_atomic_instruction(english: str) -> bool:
    """True when the controlled sentence expresses at most one spatial relation."""
    lo = english.lower().rstrip(". ")
    if ", with " in lo:
        tail = lo.split(", with ", 1)[1]
        return "," not in tail and " and " not in tail
    # The ambiguous generators use a single "Put X ..." clause.
    return lo.startswith("put ") and "," not in lo and " and " not in lo


def _rel(name: str, args: List[str], const=None) -> dict:
    return {"node": "rel", "name": name, "args": args, "const": const}


def _choice(kind: str, span: str, formulas: List[dict], priors=None,
            emphasis: bool = False) -> dict:
    priors = priors or [1.0 / len(formulas)] * len(formulas)
    return {
        "node": "choice", "kind": kind, "span": span, "emphasis": emphasis,
        "options": [
            {"prior": round(float(p), 6), "formula": f, "skip": False}
            for p, f in zip(priors, formulas)
        ],
    }


def _top_args(formula: dict) -> List[dict]:
    if isinstance(formula, dict) and formula.get("node") == "and":
        return list(formula.get("args", []))
    return [formula] if isinstance(formula, dict) else []


def _formula(args: List[dict]) -> dict:
    return args[0] if len(args) == 1 else {"node": "and", "args": args}


def _boilerplate(objects: List[dict]) -> List[dict]:
    out = []
    for o in objects:
        if o.get("status") != "new":
            continue
        if o.get("type") is not None:
            out.append({"node": "type", "obj": o["id"], "type": o["type"]})
        for p in o.get("properties", []):
            out.append({"node": "property", "obj": o["id"], "value": p})
        out.append({"node": "default", "obj": o["id"]})
    return out


def _strip_and_restore_boilerplate(args: List[dict], objects: List[dict]) -> List[dict]:
    """Make object declarations exactly match the authoritative input object table."""
    semantic = [a for a in args if a.get("node") not in ("type", "property", "default")]
    return _boilerplate(objects) + semantic


def _ensure_unsupported_choice(args: List[dict], english: str,
                               objects: List[dict]) -> List[dict]:
    if any(a.get("node") == "choice" and a.get("kind") == "unsupported_type" for a in args):
        return args
    unknown = [o for o in objects if o.get("status") == "new" and o.get("type") is None]
    if len(unknown) != 1:
        return args
    m = re.search(r"\b(?:put|add)\s+(?:a|an)\s+([a-z][a-z-]*)\b", english.lower())
    if not m:
        return args
    word, oid = m.group(1), unknown[0]["id"]
    from .profile import nearest_types
    near = nearest_types(word, k=2)
    ch = {
        "node": "choice", "kind": "unsupported_type", "span": word,
        "emphasis": False, "options": [
            {"prior": 0.5, "formula": {"node": "type", "obj": oid, "type": near[0]}, "skip": False},
            {"prior": 0.3, "formula": {"node": "type", "obj": oid, "type": near[1]}, "skip": False},
            {"prior": 0.2, "formula": None, "skip": True},
        ],
    }
    return args + [ch]


def _single_explicit_relation(english: str, objects: List[dict]) -> Optional[dict]:
    """Parse the one-new/one-existing controlled diagnostic templates exactly.

    This deliberately does not handle free-form multi-relation input.  Its purpose is to
    prevent elementary lexical distinctions (partial/complete, binary/value, x/y alignment)
    from being corrupted by a small generative model.
    """
    if not _is_atomic_instruction(english):
        return None
    new = [o for o in objects if o.get("status") == "new" and o.get("type")]
    old = [o for o in objects if o.get("status") == "existing"]
    if len(new) != 1 or len(old) != 1:
        return None
    lo = english.lower()
    if re.search(r"\b(?:near|next to|beside)\b|\bby\s+the\b", lo):
        return None
    if re.search(r"\b(?:well|far|way)\b", lo):
        return None
    a, b = new[0]["id"], old[0]["id"]
    # The controlled generator can state a relation whose subject is the existing object
    # ("a toaster shorter than a bed").  Bind by the first type mention in the relation clause.
    clause = lo.split(", with ", 1)[1] if ", with " in lo else lo
    npos = clause.find(str(new[0].get("type", "")).lower())
    opos = clause.find(str(old[0].get("type", "")).lower())
    if opos >= 0 and (npos < 0 or opos < npos):
        a, b = b, a
    offset_m = re.search(r"\bby\s+(\d+)\s+per[- ]mille\b", lo)
    const = int(offset_m.group(1)) if offset_m else None

    for word in ("left", "right", "above", "below"):
        if re.search(rf"\b{word}\b", lo):
            complete = bool(re.search(rf"\b(?:completely|fully)\b[^,.]*\b{word}\b", lo))
            return _rel(_COMPLETE[word] if complete else word, [a, b], const)
    for word in ("wider", "narrower", "taller", "shorter"):
        if re.search(rf"\b{word}\s+than\s+(?:an?|the)\b", lo):
            return _rel(word, [a, b])
    if "horizontally aligned" in lo:
        return _rel("yeq", [a, b])
    if "vertically aligned" in lo:
        return _rel("xeq", [a, b])
    if "same width" in lo:
        return _rel("weq", [a, b])
    if "same height" in lo:
        return _rel("heq", [a, b])
    return None


def _absolute_relation(english: str, objects: List[dict]) -> Optional[dict]:
    if not _is_atomic_instruction(english):
        return None
    new = [o for o in objects if o.get("status") == "new" and o.get("type")]
    if len(new) != 1:
        return None
    lo = english.lower()
    oid = new[0]["id"]
    # "half" and the generator's legacy "part" have a documented midpoint convention.
    zones = {"right": "right_value", "left": "left_value",
             "top": "above_value", "bottom": "below_value"}
    for word, name in zones.items():
        if re.search(rf"\bin the {word} (?:half|part) of the image\b", lo):
            return _rel(name, [oid], 500)
    m = re.search(r"\b(wider|narrower|taller|shorter) than (\d+) per[- ]mille\b", lo)
    if m:
        return _rel(m.group(1) + "_value", [oid], int(m.group(2)))
    return None


def _explicit_ambiguity(english: str, objects: List[dict]) -> Optional[dict]:
    lo = english.lower()
    new = [o for o in objects if o.get("status") == "new"]
    old = [o for o in objects if o.get("status") == "existing"]
    if len(new) != 1:
        return None
    a = new[0]["id"]

    # Ambiguous reference: one linguistic description, two detected candidates.
    if len(old) >= 2:
        for typ in sorted({o.get("type") for o in old if o.get("type")}, key=len, reverse=True):
            candidates = [o["id"] for o in old if o.get("type") == typ]
            if len(candidates) >= 2 and f"the {typ.lower()}" in lo:
                word = next((w for w in ("left", "right", "above", "below")
                             if re.search(rf"\b{w}\b", lo)), None)
                if word:
                    name = word
                    return _choice("reference", f"the {typ}",
                                   [_rel(name, [a, oid]) for oid in candidates],
                                   [1.0 / len(candidates)] * len(candidates))

    if len(old) != 1:
        return None
    b = old[0]["id"]
    cue = re.search(r"\b(well|far|way)\b", lo)
    if cue:
        word = next((w for w in ("left", "right", "above", "below")
                     if re.search(rf"\b{w}\b", lo)), None)
        if word:
            name = _COMPLETE[word]
            return _choice("offset", cue.group(1),
                           [_rel(name, [a, b], 0), _rel(name, [a, b], 300)],
                           [0.4, 0.6], emphasis=True)

    vague = re.search(r"\b(near|next to|beside)\b|\b(by)\s+the\b", lo)
    if vague:
        phrase = (vague.group(1) or vague.group(2)).strip()
        span = f"{phrase} the {old[0].get('type')}"
        names = ["cabove", "cbelow", "cright", "cleft"]
        return _choice("direction", span, [_rel(n, [a, b]) for n in names],
                       [0.25, 0.25, 0.25, 0.25])
    return None


def normalize_prediction(spec_json: Dict, english: str, objects: List[dict]) -> Dict:
    """Return a validated-shape prediction normalized against observable input facts."""
    raw = deepcopy(spec_json) if isinstance(spec_json, dict) else {}
    args = _top_args(raw.get("formula", {}))
    args = _strip_and_restore_boilerplate(args, objects)
    args = _ensure_unsupported_choice(args, english, objects)

    ambiguity = _explicit_ambiguity(english, objects)
    if ambiguity is not None:
        # For the diagnostic's atomic ambiguity templates, replace model guesses/duplicate
        # relations with the cue-determined CHOICE. Preserve unsupported-type CHOICE nodes.
        semantic = [a for a in args
                    if a.get("node") != "rel" and
                    not (a.get("node") == "choice" and a.get("kind") != "unsupported_type")]
        args = semantic + [ambiguity]
    else:
        explicit = _absolute_relation(english, objects) or _single_explicit_relation(english, objects)
        if explicit is not None:
            args = [a for a in args if a.get("node") != "rel"] + [explicit]

    # Stable de-duplication for boilerplate and leaves.
    seen, kept = set(), []
    for a in args:
        key = repr(a)
        if key not in seen:
            seen.add(key); kept.append(a)
    return {"objects": deepcopy(objects), "formula": _formula(kept)}
