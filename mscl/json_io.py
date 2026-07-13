"""MSCL v1 — JSON I/O.

This module defines the JSON form an English->MSCL model emits, and converts it
to/from the internal AST. The JSON is intentionally discriminated (every node has
a "node" tag) so it can be enforced by JSON-schema- or grammar-constrained decoding.

JSON node shapes
----------------
rel      : {"node":"rel","name":<str>,"args":[<id>...],"const":<int|null>}
type     : {"node":"type","obj":<id>,"type":<str>}
property : {"node":"property","obj":<id>,"value":<str>}
default  : {"node":"default","obj":<id>}
not      : {"node":"not","arg":<node>}
and      : {"node":"and","args":[<node>...]}
or       : {"node":"or","args":[<node>...]}
choice   : {"node":"choice","kind":<kind>,"span":<str>,"emphasis":<bool>,
            "options":[{"prior":<float>,"formula":<node|null>,"skip":<bool>}...]}

Top level
----------
{"objects":[{"id","status","type"?,"properties"?,"box"?}...], "formula":<node>}
"""
from __future__ import annotations
import json
from typing import Any, Dict
from .ast import (Spec, Obj, Atom, Relation, TypePred, PropertyPred, Default,
                  Not, And, Or, Choice, Option, Formula, CHOICE_KINDS)
from .relations import ALL_RELATIONS, arity, _BINARY, _UNARY
from .profile import SPRING_TYPES as _SPRING_TYPES


def spec_from_json(data: Dict[str, Any]) -> Spec:
    objs = [Obj(id=o["id"], status=o["status"], type=o.get("type"),
                properties=list(o.get("properties", [])),
                box=tuple(o["box"]) if o.get("box") else None)
            for o in data["objects"]]
    return Spec(objects=objs, formula=_node_from_json(data["formula"]))


def _node_from_json(n: Dict[str, Any]) -> Formula:
    k = n["node"]
    if k == "rel":
        return Relation(name=n["name"], args=list(n["args"]), const=n.get("const"))
    if k == "type":
        return TypePred(obj=n["obj"], type=n["type"])
    if k == "property":
        return PropertyPred(obj=n["obj"], value=n["value"])
    if k == "default":
        return Default(obj=n["obj"])
    if k == "not":
        return Not(arg=_node_from_json(n["arg"]))
    if k == "and":
        return And(args=[_node_from_json(a) for a in n["args"]])
    if k == "or":
        return Or(args=[_node_from_json(a) for a in n["args"]])
    if k == "choice":
        opts = [Option(prior=float(o["prior"]),
                       formula=(_node_from_json(o["formula"]) if o.get("formula") else None),
                       skip=bool(o.get("skip", False)))
                for o in n["options"]]
        return Choice(kind=n["kind"], span=n.get("span", ""),
                      options=opts, emphasis=bool(n.get("emphasis", False)))
    raise ValueError(f"unknown node tag: {k}")


def node_to_json(f: Formula) -> Dict[str, Any]:
    if isinstance(f, Relation):
        return {"node": "rel", "name": f.name, "args": f.args, "const": f.const}
    if isinstance(f, TypePred):
        return {"node": "type", "obj": f.obj, "type": f.type}
    if isinstance(f, PropertyPred):
        return {"node": "property", "obj": f.obj, "value": f.value}
    if isinstance(f, Default):
        return {"node": "default", "obj": f.obj}
    if isinstance(f, Not):
        return {"node": "not", "arg": node_to_json(f.arg)}
    if isinstance(f, And):
        return {"node": "and", "args": [node_to_json(a) for a in f.args]}
    if isinstance(f, Or):
        return {"node": "or", "args": [node_to_json(a) for a in f.args]}
    if isinstance(f, Choice):
        return {"node": "choice", "kind": f.kind, "span": f.span,
                "emphasis": f.emphasis,
                "options": [{"prior": o.prior,
                             "formula": (node_to_json(o.formula) if o.formula else None),
                             "skip": o.skip} for o in f.options]}
    raise TypeError(f"cannot serialize {type(f)}")


def spec_to_json(s: Spec) -> Dict[str, Any]:
    return {
        "objects": [
            {"id": o.id, "status": o.status,
             **({"type": o.type} if o.type is not None else {}),
             **({"properties": o.properties} if o.properties else {}),
             **({"box": list(o.box)} if o.box else {})}
            for o in s.objects
        ],
        "formula": node_to_json(s.formula),
    }


# ---------------------------------------------------------------------------
# JSON Schema for grammar-constrained decoding.
# This is the machine-readable contract the LLM decodes against (Outlines / xgrammar /
# structured-output APIs). It guarantees: valid node tags, valid relation names,
# relation arity is enforced during constrained decoding by separate binary/unary branches.
# ---------------------------------------------------------------------------
def json_schema() -> Dict[str, Any]:
    node_ref = {"$ref": "#/$defs/node"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "MSCL-SPRING spec",
        "type": "object",
        "required": ["objects", "formula"],
        "additionalProperties": False,
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "status"],
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"enum": ["existing", "new"]},
                        "type": {"type": ["string", "null"]},
                        "properties": {"type": "array", "items": {"type": "string"}},
                        "box": {"type": "array", "items": {"type": "integer"},
                                "minItems": 4, "maxItems": 4},
                    },
                },
            },
            "formula": node_ref,
        },
        "$defs": {
            "node": {
                "oneOf": [
                    {"type": "object", "required": ["node", "name", "args"],
                     "additionalProperties": False,
                     "properties": {
                         "node": {"const": "rel"},
                         "name": {"enum": list(_BINARY)},
                         "args": {"type": "array", "items": {"type": "string"},
                                  "minItems": 2, "maxItems": 2},
                         "const": {"type": ["integer", "null"]}}},
                    {"type": "object", "required": ["node", "name", "args", "const"],
                     "additionalProperties": False,
                     "properties": {
                         "node": {"const": "rel"},
                         "name": {"enum": list(_UNARY)},
                         "args": {"type": "array", "items": {"type": "string"},
                                  "minItems": 1, "maxItems": 1},
                         "const": {"type": "integer"}}},
                    {"type": "object", "required": ["node", "obj", "type"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "type"},
                                    "obj": {"type": "string"},
                                    "type": {"enum": list(_SPRING_TYPES)}}},
                    {"type": "object", "required": ["node", "obj", "value"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "property"},
                                    "obj": {"type": "string"},
                                    "value": {"type": "string"}}},
                    {"type": "object", "required": ["node", "obj"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "default"},
                                    "obj": {"type": "string"}}},
                    {"type": "object", "required": ["node", "arg"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "not"}, "arg": node_ref}},
                    {"type": "object", "required": ["node", "args"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "and"},
                                    "args": {"type": "array", "items": node_ref,
                                             "minItems": 1}}},
                    {"type": "object", "required": ["node", "args"],
                     "additionalProperties": False,
                     "properties": {"node": {"const": "or"},
                                    "args": {"type": "array", "items": node_ref,
                                             "minItems": 2}}},
                    {"type": "object", "required": ["node", "kind", "options"],
                     "additionalProperties": False,
                     "properties": {
                         "node": {"const": "choice"},
                         "kind": {"enum": list(CHOICE_KINDS)},
                         "span": {"type": "string"},
                         "emphasis": {"type": "boolean"},
                         "options": {"type": "array", "minItems": 2, "items": {
                             "type": "object", "required": ["prior"],
                             "additionalProperties": False,
                             "properties": {"prior": {"type": "number"},
                                            "formula": {"oneOf": [node_ref, {"type": "null"}]},
                                            "skip": {"type": "boolean"}}}}}},
                ]
            }
        },
    }


def parser_json_schema() -> Dict[str, Any]:
    """Compact constrained-output schema used by the few-shot parser.

    The input object table is authoritative and normalize_prediction() restores it after
    generation, so asking the model to copy it wastes output tokens.  The public/full schema
    returned by json_schema() remains unchanged for stored specs and external integrations.
    """
    full = json_schema()
    return {
        "$schema": full["$schema"],
        "title": "MSCL-SPRING parser output",
        "type": "object",
        "required": ["formula"],
        "additionalProperties": False,
        "properties": {"formula": {"$ref": "#/$defs/node"}},
        "$defs": full["$defs"],
    }


# ---------------------------------------------------------------------------
# Post-generation cleanup: remove duplicate / mirror-redundant relations that
# small models sometimes emit (pattern P3). Conservative: only removes provably
# redundant leaves inside a top-level AND; never changes meaning.
# ---------------------------------------------------------------------------
_MIRROR = {"above": "below", "below": "above", "left": "right", "right": "left",
           "cabove": "cbelow", "cbelow": "cabove", "cleft": "cright", "cright": "cleft"}


def dedupe_relations(spec_json: dict) -> dict:
    f = spec_json.get("formula")
    if not isinstance(f, dict) or f.get("node") != "and":
        return spec_json
    seen = set()
    kept = []
    # first pass: collect relation signatures to detect mirror pairs
    rel_sigs = set()
    for a in f["args"]:
        if a.get("node") == "rel":
            rel_sigs.add((a["name"], tuple(a.get("args", [])), a.get("const")))
    for a in f["args"]:
        if a.get("node") == "rel":
            sig = (a["name"], tuple(a.get("args", [])), a.get("const"))
            if sig in seen:
                continue                      # exact duplicate -> drop
            # mirror redundancy: cabove(x,y) present AND cbelow(y,x) present -> keep first only
            mname = _MIRROR.get(a["name"])
            if mname and len(a.get("args", [])) == 2:
                mirror_sig = (mname, (a["args"][1], a["args"][0]), a.get("const"))
                if mirror_sig in seen:
                    continue                  # its mirror already kept -> drop this one
            seen.add(sig)
            kept.append(a)
        else:
            kept.append(a)
    out = dict(spec_json)
    if len(kept) == 1:
        # Collapse the AND itself; never put a dict where the schema requires args: [...].
        out["formula"] = kept[0]
    else:
        f = dict(f); f["args"] = kept
        out["formula"] = f
    return out
