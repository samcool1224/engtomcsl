"""Build the deterministic development pilot for Evaluation 1.

The output is intentionally small enough to inspect by hand.  It is a development benchmark,
not the final paper test set.  The final set should add independently human-authored prompts and
be frozen before the parser prompt or exemplars are tuned against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rel(name, a, b=None, const=None):
    args = [a] if b is None else [a, b]
    return {"node": "rel", "name": name, "args": args, "const": const}


def typ(obj, value):
    return {"node": "type", "obj": obj, "type": value}


def default(obj):
    return {"node": "default", "obj": obj}


def prop(obj, value):
    return {"node": "property", "obj": obj, "value": value}


def conjunction(*args):
    flat = [arg for arg in args if arg is not None]
    return flat[0] if len(flat) == 1 else {"node": "and", "args": flat}


def choice(kind, span, formulas, *, emphasis=False):
    prior = round(1.0 / len(formulas), 6)
    return {
        "node": "choice",
        "kind": kind,
        "span": span,
        "emphasis": emphasis,
        "options": [
            {"prior": prior, "formula": formula, "skip": formula is None}
            for formula in formulas
        ],
    }


def existing(obj_id, obj_type, x, y, w=160, h=160):
    return {"id": obj_id, "status": "existing", "type": obj_type,
            "box": [x, y, w, h]}


def new(obj_id, obj_type=None, properties=None):
    record = {"id": obj_id, "status": "new"}
    if obj_type is not None:
        record["type"] = obj_type
    if properties:
        record["properties"] = list(properties)
    return record


def base_formula(obj_id, obj_type, relation, properties=None):
    nodes = [typ(obj_id, obj_type), default(obj_id), relation]
    nodes.extend(prop(obj_id, value) for value in (properties or []))
    return conjunction(*nodes)


def record(case_id, english, objects, formula, ambiguity_type="unambiguous",
           intended=None, template_family=None):
    return {
        "id": case_id,
        "source": "controlled_pilot",
        "split": "pilot_test",
        "template_family": template_family or ambiguity_type,
        "english": english,
        "objects": objects,
        "formula": formula,
        "ambiguous": ambiguity_type != "unambiguous",
        "ambiguity_type": ambiguity_type,
        "intended_options": ([{"kind": ambiguity_type, "formula": intended}]
                              if intended is not None else []),
    }


def build_records():
    rows = []

    # Unambiguous controls: exact directions, complete directions, size, alignment, absolute
    # position, properties, and a two-relation composition.
    unambiguous = [
        ("left", "Put a chair to the left of the desk.", "chair", "desk", "left", None),
        ("right", "Place the microwave to the right of the oven.", "microwave", "oven", "right", None),
        ("above", "Add a mirror above the couch.", "mirror", "couch", "above", None),
        ("below", "Set the toaster below the window.", "toaster", "window", "below", None),
        ("cleft", "Put the chair completely to the left of the dining table.", "chair", "dining table", "cleft", None),
        ("cright_gap", "Place the sink fully to the right of the refrigerator by 120 per-mille.", "sink", "refrigerator", "cright", 120),
        ("wider", "Make the couch wider than the bed.", "couch", "bed", "wider", None),
        ("shorter", "Add a toaster shorter than the microwave.", "toaster", "microwave", "shorter", None),
        ("xeq", "Vertically align the mirror with the door.", "mirror", "door", "xeq", None),
        ("yeq", "Horizontally align the desk with the window.", "desk", "window", "yeq", None),
    ]
    for index, (family, english, new_type, ref_type, relation_name, const) in enumerate(unambiguous, 1):
        objects = [existing("e0", ref_type, 420, 360), new("o0", new_type)]
        rows.append(record(
            f"u{index:03d}", english, objects,
            base_formula("o0", new_type, rel(relation_name, "o0", "e0", const)),
            template_family=family,
        ))
    rows.append(record(
        "u011", "Put a blue TV in the right half of the image.",
        [new("o0", "TV", ["blue"])],
        conjunction(typ("o0", "TV"), default("o0"), prop("o0", "blue"),
                    rel("right_value", "o0", const=500)),
        template_family="absolute_value",
    ))
    rows.append(record(
        "u012", "Add a chair left of the table and below the window.",
        [existing("e0", "dining table", 500, 500), existing("e1", "window", 300, 100),
         new("o0", "chair")],
        conjunction(typ("o0", "chair"), default("o0"),
                    rel("left", "o0", "e0"), rel("below", "o0", "e1")),
        template_family="relation_composition",
    ))

    # Vague direction.  "beside"/"alongside" are annotated as horizontal alternatives;
    # near/by/next-to permit the four cardinal placements.
    direction_cases = [
        ("Put a chair beside the couch.", "chair", "couch", "beside the couch", ["cleft", "cright"]),
        ("Place the lamp-sized mirror alongside the desk.", "mirror", "desk", "alongside the desk", ["cleft", "cright"]),
        ("Add a potted plant near the window.", "potted plant", "window", "near the window", ["cleft", "cright", "cabove", "cbelow"]),
        ("Put the microwave by the oven.", "microwave", "oven", "by the oven", ["cleft", "cright", "cabove", "cbelow"]),
        ("Set a toaster next to the sink.", "toaster", "sink", "next to the sink", ["cleft", "cright", "cabove", "cbelow"]),
        ("Position a TV beside the bed.", "TV", "bed", "beside the bed", ["cleft", "cright"]),
        ("Place a chair near the dining table.", "chair", "dining table", "near the dining table", ["cleft", "cright", "cabove", "cbelow"]),
        ("Add a mirror by the door.", "mirror", "door", "by the door", ["cleft", "cright", "cabove", "cbelow"]),
    ]
    for index, (english, new_type, ref_type, span, names) in enumerate(direction_cases, 1):
        objects = [existing("e0", ref_type, 430, 350), new("o0", new_type)]
        options = [rel(name, "o0", "e0") for name in names]
        ch = choice("direction", span, options)
        intended = options[(index - 1) % len(options)]
        rows.append(record(
            f"d{index:03d}", english, objects,
            conjunction(typ("o0", new_type), default("o0"), ch),
            "direction", intended, "vague_direction",
        ))

    offset_cases = [
        ("Put the chair far to the left of the table.", "chair", "dining table", "cleft", "far"),
        ("Place the microwave well to the right of the oven.", "microwave", "oven", "cright", "well"),
        ("Add a mirror way above the couch.", "mirror", "couch", "cabove", "way"),
        ("Set the toaster far below the window.", "toaster", "window", "cbelow", "far"),
        ("Move the desk well to the left of the bed.", "desk", "bed", "cleft", "well"),
        ("Position the potted plant way to the right of the door.", "potted plant", "door", "cright", "way"),
    ]
    for index, (english, new_type, ref_type, relation_name, span) in enumerate(offset_cases, 1):
        objects = [existing("e0", ref_type, 430, 350), new("o0", new_type)]
        options = [rel(relation_name, "o0", "e0", 0), rel(relation_name, "o0", "e0", 300)]
        ch = choice("offset", span, options, emphasis=True)
        rows.append(record(
            f"o{index:03d}", english, objects,
            conjunction(typ("o0", new_type), default("o0"), ch),
            "offset", options[index % 2], "vague_offset",
        ))

    reference_cases = [
        ("Add a mirror to the right of the chair.", "mirror", "chair", "right"),
        ("Put a toaster to the left of the microwave.", "toaster", "microwave", "left"),
        ("Place a desk below the window.", "desk", "window", "below"),
        ("Add a potted plant above the couch.", "potted plant", "couch", "above"),
        ("Position a TV completely right of the bed.", "TV", "bed", "cright"),
        ("Set a sink completely left of the refrigerator.", "sink", "refrigerator", "cleft"),
    ]
    for index, (english, new_type, ref_type, relation_name) in enumerate(reference_cases, 1):
        objects = [existing("e0", ref_type, 120, 350), existing("e1", ref_type, 680, 350),
                   new("o0", new_type)]
        options = [rel(relation_name, "o0", "e0"), rel(relation_name, "o0", "e1")]
        ch = choice("reference", f"the {ref_type}", options)
        rows.append(record(
            f"r{index:03d}", english, objects,
            conjunction(typ("o0", new_type), default("o0"), ch),
            "reference", options[(index - 1) % 2], "duplicate_reference",
        ))

    unsupported_cases = [
        ("Put a lamp to the left of the couch.", "lamp", "couch", ["potted plant", "mirror"]),
        ("Place a rug below the bed.", "rug", "bed", ["dining table", "couch"]),
        ("Add a vase above the desk.", "vase", "desk", ["potted plant", "blender"]),
        ("Put a bookshelf right of the door.", "bookshelf", "door", ["refrigerator", "desk"]),
        ("Position a fan above the window.", "fan", "window", ["mirror", "potted plant"]),
        ("Add a painting left of the TV.", "painting", "TV", ["mirror", "window"]),
    ]
    for index, (english, word, ref_type, alternatives) in enumerate(unsupported_cases, 1):
        objects = [existing("e0", ref_type, 430, 350), new("o0")]
        options = [typ("o0", alternatives[0]), typ("o0", alternatives[1]), None]
        ch = choice("unsupported_type", word, options)
        direction = "below" if " below " in english else "above" if " above " in english \
            else "right" if " right " in english else "left"
        rows.append(record(
            f"t{index:03d}", english, objects,
            conjunction(ch, default("o0"), rel(direction, "o0", "e0")),
            "unsupported_type", options[(index - 1) % 2], "unsupported_object_type",
        ))

    scope_cases = [
        ("Put a chair and a couch left of the TV.", "chair", "couch", "TV", "left"),
        ("Place a mirror and a desk below the window.", "mirror", "desk", "window", "below"),
        ("Add a toaster and a microwave right of the oven.", "toaster", "microwave", "oven", "right"),
        ("Position a bed and a potted plant above the couch.", "bed", "potted plant", "couch", "above"),
        ("Set a chair and a sink completely left of the refrigerator.", "chair", "sink", "refrigerator", "cleft"),
        ("Put a TV and a mirror completely right of the door.", "TV", "mirror", "door", "cright"),
    ]
    for index, (english, type0, type1, ref_type, relation_name) in enumerate(scope_cases, 1):
        objects = [existing("e0", ref_type, 430, 350), new("o0", type0), new("o1", type1)]
        nearest_only = rel(relation_name, "o1", "e0")
        both = conjunction(rel(relation_name, "o0", "e0"), nearest_only)
        ch = choice("scope", english.rstrip("."), [both, nearest_only])
        rows.append(record(
            f"s{index:03d}", english, objects,
            conjunction(typ("o0", type0), default("o0"), typ("o1", type1), default("o1"), ch),
            "scope", [both, nearest_only][index % 2], "coordination_scope",
        ))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("eval1_pilot.jsonl"))
    args = parser.parse_args()
    rows = build_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                           encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["ambiguity_type"]] = counts.get(row["ambiguity_type"], 0) + 1
    print(f"wrote {len(rows)} examples to {args.output}")
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()

