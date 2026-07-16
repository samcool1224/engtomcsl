"""MSCL v1 — tests. Run: python -m pytest -q  (or python tests/test_core.py)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (Spec, Obj, Relation, TypePred, PropertyPred, Default,
                  And, Or, Not, Choice, Option, desugar, to_spring, from_spring,
                  spec_to_json, spec_from_json, model_check, feasible, collect_atoms,
                  init_domains, validate, assert_resolved, resolve, resolve_ask_none,
                  resolve_ask_all, json_schema, ALL_RELATIONS)
from mscl.relations import _BINARY, _UNARY
from mscl.postprocess import normalize_prediction
from mscl.parser import (_exemplars, _adaptive_exemplars, build_prompt,
                         StubBackend, DEFAULT_K_SHOT)
from mscl.json_io import dedupe_relations, parser_json_schema
from mscl.local_debug import parse_with_repair


# ---------------------------------------------------------------------------
def test_all_28_relations_present():
    assert len(ALL_RELATIONS) == 28, ALL_RELATIONS
    assert len(_BINARY) == 16 and len(_UNARY) == 12


def test_desugar_cright_matches_table1():
    # cright(o1,o2,c): X(o1) >= R(o2) + c  =>  x1 - x2 - w2 >= c
    a = desugar(Relation("cright", ["o1", "o2"], 100))
    d = {(o, p): c for c, o, p in a.terms}
    assert a.op == ">=" and a.const == 100
    assert d[("o1", "x")] == 1
    assert d[("o2", "x")] == -1 and d[("o2", "w")] == -1


def test_desugar_above_smaller_y():
    # above(o1,o2,c): Y1 <= Y2 - c  => y1 - y2 <= -c
    a = desugar(Relation("above", ["o1", "o2"], 30))
    d = {(o, p): c for c, o, p in a.terms}
    assert a.op == "<=" and a.const == -30
    assert d[("o1", "y")] == 1 and d[("o2", "y")] == -1


def test_desugar_value_unary():
    a = desugar(Relation("right_value", ["o1"], 500))   # X(o1) > 500
    assert a.op == ">" and a.const == 500
    assert a.terms == [(1, "o1", "x")]


def test_model_check_concrete():
    # oven at x=100, microwave at x=400 ; cright(m,oven) with c=0 needs x_m >= x_oven+w_oven
    M = {"oven": (100, 300, 150, 180), "m": (400, 300, 120, 120)}
    assert model_check(Relation("cright", ["m", "oven"]), M)       # 400 >= 100+150 ok
    M2 = {"oven": (100, 300, 150, 180), "m": (200, 300, 120, 120)}
    assert not model_check(Relation("cright", ["m", "oven"]), M2)  # 200 >= 250 false


def test_spring_string_roundtrip():
    f = And([Relation("cright", ["m", "o0"]),
             TypePred("m", "microwave"),
             PropertyPred("m", "blue"),
             Or([Relation("cleft", ["t", "o0"]), Relation("above", ["t", "o0"], 50)])])
    s = to_spring(f)
    g = from_spring(s)
    # re-render and compare canonical strings (structure preserved)
    assert to_spring(g) == s, (s, to_spring(g))


def test_json_roundtrip():
    spec = Spec(
        objects=[Obj("o0", "existing", "oven", box=(100, 300, 150, 180)),
                 Obj("m", "new", "microwave", ["blue"])],
        formula=And([Relation("cright", ["m", "o0"]), TypePred("m", "microwave"),
                     PropertyPred("m", "blue"), Default("m")]))
    j = spec_to_json(spec)
    spec2 = spec_from_json(j)
    assert spec_to_json(spec2) == j


def test_validate_arity_and_value():
    bad = Spec([Obj("a", "new")], Relation("cright", ["a"]))   # needs 2 args
    try:
        validate(bad); assert False, "should raise"
    except Exception:
        pass
    bad2 = Spec([Obj("a", "new")], Relation("right_value", ["a"]))  # needs const
    try:
        validate(bad2); assert False
    except Exception:
        pass


def test_prompt_default_covers_every_choice_kind():
    text = str([x["formula"] for x in _exemplars(DEFAULT_K_SHOT)])
    for kind in ("direction", "offset", "reference", "unsupported_type"):
        assert f"'kind': '{kind}'" in text


def test_postprocess_repairs_explicit_binary_relation():
    objects = [{"id": "e0", "status": "existing", "type": "chair", "box": [1, 2, 3, 4]},
               {"id": "o0", "status": "new", "type": "oven", "properties": ["red"]}]
    bad = {"objects": objects, "formula":
           {"node": "rel", "name": "shorter_value", "args": ["o0"], "const": 0}}
    out = normalize_prediction(bad, "Add a red oven, with a red oven shorter than a chair.", objects)
    assert {"node": "rel", "name": "shorter", "args": ["o0", "e0"], "const": None} in out["formula"]["args"]


def test_postprocess_repairs_multiple_explicit_complete_relations():
    objects = [
        {"id": "e0", "status": "existing", "type": "dining table",
         "box": [400, 350, 250, 180]},
        {"id": "o0", "status": "new", "type": "chair", "properties": ["blue"]},
        {"id": "o1", "status": "new", "type": "potted plant"},
    ]
    bad = {"objects": objects, "formula": {"node": "and", "args": [
        _rel_for_test("cleft", ["o0", "e0"], None),
        _rel_for_test("cleft", ["o1", "e0"], 1000),
    ]}}
    english = ("Add a blue chair and a potted plant, with the blue chair completely "
               "to the left of the dining table and the potted plant completely to "
               "the right of the dining table.")

    out = normalize_prediction(bad, english, objects)
    relations = [a for a in out["formula"]["args"] if a.get("node") == "rel"]

    assert relations == [
        _rel_for_test("cleft", ["o0", "e0"], None),
        _rel_for_test("cright", ["o1", "e0"], None),
    ]


def test_postprocess_emits_offset_choice_and_canonical_span():
    objects = [{"id": "e0", "status": "existing", "type": "microwave", "box": [1, 2, 3, 4]},
               {"id": "o0", "status": "new", "type": "oven"}]
    bad = {"objects": objects, "formula": _rel_for_test("cbelow", ["o0", "e0"], 300)}
    out = normalize_prediction(bad, "Put an oven well below the microwave.", objects)
    choices = [x for x in out["formula"]["args"] if x.get("node") == "choice"]
    assert len(choices) == 1 and choices[0]["kind"] == "offset"
    assert choices[0]["span"] == "well" and choices[0]["emphasis"] is True


def test_postprocess_repairs_absolute_half_without_hidden_number():
    objects = [{"id": "o0", "status": "new", "type": "door"}]
    bad = {"objects": objects, "formula": _rel_for_test("cright", ["o0"], None)}
    out = normalize_prediction(bad, "Add a door, with a door in the right half of the image.", objects)
    assert {"node": "rel", "name": "right_value", "args": ["o0"], "const": 500} in out["formula"]["args"]


def test_postprocess_emits_full_vague_direction_set():
    objects = [{"id": "e0", "status": "existing", "type": "desk", "box": [1, 2, 3, 4]},
               {"id": "o0", "status": "new", "type": "TV"}]
    bad = {"objects": objects, "formula": _rel_for_test("cright", ["o0", "e0"], None)}
    out = normalize_prediction(bad, "Put a TV next to the desk.", objects)
    ch = [a for a in out["formula"]["args"] if a.get("kind") == "direction"][0]
    assert {o["formula"]["name"] for o in ch["options"]} == {"cleft", "cright", "cabove", "cbelow"}


def test_postprocess_wraps_duplicate_references_in_choice():
    objects = [{"id": "e0", "status": "existing", "type": "chair", "box": [1, 2, 3, 4]},
               {"id": "e1", "status": "existing", "type": "chair", "box": [5, 6, 7, 8]},
               {"id": "o0", "status": "new", "type": "mirror"}]
    bad = {"objects": objects, "formula": _rel_for_test("right", ["o0", "e0"], None)}
    out = normalize_prediction(bad, "Add a mirror to the right of the chair.", objects)
    ch = [a for a in out["formula"]["args"] if a.get("kind") == "reference"][0]
    assert {tuple(o["formula"]["args"]) for o in ch["options"]} == {("o0", "e0"), ("o0", "e1")}


def test_postprocess_adds_unsupported_type_choice():
    objects = [{"id": "e0", "status": "existing", "type": "window", "box": [1, 2, 3, 4]},
               {"id": "L", "status": "new", "type": None}]
    bad = {"objects": objects, "formula": _rel_for_test("cleft", ["L", "e0"], None)}
    out = normalize_prediction(bad, "Put a fan near the window.", objects)
    kinds = {a.get("kind") for a in out["formula"]["args"] if a.get("node") == "choice"}
    assert kinds == {"unsupported_type", "direction"}


def test_postprocess_treats_input_objects_as_authoritative():
    objects = [{"id": "o0", "status": "new", "type": "sink"}]
    bad = {"objects": [{"id": "invented", "status": "new", "type": "chair"}],
           "formula": {"node": "default", "obj": "invented"}}
    out = normalize_prediction(bad, "Add a sink.", objects)
    assert out["objects"] == objects
    assert out["formula"]["args"][0] == {"node": "type", "obj": "o0", "type": "sink"}


def test_dedupe_never_turns_and_args_into_a_dict():
    raw = {"objects": [], "formula": {"node": "and", "args": [
        _rel_for_test("left", ["a", "b"], None),
        _rel_for_test("left", ["a", "b"], None)]}}
    out = dedupe_relations(raw)
    assert out["formula"]["node"] == "rel"


def _rel_for_test(name, args, const):
    return {"node": "rel", "name": name, "args": args, "const": const}


def test_feasibility_sound_unsat():
    # left_value(a,100) AND right_value(a,800) on x: x<100 and x>800 -> UNSAT
    atoms = [desugar(Relation("left_value", ["a"], 100)),
             desugar(Relation("right_value", ["a"], 800))]
    dom = init_domains(["a"])
    assert feasible(atoms, dom) is False

def test_feasibility_sat():
    atoms = [desugar(Relation("left_value", ["a"], 800)),
             desugar(Relation("right_value", ["a"], 100))]  # 100 < x < 800
    dom = init_domains(["a"])
    assert feasible(atoms, dom) is True


# ---------------------------------------------------------------------------
# Dialogue policy
# ---------------------------------------------------------------------------
def _ambiguous_lamp_spec():
    # "put a lamp by the couch": unsupported_type + direction + offset CHOICEs
    objs = [Obj("c0", "existing", "couch", box=(400, 400, 300, 200)),
            Obj("L", "new", None)]
    direction = Choice("direction", "by the couch", [
        Option(0.4, Relation("cleft", ["L", "c0"])),
        Option(0.4, Relation("cright", ["L", "c0"])),
        Option(0.2, Relation("cabove", ["L", "c0"]))])
    utype = Choice("unsupported_type", "lamp", [
        Option(0.5, TypePred("L", "potted plant")),
        Option(0.3, TypePred("L", "mirror")),
        Option(0.2, None, skip=True)])
    offset = Choice("offset", "by", [
        Option(0.8, Relation("cleft", ["L", "c0"], 0)),
        Option(0.2, Relation("cleft", ["L", "c0"], 200))])
    return Spec(objs, And([utype, direction, offset]))


def test_offset_autoresolves_silently():
    spec = _ambiguous_lamp_spec()
    answers = iter([0, 0, 0])  # would pick first option if asked
    log = resolve(spec, oracle=lambda q: next(answers), budget=5)[1]
    asked_kinds = {q.kind for q in log.asked}
    # offset with no emphasis must NOT be asked
    assert "offset" not in asked_kinds, asked_kinds
    # offset should have been auto-resolved
    assert any(k == "offset" for k, _ in log.auto_resolved)


def test_structural_asked_before_geometric():
    spec = _ambiguous_lamp_spec()
    order = []
    def oracle(q):
        order.append(q.kind); return 0
    resolve(spec, oracle=oracle, budget=5)
    # unsupported_type must come before direction
    assert order.index("unsupported_type") < order.index("direction"), order


def test_budget_limits_questions():
    spec = _ambiguous_lamp_spec()
    n = {"c": 0}
    def oracle(q):
        n["c"] += 1; return 0
    spec, log = resolve(spec, oracle=oracle, budget=1)
    assert n["c"] == 1, n
    assert len(log.low_confidence) >= 1   # something resolved by prior


def test_ask_all_asks_more_than_policy():
    spec1 = _ambiguous_lamp_spec()
    spec2 = _ambiguous_lamp_spec()
    cnt = {"a": 0}
    resolve(spec1, oracle=lambda q: 0, budget=5)
    _, n_all = resolve_ask_all(spec2, oracle=lambda q: (cnt.__setitem__("a", cnt["a"]+1) or 0))
    # ASK_ALL confirms all 3 CHOICEs; policy should ask fewer (offset auto-resolved)
    assert n_all == 3


def test_resolved_spec_renders_to_spring():
    spec = _ambiguous_lamp_spec()
    resolve(spec, oracle=lambda q: 0, budget=5)
    assert_resolved(spec)            # no CHOICE remains
    s = to_spring(spec.formula)
    assert "lamp" not in s           # resolved to an in-vocab type
    assert from_spring(s) is not None


def test_json_schema_wellformed():
    sch = json_schema()
    assert sch["$defs"]["node"]["oneOf"]
    names = set()
    arities = {}
    for branch in sch["$defs"]["node"]["oneOf"]:
        if branch["properties"]["node"].get("const") == "rel":
            branch_names = branch["properties"]["name"]["enum"]
            names.update(branch_names)
            for name in branch_names:
                arities[name] = (branch["properties"]["args"]["minItems"],
                                 branch["properties"]["args"]["maxItems"])
    assert len(names) == 28
    assert arities["heq"] == (2, 2)
    assert arities["heq_value"] == (1, 1)


def test_parser_schema_is_formula_only():
    sch = parser_json_schema()
    assert sch["required"] == ["formula"]
    assert set(sch["properties"]) == {"formula"}


def test_adaptive_prompt_is_smaller_and_formula_only():
    objects = [{"id": "e0", "status": "existing", "type": "TV", "box": [1, 2, 3, 4]},
               {"id": "o0", "status": "new", "type": "bed"}]
    english = "Put a bed well above the TV."
    adaptive = build_prompt(english, objects, adaptive=True)
    full = build_prompt(english, objects, adaptive=False)
    assert len(_adaptive_exemplars(english, objects)) <= 4
    assert len(adaptive) < len(full) * 0.6
    assert 'OUTPUT:\n{"formula":' in adaptive


def test_parse_timing_separates_inference_and_checking():
    objects = [{"id": "o0", "status": "new", "type": "sink"}]
    backend = StubBackend({"Add a sink.": {"node": "and", "args": [
        {"node": "type", "obj": "o0", "type": "sink"},
        {"node": "default", "obj": "o0"}]}})
    _, timing = parse_with_repair("Add a sink.", objects, backend,
                                  retries=0, return_timing=True)
    assert timing.attempts == 1
    assert timing.inference_s >= 0 and timing.checking_s >= 0
    assert timing.total_s >= timing.inference_s


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
