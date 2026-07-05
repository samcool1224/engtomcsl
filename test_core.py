"""MSCL v1 — tests. Run: python -m pytest -q  (or python tests/test_core.py)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (Spec, Obj, Relation, TypePred, PropertyPred, Default,
                  And, Or, Not, Choice, Option, desugar, to_spring, from_spring,
                  spec_to_json, spec_from_json, model_check, feasible, collect_atoms,
                  init_domains, validate, assert_resolved, resolve, resolve_ask_none,
                  resolve_ask_all, json_schema, ALL_RELATIONS)
from mscl.relations import _BINARY, _UNARY


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
    names = None
    for branch in sch["$defs"]["node"]["oneOf"]:
        if branch["properties"]["node"].get("const") == "rel":
            names = branch["properties"]["name"]["enum"]
    assert names and len(names) == 28


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
