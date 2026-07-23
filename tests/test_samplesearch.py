"""Exact solver and SampleSearch tests. Run with: python -m pytest -q"""
import os
import re
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (ALL_RELATIONS, And, Atom, Choice, Default, GeometricPreference, Not,
                  Obj, Option, Or, Relation, SampleSearch, Spec, TypePred,
                  UniformPreference, UnsatError, Z3Backend, generate_layout,
                  evaluate_sampler, format_comparison_table, model_check,
                  render_layout_grid_svg, render_layout_svg, spec_from_json)
from mscl.datagen import generate_dataset
from mscl.relations import NO_OFFSET, arity
from mscl.validate import ValidationError
import mscl.z3_backend as z3_backend_module


@contextmanager
def _raises(error_type, match=None):
    try:
        yield
    except error_type as error:
        if match is not None:
            assert re.search(match, str(error)), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}")


def _fixed_spec(formula, layout):
    objects = [Obj(oid, "existing", "chair", box=box) for oid, box in layout.items()]
    return Spec(objects, formula)


def test_z3_matches_model_check_for_all_28_relations():
    layouts = [
        {"a": (100, 120, 180, 220), "b": (500, 520, 210, 170)},
        {"a": (600, 80, 150, 300), "b": (200, 400, 300, 140)},
        {"a": (350, 350, 200, 200), "b": (350, 350, 200, 200)},
    ]
    for name in ALL_RELATIONS:
        args = ["a", "b"] if arity(name) == 2 else ["a"]
        const = None if name in NO_OFFSET else (40 if arity(name) == 2 else 500)
        relation = Relation(name, args, const)
        for layout in layouts:
            backend = Z3Backend(_fixed_spec(relation, layout))
            assert backend.is_satisfiable() == model_check(relation, layout), (name, layout)


def test_z3_preserves_boolean_semantics():
    layout = {"a": (100, 100, 150, 150), "b": (600, 600, 150, 150)}
    formula = And([
        Or([Relation("cleft", ["a", "b"]), Relation("cabove", ["a", "b"])]),
        Not(Relation("cright", ["a", "b"])),
    ])
    backend = Z3Backend(_fixed_spec(formula, layout))
    assert backend.is_satisfiable()
    assert backend.verify_layout(layout)


def test_layer_one_atom_is_accepted_and_solved_exactly():
    # x(a) + w(a) <= 400
    atom = Atom([(1, "a", "x"), (1, "a", "w")], "<=", 400)
    spec = Spec([Obj("a", "new", "chair")], atom)
    result = SampleSearch(UniformPreference()).sample(spec, seed=12)
    assert result.layout["a"][0] + result.layout["a"][2] <= 400


def test_samplesearch_returns_valid_in_frame_layout_and_timings():
    spec = Spec(
        [Obj("table", "existing", "dining table", box=(400, 350, 250, 180)),
         Obj("plant", "new", "potted plant")],
        And([TypePred("plant", "potted plant"), Default("plant"),
             Relation("cleft", ["plant", "table"], 20)]),
    )
    result = generate_layout(spec, seed=7)
    x, y, w, h = result.layout["plant"]
    assert x + w <= 1000 and y + h <= 1000
    assert model_check(spec.formula, result.layout)
    assert result.layout["table"] == (400, 350, 250, 180)
    assert result.stats.solver_checks > 0
    assert result.stats.solver_time_s >= 0
    assert result.stats.preference_time_s >= 0
    assert result.stats.verification_time_s >= 0
    assert result.stats.total_time_s >= result.stats.solver_time_s
    assert result.trace


def test_samplesearch_seed_is_reproducible_and_samples_are_diverse():
    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    search = SampleSearch(UniformPreference())
    first = search.sample(spec, seed=123)
    second = search.sample(spec, seed=123)
    assert first.layout == second.layout
    assert [(e.variable, e.chosen) for e in first.trace] == [
        (e.variable, e.chosen) for e in second.trace]
    layouts = {tuple(search.sample(spec, seed=i).layout["a"]) for i in range(8)}
    assert len(layouts) > 1


def test_sample_many_uses_repeatable_child_seeds():
    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    search = SampleSearch(GeometricPreference())
    one = search.sample_many(spec, 3, seed=99)
    two = search.sample_many(spec, 3, seed=99)
    assert [r.stats.seed for r in one] == [r.stats.seed for r in two]
    assert [r.layout for r in one] == [r.layout for r in two]


def test_unsat_spec_raises_with_exact_core():
    spec = Spec([Obj("a", "new", "chair")], And([
        Relation("left_value", ["a"], 100),
        Relation("right_value", ["a"], 800),
    ]))
    caught = None
    try:
        SampleSearch().sample(spec, seed=1)
    except UnsatError as error:
        caught = error
    assert caught is not None
    assert caught.explanation.constraints
    assert any("formula" in c for c in caught.explanation.constraints)


def test_samplesearch_requires_choice_resolution_first():
    choice = Choice("direction", "near", [
        Option(0.5, Relation("left_value", ["a"], 500)),
        Option(0.5, Relation("right_value", ["a"], 500)),
    ])
    spec = Spec([Obj("a", "new", "chair")], choice)
    with _raises(ValidationError, match="CHOICE"):
        SampleSearch().sample(spec, seed=1)


def test_existing_out_of_frame_is_rejected_by_solver():
    spec = Spec([Obj("a", "existing", "chair", box=(900, 100, 200, 200))],
                TypePred("a", "chair"))
    with _raises(UnsatError):
        SampleSearch().sample(spec, seed=1)


def test_invalid_object_state_is_rejected_before_search():
    missing_box = Spec([Obj("a", "existing", "chair")], TypePred("a", "chair"))
    with _raises(ValidationError, match="observed box"):
        SampleSearch().sample(missing_box, seed=1)

    conflicting_type = Spec([Obj("a", "new", "chair")], TypePred("a", "couch"))
    with _raises(ValidationError, match="conflicts"):
        SampleSearch().sample(conflicting_type, seed=1)


def test_zero_weight_guide_cannot_remove_feasible_support():
    class ZeroGuide:
        def branch_weights(self, **kwargs):
            return [0.0 for _ in kwargs["branches"]]

    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    result = SampleSearch(ZeroGuide()).sample(spec, seed=4)
    assert model_check(spec.formula, result.layout)


def test_new_dataset_generation_uses_exact_satisfiability_filter():
    samples = generate_dataset(40, ambiguous_frac=0.5, seed=31)
    assert len(samples) == 40
    for sample in samples:
        spec = spec_from_json(sample.gold_json)
        assert Z3Backend(spec).is_satisfiable(), sample.english


def test_z3_loader_recovers_after_late_notebook_install():
    installed = z3_backend_module.z3
    z3_backend_module.z3 = None
    try:
        assert z3_backend_module.z3_available()
        assert z3_backend_module._require_z3() is not None
    finally:
        z3_backend_module.z3 = installed


def test_geometric_exploration_is_one_global_mixture():
    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    guide = GeometricPreference(exploration=1.0)
    root = guide.branch_weights(spec=spec, variable=("a", "w"),
                                branches=((1, 500), (501, 1000)), assigned={})
    lower_children = guide.branch_weights(spec=spec, variable=("a", "w"),
                                          branches=((1, 250), (251, 500)), assigned={})
    assert abs(sum(root) - 1.0) < 1e-12
    assert abs(sum(lower_children) - root[0]) < 1e-12


def test_geometric_preference_aligns_explicit_horizontal_relations():
    spec = Spec(
        [
            Obj("table", "new", "dining table"),
            Obj("chair", "new", "chair"),
            Obj("plant", "new", "potted plant"),
        ],
        And([
            Default("table"), Default("chair"), Default("plant"),
            Relation("cleft", ["chair", "table"]),
            Relation("cright", ["plant", "table"]),
        ]),
    )
    candidates = SampleSearch().sample_many(spec, 20, seed=42)
    bottom_spreads = []
    for result in candidates:
        bottoms = [y + h for _, y, _, h in result.layout.values()]
        bottom_spreads.append(max(bottoms) - min(bottoms))
    assert sum(bottom_spreads) / len(bottom_spreads) < 180


def test_layout_visualization_renders_single_and_grid_svg():
    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    result = SampleSearch().sample(spec, seed=2)
    single = render_layout_svg(spec, result.layout, title="chair sample")
    grid = render_layout_grid_svg(spec, [result.layout, result.layout], columns=2)
    assert single.startswith("<svg") and "chair sample" in single
    assert "a: chair" in single and "<rect" in single
    assert grid.startswith("<svg") and "sample 2" in grid


def test_evaluator_reports_validity_diversity_and_preference_fit():
    spec = Spec([Obj("a", "new", "chair")], Default("a"))
    uniform = evaluate_sampler(spec, SampleSearch(UniformPreference()),
                               count=24, seed=42, label="uniform")
    geometric = evaluate_sampler(spec, SampleSearch(GeometricPreference()),
                                 count=24, seed=42, label="geometric")
    assert uniform.validity_rate == geometric.validity_rate == 1.0
    assert uniform.unique_count == geometric.unique_count == 24
    assert geometric.mean_typical_size_error < uniform.mean_typical_size_error
    assert geometric.mean_diversity > 0
    table = format_comparison_table([uniform, geometric])
    assert "size err" in table and "geometric" in table


if __name__ == "__main__":
    import traceback
    functions = [v for k, v in sorted(globals().items())
                 if k.startswith("test_") and callable(v)]
    passed = 0
    for function in functions:
        try:
            function()
            passed += 1
            print(f"PASS {function.__name__}")
        except Exception as error:
            print(f"FAIL {function.__name__}: {error}")
            traceback.print_exc()
    print(f"\n{passed}/{len(functions)} passed")
    raise SystemExit(0 if passed == len(functions) else 1)
