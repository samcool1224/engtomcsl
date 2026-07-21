"""Offline tests for the from-scratch English-to-image vertical slice."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (And, Default, EnglishToImageSystem, GligenImageGenerator, Obj,
                  PropertyPred, Relation, Spec, StubBackend, TypePred,
                  extract_new_objects, layout_to_grounded_scene)


def test_extracts_supported_objects_properties_and_dedupes_references():
    english = ("Create a room with a dining table and a blue chair completely left of "
               "the dining table, with a potted plant right of the table.")
    assert extract_new_objects(english) == [
        {"id": "o0", "status": "new", "type": "dining table"},
        {"id": "o1", "status": "new", "type": "chair", "properties": ["blue"]},
        {"id": "o2", "status": "new", "type": "potted plant"},
    ]


def test_extracts_small_explicit_counts():
    assert extract_new_objects("Create a room with two chairs and a table.") == [
        {"id": "o0", "status": "new", "type": "chair"},
        {"id": "o1", "status": "new", "type": "chair"},
        {"id": "o2", "status": "new", "type": "dining table"},
    ]


def test_layout_converts_to_normalized_xyxy_boxes_and_phrases():
    spec = Spec(
        [Obj("o0", "new", "chair", ["blue"])],
        And([TypePred("o0", "chair"), PropertyPred("o0", "blue"), Default("o0")]),
    )
    scene = layout_to_grounded_scene(
        spec, {"o0": (100, 200, 300, 400)}, "Add a blue chair."
    )
    assert scene.object_ids == ("o0",)
    assert scene.phrases == ("a blue chair",)
    assert scene.boxes == ((0.1, 0.2, 0.4, 0.6),)


class _FakeImage:
    pass


class _FakeOutput:
    images = [_FakeImage()]


class _FakePipe:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _FakeOutput()


def test_gligen_adapter_passes_exact_grounding_without_loading_model():
    spec = Spec([Obj("o0", "new", "chair")], Default("o0"))
    scene = layout_to_grounded_scene(spec, {"o0": (100, 200, 300, 400)}, "chair")
    fake = _FakePipe()
    image = GligenImageGenerator(pipe=fake).generate(scene, seed=9, num_inference_steps=12)
    assert isinstance(image, _FakeImage)
    assert fake.kwargs["gligen_phrases"] == ["a chair"]
    assert fake.kwargs["gligen_boxes"] == [[0.1, 0.2, 0.4, 0.6]]
    assert fake.kwargs["num_inference_steps"] == 12


class _FakeImageGenerator:
    def __init__(self):
        self.scene = None

    def generate(self, scene, **kwargs):
        self.scene = scene
        return _FakeImage()


def test_complete_pipeline_wires_english_objects_mscl_layout_and_image():
    english = "Add a blue chair completely to the left of a potted plant."
    formula = {"node": "and", "args": [
        {"node": "type", "obj": "o0", "type": "chair"},
        {"node": "property", "obj": "o0", "value": "blue"},
        {"node": "default", "obj": "o0"},
        {"node": "type", "obj": "o1", "type": "potted plant"},
        {"node": "default", "obj": "o1"},
        {"node": "rel", "name": "cleft", "args": ["o0", "o1"], "const": None},
    ]}
    fake_image_generator = _FakeImageGenerator()
    system = EnglishToImageSystem(
        StubBackend(gold={english: formula}), image_generator=fake_image_generator
    )
    result = system.generate(english, layout_seed=3, image_seed=4)
    chair = result.plan.sample.layout["o0"]
    plant = result.plan.sample.layout["o1"]
    assert chair[0] + chair[2] <= plant[0]
    assert fake_image_generator.scene == result.plan.grounding
    assert isinstance(result.image, _FakeImage)


def test_from_scratch_multirelation_error_is_repaired_before_layout():
    english = ("Create a modern dining room with a dining table, a blue chair completely "
               "to the left of the dining table, and a potted plant completely to the "
               "right of the dining table.")
    # Reproduce the failure observed in Colab: plant direction flipped with a hidden 1000.
    bad_formula = {"node": "and", "args": [
        {"node": "rel", "name": "cleft", "args": ["o1", "o0"], "const": None},
        {"node": "rel", "name": "cleft", "args": ["o2", "o0"], "const": 1000},
    ]}
    system = EnglishToImageSystem(
        StubBackend(gold={english: bad_formula}), image_generator=_FakeImageGenerator()
    )
    plan = system.plan(english, layout_seed=8)
    table, chair, plant = (plan.sample.layout[k] for k in ("o0", "o1", "o2"))
    assert chair[0] + chair[2] <= table[0]
    assert plant[0] >= table[0] + table[2]


if __name__ == "__main__":
    tests = sorted((v for k, v in globals().items() if k.startswith("test_") and callable(v)),
                   key=lambda f: f.__name__)
    failures = 0
    for test in tests:
        try:
            test(); print("PASS", test.__name__)
        except Exception as exc:
            failures += 1; print("FAIL", test.__name__, "::", exc)
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
