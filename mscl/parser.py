"""MSCL v1 — Pieces 1 & 2: prompt harness + pluggable English->MSCL parser.

`build_prompt(english, objects)` -> the full prompt string for the model.
`parse(english, objects, backend=...)` -> Spec, using a swappable backend.

Backends:
  StubBackend  : deterministic, no GPU — echoes a gold formula when given one, else a
                 trivial guess. Lets the WHOLE pipeline + eval run in this environment.
  LocalBackend : runs a local HF model with grammar-constrained decoding via Outlines.
                 You run this on your GPU; code included, import guarded.
"""
from __future__ import annotations
import json, os
from typing import List, Dict, Optional, Callable, Any
from .ast import Spec
from .json_io import spec_from_json, json_schema
from .validate import validate

HERE = os.path.dirname(os.path.abspath(__file__))
_SCHEMA = json_schema()

SYSTEM = """You translate an interior-design instruction in English into MSCL-SPRING JSON.

Rules:
- Output ONE JSON object: {"objects":[...], "formula": <node>}. No prose.
- Object types MUST be one of the 17 SPRING types. If the user names something else,
  emit a CHOICE node with kind "unsupported_type" offering the nearest types and a SKIP.
- Bind references to the given detected objects when they match by type (and position).
- Spatial relations use these names only: above below left right cabove cbelow cleft
  cright wider narrower taller shorter xeq yeq weq heq, and the *_value forms.
- When the English is genuinely ambiguous, DO NOT guess silently — emit a CHOICE node with
  the right kind (direction / offset / reference / scope / unsupported_type), candidate
  option formulas, and a prior weight per option (summing to 1).
- Integer offsets are in per-mille (0..1000) of image width/height.
"""


def _exemplars(k: int = 4) -> List[dict]:
    pairs = json.load(open(os.path.join(HERE, "..", "examples", "seed_pairs.json")))["pairs"]
    return pairs[:k]


def build_prompt(english: str, objects: List[dict], k_shot: int = 4,
                 include_schema: bool = True) -> str:
    parts = [SYSTEM]
    if include_schema:
        parts.append("JSON schema (the output must validate against this):\n"
                     + json.dumps(_SCHEMA))
    parts.append("Examples:")
    for ex in _exemplars(k_shot):
        ex_in = {"english": ex["english"], "objects": ex["objects"]}
        ex_out = {"objects": ex["objects"], "formula": ex["formula"]}
        parts.append("INPUT:\n" + json.dumps(ex_in))
        parts.append("OUTPUT:\n" + json.dumps(ex_out))
    parts.append("Now do this one.")
    parts.append("INPUT:\n" + json.dumps({"english": english, "objects": objects}))
    parts.append("OUTPUT:")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class StubBackend:
    """No-GPU backend. If a `gold` map {english: formula_json} is supplied, it returns the
    gold formula (useful to test the pipeline + dialogue policy + eval harness end-to-end
    without a model). Otherwise emits a minimal valid guess (types + default only)."""
    def __init__(self, gold: Optional[Dict[str, dict]] = None):
        self.gold = gold or {}

    def generate(self, prompt: str, english: str, objects: List[dict]) -> dict:
        if english in self.gold:
            return {"objects": objects, "formula": self.gold[english]}
        # trivial fallback: type + default for each new object, no constraints
        conj = []
        for o in objects:
            if o.get("status") == "new" and o.get("type"):
                conj.append({"node": "type", "obj": o["id"], "type": o["type"]})
                conj.append({"node": "default", "obj": o["id"]})
        formula = conj[0] if len(conj) == 1 else {"node": "and", "args": conj or [
            {"node": "default", "obj": objects[0]["id"]}]}
        return {"objects": objects, "formula": formula}


class LocalBackend:
    """GPU backend: local HF model + Outlines grammar-constrained JSON decoding.
    Import-guarded so this module loads without torch/outlines present."""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_tokens: int = 1024, temperature: float = 0.0):
        import outlines                      # noqa: F401  (raises if absent)
        from outlines import models
        from outlines.generate import text as generate
        self._models = models
        self._generate = generate
        self.model = models.transformers(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        # constrain to the JSON schema -> guaranteed well-formed output
        self.gen = generate.json(self.model, json.dumps(_SCHEMA))

    def generate(self, prompt: str, english: str, objects: List[dict]) -> dict:
        result = self.gen(prompt, max_tokens=self.max_tokens)
        return result if isinstance(result, dict) else json.loads(result)


Backend = Any  # anything with .generate(prompt, english, objects) -> dict


def parse(english: str, objects: List[dict], backend: Backend,
          k_shot: int = 4) -> Spec:
    prompt = build_prompt(english, objects, k_shot=k_shot)
    raw = backend.generate(prompt, english, objects)
    spec = spec_from_json(raw)
    validate(spec)            # raises on malformed; grammar-decoding should prevent this
    return spec
