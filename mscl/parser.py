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

Output ONE JSON object: {"objects":[...], "formula": <node>}. No prose, no markdown.

THE ONLY LEGAL OBJECT TYPES (use these EXACT strings, including spaces and capitalization):
  "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
  "toilet", "door", "TV", "microwave", "oven", "toaster", "sink", "refrigerator", "blender"
- Never invent a type. "plant" is WRONG; the legal type is "potted plant". "tv" is WRONG; use "TV".
- If the user names an object NOT in this list (e.g. "lamp", "fan", "rug", "clock"), you MUST NOT
  emit type(o,"lamp"). Instead emit a CHOICE with kind "unsupported_type" offering the 2 nearest
  legal types and a SKIP option, e.g.:
    {"node":"choice","kind":"unsupported_type","span":"lamp","options":[
       {"prior":0.5,"formula":{"node":"type","obj":"o0","type":"potted plant"}},
       {"prior":0.3,"formula":{"node":"type","obj":"o0","type":"mirror"}},
       {"prior":0.2,"formula":null,"skip":true}]}

EMIT A CHOICE WHENEVER THE ENGLISH IS AMBIGUOUS — do NOT silently pick one reading:
- Vague direction ("by", "near", "next to", "beside" the X): kind "direction", 3-4 options over
  cleft/cright/cabove/cbelow. Do NOT emit a plain relation for these words; do NOT use "or".
- Emphasized distance — the words "well", "far", "way" before a direction (e.g. "WELL to the left",
  "FAR above", "well below"): kind "offset", "emphasis":true, exactly TWO options over the SAME
  relation: one with "const":0 and one with a larger "const" (e.g. 300). NEVER just pick a number
  yourself for these — "well"/"far"/"way" ALWAYS means an offset CHOICE.
- "the X" when TWO OR MORE detected objects have type X: kind "reference", options are the same
  relation applied to each candidate object id (e.g. cleft to e0 vs cleft to e1). If you find two
  matching objects, you MUST wrap them in a reference CHOICE — never AND them together.
Each CHOICE has 2+ options with a "prior" per option (priors sum to ~1) and a "span" (the source words).

Spatial relations (use these names only): above below left right cabove cbelow cleft cright
wider narrower taller shorter xeq yeq weq heq, plus the *_value forms (e.g. right_value, wider_value).
- "completely/fully left/right/above/below" -> cleft/cright/cabove/cbelow.
- The *_value forms are RELATIONS, not properties. "taller than 282 per-mille" ->
  {"node":"rel","name":"taller_value","args":["o0"],"const":282}. NEVER write
  property(o,"taller_value") — property() is ONLY for adjectives like "blue", "wooden", "metal".
- Emit each spatial relation ONCE. Do NOT restate the same relationship two ways (e.g. do NOT emit
  both cabove(a,b) AND cbelow(b,a) — pick the single relation that matches the sentence's subject).
  "A above B" -> ONE relation: above(a,b) (or cabove if "completely/fully"). Nothing else.
- Bind references to the given detected objects when they match by type (and position).
- EXISTING objects (status "existing"): never emit type(), property(), or default() for them —
  they are already known. Only NEW objects get type() and default().
- Every NEW object needs a type() and a default(). Integer offsets are per-mille (0..1000).
"""


def _exemplars(k: int = 6) -> List[dict]:
    """Return a DIVERSE, coverage-guaranteed exemplar set. Few-shot models imitate examples
    far more than prose rules, so we ensure each key pattern is demonstrated at least once:
    an unambiguous multi-object, a *_value relation, an offset-emphasis CHOICE, a direction
    CHOICE, a reference CHOICE, and an unsupported_type CHOICE (as many as k allows)."""
    pairs = json.load(open(os.path.join(HERE, "..", "examples", "seed_pairs.json")))["pairs"]
    by_id = {p["id"]: p for p in pairs}
    # priority coverage order (id -> what it teaches)
    preferred = [
        "fig7_blue_microwave_green_toaster",   # unambiguous multi-object + properties
        "ambiguous_offset_emphasis",           # 'well' -> offset CHOICE (P1)
        "value_relation_example",              # *_value are relations not properties (P4)
        "ambiguous_lamp_by_couch",             # direction + unsupported_type CHOICE
        "ambiguous_reference_example",         # 2 same-type -> reference CHOICE (P2)
        "ambiguous_unsupported_clock",         # unsupported_type CHOICE
        "fig8_chair_couch_coffeetable",        # more unambiguous relations
    ]
    picked = [by_id[i] for i in preferred if i in by_id][:k]
    if len(picked) >= k:
        return picked
    # top up with any remaining
    for p in pairs:
        if p not in picked:
            picked.append(p)
        if len(picked) >= k:
            break
    return picked


def _exemplars_legacy(k: int = 4) -> List[dict]:
    pairs = json.load(open(os.path.join(HERE, "..", "examples", "seed_pairs.json")))["pairs"]
    ambig = [p for p in pairs if '"choice"' in json.dumps(p["formula"])]
    unamb = [p for p in pairs if '"choice"' not in json.dumps(p["formula"])]
    n_ambig = max(1, k // 2)
    n_unamb = k - n_ambig
    return unamb[:n_unamb] + ambig[:n_ambig]


def build_prompt(english: str, objects: List[dict], k_shot: int = 6,
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
def _make_bnb_config(quantize):
    """Build a BitsAndBytesConfig for quantize in {None,'4bit','8bit'}. Returns None for full.
    4bit uses NF4 + double-quant + bf16 compute (the QLoRA-recommended inference setup)."""
    if quantize in (None, "none", "full"):
        return None
    from transformers import BitsAndBytesConfig
    import torch
    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"quantize must be None, '4bit', or '8bit'; got {quantize!r}")


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
    """GPU backend: local HF model + JSON-schema-constrained decoding via Outlines.

    Tuned for Outlines >= 1.3 (the 'new' API). Auto-detects and still supports the old API.
    Key performance fix: the constrained Generator is built ONCE (compiling a large recursive
    schema into a decoding automaton is expensive; doing it per-call is what makes it 'hang').
    """
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct",
                 max_tokens=1024, temperature=0.0, device="cuda", chat_template=True,
                 quantize=None, adapter_path=None):
        """
        quantize:     None (full fp16/bf16), "4bit" (NF4, ~5GB for 7B), or "8bit" (~8GB).
        adapter_path: path to LoRA adapters from finetune_lora.py. When set, the adapters
                      are merged onto the base model AND parse() should use the compact
                      fine-tune prompt (see examples/finetune_lora.py: ft_prompt).
        """
        import outlines
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.chat_template = chat_template
        self._schema_str = json.dumps(_SCHEMA)
        self._tokenizer = None
        self._api = None
        self.generator = None      # the compiled, reusable constrained generator
        self.quantize = quantize
        self.adapter_path = adapter_path

        # Build the bitsandbytes quantization config (shared by both API branches).
        quant_config = _make_bnb_config(quantize)

        if hasattr(outlines, "from_transformers"):
            # ---- NEW API (>=1.0, incl. 1.3.x) ----
            from transformers import AutoModelForCausalLM, AutoTokenizer
            load_kwargs = {"device_map": device}
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            if adapter_path:
                from peft import PeftModel
                hf_model = PeftModel.from_pretrained(hf_model, adapter_path)
                print(f"   loaded LoRA adapters from {adapter_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._report_footprint(hf_model)
            self.model = outlines.from_transformers(hf_model, self._tokenizer)
            self._api = "new"
            from outlines.types import JsonSchema
            self._output_type = JsonSchema(self._schema_str)
            try:
                from outlines import Generator
                self.generator = Generator(self.model, self._output_type)
            except Exception:
                self.generator = None
        else:
            # ---- OLD API (<1.0) ----
            from outlines import models, generate
            model_kwargs = {}
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config
            self.model = models.transformers(model_name, device=device,
                                             model_kwargs=model_kwargs or None)
            self._tokenizer = getattr(self.model, "tokenizer", None)
            self.gen = generate.json(self.model, self._schema_str)
            self._api = "old"

    @staticmethod
    def _report_footprint(hf_model):
        try:
            gb = hf_model.get_memory_footprint() / (1024 ** 3)
            print(f"   model memory footprint: {gb:.2f} GB")
        except Exception:
            pass

    def _apply_chat(self, prompt: str) -> str:
        if not self.chat_template or self._tokenizer is None:
            return prompt
        try:
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate(self, prompt: str, english: str, objects: List[dict]) -> dict:
        text = self._apply_chat(prompt)
        if self._api == "new":
            if self.generator is not None:
                out = self.generator(text, max_new_tokens=self.max_tokens)
            else:
                out = self.model(text, self._output_type, max_new_tokens=self.max_tokens)
        else:
            out = self.gen(text, max_tokens=self.max_tokens)
        return out if isinstance(out, dict) else json.loads(out)


Backend = Any  # anything with .generate(prompt, english, objects) -> dict


def parse(english: str, objects: List[dict], backend: Backend,
          k_shot: int = 6) -> Spec:
    # Fine-tuned backends were trained on the COMPACT prompt (no few-shot, no schema);
    # using it makes inference faster and matches the training distribution.
    if getattr(backend, "adapter_path", None):
        import importlib.util, os as _os
        _p = _os.path.join(HERE, "..", "examples", "finetune_lora.py")
        _spec = importlib.util.spec_from_file_location("finetune_lora", _p)
        _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
        prompt = _m.ft_prompt(english, objects)
    else:
        prompt = build_prompt(english, objects, k_shot=k_shot)
    raw = backend.generate(prompt, english, objects)
    from .json_io import dedupe_relations
    raw = dedupe_relations(raw)      # strip duplicate / mirror-redundant relations (pattern P3)
    spec = spec_from_json(raw)
    validate(spec)            # raises on malformed; grammar-decoding should prevent this
    return spec
