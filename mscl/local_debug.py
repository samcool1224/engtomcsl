"""Debugging + robustness helpers for the local model path.

Kept intentionally small. FreeGenBackend (unconstrained generate-then-validate) was removed:
it was a fallback for a schema-compile stall that no longer occurs now that LocalBackend
builds its constrained Generator once. If you ever need it again, it's in git history.

Provides:
  time_constrained_call(backend, sample)  -- time ONE constrained generation.
  parse_with_repair(english, objects, backend, retries=1)
                                          -- parse() that re-prompts once on a validation error.
"""
from __future__ import annotations
import json, time
from typing import List
from .parser import build_prompt, DEFAULT_K_SHOT
from .json_io import spec_from_json
from .validate import validate
from .postprocess import normalize_prediction
from .json_io import dedupe_relations


def time_constrained_call(backend, sample, max_new_tokens=512):
    """Time a single constrained generation (to check the schema automaton isn't the bottleneck)."""
    english = sample.english if hasattr(sample, "english") else sample["english"]
    objects = sample.objects_json if hasattr(sample, "objects_json") else sample["objects"]
    prompt = build_prompt(english, objects, k_shot=3)
    print("prompt chars:", len(prompt), "| max_new_tokens:", max_new_tokens)
    t0 = time.time()
    try:
        saved = backend.max_tokens
        backend.max_tokens = max_new_tokens
        out = backend.generate(prompt, english, objects)
        backend.max_tokens = saved
        print(f"constrained call returned in {time.time()-t0:.1f}s")
        print("output preview:", json.dumps(out)[:200])
        return out
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"failed after {time.time()-t0:.1f}s")
        return None


def parse_with_repair(english, objects, backend, k_shot=DEFAULT_K_SHOT, retries=1):
    """parse() with a repair retry: if validation fails, re-prompt once appending the error.
    Works with any backend (constrained or not). Returns a Spec or raises the last error."""
    prompt = build_prompt(english, objects, k_shot=k_shot)
    last_err = None
    for _ in range(retries + 1):
        try:
            raw = backend.generate(prompt, english, objects)
            raw = normalize_prediction(raw, english, objects)
            raw = dedupe_relations(raw)
            spec = spec_from_json(raw)
            validate(spec)
            return spec
        except Exception as e:
            last_err = e
            prompt = (prompt + f"\n\n(Your previous output was invalid: {e}. "
                      "Return corrected JSON only, using ONLY the legal 17 types.)")
    raise last_err
