"""Diagnostic for the LocalBackend stall, plus a schema-free fallback backend.

Run the diagnostic FIRST (in your notebook, after `backend` is created):

    from mscl.local_debug import time_constrained_call, FreeGenBackend
    time_constrained_call(backend, test[0])      # times ONE constrained generation

If that returns in a sane time -> keep using LocalBackend (constrained, guaranteed valid).
If it stalls or is very slow -> use FreeGenBackend instead (fast, validate-after):

    fb = FreeGenBackend("Qwen/Qwen2.5-7B-Instruct")   # or reuse loaded hf model, see below
"""
from __future__ import annotations
import json, time, re
from typing import List, Optional
from .parser import build_prompt, _SCHEMA
from .json_io import spec_from_json
from .validate import validate


def time_constrained_call(backend, sample, max_new_tokens=512):
    """Time a single constrained generation to see if the schema automaton is the bottleneck."""
    prompt = build_prompt(sample.english if hasattr(sample, "english") else sample["english"],
                          sample.objects_json if hasattr(sample, "objects_json") else sample["objects"],
                          k_shot=3)
    print("prompt chars:", len(prompt), "| max_new_tokens:", max_new_tokens)
    t0 = time.time()
    try:
        # temporarily shorten output cap for the test
        saved = backend.max_tokens
        backend.max_tokens = max_new_tokens
        out = backend.generate(prompt,
                               sample.english if hasattr(sample, "english") else sample["english"],
                               sample.objects_json if hasattr(sample, "objects_json") else sample["objects"])
        backend.max_tokens = saved
        print(f"constrained call returned in {time.time()-t0:.1f}s")
        print("output preview:", json.dumps(out)[:200])
        return out
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"failed after {time.time()-t0:.1f}s")
        return None




def parse_with_repair(english, objects, backend, k_shot=3, retries=1):
    """parse() with one repair retry: if validation fails, re-prompt appending the error."""
    from .parser import build_prompt
    prompt = build_prompt(english, objects, k_shot=k_shot)
    last_err = None
    for attempt in range(retries + 1):
        try:
            raw = backend.generate(prompt, english, objects)
            spec = spec_from_json(raw)
            validate(spec)
            return spec
        except Exception as e:
            last_err = e
            prompt = (prompt + f"\n\n(Your previous output was invalid: {e}. "
                      "Return corrected JSON only.)")
    raise last_err
