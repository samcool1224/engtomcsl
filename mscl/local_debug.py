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
from dataclasses import dataclass, field
from typing import List, Optional
from .parser import build_prompt, build_parse_prompt, DEFAULT_K_SHOT
from .json_io import spec_from_json
from .validate import validate
from .postprocess import normalize_prediction
from .json_io import dedupe_relations


@dataclass
class ParseTiming:
    """Wall-clock breakdown for one parse, including repair attempts."""
    prompt_build_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    validation_s: float = 0.0
    total_s: float = 0.0
    attempts: int = 0
    prompt_chars_sent: int = 0
    prompt_tokens_sent: Optional[int] = None
    output_chars: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def checking_s(self) -> float:
        return self.postprocess_s + self.validation_s


def time_constrained_call(backend, sample, max_new_tokens=512):
    """Time a single constrained generation (to check the schema automaton isn't the bottleneck)."""
    english = sample.english if hasattr(sample, "english") else sample["english"]
    objects = sample.objects_json if hasattr(sample, "objects_json") else sample["objects"]
    prompt = build_prompt(english, objects, adaptive=True)
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


def parse_with_repair(english, objects, backend, k_shot=DEFAULT_K_SHOT, retries=1,
                      adaptive_examples=True, return_timing=False):
    """parse() with a repair retry: if validation fails, re-prompt once appending the error.
    Works with any backend (constrained or not). Returns a Spec or raises the last error."""
    total_t0 = time.perf_counter()
    timing = ParseTiming()
    t0 = time.perf_counter()
    prompt = build_parse_prompt(english, objects, backend, k_shot, adaptive_examples)
    timing.prompt_build_s = time.perf_counter() - t0
    last_err = None
    for attempt in range(retries + 1):
        timing.attempts += 1
        timing.prompt_chars_sent += len(prompt)
        if hasattr(backend, "count_tokens"):
            n_tokens = backend.count_tokens(prompt)
            if n_tokens is not None:
                timing.prompt_tokens_sent = ((timing.prompt_tokens_sent or 0) + n_tokens)
        try:
            phase_t0 = time.perf_counter()
            try:
                raw = backend.generate(prompt, english, objects)
            finally:
                timing.inference_s += time.perf_counter() - phase_t0
            timing.output_chars += len(json.dumps(raw, separators=(",", ":")))

            phase_t0 = time.perf_counter()
            try:
                raw = normalize_prediction(raw, english, objects)
                raw = dedupe_relations(raw)
            finally:
                timing.postprocess_s += time.perf_counter() - phase_t0

            phase_t0 = time.perf_counter()
            try:
                spec = spec_from_json(raw)
                validate(spec)
            finally:
                timing.validation_s += time.perf_counter() - phase_t0

            timing.total_s = time.perf_counter() - total_t0
            return (spec, timing) if return_timing else spec
        except Exception as e:
            last_err = e
            timing.errors.append(f"attempt {attempt + 1}: {type(e).__name__}: {e}")
            prompt = (prompt + f"\n\n(Your previous output was invalid: {e}. "
                      "Return corrected JSON only, using ONLY the legal 17 types.)")
    timing.total_s = time.perf_counter() - total_t0
    try:
        setattr(last_err, "parse_timing", timing)
    except Exception:
        pass
    raise last_err
