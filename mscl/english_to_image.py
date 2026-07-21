"""Reusable from-scratch English-to-image orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional

from .ast import Spec
from .dialogue import Oracle, ResolutionLog, resolve, resolve_ask_none
from .image_generation import GligenImageGenerator, GroundedScene, layout_to_grounded_scene
from .local_debug import ParseTiming, parse_with_repair
from .object_plan import extract_new_objects
from .samplesearch import SampleResult, SampleSearch


@dataclass
class ScenePlan:
    english: str
    objects: List[dict]
    spec: Spec
    resolution_log: Optional[ResolutionLog]
    sample: SampleResult
    grounding: GroundedScene
    parse_timing: ParseTiming
    planning_time_s: float


@dataclass
class EnglishToImageResult:
    plan: ScenePlan
    image: Any
    image_time_s: float

    @property
    def timings(self) -> Dict[str, float]:
        return {
            "llm_inference_s": self.plan.parse_timing.inference_s,
            "postprocess_checking_s": self.plan.parse_timing.checking_s,
            "layout_search_s": self.plan.sample.stats.total_time_s,
            "planning_total_s": self.plan.planning_time_s,
            "image_generation_s": self.image_time_s,
            "end_to_end_s": self.plan.planning_time_s + self.image_time_s,
        }


class EnglishToImageSystem:
    """Compose the parser, ambiguity policy, exact layout search, and image generator."""

    def __init__(self, parser_backend, *, image_generator: Optional[Any] = None,
                 sampler: Optional[SampleSearch] = None,
                 object_extractor: Callable[[str], List[dict]] = extract_new_objects):
        self.parser_backend = parser_backend
        self.image_generator = image_generator or GligenImageGenerator()
        self.sampler = sampler or SampleSearch()
        self.object_extractor = object_extractor

    def plan(self, english: str, *, oracle: Optional[Oracle] = None,
             question_budget: int = 3, layout_seed: int = 0,
             style_prompt: Optional[str] = None) -> ScenePlan:
        started = time.perf_counter()
        objects = self.object_extractor(english)
        spec, parse_timing = parse_with_repair(
            english, objects, self.parser_backend, return_timing=True
        )
        resolution_log = None
        if oracle is None:
            spec = resolve_ask_none(spec)
        else:
            spec, resolution_log = resolve(spec, oracle, budget=question_budget)
        sample = self.sampler.sample(spec, seed=layout_seed)
        grounding = layout_to_grounded_scene(spec, sample.layout, english, style_prompt)
        return ScenePlan(
            english=english,
            objects=objects,
            spec=spec,
            resolution_log=resolution_log,
            sample=sample,
            grounding=grounding,
            parse_timing=parse_timing,
            planning_time_s=time.perf_counter() - started,
        )

    def generate(self, english: str, *, oracle: Optional[Oracle] = None,
                 question_budget: int = 3, layout_seed: int = 0, image_seed: int = 0,
                 style_prompt: Optional[str] = None, **image_kwargs) -> EnglishToImageResult:
        plan = self.plan(
            english,
            oracle=oracle,
            question_budget=question_budget,
            layout_seed=layout_seed,
            style_prompt=style_prompt,
        )
        started = time.perf_counter()
        image = self.image_generator.generate(plan.grounding, seed=image_seed, **image_kwargs)
        return EnglishToImageResult(plan, image, time.perf_counter() - started)
