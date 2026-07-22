"""Reusable from-scratch English-to-image orchestration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .ast import Spec
from .dialogue import Oracle, ResolutionLog, resolve, resolve_ask_none
from .image_generation import GligenImageGenerator, GroundedScene, layout_to_grounded_scene
from .json_io import spec_to_json
from .layout_viz import render_layout_svg, save_svg
from .local_debug import ParseTiming, parse_with_repair
from .object_plan import extract_new_objects
from .samplesearch import SampleResult, SampleSearch
from .z3_backend import Z3Backend


@dataclass
class ScenePlan:
    english: str
    objects: List[dict]
    parsed_mscl: Dict[str, Any]
    spec: Spec
    resolution_log: Optional[ResolutionLog]
    sample: SampleResult
    grounding: GroundedScene
    parse_timing: ParseTiming
    object_extraction_time_s: float
    ambiguity_resolution_time_s: float
    grounding_time_s: float
    planning_time_s: float


@dataclass
class EnglishToImageResult:
    plan: ScenePlan
    image: Any
    image_time_s: float
    generation_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def timings(self) -> Dict[str, float]:
        return {
            "object_extraction_s": self.plan.object_extraction_time_s,
            "prompt_build_s": self.plan.parse_timing.prompt_build_s,
            "llm_inference_s": self.plan.parse_timing.inference_s,
            "parser_postprocess_s": self.plan.parse_timing.postprocess_s,
            "parser_validation_s": self.plan.parse_timing.validation_s,
            "ambiguity_resolution_s": self.plan.ambiguity_resolution_time_s,
            "layout_search_s": self.plan.sample.stats.total_time_s,
            "grounding_adapter_s": self.plan.grounding_time_s,
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
        phase_started = time.perf_counter()
        objects = self.object_extractor(english)
        object_extraction_time_s = time.perf_counter() - phase_started
        spec, parse_timing = parse_with_repair(
            english, objects, self.parser_backend, return_timing=True
        )
        parsed_mscl = spec_to_json(spec)
        phase_started = time.perf_counter()
        resolution_log = None
        if oracle is None:
            spec = resolve_ask_none(spec)
        else:
            spec, resolution_log = resolve(spec, oracle, budget=question_budget)
        ambiguity_resolution_time_s = time.perf_counter() - phase_started
        sample = self.sampler.sample(spec, seed=layout_seed)
        phase_started = time.perf_counter()
        grounding = layout_to_grounded_scene(spec, sample.layout, english, style_prompt)
        grounding_time_s = time.perf_counter() - phase_started
        return ScenePlan(
            english=english,
            objects=objects,
            parsed_mscl=parsed_mscl,
            spec=spec,
            resolution_log=resolution_log,
            sample=sample,
            grounding=grounding,
            parse_timing=parse_timing,
            object_extraction_time_s=object_extraction_time_s,
            ambiguity_resolution_time_s=ambiguity_resolution_time_s,
            grounding_time_s=grounding_time_s,
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
        image_time_s = time.perf_counter() - started
        generation_config = {
            "parser_backend": type(self.parser_backend).__name__,
            "parser_model": getattr(self.parser_backend, "model_name", None),
            "parser_device": getattr(self.parser_backend, "device", None),
            "parser_quantization": getattr(self.parser_backend, "quantize", None),
            "sampler": type(self.sampler).__name__,
            "preference_model": type(self.sampler.preference).__name__,
            "layout_seed": int(layout_seed),
            "question_budget": int(question_budget),
            "style_prompt": style_prompt,
            "image_generator": type(self.image_generator).__name__,
            "image_model": getattr(self.image_generator, "model_id", None),
            "image_seed": int(image_seed),
            "image_parameters": dict(image_kwargs),
        }
        return EnglishToImageResult(plan, image, image_time_s, generation_config)


def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in ("torch", "transformers", "outlines", "diffusers", "accelerate",
                    "bitsandbytes", "z3-solver", "lark", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _environment_record() -> Dict[str, Any]:
    environment: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda_available": None,
        "cuda_runtime": None,
        "gpu": None,
    }
    try:
        import torch
        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            environment["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return environment


def _resolution_record(log: Optional[ResolutionLog]) -> Dict[str, Any]:
    if log is None:
        return {
            "mode": "automatic_max_prior",
            "asked_count": 0,
            "questions": [],
            "auto_resolved": [],
            "low_confidence": [],
        }
    questions = []
    for index, question in enumerate(log.asked):
        answer_index = None
        answer_text = None
        if index < len(log.selected_answers):
            answer_index, answer_text = log.selected_answers[index]
        questions.append({
            "kind": question.kind,
            "span": question.span,
            "options": list(question.options_text),
            "selected_index": answer_index,
            "selected_text": answer_text,
        })
    return {
        "mode": "interactive",
        "asked_count": len(questions),
        "questions": questions,
        "auto_resolved": [
            {"kind": kind, "reason": reason} for kind, reason in log.auto_resolved
        ],
        "low_confidence": list(log.low_confidence),
    }


def build_run_record(result: EnglishToImageResult, *,
                     output_paths: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Build the complete JSON-serializable audit record for one successful run."""
    plan = result.plan
    layout = plan.sample.layout
    verified = Z3Backend(plan.spec).verify_layout(layout)
    image_size = getattr(result.image, "size", None)
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success" if verified else "failed_verification",
        "input": {"english": plan.english},
        "configuration": dict(result.generation_config),
        "object_plan": list(plan.objects),
        "parsed_mscl_before_resolution": plan.parsed_mscl,
        "resolved_mscl": spec_to_json(plan.spec),
        "ambiguity_resolution": _resolution_record(plan.resolution_log),
        "layout": {
            "coordinate_format": "per-mille [x, y, width, height] on a 1000x1000 canvas",
            "boxes": {key: list(value) for key, value in layout.items()},
            "exact_z3_verification": verified,
            "search_stats": asdict(plan.sample.stats),
            "search_trace_event_count": len(plan.sample.trace),
        },
        "grounding": {
            "prompt": plan.grounding.prompt,
            "object_ids": list(plan.grounding.object_ids),
            "phrases": list(plan.grounding.phrases),
            "normalized_xyxy_boxes": [list(box) for box in plan.grounding.boxes],
        },
        "image": {
            "width": image_size[0] if image_size else None,
            "height": image_size[1] if image_size else None,
        },
        "timings_seconds": dict(result.timings),
        "parser_diagnostics": asdict(plan.parse_timing),
        "environment": _environment_record(),
        "outputs": dict(output_paths or {}),
    }


def format_run_report(record: Mapping[str, Any]) -> str:
    """Render the important run record fields as readable console/text output."""
    lines = [
        "ENGLISH -> IMAGE RUN REPORT",
        "=" * 80,
        f"Status: {str(record['status']).upper()}",
        f"Created (UTC): {record['created_at_utc']}",
        "",
        "INPUT ENGLISH",
        str(record["input"]["english"]),
        "",
        "CONFIGURATION",
    ]
    for key, value in record["configuration"].items():
        lines.append(f"  {key}: {value}")

    lines.extend(("", "EXTRACTED OBJECT PLAN"))
    lines.append(json.dumps(record["object_plan"], indent=2))
    lines.extend(("", "PARSED MSCL BEFORE AMBIGUITY RESOLUTION"))
    lines.append(json.dumps(record["parsed_mscl_before_resolution"], indent=2))
    lines.extend(("", "RESOLVED MSCL"))
    lines.append(json.dumps(record["resolved_mscl"], indent=2))

    resolution = record["ambiguity_resolution"]
    lines.extend(("", "AMBIGUITY RESOLUTION",
                  f"  mode: {resolution['mode']}",
                  f"  questions asked: {resolution['asked_count']}"))
    for question in resolution["questions"]:
        lines.append(f"  - {question['kind']} @ {question['span']!r}")
        for index, option in enumerate(question["options"]):
            marker = "*" if index == question["selected_index"] else " "
            lines.append(f"    {marker} [{index}] {option}")
    lines.append(f"  automatically resolved: {resolution['auto_resolved']}")
    lines.append(f"  low confidence: {resolution['low_confidence']}")

    layout = record["layout"]
    lines.extend(("", "VERIFIED LAYOUT",
                  f"  coordinate format: {layout['coordinate_format']}",
                  f"  exact Z3 verification: {'PASS' if layout['exact_z3_verification'] else 'FAIL'}"))
    for object_id, box in layout["boxes"].items():
        lines.append(f"  {object_id}: {box}")
    lines.append("  search statistics:")
    for key, value in layout["search_stats"].items():
        lines.append(f"    {key}: {value}")
    lines.append(f"    trace_events: {layout['search_trace_event_count']}")

    grounding = record["grounding"]
    lines.extend(("", "IMAGE GROUNDING", f"  prompt: {grounding['prompt']}"))
    for object_id, phrase, box in zip(
            grounding["object_ids"], grounding["phrases"],
            grounding["normalized_xyxy_boxes"]):
        lines.append(f"  {object_id}: {phrase!r} -> {box}")

    image = record["image"]
    lines.extend(("", "GENERATED IMAGE",
                  f"  size: {image['width']} x {image['height']} pixels",
                  "", "TIMINGS"))
    for key, value in record["timings_seconds"].items():
        lines.append(f"  {key}: {value:.4f}s")

    diagnostics = record["parser_diagnostics"]
    lines.extend(("", "PARSER DIAGNOSTICS"))
    for key, value in diagnostics.items():
        lines.append(f"  {key}: {value}")

    environment = record["environment"]
    lines.extend(("", "ENVIRONMENT",
                  f"  Python: {environment['python']}",
                  f"  platform: {environment['platform']}",
                  f"  CUDA available: {environment['cuda_available']}",
                  f"  CUDA runtime: {environment['cuda_runtime']}",
                  f"  GPU: {environment['gpu']}"))
    for package, version in environment["packages"].items():
        lines.append(f"  {package}: {version or 'not installed'}")

    lines.extend(("", "SAVED OUTPUTS"))
    for label, path in record["outputs"].items():
        lines.append(f"  {label}: {path}")
    return "\n".join(lines) + "\n"


def save_run_bundle(result: EnglishToImageResult, directory: str,
                    *, prefix: str = "english_to_image") -> Tuple[Dict[str, str],
                                                                   Dict[str, Any], str]:
    """Save the PNG, SVG, JSON manifest, and matching human-readable report."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "image_png": str(output_dir / f"{prefix}_result.png"),
        "layout_svg": str(output_dir / f"{prefix}_layout.svg"),
        "run_json": str(output_dir / f"{prefix}_run.json"),
        "report_txt": str(output_dir / f"{prefix}_report.txt"),
    }
    result.image.save(paths["image_png"])
    save_svg(render_layout_svg(result.plan.spec, result.plan.sample.layout),
             paths["layout_svg"])
    record = build_run_record(result, output_paths=paths)
    report = format_run_report(record)
    Path(paths["run_json"]).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    Path(paths["report_txt"]).write_text(report, encoding="utf-8")
    return paths, record, report
