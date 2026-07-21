"""Layout-to-image grounding adapter.

The exact MSCL layout remains the source of truth.  This module only translates its
per-mille boxes into the normalized ``[xmin, ymin, xmax, ymax]`` representation accepted
by a grounded diffusion pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

from .ast import Spec
from .feasibility import Layout
from .object_plan import object_phrase
from .profile import COORD_MAX


DEFAULT_GLIGEN_MODEL = "masterful/gligen-1-4-generation-text-box"
DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, logo, people, duplicate furniture, malformed furniture, "
    "floating objects, cropped objects, low quality, blurry"
)


@dataclass(frozen=True)
class GroundedScene:
    prompt: str
    phrases: Tuple[str, ...]
    boxes: Tuple[Tuple[float, float, float, float], ...]
    object_ids: Tuple[str, ...]


def layout_to_grounded_scene(spec: Spec, layout: Layout, english: str,
                             style_prompt: Optional[str] = None) -> GroundedScene:
    """Convert a verified per-mille layout into grounded text and normalized boxes."""
    phrases: List[str] = []
    boxes: List[Tuple[float, float, float, float]] = []
    object_ids: List[str] = []
    for obj in spec.objects:
        if obj.id not in layout:
            raise ValueError(f"layout is missing object {obj.id!r}")
        x, y, width, height = layout[obj.id]
        x2, y2 = x + width, y + height
        if width <= 0 or height <= 0 or min(x, y) < 0 or max(x2, y2) > COORD_MAX:
            raise ValueError(f"object {obj.id!r} has an invalid or out-of-frame box")
        phrases.append(object_phrase(obj))
        boxes.append(tuple(v / COORD_MAX for v in (x, y, x2, y2)))
        object_ids.append(obj.id)

    style = style_prompt or (
        "photorealistic interior design photograph, coherent furnished room, realistic "
        "materials, natural lighting, wide-angle view"
    )
    prompt = f"{style}. User design: {english.strip()}"
    return GroundedScene(prompt, tuple(phrases), tuple(boxes), tuple(object_ids))


class GligenImageGenerator:
    """Lazy, reusable Diffusers GLIGEN generator with a Colab-friendly offload mode."""

    def __init__(self, model_id: str = DEFAULT_GLIGEN_MODEL, *, device: Optional[str] = None,
                 low_vram: bool = True, pipe: Any = None):
        self.model_id = model_id
        self.device = device
        self.low_vram = low_vram
        self.pipe = pipe

    def load(self):
        if self.pipe is not None:
            return self.pipe
        try:
            import torch
            from diffusers import StableDiffusionGLIGENPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Image generation dependencies are missing. Install requirements-image.txt"
            ) from exc

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        kwargs = {"torch_dtype": dtype}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        self.pipe = StableDiffusionGLIGENPipeline.from_pretrained(self.model_id, **kwargs)
        if device.startswith("cuda") and self.low_vram:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(device)
        return self.pipe

    def generate(self, scene: GroundedScene, *, seed: int = 0,
                 num_inference_steps: int = 40, guidance_scale: float = 7.5,
                 scheduled_sampling_beta: float = 0.3,
                 negative_prompt: str = DEFAULT_NEGATIVE_PROMPT):
        pipe = self.load()
        try:
            import torch
            generator_device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(int(seed))
        except ImportError:
            # A supplied test/dummy pipeline may not need PyTorch. A real Diffusers pipeline
            # cannot reach this branch because load() imports torch first.
            generator = None

        call_kwargs = dict(
            prompt=scene.prompt,
            gligen_phrases=list(scene.phrases),
            gligen_boxes=[list(box) for box in scene.boxes],
            gligen_scheduled_sampling_beta=float(scheduled_sampling_beta),
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            output_type="pil",
        )
        if generator is not None:
            call_kwargs["generator"] = generator
        output = pipe(**call_kwargs)
        return output.images[0]
