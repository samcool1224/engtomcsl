"""Layout-to-image grounding adapter.

The exact MSCL layout remains the source of truth.  This module only translates its
per-mille boxes into the normalized ``[xmin, ymin, xmax, ymax]`` representation accepted
by a grounded diffusion pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
import colorsys
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .ast import Spec
from .feasibility import Layout
from .object_plan import object_phrase
from .profile import COORD_MAX


DEFAULT_GLIGEN_MODEL = "masterful/gligen-1-4-generation-text-box"
DEFAULT_NEGATIVE_PROMPT = (
    "text, watermark, logo, people, duplicate furniture, malformed furniture, "
    "floating objects, cropped objects, incorrect object colors, wrong colors, "
    "low quality, blurry"
)
_COLOR_PROPERTIES = {"red", "blue", "green", "black", "white", "brown"}


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
        "materials, natural lighting, eye-level wide-angle view"
    )
    inventory = "; ".join(
        f"one {object_phrase(obj).removeprefix('a ')}" for obj in spec.objects
    )
    chair_count = sum(obj.type == "chair" for obj in spec.objects)
    count_guard = (
        f" Show exactly {chair_count} chair{'s' if chair_count != 1 else ''} total."
        if chair_count else " Show no chairs."
    )
    prompt = (
        f"{style}. Required inventory: {inventory}. "
        "Do not add unrequested or duplicate furniture."
        f"{count_guard} Render exactly one instance of each grounded object. "
        "The specified object colors are mandatory. Keep the furniture in one coherent "
        f"composition and respect every grounded region. User design: {english.strip()}"
    )
    return GroundedScene(prompt, tuple(phrases), tuple(boxes), tuple(object_ids))


def grounded_color_fidelity(image: Any, scene: GroundedScene,
                            spec: Spec) -> Tuple[float, Dict[str, float]]:
    """Measure requested named colors inside their grounded regions.

    This intentionally checks only explicit, inexpensive color attributes.  It is a
    candidate-ranking signal, not a claim that the image is semantically verified.
    """
    if not hasattr(image, "convert") or not hasattr(image, "crop"):
        return 0.0, {}
    width, height = image.size
    per_object: Dict[str, float] = {}
    for object_id, box in zip(scene.object_ids, scene.boxes):
        obj = spec.obj(object_id)
        colors = [value for value in obj.properties if value in _COLOR_PROPERTIES]
        if not colors:
            continue
        x1, y1, x2, y2 = box
        crop = image.crop((
            max(0, int(x1 * width)), max(0, int(y1 * height)),
            min(width, int(x2 * width)), min(height, int(y2 * height)),
        )).convert("RGB")
        if crop.width <= 0 or crop.height <= 0:
            per_object[object_id] = 0.0
            continue
        crop.thumbnail((128, 128))
        pixels = list(crop.getdata())
        scores = [_color_fraction(pixels, color) for color in colors]
        per_object[object_id] = sum(scores) / len(scores)
    if not per_object:
        return 0.0, {}
    return sum(per_object.values()) / len(per_object), per_object


def _color_fraction(pixels, color: str) -> float:
    if not pixels:
        return 0.0
    matches = 0
    for red, green, blue in pixels:
        hue, saturation, value = colorsys.rgb_to_hsv(
            red / 255.0, green / 255.0, blue / 255.0
        )
        degrees = hue * 360.0
        if color == "red":
            ok = saturation > 0.38 and value > 0.20 and (
                degrees <= 22.0 or degrees >= 338.0)
        elif color == "blue":
            ok = saturation > 0.30 and value > 0.18 and 185.0 <= degrees <= 255.0
        elif color == "green":
            ok = saturation > 0.28 and value > 0.16 and 70.0 <= degrees <= 165.0
        elif color == "black":
            ok = value < 0.23
        elif color == "white":
            ok = value > 0.78 and saturation < 0.18
        elif color == "brown":
            ok = (0.12 < value < 0.72 and saturation > 0.28
                  and 12.0 <= degrees <= 48.0)
        else:
            ok = False
        matches += int(ok)
    return matches / len(pixels)


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
