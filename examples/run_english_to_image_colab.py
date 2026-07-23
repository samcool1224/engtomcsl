"""First complete English -> image run for Colab or local Jupyter.

Run from the project after installing requirements.txt and
requirements-image.txt. Loading the two models is intentionally done once.
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get("MSCL_OUTPUT_DIR", str(PROJECT_ROOT))
).expanduser().resolve()

sys.path.insert(0, str(PROJECT_ROOT))

from mscl import (DEFAULT_NEGATIVE_PROMPT, EnglishToImageSystem,
                  GligenImageGenerator, LocalBackend, save_run_bundle)


ENGLISH = (
    "Create a modern dining room with a dining table, a blue chair completely to the "
    "left of the dining table, and a potted plant completely to the right of the "
    "dining table."
)


def terminal_oracle(question):
    """Ask only when the parser emitted a genuine MSCL CHOICE."""
    print(f"\nClarification needed for {question.span!r}:")
    for index, option in enumerate(question.options_text):
        print(f"  [{index}] {option}")
    while True:
        try:
            selected = int(input("Choose an option number: "))
            if 0 <= selected < len(question.options_text):
                return selected
        except ValueError:
            pass
        print("Please enter one of the displayed option numbers.")

print("Loading English parser...")
parser_load_started = time.perf_counter()
parser_backend = LocalBackend("Qwen/Qwen2.5-7B-Instruct", quantize="4bit")
parser_model_load_s = time.perf_counter() - parser_load_started

# CPU offload lets GLIGEN share a typical Colab GPU with the quantized parser.
image_generator = GligenImageGenerator(low_vram=True)
system = EnglishToImageSystem(parser_backend, image_generator=image_generator)

print("Planning and generating...")
result = system.generate(
    ENGLISH,
    oracle=terminal_oracle,
    layout_seed=42,
    layout_candidate_count=24,
    image_seed=7,
    image_candidate_count=4,
    num_inference_steps=40,
    guidance_scale=7.5,
    scheduled_sampling_beta=0.3,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
)
result.generation_config["parser_model_load_s"] = parser_model_load_s

paths, record, report = save_run_bundle(
    result,
    str(OUTPUT_DIR),
    prefix="english_to_image",
)
print("\n" + report)

try:
    from IPython.display import SVG, display
    display(SVG(filename=paths["layout_svg"]))
    display(result.image)
except ImportError:
    pass
