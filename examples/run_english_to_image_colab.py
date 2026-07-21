"""First complete English -> image run for Google Colab.

Run from /content/engtomcsl after installing requirements.txt and
requirements-image.txt.  Loading the two models is intentionally done once.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mscl import (EnglishToImageSystem, GligenImageGenerator, LocalBackend,
                  render_layout_svg, save_svg, spec_to_json)


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
parser_backend = LocalBackend("Qwen/Qwen2.5-7B-Instruct", quantize="4bit")

# CPU offload lets GLIGEN share a typical Colab GPU with the quantized parser.
image_generator = GligenImageGenerator(low_vram=True)
system = EnglishToImageSystem(parser_backend, image_generator=image_generator)

print("Planning and generating...")
result = system.generate(
    ENGLISH,
    oracle=terminal_oracle,
    layout_seed=42,
    image_seed=7,
    num_inference_steps=40,
)

output_path = "/content/engtomcsl/english_to_image_result.png"
result.image.save(output_path)
layout_path = "/content/engtomcsl/english_to_image_layout.svg"
save_svg(render_layout_svg(result.plan.spec, result.plan.sample.layout), layout_path)

print("\nObjects:")
print(json.dumps(result.plan.objects, indent=2))
print("\nResolved MSCL:")
print(json.dumps(spec_to_json(result.plan.spec)["formula"], indent=2))
print("\nLayout:", result.plan.sample.layout)
print("\nGrounding phrases:", result.plan.grounding.phrases)
print("Grounding boxes:", result.plan.grounding.boxes)
print("\nTimings:")
for name, seconds in result.timings.items():
    print(f"  {name:24s} {seconds:.3f}s")
print("\nSaved:", output_path)
print("Saved layout:", layout_path)

try:
    from IPython.display import SVG, display
    display(SVG(filename=layout_path))
    display(result.image)
except ImportError:
    pass
