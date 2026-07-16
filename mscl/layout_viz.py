"""Dependency-free SVG rendering for MSCL box layouts.

The renderer is intentionally diagnostic rather than photorealistic: it makes spatial
constraints, object identities, existing/new status, and odd size choices immediately
visible before layouts are handed to an image generator.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from .ast import Obj, Spec
from .feasibility import Layout


_PALETTE = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
            "#0891b2", "#be185d", "#4d7c0f", "#9333ea", "#0f766e")


def render_layout_svg(spec: Spec, layout: Mapping[str, Tuple[int, int, int, int]], *,
                      width: int = 640, height: int = 640,
                      title: Optional[str] = None, show_coordinates: bool = True) -> str:
    """Return one layout as an SVG string."""
    if width < 200 or height < 200:
        raise ValueError("SVG dimensions must be at least 200x200")
    title_height = 34 if title else 10
    padding = 28
    canvas_x = padding
    canvas_y = padding + title_height
    canvas_w = width - 2 * padding
    canvas_h = height - canvas_y - padding
    body = _layout_body(spec, layout, canvas_x, canvas_y, canvas_w, canvas_h,
                        show_coordinates=show_coordinates)
    title_node = (f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="16" font-weight="600" '
                  f'fill="#111827">{escape(title)}</text>') if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'{title_node}{body}</svg>'
    )


def render_layout_grid_svg(spec: Spec, layouts: Sequence[Mapping[str, Tuple[int, int, int, int]]], *,
                           columns: int = 4, cell_size: int = 280,
                           titles: Optional[Sequence[str]] = None,
                           show_coordinates: bool = False) -> str:
    """Render several layouts in a compact comparison grid."""
    if columns < 1:
        raise ValueError("columns must be positive")
    if cell_size < 180:
        raise ValueError("cell_size must be at least 180")
    if titles is not None and len(titles) != len(layouts):
        raise ValueError("titles must have one entry per layout")
    if not layouts:
        raise ValueError("at least one layout is required")

    rows = (len(layouts) + columns - 1) // columns
    width = columns * cell_size
    height = rows * cell_size
    panels = []
    for i, layout in enumerate(layouts):
        col, row = i % columns, i // columns
        x0, y0 = col * cell_size, row * cell_size
        panel_title = titles[i] if titles is not None else f"sample {i + 1}"
        panels.append(
            f'<g transform="translate({x0},{y0})">'
            f'<rect x="4" y="4" width="{cell_size - 8}" height="{cell_size - 8}" '
            'rx="8" fill="#ffffff" stroke="#cbd5e1"/>'
            f'<text x="{cell_size / 2:.1f}" y="23" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13" font-weight="600" '
            f'fill="#334155">{escape(panel_title)}</text>'
            f'{_layout_body(spec, layout, 18, 34, cell_size - 36, cell_size - 52, show_coordinates)}'
            '</g>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">'
        '<rect width="100%" height="100%" fill="#e2e8f0"/>'
        f'{"".join(panels)}</svg>'
    )


def save_svg(svg: str, path: Union[str, Path]) -> Path:
    """Write an SVG string and return the resolved output path."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def _layout_body(spec: Spec, layout: Mapping[str, Tuple[int, int, int, int]],
                 x0: float, y0: float, width: float, height: float,
                 show_coordinates: bool) -> str:
    sx, sy = width / 1000.0, height / 1000.0
    pieces = [
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'fill="#ffffff" stroke="#64748b" stroke-width="1.5"/>'
    ]
    for index, obj in enumerate(spec.objects):
        if obj.id not in layout:
            continue
        x, y, w, h = layout[obj.id]
        px, py, pw, ph = x0 + x * sx, y0 + y * sy, w * sx, h * sy
        color = _PALETTE[index % len(_PALETTE)]
        dash = ' stroke-dasharray="6 4"' if obj.status == "existing" else ""
        pieces.append(
            f'<rect x="{px:.2f}" y="{py:.2f}" width="{pw:.2f}" height="{ph:.2f}" '
            f'fill="{color}" fill-opacity="0.20" stroke="{color}" stroke-width="2"{dash}/>'
        )
        label = f"{obj.id}: {obj.type or 'unknown'}"
        if show_coordinates:
            label += f" [{x},{y},{w},{h}]"
        label_y = max(y0 + 13, min(y0 + height - 4, py + 14))
        pieces.append(
            f'<text x="{px + 4:.2f}" y="{label_y:.2f}" font-family="sans-serif" '
            f'font-size="11" font-weight="600" fill="#0f172a">{escape(label)}</text>'
        )
    return "".join(pieces)
