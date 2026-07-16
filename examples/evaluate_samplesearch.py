"""Compare SampleSearch guides and render a sample grid.

Run from the repository root:
    python -m examples.evaluate_samplesearch --count 40
"""
import argparse

from mscl import (Default, GeometricPreference, Obj, Spec, UniformPreference,
                  compare_preferences, format_comparison_table,
                  render_layout_grid_svg, save_svg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=40, help="samples per guide")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="samplesearch_layouts.svg")
    args = parser.parse_args()

    spec = Spec([Obj("chair", "new", "chair")], Default("chair"))
    reports = compare_preferences(
        spec,
        {"uniform": UniformPreference(), "geometric": GeometricPreference()},
        count=args.count,
        seed=args.seed,
    )
    print(format_comparison_table(reports))

    geometric = next(report for report in reports if report.label == "geometric")
    shown = geometric.results[:min(12, len(geometric.results))]
    svg = render_layout_grid_svg(
        spec,
        [result.layout for result in shown],
        titles=[f"seed {result.stats.seed}" for result in shown],
        columns=4,
    )
    output = save_svg(svg, args.output)
    print(f"\nSaved visual grid: {output}")
    print("Colab: from IPython.display import SVG, display; display(SVG(filename=str(output)))")


if __name__ == "__main__":
    main()
