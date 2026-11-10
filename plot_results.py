#!/usr/bin/env python3
"""Plot VectorDBBench JSON results: recall on x, peak QPS on y.

Usage: venv/bin/python plot_results.py [--results-dir results] [--output plot.html]
Identical configurations keep the fastest run and its corresponding recall.
Different datasets and metrics are plotted in separate panels.
Top-k values share a panel; curves split by top-k, build parameters and RaBitQ bits.
"""

import argparse
import json
import math
from collections import defaultdict
from html import escape
from pathlib import Path


def load_points(results_dir):
    points = {}
    files = sorted(results_dir.glob("result_*.json"))
    if not files:
        raise ValueError(f"No result_*.json files in {results_dir}")
    for path in files:
        for result in json.loads(path.read_text())["results"]:
            metrics = result["metrics"]
            task = result["task_config"]
            config = task["db_case_config"]
            case = task["case_config"]
            candidates = metrics.get("conc_qps_list") or [metrics.get("qps")]
            qps = max(
                (float(q) for q in candidates
                 if q is not None and math.isfinite(float(q))),
                default=0,
            )
            recall = metrics.get("recall")
            if qps <= 0 or recall is None or not math.isfinite(float(recall)):
                print(f"Skip {path.name}: missing valid QPS/recall")
                continue
            recall = float(recall)
            if not 0 <= recall <= 1:
                raise ValueError(f"{path}: recall must be between 0 and 1")
            dataset = (
                result.get("dataset_metadata", {}).get("name")
                or case.get("custom_case", {}).get("dataset_name")
                or str(case["case_id"])
            )
            panel = (dataset, config.get("metric_type", ""))
            # Concurrency is swept to find peak QPS, not a separate query point.
            query_case = {
                key: value for key, value in case.items()
                if key != "concurrency_search_config"
            }
            query_case.setdefault("nq", 1)
            identity = json.dumps(
                [panel, task.get("db"), task.get("db_config"), config, query_case],
                sort_keys=True,
            )
            point = {
                "panel": panel, "index": config["index"], "config": config,
                "topk": case["k"],
                "qps": qps, "recall": recall, "source": path.name,
            }
            if identity not in points or qps > points[identity]["qps"]:
                points[identity] = point
    if not points:
        raise ValueError("No valid search results found")
    return list(points.values()), len(files)


def curve_label(point):
    config = point["config"]
    parameters = [f"top-k={point['topk']}"]
    for label, aliases in (
        ("M", ("M", "m")),
        ("efConstruction", ("efConstruction", "ef_construction")),
        ("nlist", ("nlist",)),
    ):
        for key in aliases:
            if key in config:
                parameters.append(f"{label}={config[key]}")
                break
    if point["index"] == "IVF_RABITQ":
        bits = config.get("rbq_bits_query", "default")
        parameters.append(f"rbq_bits_query={bits}")
    if parameters:
        return f"{point['index']} ({', '.join(parameters)})"
    return point["index"]



def plot_interactive(points, output):
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots

    panels = defaultdict(list)
    for point in points:
        panels[point["panel"]].append(point)
    panel_names = sorted(panels)
    fig = make_subplots(
        rows=1, cols=len(panel_names),
        subplot_titles=[escape(f"{d} | {m}") for d, m in panel_names],
    )
    indices = sorted({curve_label(p) for p in points})
    colors = {index: qualitative.Plotly[i % len(qualitative.Plotly)]
              for i, index in enumerate(indices)}
    shown = set()
    for col, panel in enumerate(panel_names, 1):
        groups = defaultdict(list)
        for point in panels[panel]:
            groups[curve_label(point)].append(point)
        for index, group in sorted(groups.items()):
            group.sort(key=lambda p: (p["recall"], p["qps"]))
            details = [
                "<br>".join(
                    f"{escape(str(key))}={escape(str(value))}"
                    for key, value in sorted(p["config"].items())
                    if key != "index"
                ) + f"<br>top-k={p['topk']}<br>Source: " + escape(p["source"])
                for p in group
            ]
            fig.add_trace(go.Scatter(
                x=[p["recall"] for p in group], y=[p["qps"] for p in group],
                mode="lines+markers", name=index, legendgroup=index,
                showlegend=index not in shown, line={"color": colors[index]},
                marker={"size": 8}, customdata=details,
                hovertemplate=(
                    "Recall: %{x:.2%}<br>QPS: %{y:,.2f}<br>"
                    "%{customdata}<extra>%{fullData.name}</extra>"
                ),
            ), row=1, col=col)
            shown.add(index)
        fig.update_xaxes(title_text="Recall", tickformat=".1%", row=1, col=col)
        fig.update_yaxes(title_text="Peak QPS", rangemode="tozero", row=1, col=col)
    fig.update_layout(
        template="plotly_white", height=700, hovermode="closest",
        title="Peak QPS vs Recall", margin={"t": 100, "b": 80},
        legend={"title": {"text": "Index / top-k / build parameters (click to toggle)"},
                "orientation": "h", "y": -0.2, "x": 0},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Embed Plotly so the resulting file also works offline.
    fig.write_html(output, include_plotlyjs=True, full_html=True,
                   config={"scrollZoom": True, "responsive": True, "displaylogo": False})
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--output", type=Path, help="Interactive HTML output path")
    args = parser.parse_args()
    output = args.output or args.results_dir / "qps_recall.html"
    if output.suffix.lower() != ".html":
        parser.error("--output must have an .html extension")
    try:
        points, file_count = load_points(args.results_dir)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    plot_interactive(points, output)
    print(f"Read {file_count} files; plotted {len(points)} configurations -> {output}")
    for label in sorted({curve_label(p) for p in points}):
        print(f"  {label}: {sum(curve_label(p) == label for p in points)} points")


if __name__ == "__main__":
    main()
