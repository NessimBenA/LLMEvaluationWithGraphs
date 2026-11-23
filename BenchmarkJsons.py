import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import networkx as nx
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from CleanJsons import DEFAULT_REPLY_DIR, STANDARDIZED_PREFIX, StandardGraph, standardize_reply
from GenerateJsons import DEFAULT_LOG_FILE, reply_path

DEFAULT_RESULTS_DIR: Final = "benchmark_results"
SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"
BAR: Final = "#2a78d6"
NO_GRAPH: Final = "#f0efec"
SEQUENTIAL: Final = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
LOG_COLUMNS: Final = ("subject", "model", "num_retries_attempted")


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    weight: float
    higher_is_better: bool


METRICS: Final = (
    Metric("average_degree", "Average Node Degree", 2.0, True),
    Metric("num_nodes", "Node Count", 1.0, True),
    Metric("num_edges", "Edge Count", 1.0, True),
    Metric("density", "Graph Density", 0.8, True),
    Metric("num_components", "Connected Components", 1.0, False),
    Metric("largest_component_ratio", "Largest Component Ratio", 0.8, True),
)
TOTAL_WEIGHT: Final = sum(metric.weight for metric in METRICS)


@dataclass(frozen=True)
class Run:
    subject: str
    model: str
    attempts: int


@dataclass(frozen=True)
class GraphResult:
    run: Run
    retry: int
    path: Path
    metrics: dict[str, float]


@dataclass(frozen=True)
class SubjectResult:
    run: Run
    graphs: int
    metrics: dict[str, float] | None
    score: float | None


@dataclass(frozen=True)
class ModelResult:
    rank: int
    model: str
    score: float
    subjects_scored: int
    subjects_without_graph: int


def graph_metrics(graph: StandardGraph) -> dict[str, float]:
    digraph: nx.DiGraph[str] = nx.DiGraph()
    digraph.add_nodes_from(node["id"] for node in graph["nodes"])
    digraph.add_edges_from((edge["source"], edge["target"]) for edge in graph["edges"])
    nodes = digraph.number_of_nodes()
    edges = digraph.number_of_edges()
    components = [len(component) for component in nx.weakly_connected_components(digraph)]
    return {
        "average_degree": 2 * edges / nodes,
        "num_nodes": float(nodes),
        "num_edges": float(edges),
        "density": float(nx.density(digraph)),
        "num_components": float(len(components)),
        "largest_component_ratio": max(components) / nodes,
    }


def latest_runs(log_path: Path) -> list[Run]:
    runs: dict[tuple[str, str], Run] = {}
    with log_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in LOG_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{log_path} has no column {', '.join(missing)}; it is not a GenerateJsons.py log.")
        for row in reader:
            attempts = str(row["num_retries_attempted"]).strip()
            run = Run(str(row["subject"]), str(row["model"]), int(attempts) if attempts.isdigit() else 0)
            runs[(run.subject, run.model)] = run
    return list(runs.values())


def load_graphs(runs: list[Run], reply_dir: Path) -> list[GraphResult]:
    results: list[GraphResult] = []
    for run in runs:
        for retry in range(1, run.attempts + 1):
            raw = reply_path(reply_dir, run.subject, run.model, retry)
            path = raw.with_name(STANDARDIZED_PREFIX + raw.name)
            if not path.is_file():
                continue
            graph, reason = standardize_reply(path.read_text(encoding="utf-8"))
            if graph is None:
                print(f"Skipped {path.name}: {reason}")
                continue
            results.append(GraphResult(run, retry, path, graph_metrics(graph)))
    return results


def normalized(value: float, low: float, high: float, higher_is_better: bool) -> float:
    if high == low:
        return 1.0
    scaled = (value - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def composite_scores(averaged: dict[Run, dict[str, float]]) -> dict[Run, float]:
    scores: dict[Run, float] = {}
    for subject in sorted({run.subject for run in averaged}):
        group = {run: metrics for run, metrics in averaged.items() if run.subject == subject}
        for run, metrics in group.items():
            weighted = 0.0
            for metric in METRICS:
                values = [other[metric.key] for other in group.values()]
                weighted += metric.weight * normalized(metrics[metric.key], min(values), max(values), metric.higher_is_better)
            scores[run] = weighted / TOTAL_WEIGHT
    return scores


def subject_results(runs: list[Run], graphs: list[GraphResult]) -> list[SubjectResult]:
    averaged: dict[Run, dict[str, float]] = {}
    counts: dict[Run, int] = {}
    for run in runs:
        own = [graph.metrics for graph in graphs if graph.run == run]
        counts[run] = len(own)
        if own:
            averaged[run] = {metric.key: statistics.mean(values[metric.key] for values in own) for metric in METRICS}
    scores = composite_scores(averaged)
    ordered = sorted(runs, key=lambda run: (run.subject, run.model))
    return [SubjectResult(run, counts[run], averaged.get(run), scores.get(run)) for run in ordered]


def model_ranking(results: list[SubjectResult]) -> list[ModelResult]:
    rows: list[tuple[float, str, int, int]] = []
    for model in sorted({result.run.model for result in results}):
        own = [result for result in results if result.run.model == model]
        scores = [result.score for result in own if result.score is not None]
        if scores:
            rows.append((statistics.mean(scores), model, len(scores), len(own) - len(scores)))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [ModelResult(rank, model, score, scored, missing) for rank, (score, model, scored, missing) in enumerate(rows, start=1)]


def number(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_tables(output_dir: Path, graphs: list[GraphResult], results: list[SubjectResult], ranking: list[ModelResult]) -> None:
    keys = [metric.key for metric in METRICS]
    write_csv(
        output_dir / "graph_metrics.csv",
        ["subject", "model", "retry", "file", *keys],
        [[graph.run.subject, graph.run.model, str(graph.retry), graph.path.name, *(number(graph.metrics[key]) for key in keys)] for graph in graphs],
    )
    write_csv(
        output_dir / "subject_scores.csv",
        ["subject", "model", "attempts", "graphs", *keys, "composite_score"],
        [
            [
                result.run.subject,
                result.run.model,
                str(result.run.attempts),
                str(result.graphs),
                *(number(None if result.metrics is None else result.metrics[key]) for key in keys),
                number(result.score),
            ]
            for result in results
        ],
    )
    write_csv(
        output_dir / "model_ranking.csv",
        ["rank", "model", "score", "subjects_scored", "subjects_without_graph"],
        [[str(row.rank), row.model, number(row.score), str(row.subjects_scored), str(row.subjects_without_graph)] for row in ranking],
    )


def style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def plot_ranking(ranking: list[ModelResult], path: Path) -> None:
    figure = Figure(figsize=(9, 1.6 + 0.45 * len(ranking)), facecolor=SURFACE)
    axes = figure.add_subplot()
    style(axes)
    bars = axes.barh([row.model for row in ranking], [row.score for row in ranking], height=0.55, color=BAR, zorder=2)
    axes.bar_label(bars, labels=[f"{row.score:.3f}" for row in ranking], padding=4, color=INK_SECONDARY, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlim(0, 1.1)
    axes.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    axes.tick_params(axis="y", colors=INK, labelsize=10)
    axes.set_xlabel("Mean composite score over subjects (0 to 1, higher is better)", color=INK_SECONDARY, fontsize=9)
    axes.set_title("Model ranking by composite graph score", color=INK, fontsize=12, loc="left", pad=12)
    figure.tight_layout()
    figure.savefig(path, dpi=200)


def plot_subject_scores(results: list[SubjectResult], ranking: list[ModelResult], path: Path) -> None:
    subjects = sorted({result.run.subject for result in results})
    ranked = [row.model for row in ranking]
    models = ranked + sorted({result.run.model for result in results} - set(ranked))
    scores = {(result.run.model, result.run.subject): result.score for result in results}
    cells = [[scores.get((model, subject)) for subject in subjects] for model in models]
    figure = Figure(figsize=(3 + 1.5 * len(subjects), 1.8 + 0.5 * len(models)), facecolor=SURFACE)
    axes = figure.add_subplot()
    style(axes)
    colormap = LinearSegmentedColormap.from_list("composite", list(SEQUENTIAL)).with_extremes(bad=NO_GRAPH)
    image = axes.imshow([[math.nan if cell is None else cell for cell in row] for row in cells], cmap=colormap, vmin=0.0, vmax=1.0, aspect="auto")
    for row_index, row in enumerate(cells):
        for column_index, cell in enumerate(row):
            text = "no graph" if cell is None else f"{cell:.2f}"
            color = "#ffffff" if cell is not None and cell >= 0.55 else INK
            axes.text(column_index, row_index, text, ha="center", va="center", color=color, fontsize=9)
    axes.set_xticks(range(len(subjects)), subjects, rotation=20, ha="right")
    axes.set_yticks(range(len(models)), models)
    axes.tick_params(axis="y", colors=INK, labelsize=9)
    for spine in axes.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.04, pad=0.02)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    colorbar.set_label("Composite score", color=INK_SECONDARY, fontsize=9)
    axes.set_title("Composite score per subject (models compared within each subject)", color=INK, fontsize=12, loc="left", pad=12)
    figure.tight_layout()
    figure.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every standardized graph on six metrics and rank the models by the weighted composite score.")
    parser.add_argument("--log", default=DEFAULT_LOG_FILE, help=f"GenerateJsons.py log (default: {DEFAULT_LOG_FILE}).")
    parser.add_argument("--json-dir", default=DEFAULT_REPLY_DIR, help=f"Folder with the standardized_*.json files (default: {DEFAULT_REPLY_DIR}).")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help=f"Folder for the CSV tables and charts (default: {DEFAULT_RESULTS_DIR}).")
    args = parser.parse_args()
    log_path = Path(str(args.log))
    reply_dir = Path(str(args.json_dir))
    output_dir = Path(str(args.output_dir))
    if not log_path.is_file():
        sys.exit(f"Log not found: {log_path}. Run GenerateJsons.py first.")
    if not reply_dir.is_dir():
        sys.exit(f"Folder not found: {reply_dir}. Run GenerateJsons.py and CleanJsons.py first.")
    runs = latest_runs(log_path)
    graphs = load_graphs(runs, reply_dir)
    if not graphs:
        sys.exit(f"No standardized graph in {reply_dir} matches the runs in {log_path}. Run CleanJsons.py first.")
    results = subject_results(runs, graphs)
    ranking = model_ranking(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(output_dir, graphs, results, ranking)
    plot_ranking(ranking, output_dir / "model_ranking.png")
    plot_subject_scores(results, ranking, output_dir / "subject_scores.png")
    print(f"{len(runs)} subject and model runs in the log, {len(graphs)} graphs scored.\n")
    print(f"{'Rank':<6}{'Score':<9}{'Subjects':<10}Model")
    for row in ranking:
        print(f"{row.rank:<6}{row.score:<9.4f}{f'{row.subjects_scored}/{row.subjects_scored + row.subjects_without_graph}':<10}{row.model}")
    print(f"\nTables and charts written to {output_dir}")


if __name__ == "__main__":
    main()
