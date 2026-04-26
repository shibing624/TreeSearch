# -*- coding: utf-8 -*-
"""Report formatting for GraphRAG experiment outputs."""

from pathlib import Path


SUMMARY_COLUMNS = (
    "method",
    "count",
    "node_recall",
    "source_path_recall",
    "citation_precision",
    "citation_recall",
    "line_grounding_accuracy",
    "task_success_rate",
    "latency_seconds",
    "avg_latency_seconds",
    "llm_calls",
    "llm_calls_per_query",
)


def write_reports(
    report: dict,
    output_path: Path | None = None,
    markdown_path: Path | None = None,
    latex_path: Path | None = None,
) -> None:
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_markdown_summary(report), encoding="utf-8")
    if latex_path is not None:
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        latex_path.write_text(format_latex_summary(report), encoding="utf-8")


def format_markdown_summary(report: dict) -> str:
    lines = [
        "# GraphRAG RealRepoBench Summary",
        "",
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---:" if column != "method" else "---" for column in SUMMARY_COLUMNS) + " |",
    ]
    for method, summary in report["summary"].items():
        lines.append("| " + " | ".join(_format_cell(method, summary, column) for column in SUMMARY_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def format_latex_summary(report: dict) -> str:
    header = " & ".join(SUMMARY_COLUMNS) + r" \\"
    rows = [r"\begin{tabular}{lrrrrrrrrrrr}", r"\toprule", header, r"\midrule"]
    for method, summary in report["summary"].items():
        rows.append(" & ".join(_format_cell(method, summary, column) for column in SUMMARY_COLUMNS) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def _format_cell(method: str, summary: dict, column: str) -> str:
    if column == "method":
        return method
    value = summary[column]
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
