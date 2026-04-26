# -*- coding: utf-8 -*-
"""Public external baseline references used for paper comparison tables.

These are not local reproductions. They document published/leaderboard numbers
so experiment reports can clearly separate "ours ran locally" from "external
reference".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "evaluation" / "output" / "external_baselines.json"


EXTERNAL_BASELINES: list[dict[str, Any]] = [
    {
        "benchmark": "GraphRAG-Benchmark",
        "source": "GraphRAG-Bench/GraphRAG-Benchmark README and leaderboard",
        "url": "https://graphrag-bench.github.io/",
        "method": "Fast-GraphRAG",
        "dataset": "Novel",
        "metric": "Fact Retrieval Accuracy",
        "value": 60.08,
        "unit": "percent",
        "note": "Public reference from benchmark paper/leaderboard; not reproduced locally.",
    },
    {
        "benchmark": "GraphRAG-Benchmark",
        "source": "GraphRAG-Bench/GraphRAG-Benchmark README and leaderboard",
        "url": "https://graphrag-bench.github.io/",
        "method": "LightRAG",
        "dataset": "Medical",
        "metric": "Contextual Summarization Accuracy",
        "value": 69.37,
        "unit": "percent",
        "note": "Public reference from benchmark paper/leaderboard; not reproduced locally.",
    },
    {
        "benchmark": "GraphRAG-Benchmark",
        "source": "GraphRAG-Bench/GraphRAG-Benchmark README and leaderboard",
        "url": "https://graphrag-bench.github.io/",
        "method": "LightRAG",
        "dataset": "Medical",
        "metric": "Creative Generation Factual Score",
        "value": 70.84,
        "unit": "percent",
        "note": "Public reference from benchmark paper/leaderboard; not reproduced locally.",
    },
    {
        "benchmark": "GraphRAG-Benchmark",
        "source": "GraphRAG-Bench/GraphRAG-Benchmark README and leaderboard",
        "url": "https://graphrag-bench.github.io/",
        "method": "HippoRAG2",
        "dataset": "Medical",
        "metric": "Fact Retrieval ROUGE-L",
        "value": 33.92,
        "unit": "percent",
        "note": "Public reference from benchmark paper/leaderboard; not reproduced locally.",
    },
]


def write_external_baselines(output_path: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "external_reference_only",
        "baselines": EXTERNAL_BASELINES,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), **payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write public external baseline references")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(write_external_baselines(args.output_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
