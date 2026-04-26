# -*- coding: utf-8 -*-
"""Evaluation-scoped CodeSearchNet runner.

This is a thin wrapper around the existing benchmark implementation that keeps
dataset cache, indexes, and result reports under ``evaluation/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.benchmark.codesearchnet_benchmark import (
    EmbeddingIndex,
    TreeSearchCodeAutoIndex,
    TreeSearchCodeGraphRAGIndex,
    TreeSearchCodeIndex,
    TreeSearchCodeTreeIndex,
    ZhipuEmbeddingClient,
    aevaluate_retrieval,
    evaluate_retrieval,
    flatten_tree,
    load_codesearchnet_from_hf,
)
from evaluation.evaluate import ZHIPU_API_KEY_ENV, load_env_file


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CodeSearchNetPaths:
    cache_dir: Path
    output_dir: Path
    index_dir: Path


def default_paths(language: str) -> CodeSearchNetPaths:
    return CodeSearchNetPaths(
        cache_dir=REPO_ROOT / "evaluation" / "data" / "codesearchnet_cache",
        output_dir=REPO_ROOT / "evaluation" / "output" / "codesearchnet" / language,
        index_dir=REPO_ROOT / "evaluation" / "output" / "indexes" / "codesearchnet" / language,
    )


async def run_codesearchnet_eval(
    language: str = "python",
    split: str = "test",
    max_samples: int = 50,
    max_corpus: int = 1000,
    with_graphrag: bool = True,
    with_embedding: bool = False,
    zhipu_api_key: str | None = None,
) -> dict[str, Any]:
    paths = default_paths(language)
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.index_dir.mkdir(parents=True, exist_ok=True)

    query_samples, corpus = load_codesearchnet_from_hf(
        language=language,
        split=split,
        max_samples=max_samples,
        max_corpus=max_corpus,
        cache_dir=str(paths.cache_dir),
    )

    results = {}
    ts_index = TreeSearchCodeIndex()
    index_time = await ts_index.index(corpus, str(paths.index_dir))
    ts_metrics = evaluate_retrieval(query_samples, ts_index)
    ts_metrics["index_time"] = index_time
    ts_metrics["num_nodes"] = sum(len(flatten_tree(d.structure)) for d in ts_index.documents)
    results["treesearch_fts5"] = ts_metrics

    tree_index = TreeSearchCodeTreeIndex(ts_index)
    tree_metrics = evaluate_retrieval(query_samples, tree_index)
    tree_metrics["index_time"] = index_time
    tree_metrics["num_nodes"] = ts_metrics["num_nodes"]
    results["treesearch_tree"] = tree_metrics

    auto_index = TreeSearchCodeAutoIndex(ts_index)
    auto_metrics = evaluate_retrieval(query_samples, auto_index)
    auto_metrics["index_time"] = index_time
    auto_metrics["num_nodes"] = ts_metrics["num_nodes"]
    results["treesearch_auto"] = auto_metrics

    if with_graphrag:
        graph_index = TreeSearchCodeGraphRAGIndex(ts_index)
        graph_metrics = await aevaluate_retrieval(query_samples, graph_index)
        graph_metrics["index_time"] = index_time
        graph_metrics["num_nodes"] = ts_metrics["num_nodes"]
        results["treesearch_graphrag"] = graph_metrics

    if with_embedding:
        env = load_env_file(REPO_ROOT / ".env")
        api_key = zhipu_api_key or os.environ.get(ZHIPU_API_KEY_ENV, "") or env.get(ZHIPU_API_KEY_ENV, "")
        if not api_key:
            raise ValueError(f"zhipu_api_key or {ZHIPU_API_KEY_ENV} is required when with_embedding=True")
        emb_index = EmbeddingIndex(
            emb_client=ZhipuEmbeddingClient(api_key=api_key),
            cache_dir=str(paths.output_dir / "embedding_cache"),
        )
        emb_index_time = emb_index.index(corpus)
        emb_metrics = evaluate_retrieval(query_samples, emb_index)
        emb_metrics["index_time"] = emb_index_time
        emb_metrics["num_embeddings"] = len(corpus.samples)
        results["embedding"] = emb_metrics

    report_path = paths.output_dir / f"{language}_benchmark_report.json"
    report = {
        "language": language,
        "split": split,
        "max_samples": max_samples,
        "max_corpus": max_corpus,
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_path": str(report_path), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CodeSearchNet evaluation under evaluation/")
    parser.add_argument("--language", default="python", choices=["python", "java", "javascript", "go", "ruby", "php"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-corpus", type=int, default=1000)
    parser.add_argument("--without-graphrag", action="store_true")
    parser.add_argument("--with-embedding", action="store_true")
    parser.add_argument("--zhipu-api-key", default=None)
    args = parser.parse_args()

    report = asyncio.run(
        run_codesearchnet_eval(
            language=args.language,
            split=args.split,
            max_samples=args.max_samples,
            max_corpus=args.max_corpus,
            with_graphrag=not args.without_graphrag,
            with_embedding=args.with_embedding,
            zhipu_api_key=args.zhipu_api_key,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
