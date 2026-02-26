# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Long-document retrieval benchmark example.

Compares TreeSearch strategies (BM25, BestFirst, MCTS) on QASPER/QuALITY/custom datasets,
with automatic token consumption and latency tracking.

Usage:
    # QASPER benchmark:
    python examples/benchmark/qa_benchmark.py --dataset qasper --data-path data/qasper.json --index-dir ./indexes/

    # QuALITY benchmark:
    python examples/benchmark/qa_benchmark.py --dataset quality --data-path data/quality.jsonl --index-dir ./indexes/

    # Custom JSONL dataset (BM25-only, no API key needed):
    python examples/benchmark/qa_benchmark.py --dataset custom --data-path data.jsonl --index-dir ./indexes/ --strategies bm25

    # Compare BM25 vs BestFirst:
    python examples/benchmark/qa_benchmark.py --dataset qasper --data-path data/qasper.json --index-dir ./indexes/ --strategies bm25 best_first
"""
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from treesearch import load_documents
from treesearch.benchmark import (
    run_benchmark,
    print_report,
    print_comparison,
)


async def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Long-document retrieval benchmark with cost tracking"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["qasper", "quality", "custom"],
        help="Benchmark dataset (qasper, quality, custom)"
    )
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--index-dir", type=str, required=True, help="Directory containing index JSON files")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument(
        "--strategies", type=str, nargs="+", default=["bm25", "best_first"],
        help="Search strategies to compare (bm25, best_first, mcts, llm)"
    )
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results per query")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent evaluations")
    args = parser.parse_args()

    # Load pre-indexed documents
    documents = load_documents(args.index_dir)
    if not documents:
        print(f"No indexes found in {args.index_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(documents)} document(s) from {args.index_dir}")
    print(f"Dataset: {args.dataset} | Strategies: {args.strategies} | Max samples: {args.max_samples}\n")

    # Run benchmark
    reports = await run_benchmark(
        dataset=args.dataset,
        documents=documents,
        data_path=args.data_path,
        strategies=args.strategies,
        model=args.model,
        max_samples=args.max_samples,
        top_k=args.top_k,
        max_concurrency=args.concurrency,
        output_dir=args.output_dir,
    )

    # Print individual reports
    for report in reports:
        print_report(report)

    # Print comparison table
    if len(reports) > 1:
        print_comparison(reports)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
