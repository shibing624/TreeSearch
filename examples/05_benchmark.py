# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Benchmark comparing retrieval strategies on tree-structured documents.

Strategies compared:
  1. BM25-only: NodeBM25Index standalone search (zero LLM cost)
  2. BestFirst (default): BM25 pre-scoring + LLM tree search
  3. MCTS: Monte Carlo Tree Search with LLM evaluation
  4. LLM single-pass: one LLM call per document

Metrics: Precision@K, Recall@K, MRR, NDCG@K, Hit@K, F1@K, LLM calls, latency.

Usage:
    # Default: BM25-only, top 3 queries (no API key needed):
    python examples/05_benchmark.py

    # All queries:
    python examples/05_benchmark.py --max-queries 0

    # With specific strategies (needs OPENAI_API_KEY):
    python examples/05_benchmark.py --strategies bm25 best_first

    # Force re-index:
    python examples/05_benchmark.py --force-index
"""
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import (
    Document,
    NodeBM25Index,
    build_index,
    load_index,
    search,
    evaluate_query,
    flatten_tree,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

pwd_path = os.path.dirname(__file__)

DATA_DIR = os.path.join(pwd_path, "data", "markdowns")
INDEX_DIR = os.path.join(pwd_path, "indexes", "benchmark")
GT_PATH = os.path.join(pwd_path, "data", "benchmark_ground_truth.json")
RESULT_PATH = os.path.join(pwd_path, "results", "benchmark_results.json")


def load_ground_truth() -> list[dict]:
    """Load benchmark ground truth queries."""
    with open(GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


async def ensure_indexes(force: bool = False) -> None:
    """Build indexes if not exist."""
    if not force and os.path.isdir(INDEX_DIR) and any(f.endswith(".json") for f in os.listdir(INDEX_DIR)):
        print(f"Indexes exist in {INDEX_DIR}, skipping build. Use --force-index to rebuild.")
        return
    print("Building indexes...")
    pattern = os.path.join(DATA_DIR, "*.md")
    await build_index(
        paths=[pattern],
        output_dir=INDEX_DIR,
        if_add_node_summary=True,
        if_add_node_text=True,
        if_add_doc_description=True,
        if_add_node_id=True,
    )
    print(f"Indexes built in {INDEX_DIR}/")


def load_documents() -> list[Document]:
    """Load all indexed documents."""
    documents = []
    for fp in sorted(os.listdir(INDEX_DIR)):
        if not fp.endswith(".json") or fp.startswith("_"):
            continue
        data = load_index(os.path.join(INDEX_DIR, fp))
        doc = Document(
            doc_id=os.path.splitext(fp)[0].replace("_structure", ""),
            doc_name=data["doc_name"],
            structure=data["structure"],
            doc_description=data.get("doc_description", ""),
        )
        documents.append(doc)
    return documents


def resolve_ground_truth_node_ids(documents: list[Document], gt_queries: list[dict]) -> list[dict]:
    """Map ground truth titles to actual node_ids from indexed documents."""
    # Build title -> node_id map per doc
    doc_title_map: dict[str, dict[str, str]] = {}
    for doc in documents:
        title_map = {}
        for node in flatten_tree(doc.structure):
            title = node.get("title", "")
            nid = node.get("node_id", "")
            if title and nid:
                title_map[title.lower().strip()] = nid
        doc_title_map[doc.doc_id] = title_map

    resolved = []
    for q in gt_queries:
        relevant_node_ids = []
        for doc_name in q["relevant_docs"]:
            # Match doc_id by checking if doc_id contains the doc_name
            matched_doc_id = None
            for doc in documents:
                if doc_name in doc.doc_id.lower() or doc_name in doc.doc_name.lower():
                    matched_doc_id = doc.doc_id
                    break
            if not matched_doc_id:
                continue
            title_map = doc_title_map.get(matched_doc_id, {})
            for title in q["relevant_titles"]:
                nid = title_map.get(title.lower().strip())
                if nid:
                    relevant_node_ids.append(nid)
        resolved.append({
            "query": q["query"],
            "relevant_node_ids": relevant_node_ids,
            "relevant_docs": q["relevant_docs"],
            "difficulty": q.get("difficulty", "medium"),
        })
    return resolved


def run_bm25_search(documents: list[Document], query: str, top_k: int = 10) -> list[str]:
    """BM25-only search: return node_ids ranked by BM25 score."""
    index = NodeBM25Index(documents)
    results = index.search(query, top_k=top_k)
    return [r["node_id"] for r in results]


async def run_strategy_search(
    documents: list[Document],
    query: str,
    strategy: str,
    top_k: int = 10,
) -> tuple[list[str], int]:
    """Run search with a given strategy. Returns (node_ids, llm_calls)."""
    kwargs = {
        "strategy": strategy,
        "top_k_docs": min(3, len(documents)),
        "max_nodes_per_doc": top_k,
        "use_bm25": True,
    }
    if strategy == "best_first":
        kwargs["max_llm_calls"] = 30
    elif strategy == "mcts":
        kwargs["mcts_iterations"] = 10

    result = await search(query=query, documents=documents, **kwargs)

    node_ids = []
    for doc_result in result.documents:
        for node in doc_result["nodes"]:
            node_ids.append(node["node_id"])
    return node_ids[:top_k], result.total_llm_calls


async def run_benchmark(
    documents: list[Document],
    gt_queries: list[dict],
    strategies: list[str],
    top_k: int = 5,
) -> dict:
    """Run full benchmark across all strategies and queries."""
    k_values = [1, 3, 5]
    results = {}

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        query_metrics = []
        total_llm_calls = 0
        total_time = 0.0

        for i, q in enumerate(gt_queries):
            query = q["query"]
            relevant = q["relevant_node_ids"]

            if not relevant:
                print(f"  [{i+1}/{len(gt_queries)}] Skipped (no ground truth): {query[:50]}")
                continue

            t0 = time.time()
            if strategy == "bm25":
                retrieved = run_bm25_search(documents, query, top_k=top_k)
                llm_calls = 0
            else:
                retrieved, llm_calls = await run_strategy_search(
                    documents, query, strategy, top_k=top_k
                )
            elapsed = time.time() - t0

            total_llm_calls += llm_calls
            total_time += elapsed

            metrics = evaluate_query(retrieved, relevant, k_values)
            metrics["llm_calls"] = llm_calls
            metrics["latency"] = round(elapsed, 3)
            metrics["query"] = query
            metrics["difficulty"] = q["difficulty"]
            metrics["retrieved"] = retrieved
            metrics["relevant"] = relevant

            query_metrics.append(metrics)

            hit = "HIT" if metrics["hit@1"] else "miss"
            print(f"  [{i+1}/{len(gt_queries)}] MRR={metrics['mrr']:.2f} "
                  f"P@3={metrics['precision@3']:.2f} R@3={metrics['recall@3']:.2f} "
                  f"NDCG@3={metrics['ndcg@3']:.2f} [{hit}] "
                  f"LLM={llm_calls} {elapsed:.1f}s | {query[:50]}")

        # Aggregate metrics
        if query_metrics:
            avg = {}
            metric_keys = [k for k in query_metrics[0] if isinstance(query_metrics[0][k], (int, float))]
            for key in metric_keys:
                avg[key] = sum(m[key] for m in query_metrics) / len(query_metrics)
            avg["total_llm_calls"] = total_llm_calls
            avg["total_time"] = round(total_time, 2)
            avg["num_queries"] = len(query_metrics)

            # Per-difficulty breakdown
            for diff in ["easy", "medium", "hard"]:
                subset = [m for m in query_metrics if m["difficulty"] == diff]
                if subset:
                    for key in metric_keys:
                        if key not in ("llm_calls", "latency"):
                            avg[f"{diff}_{key}"] = sum(m[key] for m in subset) / len(subset)

            results[strategy] = {
                "summary": {k: round(v, 4) if isinstance(v, float) else v for k, v in avg.items()},
                "per_query": query_metrics,
            }

    return results


def print_comparison_table(results: dict) -> None:
    """Print a comparison table of all strategies."""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}")

    strategies = list(results.keys())
    metrics_to_show = ["mrr", "precision@3", "recall@3", "ndcg@3", "hit@1", "f1@3",
                       "llm_calls", "latency"]

    # Header
    header = f"{'Metric':<20}"
    for s in strategies:
        header += f"{s.upper():>15}"
    print(header)
    print("-" * (20 + 15 * len(strategies)))

    # Rows
    for metric in metrics_to_show:
        row = f"{metric:<20}"
        values = []
        for s in strategies:
            val = results[s]["summary"].get(metric, 0)
            values.append(val)
            if isinstance(val, float):
                row += f"{val:>15.4f}"
            else:
                row += f"{val:>15}"
        # Mark best
        print(row)

    # Total LLM calls and time
    print("-" * (20 + 15 * len(strategies)))
    row = f"{'total_llm_calls':<20}"
    for s in strategies:
        val = results[s]["summary"].get("total_llm_calls", 0)
        row += f"{val:>15}"
    print(row)

    row = f"{'total_time (s)':<20}"
    for s in strategies:
        val = results[s]["summary"].get("total_time", 0)
        row += f"{val:>15.2f}"
    print(row)

    row = f"{'num_queries':<20}"
    for s in strategies:
        val = results[s]["summary"].get("num_queries", 0)
        row += f"{val:>15}"
    print(row)

    # Per-difficulty breakdown
    print(f"\n{'='*80}")
    print("PER-DIFFICULTY BREAKDOWN (MRR / NDCG@3)")
    print(f"{'='*80}")
    header = f"{'Difficulty':<20}"
    for s in strategies:
        header += f"{s.upper():>15}"
    print(header)
    print("-" * (20 + 15 * len(strategies)))
    for diff in ["easy", "medium", "hard"]:
        row = f"{diff:<20}"
        for s in strategies:
            mrr = results[s]["summary"].get(f"{diff}_mrr", 0)
            ndcg = results[s]["summary"].get(f"{diff}_ndcg@3", 0)
            row += f"{mrr:.2f}/{ndcg:.2f}".rjust(15)
        print(row)

    # Key takeaways
    print(f"\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}")
    if "bm25" in results and len(strategies) > 1:
        bm25_mrr = results["bm25"]["summary"].get("mrr", 0)
        for s in strategies:
            if s != "bm25":
                s_mrr = results[s]["summary"].get("mrr", 0)
                delta = s_mrr - bm25_mrr
                print(f"  {s.upper()} vs BM25: MRR delta = {delta:+.4f}")
    print(f"  No vector embeddings needed. No chunk splitting.")
    print(f"  Tree structure preserves document hierarchy for precise retrieval.")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="TreeSearch benchmark")
    parser.add_argument("--bm25-only", action="store_true", help="Run BM25-only benchmark (no API key needed)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Strategies to benchmark (bm25, best_first, mcts, llm)")
    parser.add_argument("--force-index", action="store_true", help="Force rebuild indexes")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results per query")
    parser.add_argument("--max-queries", type=int, default=3, help="Max number of queries to evaluate (default: 3)")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    args = parser.parse_args()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # Determine strategies
    if args.bm25_only:
        strategies = ["bm25"]
    elif args.strategies:
        strategies = args.strategies
    else:
        strategies = ["bm25"]

    # Build indexes (needs API key for first time)
    needs_index = not os.path.isdir(INDEX_DIR) or not any(
        f.endswith(".json") and not f.startswith("_") for f in os.listdir(INDEX_DIR)
    ) if os.path.isdir(INDEX_DIR) else True

    if needs_index or args.force_index:
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY needed to build indexes for the first time.")
            print("Set OPENAI_API_KEY and run again, or build indexes via:")
            print("  python examples/02_multi_doc_search.py")
            sys.exit(1)
        await ensure_indexes(force=args.force_index)
    else:
        print(f"Using existing indexes in {INDEX_DIR}/")

    # Load documents and ground truth
    documents = load_documents()
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        node_count = len(flatten_tree(doc.structure))
        print(f"  - [{doc.doc_id}] {doc.doc_name} ({node_count} nodes)")

    gt_queries = load_ground_truth()
    resolved = resolve_ground_truth_node_ids(documents, gt_queries)
    valid = [q for q in resolved if q["relevant_node_ids"]]
    if args.max_queries > 0 and len(valid) > args.max_queries:
        valid = valid[:args.max_queries]
    print(f"\nGround truth: {len(gt_queries)} queries, {len(valid)} selected for evaluation")

    if not valid:
        print("ERROR: No ground truth queries could be resolved. Check indexes.")
        sys.exit(1)

    # Run benchmark
    t_start = time.time()
    results = await run_benchmark(documents, valid, strategies, top_k=args.top_k)
    total_elapsed = time.time() - t_start

    # Print comparison
    print_comparison_table(results)
    print(f"\nTotal benchmark time: {total_elapsed:.1f}s")

    # Save results
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    output = {
        "benchmark_info": {
            "strategies": strategies,
            "num_documents": len(documents),
            "num_queries": len(valid),
            "top_k": args.top_k,
            "total_time": round(total_elapsed, 2),
        },
        "results": {
            s: results[s]["summary"] for s in results
        },
        "per_query": {
            s: [{
                "query": m["query"],
                "difficulty": m["difficulty"],
                "mrr": m["mrr"],
                "precision@3": m["precision@3"],
                "recall@3": m["recall@3"],
                "ndcg@3": m["ndcg@3"],
                "llm_calls": m["llm_calls"],
                "latency": m["latency"],
            } for m in results[s]["per_query"]] for s in results
        },
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
