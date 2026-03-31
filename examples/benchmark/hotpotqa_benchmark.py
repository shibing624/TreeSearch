# -*- coding: utf-8 -*-
"""
HotpotQA Supporting Passage Recall Benchmark for TreeSearch.

Evaluates the ability to retrieve ALL relevant passages needed for multi-hop
reasoning — not just one, but both supporting passages simultaneously.

Dataset: HotpotQA (fullwiki or distractor split)
  - Each question has exactly 2 supporting facts (from 2 different articles)
  - supporting_facts: [[title, sent_idx], ...]
  - context: [[title, [sent0, sent1, ...]], ...]

Key Metrics:
  SP-Recall@k     Supporting Passage Recall — fraction of supporting passages
                  found in top-k retrieved nodes
  2-hop-Cov@k    Two-hop Coverage — whether BOTH supporting passages appear
                  in top-k (both required articles covered)
  MRR             Standard retrieval metric (comparable to other benchmarks)

Usage:
  # Quick test (50 samples, fullwiki)
  python examples/benchmark/hotpotqa_benchmark.py --max-samples 50

  # Full evaluation (500 samples)
  python examples/benchmark/hotpotqa_benchmark.py --max-samples 500 --strategies fts5 tree


  HOTPOTQA BENCHMARK RESULTS
======================================================================
Strategy                 MRR    SP-Rec@3    2hop-Cov@3    SP-Rec@5    2hop-Cov@5   SP-Rec@10   2hop-Cov@10     Latency
----------------------------------------------------------------------
fts5                  1.0000      1.0000        1.0000      1.0000        1.0000      1.0000        1.0000      0.015s
tree                  0.9056      0.9762        0.9762      1.0000        1.0000      1.0000        1.0000      0.005s

======================================================================
FINANCEBENCH END-TO-END RAG EVALUATION
======================================================================
Strategy                         EM        F1     Ret.Lat   PromptTok
----------------------------------------------------------------------
naive_rag                    0.0000    0.1555      0.048s        2308
treesearch_flat              0.0000    0.1479      0.040s        2308
treesearch_tree              0.0000    0.1498      0.071s        2148
======================================================================
"""
import asyncio
import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from treesearch.tree import Document, flatten_tree
from treesearch.fts import FTS5Index
from treesearch.tree_searcher import TreeSearcher
from treesearch.config import set_config, TreeSearchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "benchmark_results", "hotpotqa")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HotpotSample:
    """One HotpotQA question with multi-hop supporting evidence."""
    id: str
    question: str
    answer: str
    level: str                          # easy / medium / hard
    # supporting_facts: list of [title, sent_idx]
    supporting_titles: list[str] = field(default_factory=list)   # unique article titles needed
    supporting_facts: list[list] = field(default_factory=list)   # [[title, sent_idx], ...]
    # context: list of [title, sentences] — candidate articles
    context: list[list] = field(default_factory=list)


@dataclass
class HotpotResult:
    """Evaluation result for one HotpotQA sample."""
    sample: HotpotSample
    retrieved_node_ids: list[str] = field(default_factory=list)
    supporting_node_ids: list[str] = field(default_factory=list)   # ground-truth nodes
    # Core metrics
    mrr: float = 0.0
    sp_recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> recall
    twohop_cov_at_k: dict[int, bool] = field(default_factory=dict)  # k -> both_found
    latency: float = 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hotpotqa(
    split: str = "validation",
    max_samples: int = 200,
    cache: bool = True,
) -> list[HotpotSample]:
    """Load HotpotQA from HuggingFace (hotpot_qa dataset).

    Args:
        split: 'validation' (recommended) or 'train'
        max_samples: maximum number of samples to load
        cache: whether to use local pkl cache

    Returns:
        list of HotpotSample
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"hotpotqa_{split}_{max_samples}.pkl")

    if cache and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        logger.info("Loaded %d HotpotQA samples from cache: %s", len(samples), cache_path)
        return samples

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required: pip install datasets"
        )

    logger.info("Loading HotpotQA (%s split) from HuggingFace...", split)
    # hotpot_qa fullwiki: each item has full Wikipedia context
    ds = hf_load("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)

    samples = []
    for row in ds:
        if len(samples) >= max_samples:
            break

        # supporting_facts: {"title": [...], "sent_id": [...]}
        sf = row.get("supporting_facts", {})
        sf_titles = sf.get("title", [])
        sf_sent_ids = sf.get("sent_id", [])
        supporting_facts = list(zip(sf_titles, sf_sent_ids))
        # unique titles needed to answer the question
        supporting_titles = list(dict.fromkeys(sf_titles))

        # context: {"title": [...], "sentences": [[s0, s1, ...], ...]}
        ctx = row.get("context", {})
        ctx_titles = ctx.get("title", [])
        ctx_sentences = ctx.get("sentences", [])
        context = list(zip(ctx_titles, ctx_sentences))

        samples.append(HotpotSample(
            id=row.get("id", f"hp_{len(samples)}"),
            question=row.get("question", ""),
            answer=row.get("answer", ""),
            level=row.get("level", ""),
            supporting_titles=supporting_titles,
            supporting_facts=supporting_facts,
            context=list(context),
        ))

    if cache:
        with open(cache_path, "wb") as f:
            pickle.dump(samples, f)
        logger.info("Cached %d HotpotQA samples to %s", len(samples), cache_path)

    logger.info("Loaded %d HotpotQA samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

def build_documents_for_sample(sample: HotpotSample) -> list[Document]:
    """Build TreeSearch Documents from a HotpotQA sample's context.

    Each candidate article becomes one Document.
    Each sentence within the article becomes a leaf node.

    The fullwiki split provides 10 candidate articles per question.
    The distractor split already limits to relevant + distractor articles.
    """
    documents = []
    for title, sentences in sample.context:
        if not title or not sentences:
            continue
        # Use markdown format so md_to_tree parses sentences as nodes
        text = f"# {title}\n\n" + "\n\n".join(
            f"{i+1}. {s}" for i, s in enumerate(sentences) if s.strip()
        )
        result = asyncio.run(_async_text_to_tree(text))
        structure = result.get("structure", [])

        doc_id = f"{sample.id}_{title[:40].replace(' ', '_')}"
        doc = Document(
            doc_id=doc_id,
            doc_name=title,
            structure=structure,
            metadata={"source_path": "", "article_title": title},
            source_type="text",
        )
        documents.append(doc)
    return documents


async def _async_text_to_tree(text: str) -> dict:
    from treesearch.indexer import text_to_tree
    return await text_to_tree(text_content=text)


# ---------------------------------------------------------------------------
# Ground truth node resolution
# ---------------------------------------------------------------------------

def resolve_supporting_nodes(
    sample: HotpotSample,
    documents: list[Document],
) -> dict[str, list[str]]:
    """Map supporting facts to node_ids, grouped by article title.

    Returns:
        {article_title: [node_id, ...]} for each supporting article
    """
    title_to_doc = {doc.doc_name: doc for doc in documents}
    supporting_by_title: dict[str, list[str]] = defaultdict(list)

    for title, sent_idx in sample.supporting_facts:
        doc = title_to_doc.get(title)
        if doc is None:
            continue
        all_nodes = flatten_tree(doc.structure)
        # Sentence nodes are numbered; find by position or text matching
        for node in all_nodes:
            nid = node.get("node_id", "")
            node_text = (node.get("text", "") or "").strip()
            # Match by sentence index in text (numbered "1. ...", "2. ...")
            expected_prefix = f"{sent_idx + 1}."
            if node_text.startswith(expected_prefix):
                supporting_by_title[title].append(nid)
                break
        else:
            # Fallback: use any node from this article
            if all_nodes:
                supporting_by_title[title].append(all_nodes[0].get("node_id", ""))

    return dict(supporting_by_title)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_sp_recall(
    retrieved_node_ids: list[str],
    supporting_by_title: dict[str, list[str]],
    k: int,
) -> float:
    """SP-Recall@k: fraction of supporting articles that have at least one
    node in the top-k retrieved nodes."""
    if not supporting_by_title:
        return 0.0
    top_k_set = set(retrieved_node_ids[:k])
    hits = sum(
        1 for nodes in supporting_by_title.values()
        if any(n in top_k_set for n in nodes)
    )
    return hits / len(supporting_by_title)


def compute_twohop_coverage(
    retrieved_node_ids: list[str],
    supporting_by_title: dict[str, list[str]],
    k: int,
) -> bool:
    """2-hop Coverage@k: whether ALL supporting articles have at least one node
    in the top-k retrieved nodes.

    This is the key metric: did we find evidence from BOTH required articles?
    """
    if not supporting_by_title:
        return False
    top_k_set = set(retrieved_node_ids[:k])
    return all(
        any(n in top_k_set for n in nodes)
        for nodes in supporting_by_title.values()
    )


def compute_mrr(
    retrieved_node_ids: list[str],
    supporting_by_title: dict[str, list[str]],
) -> float:
    """MRR: reciprocal rank of the first supporting node."""
    all_supporting = {n for nodes in supporting_by_title.values() for n in nodes}
    for rank, nid in enumerate(retrieved_node_ids, 1):
        if nid in all_supporting:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def _retrieve_flat(
    question: str,
    documents: list[Document],
    fts_index: FTS5Index,
    top_k: int,
) -> list[str]:
    """FTS5 flat retrieval: score all nodes, return top-k by score."""
    all_scored: list[tuple[str, float]] = []
    for doc in documents:
        scores = fts_index.score_nodes(question, doc.doc_id)
        all_scored.extend(scores.items())
    all_scored.sort(key=lambda x: -x[1])
    return [nid for nid, _ in all_scored[:top_k]]


def _retrieve_tree(
    question: str,
    documents: list[Document],
    fts_index: FTS5Index,
    searcher: TreeSearcher,
    top_k: int,
) -> list[str]:
    """Tree mode retrieval: anchor -> walk -> path aggregation."""
    fts_score_map: dict[str, dict[str, float]] = {}
    for doc in documents:
        scores = fts_index.score_nodes(question, doc.doc_id)
        if scores:
            fts_score_map[doc.doc_id] = scores
    _, flat_nodes = searcher.search(question, documents, fts_score_map)
    return [fn["node_id"] for fn in flat_nodes[:top_k]]


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample(
    sample: HotpotSample,
    strategy: str,
    searcher: Optional[TreeSearcher],
    k_values: list[int] = (1, 3, 5, 10),
) -> HotpotResult:
    """Evaluate one HotpotQA sample."""
    t0 = time.time()

    # Build documents for this sample's context
    documents = build_documents_for_sample(sample)
    if not documents:
        return HotpotResult(sample=sample)

    # Index documents (in-memory, per-sample)
    fts_index_local = FTS5Index(db_path=None)
    for doc in documents:
        fts_index_local.save_document(doc)
        fts_index_local.index_document(doc)

    # Retrieve
    max_k = max(k_values)
    if strategy == "tree" and searcher is not None:
        retrieved = _retrieve_tree(sample.question, documents, fts_index_local, searcher, max_k)
    else:
        retrieved = _retrieve_flat(sample.question, documents, fts_index_local, max_k)

    fts_index_local.close()

    # Ground truth
    supporting_by_title = resolve_supporting_nodes(sample, documents)

    # Compute metrics
    sp_recall = {k: compute_sp_recall(retrieved, supporting_by_title, k) for k in k_values}
    twohop_cov = {k: compute_twohop_coverage(retrieved, supporting_by_title, k) for k in k_values}
    mrr = compute_mrr(retrieved, supporting_by_title)

    all_supporting = [n for nodes in supporting_by_title.values() for n in nodes]

    return HotpotResult(
        sample=sample,
        retrieved_node_ids=retrieved,
        supporting_node_ids=all_supporting,
        mrr=mrr,
        sp_recall_at_k=sp_recall,
        twohop_cov_at_k=twohop_cov,
        latency=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_hotpotqa_benchmark(
    samples: list[HotpotSample],
    strategies: list[str],
    k_values: list[int],
) -> dict[str, dict]:
    """Run HotpotQA benchmark for given strategies.

    Returns:
        {strategy: {metric: value}} aggregated results
    """
    all_results: dict[str, list[HotpotResult]] = {s: [] for s in strategies}

    # Build shared searcher for tree mode
    searcher = None
    if "tree" in strategies:
        set_config(TreeSearchConfig(
            path_top_k=10,
            anchor_top_k=10,
            max_expansions=60,
        ))
        searcher = TreeSearcher()

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()} | HotpotQA | {len(samples)} samples")
        print(f"{'='*60}")

        for i, sample in enumerate(samples):
            result = evaluate_sample(
                sample, strategy,
                searcher if strategy == "tree" else None,
                k_values,
            )
            all_results[strategy].append(result)

            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                valid = [r for r in all_results[strategy] if r.mrr > 0 or r.supporting_node_ids]
                if valid:
                    avg_mrr = sum(r.mrr for r in valid) / len(valid)
                    avg_2hop5 = sum(r.twohop_cov_at_k.get(5, False) for r in valid) / len(valid)
                    print(f"  [{i+1}/{len(samples)}] MRR={avg_mrr:.3f} 2hop-Cov@5={avg_2hop5:.3f}")

    # Aggregate
    aggregated: dict[str, dict] = {}
    for strategy, results in all_results.items():
        valid = [r for r in results if r.supporting_node_ids]
        if not valid:
            aggregated[strategy] = {}
            continue

        agg: dict = {
            "num_samples": len(valid),
            "mrr": sum(r.mrr for r in valid) / len(valid),
            "avg_latency": sum(r.latency for r in valid) / len(valid),
        }
        for k in k_values:
            agg[f"sp_recall@{k}"] = sum(r.sp_recall_at_k.get(k, 0) for r in valid) / len(valid)
            agg[f"2hop_cov@{k}"] = sum(r.twohop_cov_at_k.get(k, False) for r in valid) / len(valid)

        # By difficulty level
        by_level: dict[str, dict] = {}
        for r in valid:
            lvl = r.sample.level or "unknown"
            by_level.setdefault(lvl, {"count": 0, "mrr": 0.0, "2hop_cov_5": 0.0})
            by_level[lvl]["count"] += 1
            by_level[lvl]["mrr"] += r.mrr
            by_level[lvl]["2hop_cov_5"] += float(r.twohop_cov_at_k.get(5, False))
        for lvl in by_level:
            cnt = by_level[lvl]["count"]
            by_level[lvl]["mrr"] /= cnt
            by_level[lvl]["2hop_cov_5"] /= cnt
        agg["by_level"] = by_level

        aggregated[strategy] = agg

    return aggregated


def print_hotpotqa_summary(aggregated: dict[str, dict], k_values: list[int]) -> None:
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("HOTPOTQA BENCHMARK RESULTS")
    print(f"{'='*70}")

    header = f"{'Strategy':<20} {'MRR':>7}"
    for k in k_values:
        header += f"  {'SP-Rec@'+str(k):>10}  {'2hop-Cov@'+str(k):>12}"
    header += f"  {'Latency':>10}"
    print(header)
    print("-" * 70)

    for strategy, agg in aggregated.items():
        if not agg:
            continue
        row = f"{strategy:<20} {agg['mrr']:>7.4f}"
        for k in k_values:
            row += f"  {agg.get(f'sp_recall@{k}', 0):>10.4f}  {agg.get(f'2hop_cov@{k}', 0):>12.4f}"
        row += f"  {agg['avg_latency']:>9.3f}s"
        print(row)

    print(f"{'='*70}")

    # Highlight 2-hop Coverage (the key metric)
    print("\n📊 Key Metric: 2-hop Coverage@5 (both supporting passages in top-5)")
    for strategy, agg in aggregated.items():
        if agg:
            val = agg.get("2hop_cov@5", 0)
            bar = "█" * int(val * 30)
            print(f"  {strategy:<20} {val:.1%} {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA Supporting Passage Recall benchmark for TreeSearch"
    )
    parser.add_argument("--split", default="validation",
                        choices=["validation", "train"],
                        help="HotpotQA split to use")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max number of questions to evaluate")
    parser.add_argument("--strategies", nargs="+", default=["fts5", "tree"],
                        choices=["fts5", "tree"],
                        help="Retrieval strategies to compare")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10],
                        help="K values for @K metrics")
    parser.add_argument("--output-dir", default=_RESULTS_DIR,
                        help="Directory to save JSON results")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable local dataset cache")
    args = parser.parse_args()

    # Load dataset
    samples = load_hotpotqa(
        split=args.split,
        max_samples=args.max_samples,
        cache=not args.no_cache,
    )

    t_start = time.time()
    aggregated = run_hotpotqa_benchmark(
        samples=samples,
        strategies=args.strategies,
        k_values=args.k_values,
    )
    elapsed = time.time() - t_start

    print_hotpotqa_summary(aggregated, k_values=[3, 5, 10])
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"hotpotqa_{args.split}_{args.max_samples}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "num_samples": args.max_samples,
            "strategies": args.strategies,
            "results": aggregated,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
