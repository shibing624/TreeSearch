# -*- coding: utf-8 -*-
"""Diagnostic script: compare FTS5 vs Tree per-sample on FinanceBench.

For each sample, shows:
1. Whether FTS5 hit and Tree missed (or vice versa)
2. The FTS5 rank of the relevant node vs Tree rank
3. Walk-only injected nodes and their scores
4. Generic section demotion effects

Usage:
    python examples/benchmark/_diag_finance.py --max-samples 50
"""
import asyncio
import argparse
import logging
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from treesearch.fts import FTS5Index
from treesearch.tree import Document, flatten_tree
from treesearch.tree_searcher import TreeSearcher
from treesearch.config import set_config, get_config, TreeSearchConfig

from financebench_benchmark import (
    FinanceBenchSample,  # needed for pickle deserialization
    load_financebench_from_hf, download_pdfs, build_pdf_indexes,
)
from benchmark_utils import BenchmarkSample, resolve_relevant_nodes

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=50)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--pdf-dir", type=str,
                        default=os.path.join(_script_dir, "data", "financebench_pdfs"))
    parser.add_argument("--index-dir", type=str,
                        default=os.path.join(_script_dir, "indexes", "financebench"))
    args = parser.parse_args()

    # Load data
    raw_samples, benchmark_samples = load_financebench_from_hf(max_samples=args.max_samples)
    pdf_paths = download_pdfs(raw_samples, args.pdf_dir, skip_existing=True)
    available_docs = set(pdf_paths.keys())
    benchmark_samples = [s for s in benchmark_samples if s.doc_id in available_docs]

    documents = await build_pdf_indexes(pdf_paths, args.index_dir)

    # Build FTS5 index
    fts_index = FTS5Index()
    fts_index.index_documents(documents)

    # Configure tree searcher
    old_cfg = get_config()
    set_config(TreeSearchConfig(
        path_top_k=max(10, old_cfg.path_top_k),
        anchor_top_k=max(10, old_cfg.anchor_top_k),
        max_expansions=max(60, old_cfg.max_expansions),
    ))

    k = 5  # Compare at top-5

    # Diagnostic counters
    fts_wins = 0
    tree_wins = 0
    both_hit = 0
    both_miss = 0
    tree_rank_improvements = []
    tree_rank_degradations = []
    walk_only_injections = 0
    walk_only_hits = 0

    # Detailed per-sample analysis
    details = []

    for i, sample in enumerate(benchmark_samples):
        has_evidence = bool(sample.evidence_texts and any(e for e in sample.evidence_texts))
        if not has_evidence:
            continue

        # Get target docs
        target_docs = documents
        if sample.doc_id:
            matched = [d for d in documents if sample.doc_id in d.doc_id or sample.doc_id in d.doc_name]
            if matched:
                target_docs = matched

        # Resolve ground truth
        relevant_node_ids = resolve_relevant_nodes(sample, documents)
        if not relevant_node_ids:
            continue

        # --- FTS5 ---
        fts_scored = []
        for doc in target_docs:
            node_scores = fts_index.score_nodes(sample.question, doc.doc_id)
            fts_scored.extend(node_scores.items())
        fts_scored.sort(key=lambda x: -x[1])
        fts_top_k = [nid for nid, _ in fts_scored[:k]]

        fts_hit = any(nid in relevant_node_ids for nid in fts_top_k)
        fts_rank = None
        for rank, nid in enumerate(fts_top_k, 1):
            if nid in relevant_node_ids:
                fts_rank = rank
                break

        # --- Tree ---
        fts_score_map = {}
        for doc in target_docs:
            scores = fts_index.score_nodes(sample.question, doc.doc_id)
            if scores:
                fts_score_map[doc.doc_id] = scores

        searcher = TreeSearcher()
        paths, flat_nodes = searcher.search(sample.question, target_docs, fts_score_map)
        tree_top_k = [fn["node_id"] for fn in flat_nodes[:k]]

        tree_hit = any(nid in relevant_node_ids for nid in tree_top_k)
        tree_rank = None
        for rank, nid in enumerate(tree_top_k, 1):
            if nid in relevant_node_ids:
                tree_rank = rank
                break

        # Count walk-only injections in top-k
        fts_all_nids = {nid for nid, _ in fts_scored}
        tree_only_in_top_k = [fn for fn in flat_nodes[:k] if fn["node_id"] not in fts_all_nids]
        walk_only_injections += len(tree_only_in_top_k)
        for fn in tree_only_in_top_k:
            if fn["node_id"] in relevant_node_ids:
                walk_only_hits += 1

        # Classify
        if fts_hit and tree_hit:
            both_hit += 1
        elif fts_hit and not tree_hit:
            fts_wins += 1
        elif not fts_hit and tree_hit:
            tree_wins += 1
        else:
            both_miss += 1

        # Rank comparison
        if fts_rank and tree_rank:
            if tree_rank < fts_rank:
                tree_rank_improvements.append(fts_rank - tree_rank)
            elif tree_rank > fts_rank:
                tree_rank_degradations.append(tree_rank - fts_rank)

        # Detailed info for FTS5-wins (Tree failures)
        if fts_hit and not tree_hit:
            # Find the relevant node in FTS5 results and its tree position
            detail = {
                "sample_idx": i,
                "question": sample.question[:80],
                "fts_rank": fts_rank,
                "relevant_nids": relevant_node_ids[:3],
                "tree_top5_nids": tree_top_k,
                "tree_top5_scores": [fn["score"] for fn in flat_nodes[:k]],
            }
            # Check where relevant node ended up in tree ranking
            for rank, fn in enumerate(flat_nodes, 1):
                if fn["node_id"] in relevant_node_ids:
                    detail["tree_actual_rank"] = rank
                    detail["tree_actual_score"] = fn["score"]
                    break
            else:
                detail["tree_actual_rank"] = "NOT_IN_LIST"
                detail["tree_actual_score"] = 0

            # Check FTS5 score of the relevant node
            for nid, s in fts_scored:
                if nid in relevant_node_ids:
                    detail["relevant_fts_score"] = round(s, 4)
                    break

            # Check if relevant node was in walk
            detail["relevant_in_fts_map"] = any(
                nid in fts_score_map.get(doc.doc_id, {})
                for doc in target_docs
                for nid in relevant_node_ids
            )

            details.append(detail)

    # Analyze both-miss cases: where are the relevant nodes?
    both_miss_details = []
    for i, sample in enumerate(benchmark_samples):
        has_evidence = bool(sample.evidence_texts and any(e for e in sample.evidence_texts))
        if not has_evidence:
            continue
        target_docs = documents
        if sample.doc_id:
            matched = [d for d in documents if sample.doc_id in d.doc_id or sample.doc_id in d.doc_name]
            if matched:
                target_docs = matched
        relevant_node_ids = resolve_relevant_nodes(sample, documents)
        if not relevant_node_ids:
            continue

        # FTS5 results
        fts_scored_bm = []
        for doc in target_docs:
            node_scores = fts_index.score_nodes(sample.question, doc.doc_id)
            fts_scored_bm.extend(node_scores.items())
        fts_scored_bm.sort(key=lambda x: -x[1])
        fts_top_k_bm = [nid for nid, _ in fts_scored_bm[:k]]
        fts_hit_bm = any(nid in relevant_node_ids for nid in fts_top_k_bm)

        # Tree results
        fts_score_map_bm = {}
        for doc in target_docs:
            scores = fts_index.score_nodes(sample.question, doc.doc_id)
            if scores:
                fts_score_map_bm[doc.doc_id] = scores
        searcher_bm = TreeSearcher()
        _, flat_nodes_bm = searcher_bm.search(sample.question, target_docs, fts_score_map_bm)
        tree_top_k_bm = [fn["node_id"] for fn in flat_nodes_bm[:k]]
        tree_hit_bm = any(nid in relevant_node_ids for nid in tree_top_k_bm)

        if not fts_hit_bm and not tree_hit_bm:
            # Both miss — find where relevant nodes are
            fts_rank_bm = None
            for rank, (nid, _) in enumerate(fts_scored_bm, 1):
                if nid in relevant_node_ids:
                    fts_rank_bm = rank
                    break
            tree_rank_bm = None
            for rank, fn in enumerate(flat_nodes_bm, 1):
                if fn["node_id"] in relevant_node_ids:
                    tree_rank_bm = rank
                    break
            # Check if relevant node has ANY FTS5 score
            rel_fts_scores = []
            for nid, s in fts_scored_bm:
                if nid in relevant_node_ids:
                    rel_fts_scores.append((nid, round(s, 4)))
            # Check if relevant node is NOT in FTS5 at all
            all_fts_nids = {nid for nid, _ in fts_scored_bm}
            not_in_fts = [nid for nid in relevant_node_ids if nid not in all_fts_nids]

            both_miss_details.append({
                "sample_idx": i,
                "question": sample.question[:100],
                "relevant_nids": relevant_node_ids[:3],
                "fts_rank": fts_rank_bm or ">ALL",
                "tree_rank": tree_rank_bm or ">ALL",
                "total_fts_nodes": len(fts_scored_bm),
                "total_tree_nodes": len(flat_nodes_bm),
                "rel_fts_scores": rel_fts_scores[:3],
                "not_in_fts": not_in_fts[:3],
            })

    # Print report
    total = fts_wins + tree_wins + both_hit + both_miss
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC REPORT: FinanceBench Tree vs FTS5 (top-{k})")
    print(f"{'='*70}")
    print(f"  Total samples with evidence: {total}")
    print(f"  Both hit:   {both_hit:3d} ({both_hit/total*100:.1f}%)")
    print(f"  FTS5 only:  {fts_wins:3d} ({fts_wins/total*100:.1f}%)  ← Tree failures")
    print(f"  Tree only:  {tree_wins:3d} ({tree_wins/total*100:.1f}%)  ← Tree wins")
    print(f"  Both miss:  {both_miss:3d} ({both_miss/total*100:.1f}%)")
    print()
    print(f"  Walk-only injections in top-{k}: {walk_only_injections}")
    print(f"  Walk-only hits (correct):        {walk_only_hits}")
    print()
    if tree_rank_improvements:
        print(f"  Tree rank improvements: {len(tree_rank_improvements)} cases, "
              f"avg +{sum(tree_rank_improvements)/len(tree_rank_improvements):.1f} positions")
    if tree_rank_degradations:
        print(f"  Tree rank degradations: {len(tree_rank_degradations)} cases, "
              f"avg -{sum(tree_rank_degradations)/len(tree_rank_degradations):.1f} positions")

    print(f"\n{'='*70}")
    print(f"  FTS5-ONLY HITS (Tree missed) — {len(details)} cases")
    print(f"{'='*70}")
    for d in details:
        print(f"\n  Sample #{d['sample_idx']}: {d['question']}")
        print(f"    FTS5 rank: {d['fts_rank']}")
        print(f"    Tree actual rank: {d['tree_actual_rank']}")
        print(f"    Tree actual score: {d['tree_actual_score']}")
        print(f"    Relevant FTS5 score: {d.get('relevant_fts_score', 'N/A')}")
        print(f"    Relevant in FTS map: {d['relevant_in_fts_map']}")
        print(f"    Tree top-5 scores: {d['tree_top5_scores']}")
        print(f"    Relevant node IDs: {d['relevant_nids']}")

    set_config(old_cfg)

    # Print both-miss analysis
    print(f"\n{'='*70}")
    print(f"  BOTH-MISS ANALYSIS — {len(both_miss_details)} cases")
    print(f"{'='*70}")

    # Categorize
    in_fts_but_low = 0  # relevant node has FTS5 score but ranked too low
    not_in_fts_at_all = 0  # relevant node not in FTS5 results at all
    close_miss = 0  # rank 6-10 (just outside top-5)

    for d in both_miss_details:
        if d["not_in_fts"]:
            not_in_fts_at_all += 1
        elif isinstance(d["fts_rank"], int) and d["fts_rank"] <= 10:
            close_miss += 1
            in_fts_but_low += 1
        else:
            in_fts_but_low += 1

    print(f"  Relevant node NOT in FTS5 at all: {not_in_fts_at_all}")
    print(f"  Relevant node in FTS5 but ranked low: {in_fts_but_low}")
    print(f"  Close misses (rank 6-10): {close_miss}")
    print()

    # Show first 15 detailed
    for d in both_miss_details[:15]:
        print(f"  #{d['sample_idx']}: {d['question']}")
        print(f"    FTS5 rank: {d['fts_rank']}/{d['total_fts_nodes']}  "
              f"Tree rank: {d['tree_rank']}/{d['total_tree_nodes']}")
        print(f"    Relevant FTS5 scores: {d['rel_fts_scores']}")
        if d["not_in_fts"]:
            print(f"    ⚠️ NOT IN FTS5: {d['not_in_fts']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
