# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Benchmark module for evaluating TreeSearch on long-document QA/retrieval tasks.

Compares TreeSearch vs. BM25 vs. full-context approaches with automatic
cost (token consumption) and latency tracking.

Supported datasets: QASPER, QuALITY, custom JSONL.
Key metrics: retrieval accuracy (P@K, R@K, NDCG@K, MRR, Hit@K) + cost (tokens, LLM calls, latency).

Usage:
    from treesearch.benchmark import load_dataset, run_benchmark, print_report, print_comparison
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .llm import count_tokens, DEFAULT_MODEL
from .metrics import (
    CostStats, CostTracker, aggregate_cost_stats,
    evaluate_query,
)
from .search import search, SearchResult
from .tree import Document, flatten_tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkSample:
    """A single benchmark question with ground-truth retrieval targets."""
    question: str = ""
    answer: str = ""
    evidence_texts: list[str] = field(default_factory=list)
    relevant_section_titles: list[str] = field(default_factory=list)
    question_type: str = ""
    doc_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class SampleResult:
    """Evaluation result for a single benchmark sample."""
    sample: Optional[BenchmarkSample] = None
    retrieved_node_ids: list[str] = field(default_factory=list)
    relevant_node_ids: list[str] = field(default_factory=list)
    retrieval_metrics: dict = field(default_factory=dict)
    cost: CostStats = field(default_factory=CostStats)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report for one strategy."""
    dataset: str = ""
    strategy: str = ""
    model: str = ""
    num_samples: int = 0
    avg_retrieval_metrics: dict = field(default_factory=dict)
    avg_cost: CostStats = field(default_factory=CostStats)
    total_cost: CostStats = field(default_factory=CostStats)
    per_type_metrics: dict = field(default_factory=dict)
    individual_results: list[SampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "strategy": self.strategy,
            "model": self.model,
            "num_samples": self.num_samples,
            "avg_retrieval_metrics": self.avg_retrieval_metrics,
            "avg_cost": self.avg_cost.to_dict(),
            "total_cost": self.total_cost.to_dict(),
            "per_type_metrics": self.per_type_metrics,
        }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_qasper(data_path: str, max_samples: int = 200) -> list[BenchmarkSample]:
    """Load QASPER dataset (academic paper QA with section-level evidence).

    QASPER JSON format: {paper_id: {title, abstract, full_text, qas: [{question, answers}]}}
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for paper_id, paper in data.items():
        if len(samples) >= max_samples:
            break
        for qa in paper.get("qas", []):
            if len(samples) >= max_samples:
                break
            question = qa.get("question", "")
            answers = qa.get("answers", [])
            answer_texts = []
            evidence_texts = []
            for ans in answers:
                answer_obj = ans.get("answer", {})
                if isinstance(answer_obj, dict):
                    free_text = answer_obj.get("free_form_answer", "")
                    extractive = answer_obj.get("extractive_spans", [])
                    if free_text:
                        answer_texts.append(free_text)
                    answer_texts.extend(extractive)
                    evidence_texts.extend(ans.get("evidence", []))

            # Extract section titles from full_text for ground truth matching
            section_titles = []
            for section in paper.get("full_text", {}).get("section_name", []):
                if section:
                    section_titles.append(section)

            samples.append(BenchmarkSample(
                question=question,
                answer=" | ".join(answer_texts) if answer_texts else "",
                evidence_texts=evidence_texts,
                relevant_section_titles=evidence_texts,  # QASPER evidence = section titles
                question_type=qa.get("question_type", ""),
                doc_id=paper_id,
                metadata={"paper_title": paper.get("title", "")},
            ))
    logger.info("Loaded %d QASPER samples from %s", len(samples), data_path)
    return samples


def load_quality(data_path: str, max_samples: int = 200) -> list[BenchmarkSample]:
    """Load QuALITY dataset (multiple-choice QA on long documents).

    QuALITY JSONL format: each line is {article_id, article, questions: [{question, options, gold_label}]}
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            item = json.loads(line.strip())
            article_id = item.get("article_id", "")
            for q in item.get("questions", []):
                if len(samples) >= max_samples:
                    break
                question = q.get("question", "")
                options = q.get("options", [])
                gold_label = q.get("gold_label", 0)
                # gold_label is 1-indexed
                answer = options[gold_label - 1] if 0 < gold_label <= len(options) else ""
                samples.append(BenchmarkSample(
                    question=question,
                    answer=answer,
                    question_type="multiple_choice",
                    doc_id=article_id,
                    metadata={
                        "options": options,
                        "gold_label": gold_label,
                        "difficulty": q.get("difficult", 0),
                    },
                ))
    logger.info("Loaded %d QuALITY samples from %s", len(samples), data_path)
    return samples


def load_custom(data_path: str, max_samples: int = 200) -> list[BenchmarkSample]:
    """Load custom JSONL dataset.

    Each line: {question, answer, evidence_texts?, relevant_section_titles?, doc_id?, question_type?}
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            item = json.loads(line.strip())
            samples.append(BenchmarkSample(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                evidence_texts=item.get("evidence_texts", []),
                relevant_section_titles=item.get("relevant_section_titles", []),
                question_type=item.get("question_type", ""),
                doc_id=item.get("doc_id", ""),
                metadata=item.get("metadata", {}),
            ))
    logger.info("Loaded %d custom samples from %s", len(samples), data_path)
    return samples


def load_dataset(
    dataset: str,
    data_path: str,
    max_samples: int = 200,
) -> list[BenchmarkSample]:
    """Load benchmark dataset by name.

    Args:
        dataset: 'qasper', 'quality', or 'custom'
        data_path: path to dataset file
        max_samples: max number of samples to load

    Returns:
        list of BenchmarkSample
    """
    loaders = {
        "qasper": load_qasper,
        "quality": load_quality,
        "custom": load_custom,
    }
    loader = loaders.get(dataset)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(loaders.keys())}")
    return loader(data_path, max_samples=max_samples)


# ---------------------------------------------------------------------------
# Ground truth resolution
# ---------------------------------------------------------------------------

def resolve_relevant_nodes(
    sample: BenchmarkSample,
    documents: list[Document],
) -> list[str]:
    """Map a sample's evidence/section titles to actual node_ids in indexed documents.

    Matching strategy:
      1. Exact title match (case-insensitive)
      2. Substring match on node text (for evidence_texts)
    """
    relevant_ids = []
    target_docs = documents
    if sample.doc_id:
        matched = [d for d in documents if sample.doc_id in d.doc_id or sample.doc_id in d.doc_name]
        if matched:
            target_docs = matched

    for doc in target_docs:
        all_nodes = flatten_tree(doc.structure)
        for node in all_nodes:
            nid = node.get("node_id", "")
            title = (node.get("title", "") or "").lower().strip()
            text = (node.get("text", "") or "").lower()

            # Match by section title
            for ref_title in sample.relevant_section_titles:
                if ref_title.lower().strip() == title:
                    relevant_ids.append(nid)
                    break

            # Match by evidence text (substring in node text)
            if nid not in relevant_ids:
                for evidence in sample.evidence_texts:
                    if evidence and evidence.lower() in text:
                        relevant_ids.append(nid)
                        break

    return list(dict.fromkeys(relevant_ids))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Single sample evaluation
# ---------------------------------------------------------------------------

async def _evaluate_sample(
    sample: BenchmarkSample,
    documents: list[Document],
    strategy: str,
    model: str,
    top_k: int = 5,
    k_values: Optional[list[int]] = None,
    use_bm25: bool = True,
) -> SampleResult:
    """Evaluate a single benchmark sample with cost tracking."""
    if k_values is None:
        k_values = [1, 3, 5]

    relevant_node_ids = resolve_relevant_nodes(sample, documents)

    tracker = CostTracker()
    retrieved_node_ids = []

    with tracker:
        if strategy == "bm25":
            from .rank_bm25 import NodeBM25Index
            index = NodeBM25Index(documents)
            bm25_results = index.search(sample.question, top_k=top_k)
            retrieved_node_ids = [r["node_id"] for r in bm25_results]
        else:
            result = await search(
                query=sample.question,
                documents=documents,
                model=model,
                strategy=strategy,
                top_k_docs=min(3, len(documents)),
                max_nodes_per_doc=top_k,
                use_bm25=use_bm25,
            )
            for doc_result in result.documents:
                for node in doc_result.get("nodes", []):
                    retrieved_node_ids.append(node["node_id"])
            retrieved_node_ids = retrieved_node_ids[:top_k]

            # Record token usage from LLM calls
            tracker.stats.llm_calls = result.total_llm_calls

    metrics = evaluate_query(retrieved_node_ids, relevant_node_ids, k_values) if relevant_node_ids else {}

    return SampleResult(
        sample=sample,
        retrieved_node_ids=retrieved_node_ids,
        relevant_node_ids=relevant_node_ids,
        retrieval_metrics=metrics,
        cost=tracker.stats,
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    dataset: str,
    documents: list[Document],
    data_path: str,
    strategies: Optional[list[str]] = None,
    model: str = DEFAULT_MODEL,
    max_samples: int = 50,
    top_k: int = 5,
    k_values: Optional[list[int]] = None,
    max_concurrency: int = 3,
    output_dir: Optional[str] = None,
) -> list[BenchmarkReport]:
    """Run benchmark across multiple strategies with cost tracking.

    Args:
        dataset: dataset name ('qasper', 'quality', 'custom')
        documents: pre-indexed Document list
        data_path: path to dataset file
        strategies: list of strategies to compare (default: ['bm25', 'best_first'])
        model: LLM model name
        max_samples: max samples to evaluate
        top_k: top-K results per query
        k_values: list of K values for @K metrics
        max_concurrency: max concurrent evaluations
        output_dir: if set, save JSON reports here

    Returns:
        list of BenchmarkReport (one per strategy)
    """
    if strategies is None:
        strategies = ["bm25", "best_first"]
    if k_values is None:
        k_values = [1, 3, 5]

    # Load dataset
    samples = load_dataset(dataset, data_path, max_samples=max_samples)
    if not samples:
        logger.warning("No samples loaded from %s", data_path)
        return []

    logger.info("Benchmark: %d samples, strategies=%s, model=%s", len(samples), strategies, model)

    reports = []
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()} | Dataset: {dataset} | Samples: {len(samples)}")
        print(f"{'='*60}")

        # Run evaluation with concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[SampleResult] = []

        async def _eval_with_limit(s: BenchmarkSample) -> SampleResult:
            async with semaphore:
                return await _evaluate_sample(
                    s, documents, strategy, model, top_k, k_values,
                )

        tasks = [_eval_with_limit(s) for s in samples]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            m = result.retrieval_metrics
            if m:
                hit = "HIT" if m.get("hit@1", 0) > 0 else "miss"
                print(f"  [{i+1}/{len(samples)}] MRR={m.get('mrr', 0):.2f} "
                      f"P@3={m.get('precision@3', 0):.2f} R@3={m.get('recall@3', 0):.2f} "
                      f"[{hit}] LLM={result.cost.llm_calls} "
                      f"{result.cost.latency_seconds:.1f}s")
            else:
                print(f"  [{i+1}/{len(samples)}] Skipped (no ground truth)")

        # Aggregate metrics
        valid_results = [r for r in results if r.retrieval_metrics]
        if valid_results:
            avg_metrics = {}
            for key in valid_results[0].retrieval_metrics:
                avg_metrics[key] = sum(
                    r.retrieval_metrics.get(key, 0) for r in valid_results
                ) / len(valid_results)

            cost_list = [r.cost for r in valid_results]
            avg_cost = aggregate_cost_stats(cost_list)
            total_cost = CostStats(
                total_tokens=sum(c.total_tokens for c in cost_list),
                prompt_tokens=sum(c.prompt_tokens for c in cost_list),
                completion_tokens=sum(c.completion_tokens for c in cost_list),
                llm_calls=sum(c.llm_calls for c in cost_list),
                latency_seconds=sum(c.latency_seconds for c in cost_list),
            )

            # Per question-type breakdown
            per_type: dict[str, dict] = {}
            for r in valid_results:
                qtype = r.sample.question_type or "unknown"
                if qtype not in per_type:
                    per_type[qtype] = {"count": 0, "metrics": {}}
                per_type[qtype]["count"] += 1
                for key, val in r.retrieval_metrics.items():
                    per_type[qtype]["metrics"].setdefault(key, 0.0)
                    per_type[qtype]["metrics"][key] += val
            for qtype in per_type:
                cnt = per_type[qtype]["count"]
                for key in per_type[qtype]["metrics"]:
                    per_type[qtype]["metrics"][key] /= cnt
        else:
            avg_metrics = {}
            avg_cost = CostStats()
            total_cost = CostStats()
            per_type = {}

        report = BenchmarkReport(
            dataset=dataset,
            strategy=strategy,
            model=model,
            num_samples=len(valid_results),
            avg_retrieval_metrics=avg_metrics,
            avg_cost=avg_cost,
            total_cost=total_cost,
            per_type_metrics=per_type,
            individual_results=results,
        )
        reports.append(report)

    # Save reports
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for report in reports:
            path = os.path.join(output_dir, f"{dataset}_{report.strategy}_report.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info("Report saved: %s", path)

    return reports


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(report: BenchmarkReport) -> None:
    """Print a single benchmark report."""
    print(f"\n{'='*60}")
    print(f"Report: {report.dataset} | {report.strategy.upper()} | {report.model}")
    print(f"Samples: {report.num_samples}")
    print(f"{'='*60}")

    if report.avg_retrieval_metrics:
        print("\nRetrieval Metrics (averaged):")
        for key in sorted(report.avg_retrieval_metrics):
            print(f"  {key:<20} {report.avg_retrieval_metrics[key]:.4f}")

    print("\nCost (per query avg):")
    cost = report.avg_cost
    print(f"  {'LLM calls':<20} {cost.llm_calls}")
    print(f"  {'Total tokens':<20} {cost.total_tokens}")
    print(f"  {'Latency (s)':<20} {cost.latency_seconds:.2f}")

    print("\nCost (total):")
    tc = report.total_cost
    print(f"  {'LLM calls':<20} {tc.llm_calls}")
    print(f"  {'Total tokens':<20} {tc.total_tokens}")
    print(f"  {'Latency (s)':<20} {tc.latency_seconds:.2f}")

    if report.per_type_metrics:
        print("\nPer question-type breakdown:")
        for qtype, info in report.per_type_metrics.items():
            print(f"  [{qtype}] (n={info['count']}): "
                  f"MRR={info['metrics'].get('mrr', 0):.3f} "
                  f"P@3={info['metrics'].get('precision@3', 0):.3f} "
                  f"R@3={info['metrics'].get('recall@3', 0):.3f}")
    print()


def print_comparison(reports: list[BenchmarkReport]) -> None:
    """Print a comparison table across multiple strategies."""
    if not reports:
        return

    print(f"\n{'='*80}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*80}")

    metrics_to_show = ["mrr", "precision@3", "recall@3", "ndcg@3", "hit@1", "f1@3"]
    cost_fields = ["llm_calls", "total_tokens", "latency_seconds"]

    header = f"{'Metric':<22}"
    for r in reports:
        header += f"{r.strategy.upper():>15}"
    print(header)
    print("-" * (22 + 15 * len(reports)))

    for metric in metrics_to_show:
        row = f"{metric:<22}"
        for r in reports:
            val = r.avg_retrieval_metrics.get(metric, 0)
            row += f"{val:>15.4f}"
        print(row)

    print("-" * (22 + 15 * len(reports)))
    for cf in cost_fields:
        row = f"avg_{cf:<18}"
        for r in reports:
            val = getattr(r.avg_cost, cf, 0)
            if isinstance(val, float):
                row += f"{val:>15.2f}"
            else:
                row += f"{val:>15}"
        print(row)

    print("-" * (22 + 15 * len(reports)))
    for cf in cost_fields:
        row = f"total_{cf:<16}"
        for r in reports:
            val = getattr(r.total_cost, cf, 0)
            if isinstance(val, float):
                row += f"{val:>15.2f}"
            else:
                row += f"{val:>15}"
        print(row)

    print(f"{'='*80}\n")
