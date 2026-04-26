# -*- coding: utf-8 -*-
"""Run a small RealRepoBench-style evaluation for TreeSearchGraphRAG.

Example:
    python examples/graphrag/real_repo_bench.py \
        --paths treesearch docs \
        --queries examples/graphrag/sample_queries.json \
        --triplets examples/graphrag/sample_triplets.json \
        --output benchmark_results/graphrag/real_repo_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.rag import (
    ExpansionConfig,
    PreExtractedTripletExtractor,
    RealRepoSample,
    RuleBasedTripletExtractor,
    aggregate_eval_results,
    evaluate_grounded_answer,
)


def _load_samples(path: Path) -> list[RealRepoSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in data:
        samples.append(
            RealRepoSample(
                query_id=str(item["query_id"]),
                query=str(item["query"]),
                gold_node_ids=tuple(item.get("gold_node_ids", [])),
                gold_source_paths=tuple(item.get("gold_source_paths", [])),
                gold_answer=str(item.get("gold_answer", "")),
                query_type=str(item.get("query_type", "")),
                requires_cross_source=bool(item.get("requires_cross_source", False)),
                needs_line_grounding=bool(item.get("needs_line_grounding", False)),
            )
        )
    return samples


def _load_triplets(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("triplets file must be a JSON object keyed by TreeSearch node_id")
    return {str(node_id): triplets for node_id, triplets in data.items()}


def run(paths: list[str], queries_path: Path, triplets_path: Optional[Path] = None) -> list[dict]:
    samples = _load_samples(queries_path)
    ts = TreeSearch(db_path=None)
    ts.index(*paths)

    extractor = (
        PreExtractedTripletExtractor(_load_triplets(triplets_path))
        if triplets_path is not None
        else RuleBasedTripletExtractor()
    )
    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=extractor,
        expansion_config=ExpansionConfig(max_relations=50),
    )
    build_stats = rag.build_graph()

    rows = []
    metric_results = []
    for sample in samples:
        answer = rag.query(sample.query)
        metrics = evaluate_grounded_answer(sample, answer)
        metric_results.append(metrics)
        rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "answer": answer.answer,
                "verification_ok": answer.verification.ok,
                "verification_errors": list(answer.verification.errors),
                "node_recall": metrics.node_recall,
                "source_path_recall": metrics.source_path_recall,
                "citation_precision": metrics.citation_precision,
                "citation_recall": metrics.citation_recall,
                "line_grounding_accuracy": metrics.line_grounding_accuracy,
                "task_success": metrics.task_success,
                "citations": [
                    {
                        "node_id": citation.node_id,
                        "source_path": citation.source_path,
                        "line_start": citation.line_start,
                        "line_end": citation.line_end,
                    }
                    for citation in answer.evidence_chain.citations
                ],
            }
        )

    return [
        {
            "build_stats": {
                "documents": build_stats.documents,
                "passages": build_stats.passages,
                "relations": build_stats.relations,
                "structural_edges": build_stats.structural_edges,
            },
            "summary": aggregate_eval_results(metric_results),
            "results": rows,
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--triplets")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triplets_path = Path(args.triplets) if args.triplets else None
    report = run(args.paths, Path(args.queries), triplets_path)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
