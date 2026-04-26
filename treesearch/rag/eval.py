# -*- coding: utf-8 -*-
"""Evaluation helpers for repository-level GraphRAG experiments."""

from dataclasses import dataclass

from treesearch.rag.models import GroundedAnswer, make_graph_node_id


@dataclass(frozen=True)
class RealRepoSample:
    query_id: str
    query: str
    gold_node_ids: tuple[str, ...] = ()
    gold_source_paths: tuple[str, ...] = ()
    gold_answer: str = ""
    query_type: str = ""
    requires_cross_source: bool = False
    needs_line_grounding: bool = False


@dataclass(frozen=True)
class RealRepoEvalResult:
    query_id: str
    node_recall: float
    source_path_recall: float
    citation_precision: float
    citation_recall: float
    line_grounding_accuracy: float
    verification_ok: bool
    task_success: bool


def aggregate_eval_results(results: list[RealRepoEvalResult]) -> dict[str, float | int]:
    """Compute macro averages for RealRepoBench-style results."""
    if not results:
        return {
            "count": 0,
            "node_recall": 0.0,
            "source_path_recall": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "line_grounding_accuracy": 0.0,
            "verification_rate": 0.0,
            "task_success_rate": 0.0,
        }

    count = len(results)
    return {
        "count": count,
        "node_recall": sum(result.node_recall for result in results) / count,
        "source_path_recall": sum(result.source_path_recall for result in results) / count,
        "citation_precision": sum(result.citation_precision for result in results) / count,
        "citation_recall": sum(result.citation_recall for result in results) / count,
        "line_grounding_accuracy": sum(result.line_grounding_accuracy for result in results) / count,
        "verification_rate": sum(1.0 for result in results if result.verification_ok) / count,
        "task_success_rate": sum(1.0 for result in results if result.task_success) / count,
    }


def evaluate_grounded_answer(sample: RealRepoSample, answer: GroundedAnswer) -> RealRepoEvalResult:
    selected_node_ids = set(answer.evidence_chain.selected_node_ids)
    citation_node_ids = {
        make_graph_node_id(citation.doc_id, citation.node_id)
        for citation in answer.evidence_chain.citations
    }
    citation_node_ids.update(citation.node_id for citation in answer.evidence_chain.citations)
    citation_paths = {citation.source_path for citation in answer.evidence_chain.citations}
    gold_node_ids = set(sample.gold_node_ids)
    gold_source_paths = set(sample.gold_source_paths)

    node_recall = _recall(gold_node_ids, selected_node_ids | citation_node_ids)
    source_path_recall = _recall(gold_source_paths, citation_paths)
    citation_precision = _precision(citation_paths, gold_source_paths)
    citation_recall = _recall(gold_source_paths, citation_paths)
    line_grounding_accuracy = _line_grounding_accuracy(sample, answer)
    verification_ok = answer.verification.ok
    line_grounding_ok = not sample.needs_line_grounding or line_grounding_accuracy == 1.0
    task_success = verification_ok and node_recall == 1.0 and source_path_recall == 1.0 and line_grounding_ok

    return RealRepoEvalResult(
        query_id=sample.query_id,
        node_recall=node_recall,
        source_path_recall=source_path_recall,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        line_grounding_accuracy=line_grounding_accuracy,
        verification_ok=verification_ok,
        task_success=task_success,
    )


def _recall(gold: set[str], predicted: set[str]) -> float:
    if not gold:
        return 1.0
    return len(gold & predicted) / len(gold)


def _precision(predicted: set[str], gold: set[str]) -> float:
    if not gold:
        return 1.0
    if not predicted:
        return 0.0
    return len(predicted & gold) / len(predicted)


def _line_grounding_accuracy(sample: RealRepoSample, answer: GroundedAnswer) -> float:
    if not sample.needs_line_grounding:
        return 1.0
    citations = answer.evidence_chain.citations
    if not citations:
        return 0.0
    grounded = 0
    for citation in citations:
        if citation.line_start is not None and citation.line_end is not None and citation.line_start <= citation.line_end:
            grounded += 1
    return grounded / len(citations)
