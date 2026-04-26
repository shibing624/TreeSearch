# -*- coding: utf-8 -*-
"""Evaluation helpers for repository-level GraphRAG experiments."""

from treesearch.rag.eval import RealRepoSample, aggregate_eval_results, evaluate_grounded_answer
from treesearch.rag.models import EvidenceChain, EvidenceCitation, GroundedAnswer, VerificationResult


def test_evaluate_grounded_answer_scores_node_source_and_citation_precision():
    sample = RealRepoSample(
        query_id="q1",
        query="Where is max_concurrency documented?",
        gold_node_ids=("cfg", "doc"),
        gold_source_paths=("treesearch/config.py", "docs/runtime.md"),
    )
    answer = GroundedAnswer(
        query=sample.query,
        answer="max_concurrency defined_in TreeSearchConfig",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=("TreeSearchConfig",),
            selected_relation_ids=("r1",),
            selected_node_ids=("cfg", "doc"),
            reasoning_chain=("max_concurrency defined_in TreeSearchConfig",),
            citations=(
                EvidenceCitation("cfg", "code", "treesearch/config.py", 10, 14),
                EvidenceCitation("doc", "docs", "docs/runtime.md", 42, 58),
            ),
            evidence_sufficiency=True,
        ),
        verification=VerificationResult(ok=True),
    )

    result = evaluate_grounded_answer(sample, answer)

    assert result.query_id == "q1"
    assert result.node_recall == 1.0
    assert result.source_path_recall == 1.0
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.line_grounding_accuracy == 1.0
    assert result.task_success


def test_evaluate_grounded_answer_penalizes_unverified_answer():
    sample = RealRepoSample(
        query_id="q1",
        query="Where is max_concurrency documented?",
        gold_node_ids=("cfg",),
        gold_source_paths=("treesearch/config.py",),
    )
    answer = GroundedAnswer(
        query=sample.query,
        answer="forged",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=(),
            selected_relation_ids=(),
            selected_node_ids=("wrong",),
            reasoning_chain=(),
            citations=(EvidenceCitation("wrong", "code", "wrong.py", 1, 2),),
            evidence_sufficiency=False,
        ),
        verification=VerificationResult(ok=False, errors=("missing node",)),
    )

    result = evaluate_grounded_answer(sample, answer)

    assert result.node_recall == 0.0
    assert result.source_path_recall == 0.0
    assert result.citation_precision == 0.0
    assert result.citation_recall == 0.0
    assert not result.task_success


def test_aggregate_eval_results_averages_metrics():
    sample = RealRepoSample(
        query_id="q1",
        query="Where is max_concurrency documented?",
        gold_node_ids=("cfg",),
        gold_source_paths=("treesearch/config.py",),
    )
    ok_answer = GroundedAnswer(
        query=sample.query,
        answer="ok",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=(),
            selected_relation_ids=("r1",),
            selected_node_ids=("cfg",),
            reasoning_chain=("ok",),
            citations=(EvidenceCitation("cfg", "code", "treesearch/config.py", 10, 14),),
            evidence_sufficiency=True,
        ),
        verification=VerificationResult(ok=True),
    )
    bad_answer = GroundedAnswer(
        query=sample.query,
        answer="bad",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=(),
            selected_relation_ids=(),
            selected_node_ids=(),
            reasoning_chain=(),
            citations=(),
            evidence_sufficiency=False,
        ),
        verification=VerificationResult(ok=False),
    )

    summary = aggregate_eval_results(
        [
            evaluate_grounded_answer(sample, ok_answer),
            evaluate_grounded_answer(sample, bad_answer),
        ]
    )

    assert summary["count"] == 2
    assert summary["node_recall"] == 0.5
    assert summary["citation_recall"] == 0.5
    assert summary["task_success_rate"] == 0.5


def test_evaluate_grounded_answer_treats_unlabeled_gold_as_smoke_sample():
    sample = RealRepoSample(
        query_id="smoke",
        query="Where is max_concurrency documented?",
    )
    answer = GroundedAnswer(
        query=sample.query,
        answer="ok",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=(),
            selected_relation_ids=("r1",),
            selected_node_ids=("cfg",),
            reasoning_chain=("ok",),
            citations=(EvidenceCitation("cfg", "code", "treesearch/config.py", 10, 14),),
            evidence_sufficiency=True,
        ),
        verification=VerificationResult(ok=True),
    )

    result = evaluate_grounded_answer(sample, answer)

    assert result.node_recall == 1.0
    assert result.source_path_recall == 1.0
    assert result.citation_precision == 1.0
    assert result.task_success


def test_evaluate_grounded_answer_requires_line_grounding_when_requested():
    sample = RealRepoSample(
        query_id="line",
        query="Where is max_concurrency documented?",
        gold_source_paths=("treesearch/config.py",),
        needs_line_grounding=True,
    )
    answer = GroundedAnswer(
        query=sample.query,
        answer="ok",
        evidence_chain=EvidenceChain(
            query=sample.query,
            bridge_entities=(),
            selected_relation_ids=("r1",),
            selected_node_ids=("code::cfg",),
            reasoning_chain=("ok",),
            citations=(EvidenceCitation("cfg", "code", "treesearch/config.py", None, None),),
            evidence_sufficiency=True,
        ),
        verification=VerificationResult(ok=True),
    )

    result = evaluate_grounded_answer(sample, answer)

    assert result.line_grounding_accuracy == 0.0
    assert not result.task_success
