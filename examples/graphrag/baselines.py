# -*- coding: utf-8 -*-
"""Baseline adapters for RealRepoBench-style experiments."""

import re
import time
from collections import Counter

from treesearch import TreeSearch
from treesearch.rag import (
    EvidenceChain,
    EvidenceCitation,
    GroundedAnswer,
    RealRepoSample,
    VerificationResult,
    evaluate_grounded_answer,
)
from treesearch.rag.models import make_graph_node_id
from treesearch.tree import Document, flatten_tree


def baseline_methods() -> tuple[str, ...]:
    return ("graphrag", "treesearch", "fts5", "dense", "hybrid")


def run_retrieval_baseline(
    method: str,
    tree_search: TreeSearch,
    samples: list[RealRepoSample],
) -> tuple[list[dict], list, dict]:
    started = time.perf_counter()
    rows = []
    metrics = []
    doc_map = {document.doc_id: document for document in tree_search.documents}
    dense_index = _build_dense_index(tree_search.documents)

    for sample in samples:
        answer = _baseline_answer(method, tree_search, sample, doc_map, dense_index)
        result = evaluate_grounded_answer(sample, answer)
        metrics.append(result)
        rows.append(answer_row(method, sample, answer, result))

    return rows, metrics, {
        "latency_seconds": time.perf_counter() - started,
        "llm_calls": 0,
    }


def answer_row(method: str, sample: RealRepoSample, answer: GroundedAnswer, metrics) -> dict:
    return {
        "method": method,
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
                "doc_id": citation.doc_id,
                "source_path": citation.source_path,
                "line_start": citation.line_start,
                "line_end": citation.line_end,
            }
            for citation in answer.evidence_chain.citations
        ],
    }


def _baseline_answer(
    method: str,
    tree_search: TreeSearch,
    sample: RealRepoSample,
    doc_map: dict[str, Document],
    dense_index: list[dict],
) -> GroundedAnswer:
    if method == "treesearch":
        result = tree_search.search(
            sample.query,
            top_k_docs=5,
            max_nodes_per_doc=5,
            include_ancestors=True,
            merge_strategy="global_score",
        )
        return _treesearch_result_to_answer(sample.query, result, doc_map)
    if method == "fts5":
        result = tree_search.search(
            sample.query,
            top_k_docs=5,
            max_nodes_per_doc=5,
            search_mode="flat",
            merge_strategy="global_score",
        )
        return _treesearch_result_to_answer(sample.query, result, doc_map)
    if method == "dense":
        return _dense_answer(sample.query, dense_index)
    if method == "hybrid":
        dense_answer = _dense_answer(sample.query, dense_index, top_k=3)
        result = tree_search.search(
            sample.query,
            top_k_docs=5,
            max_nodes_per_doc=3,
            include_ancestors=True,
            merge_strategy="global_score",
        )
        sparse_answer = _treesearch_result_to_answer(sample.query, result, doc_map)
        return _merge_answers(sample.query, sparse_answer, dense_answer)
    raise ValueError(f"unsupported baseline method: {method}")


def _treesearch_result_to_answer(query: str, result: dict, doc_map: dict[str, Document]) -> GroundedAnswer:
    citations = []
    selected_node_ids = []
    snippets = []
    for node in result.get("flat_nodes", [])[:5]:
        node_id = str(node.get("node_id", ""))
        doc_id = str(node.get("doc_id", ""))
        if not node_id or not doc_id:
            continue
        source_node = doc_map[doc_id].get_node_by_id(node_id) if doc_id in doc_map else None
        selected_node_ids.append(make_graph_node_id(doc_id, node_id))
        citations.append(
            EvidenceCitation(
                node_id=node_id,
                doc_id=doc_id,
                source_path=str(node.get("source_path", "")),
                line_start=node.get("line_start") or (source_node.get("line_start") if source_node else None),
                line_end=node.get("line_end") or (source_node.get("line_end") if source_node else None),
            )
        )
        snippets.append(str(node.get("text", "") or node.get("snippet", "")))
    return _make_answer(query, snippets, selected_node_ids, citations)


def _dense_answer(query: str, dense_index: list[dict], top_k: int = 5) -> GroundedAnswer:
    query_vector = _term_vector(query)
    scored = sorted(
        dense_index,
        key=lambda item: (-_cosine(query_vector, item["vector"]), item["graph_node_id"]),
    )[:top_k]
    return _make_answer(
        query,
        [item["text"] for item in scored],
        [item["graph_node_id"] for item in scored],
        [item["citation"] for item in scored],
    )


def _merge_answers(query: str, left: GroundedAnswer, right: GroundedAnswer) -> GroundedAnswer:
    selected_node_ids = tuple(dict.fromkeys(left.evidence_chain.selected_node_ids + right.evidence_chain.selected_node_ids))
    citations = tuple(dict.fromkeys(left.evidence_chain.citations + right.evidence_chain.citations))
    reasoning_chain = tuple(dict.fromkeys(left.evidence_chain.reasoning_chain + right.evidence_chain.reasoning_chain))
    chain = EvidenceChain(
        query=query,
        bridge_entities=(),
        selected_relation_ids=(),
        selected_node_ids=selected_node_ids,
        reasoning_chain=reasoning_chain,
        citations=citations,
        evidence_sufficiency=bool(citations),
    )
    return GroundedAnswer(
        query=query,
        answer="\n".join(chain.reasoning_chain),
        evidence_chain=chain,
        verification=VerificationResult(ok=bool(citations)),
    )


def _make_answer(
    query: str,
    snippets: list[str],
    selected_node_ids: list[str],
    citations: list[EvidenceCitation],
) -> GroundedAnswer:
    chain = EvidenceChain(
        query=query,
        bridge_entities=(),
        selected_relation_ids=(),
        selected_node_ids=tuple(dict.fromkeys(selected_node_ids)),
        reasoning_chain=tuple(snippet for snippet in snippets if snippet),
        citations=tuple(citations),
        evidence_sufficiency=bool(citations),
    )
    return GroundedAnswer(
        query=query,
        answer="\n".join(chain.reasoning_chain),
        evidence_chain=chain,
        verification=VerificationResult(ok=bool(citations)),
    )


def _build_dense_index(documents: list[Document]) -> list[dict]:
    rows = []
    for document in documents:
        source_path = str(document.metadata.get("source_path", ""))
        for node in flatten_tree(document.structure):
            node_id = str(node.get("node_id", ""))
            if not node_id:
                continue
            text = str(node.get("text", ""))
            graph_node_id = make_graph_node_id(document.doc_id, node_id)
            rows.append(
                {
                    "graph_node_id": graph_node_id,
                    "text": text,
                    "vector": _term_vector(" ".join([str(node.get("title", "")), text])),
                    "citation": EvidenceCitation(
                        node_id=node_id,
                        doc_id=document.doc_id,
                        source_path=source_path,
                        line_start=node.get("line_start"),
                        line_end=node.get("line_end"),
                    ),
                }
            )
    return rows


def _term_vector(text: str) -> Counter:
    return Counter(term.casefold() for term in re.findall(r"[\w_]+", text))


def _cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
