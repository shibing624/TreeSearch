# -*- coding: utf-8 -*-
"""Evidence-chain selection for GraphRAG."""

import json
from typing import Protocol

from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.llm import LLMClient, OpenAIChatClient
from treesearch.rag.models import CandidateRelation, EvidenceChain, EvidenceCitation


class EvidenceSelector(Protocol):
    def select(
        self,
        query: str,
        candidates: list[CandidateRelation],
        store: InMemoryGraphStore,
    ) -> EvidenceChain:
        ...


class HeuristicEvidenceSelector:
    """Deterministic selector used for tests and no-LLM demos."""

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def select(
        self,
        query: str,
        candidates: list[CandidateRelation],
        store: InMemoryGraphStore,
    ) -> EvidenceChain:
        selected = sorted(candidates, key=lambda item: (-item.total_score, item.relation.relation_id))[: self.top_k]
        relation_ids = tuple(candidate.relation.relation_id for candidate in selected)
        graph_node_ids = tuple(dict.fromkeys(candidate.relation.graph_node_id for candidate in selected))
        reasoning_chain = tuple(candidate.relation.text for candidate in selected)
        bridge_entities = tuple(
            dict.fromkeys(
                entity
                for candidate in selected
                for entity in (candidate.relation.subject, candidate.relation.object)
            )
        )

        citations = []
        for graph_node_id in graph_node_ids:
            passage = store.get_passage_by_graph_node_id(graph_node_id)
            if not passage:
                continue
            citations.append(
                EvidenceCitation(
                    node_id=passage.node_id,
                    doc_id=passage.doc_id,
                    source_path=passage.source_path,
                    line_start=passage.line_start,
                    line_end=passage.line_end,
                    section_path=passage.path_titles,
                )
            )

        return EvidenceChain(
            query=query,
            bridge_entities=bridge_entities,
            selected_relation_ids=relation_ids,
            selected_node_ids=graph_node_ids,
            reasoning_chain=reasoning_chain,
            citations=tuple(citations),
            evidence_sufficiency=bool(selected),
        )


class LLMEvidenceSelector:
    """Select evidence relations with an OpenAI-compatible LLM."""

    def __init__(self, llm_client: LLMClient | None = None, top_k: int = 5, model: str | None = None):
        self.llm_client = llm_client or OpenAIChatClient()
        self.top_k = top_k
        self.model = model

    def select(
        self,
        query: str,
        candidates: list[CandidateRelation],
        store: InMemoryGraphStore,
    ) -> EvidenceChain:
        if not candidates:
            return HeuristicEvidenceSelector(top_k=self.top_k).select(query, candidates, store)

        candidate_rows = [
            {
                "relation_id": candidate.relation.relation_id,
                "relation": candidate.relation.text,
                "score": candidate.total_score,
            }
            for candidate in candidates[: max(self.top_k * 4, self.top_k)]
        ]
        response = self.llm_client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Select the smallest useful set of relation_id values for a grounded answer. "
                        "Return strict JSON: {\"selected_relation_ids\": [\"...\"]}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({"query": query, "candidates": candidate_rows}, ensure_ascii=False),
                },
            ],
            model=self.model,
        )
        data = json.loads(_extract_json_object(response))
        selected_ids = set(str(relation_id) for relation_id in data["selected_relation_ids"])
        selected_candidates = [candidate for candidate in candidates if candidate.relation.relation_id in selected_ids]
        selected_candidates = selected_candidates[: self.top_k]
        return HeuristicEvidenceSelector(top_k=self.top_k).select(query, selected_candidates, store)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.index("{")
    end = stripped.rindex("}") + 1
    return stripped[start:end]
