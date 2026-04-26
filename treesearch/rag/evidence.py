# -*- coding: utf-8 -*-
"""Evidence-chain selection for GraphRAG."""

from typing import Protocol

from treesearch.rag.graph_store import InMemoryGraphStore
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
