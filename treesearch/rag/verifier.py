# -*- coding: utf-8 -*-
"""Deterministic verification for selected evidence chains."""

from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.models import EvidenceChain, VerificationResult, make_graph_node_id


class EvidenceVerifier:
    """Check that evidence chains reference real relations, nodes, and citations."""

    def __init__(self, store: InMemoryGraphStore):
        self.store = store

    def verify(self, chain: EvidenceChain) -> VerificationResult:
        errors: list[str] = []

        for relation_id in chain.selected_relation_ids:
            if relation_id not in self.store.relations:
                errors.append(f"missing relation: {relation_id}")

        for node_id in chain.selected_node_ids:
            if node_id not in self.store.passages:
                errors.append(f"missing node: {node_id}")

        selected_node_ids = set(chain.selected_node_ids)
        for citation in chain.citations:
            citation_graph_node_id = make_graph_node_id(citation.doc_id, citation.node_id)
            passage = self.store.get_passage_by_graph_node_id(citation_graph_node_id)
            if not passage:
                errors.append(f"missing citation node: {citation_graph_node_id}")
                continue
            if citation_graph_node_id not in selected_node_ids:
                errors.append(f"citation node not selected: {citation_graph_node_id}")
            if citation.doc_id != passage.doc_id:
                errors.append(f"citation doc_id mismatch: {citation_graph_node_id}")
            if citation.source_path != passage.source_path:
                errors.append(f"citation source_path mismatch: {citation_graph_node_id}")
            if citation.line_start != passage.line_start or citation.line_end != passage.line_end:
                errors.append(f"citation line range mismatch: {citation_graph_node_id}")

        for relation_id in chain.selected_relation_ids:
            relation = self.store.relations.get(relation_id)
            if relation and relation.graph_node_id not in selected_node_ids:
                errors.append(f"relation node not selected: {relation_id}")

        if chain.selected_relation_ids and not chain.evidence_sufficiency:
            errors.append("selected evidence marked insufficient")

        if len(chain.selected_relation_ids) > 1 and not self._has_entity_or_structural_bridge(chain):
            errors.append("selected relations do not share an entity or structural bridge")

        return VerificationResult(ok=not errors, errors=tuple(errors))

    def _has_entity_or_structural_bridge(self, chain: EvidenceChain) -> bool:
        seen: set[str] = set()
        seen_node_ids: set[str] = set()
        for relation_id in chain.selected_relation_ids:
            relation = self.store.relations.get(relation_id)
            if not relation:
                continue
            entities = {relation.subject.casefold(), relation.object.casefold()}
            if seen & entities:
                return True
            for seen_node_id in seen_node_ids:
                if self.store.are_structurally_adjacent(seen_node_id, relation.graph_node_id):
                    return True
            seen.update(entities)
            seen_node_ids.add(relation.graph_node_id)
        return False
