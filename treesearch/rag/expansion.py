# -*- coding: utf-8 -*-
"""Structure-constrained subgraph expansion."""

import re

from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.models import CandidateRelation, ExpansionConfig, GraphRelation, GraphSeed


class StructureConstrainedExpander:
    """Expand candidate relations from TreeSearch seeds with structural scores."""

    def __init__(self, store: InMemoryGraphStore, config: ExpansionConfig | None = None):
        self.store = store
        self.config = config or ExpansionConfig()

    def expand(self, query: str, seeds: list[GraphSeed]) -> list[CandidateRelation]:
        relation_ids: set[str] = set()
        frontier_nodes = {(seed.doc_id, seed.node_id) for seed in seeds}

        for seed in seeds:
            for relation in self.store.get_relations_by_node(seed.doc_id, seed.node_id):
                relation_ids.add(relation.relation_id)
                for linked in self.store.get_relations_by_entity(relation.subject):
                    relation_ids.add(linked.relation_id)
                for linked in self.store.get_relations_by_entity(relation.object):
                    relation_ids.add(linked.relation_id)

        for _ in range(self.config.max_hops):
            next_frontier: set[tuple[str, str]] = set()
            for doc_id, node_id in frontier_nodes:
                neighbors = self.store.get_neighbor_nodes(doc_id, node_id, {"parent", "child", "sibling"})
                next_frontier.update(neighbors)
                for neighbor_doc_id, neighbor_node_id in neighbors:
                    for relation in self.store.get_relations_by_node(neighbor_doc_id, neighbor_node_id):
                        relation_ids.add(relation.relation_id)
            frontier_nodes = next_frontier

        candidates = [
            self._score_relation(query, relation, seeds)
            for relation in self.store.get_relations_by_ids(sorted(relation_ids))
        ]
        candidates.sort(key=lambda item: (-item.total_score, item.relation.relation_id))
        return candidates[: self.config.max_relations]

    def _score_relation(
        self,
        query: str,
        relation: GraphRelation,
        seeds: list[GraphSeed],
    ) -> CandidateRelation:
        seed_by_node = {seed.graph_node_id: seed for seed in seeds}
        seed = seed_by_node.get(relation.graph_node_id)
        sparse_seed_score = (seed.score if seed else 0.0) * self.config.seed_weight
        structure_score = self._structure_score(relation.doc_id, relation.node_id, seeds) * self.config.structure_weight
        source_type_score = self._source_type_score(query, relation.source_type) * self.config.source_type_weight
        grounding_score = self._grounding_score(relation.doc_id, relation.node_id) * self.config.grounding_weight
        semantic_score = self._semantic_overlap(query, relation.text)
        return CandidateRelation(
            relation=relation,
            semantic_score=semantic_score,
            sparse_seed_score=sparse_seed_score,
            structure_score=structure_score,
            source_type_score=source_type_score,
            grounding_score=grounding_score,
        )

    def _structure_score(self, doc_id: str, node_id: str, seeds: list[GraphSeed]) -> float:
        if any(seed.doc_id == doc_id and seed.node_id == node_id for seed in seeds):
            return 1.0
        for seed in seeds:
            parent_child = self.store.get_neighbor_nodes(seed.doc_id, seed.node_id, {"parent", "child"})
            if (doc_id, node_id) in parent_child:
                return 0.8
            siblings = self.store.get_neighbor_nodes(seed.doc_id, seed.node_id, {"sibling"})
            if (doc_id, node_id) in siblings:
                return 0.6
            seed_passage = self.store.get_passage(seed.doc_id, seed.node_id)
            candidate_passage = self.store.get_passage(doc_id, node_id)
            if seed_passage and candidate_passage and seed_passage.doc_id == candidate_passage.doc_id:
                return 0.3
        return 0.0

    def _grounding_score(self, doc_id: str, node_id: str) -> float:
        passage = self.store.get_passage(doc_id, node_id)
        if not passage:
            return 0.0
        if passage.source_path and passage.line_start is not None and passage.line_end is not None:
            return 1.0
        if passage.source_path:
            return 0.5
        return 0.0

    @staticmethod
    def _source_type_score(query: str, source_type: str) -> float:
        code_like = bool(re.search(r"[A-Za-z_][\w]*\(|[A-Za-z_][\w]*\.[A-Za-z_]", query))
        config_like = any(token in query for token in ("_", "config", "Config", "YAML", "JSON", "TOML"))
        if code_like and source_type == "code":
            return 1.0
        if config_like and source_type in {"code", "json", "yaml", "toml", "markdown"}:
            return 1.0
        if source_type in {"markdown", "pdf", "docx", "text"}:
            return 0.3
        return 0.0

    @staticmethod
    def _semantic_overlap(query: str, text: str) -> float:
        query_terms = {term.casefold() for term in re.findall(r"[\w_]+", query)}
        text_terms = {term.casefold() for term in re.findall(r"[\w_]+", text)}
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)
