# -*- coding: utf-8 -*-
"""In-memory graph store for the GraphRAG MVP."""

import hashlib
from collections import defaultdict

from treesearch.rag.models import (
    GraphEntity,
    GraphNodePassage,
    GraphRelation,
    StructuralEdge,
    make_graph_node_id,
)


class InMemoryGraphStore:
    """Store TreeSearch-node graph data without external dependencies."""

    def __init__(self):
        self.passages: dict[str, GraphNodePassage] = {}
        self.entities: dict[str, GraphEntity] = {}
        self.relations: dict[str, GraphRelation] = {}
        self.node_to_relation_ids: dict[str, set[str]] = defaultdict(set)
        self.entity_to_relation_ids: dict[str, set[str]] = defaultdict(set)
        self.structural_edges: list[StructuralEdge] = []

    def clear(self) -> None:
        self.passages.clear()
        self.entities.clear()
        self.relations.clear()
        self.node_to_relation_ids.clear()
        self.entity_to_relation_ids.clear()
        self.structural_edges.clear()

    def stats(self) -> dict[str, int]:
        return {
            "passages": len(self.passages),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "structural_edges": len(self.structural_edges),
        }

    def add_passages(self, passages: list[GraphNodePassage]) -> None:
        for passage in passages:
            self.passages[passage.graph_node_id] = passage

    def add_structural_edges(self, edges: list[StructuralEdge]) -> None:
        self.structural_edges.extend(edges)

    def add_relations(self, relations: list[GraphRelation]) -> None:
        for relation in relations:
            self.relations[relation.relation_id] = relation
            self.node_to_relation_ids[relation.graph_node_id].add(relation.relation_id)
            self._add_entity_link(relation.subject, relation.graph_node_id, relation.relation_id)
            self._add_entity_link(relation.object, relation.graph_node_id, relation.relation_id)

    def get_passage(self, doc_id: str, node_id: str) -> GraphNodePassage | None:
        return self.passages.get(make_graph_node_id(doc_id, node_id))

    def get_passage_by_graph_node_id(self, graph_node_id: str) -> GraphNodePassage | None:
        return self.passages.get(graph_node_id)

    def get_relations_by_node(self, doc_id: str, node_id: str) -> list[GraphRelation]:
        graph_node_id = make_graph_node_id(doc_id, node_id)
        return self.get_relations_by_ids(sorted(self.node_to_relation_ids.get(graph_node_id, set())))

    def get_relations_by_entity(self, entity_text: str) -> list[GraphRelation]:
        entity_id = _entity_id(entity_text)
        return self.get_relations_by_ids(sorted(self.entity_to_relation_ids.get(entity_id, set())))

    def get_relations_by_ids(self, relation_ids: list[str]) -> list[GraphRelation]:
        return [self.relations[rid] for rid in relation_ids if rid in self.relations]

    def get_neighbor_nodes(
        self,
        doc_id: str,
        node_id: str,
        edge_types: set[str] | None = None,
    ) -> set[tuple[str, str]]:
        neighbors: set[tuple[str, str]] = set()
        for edge in self.structural_edges:
            if edge.doc_id != doc_id or edge.src_node_id != node_id:
                continue
            if edge_types is not None and edge.edge_type not in edge_types:
                continue
            neighbors.add((edge.doc_id, edge.dst_node_id))
        return neighbors

    def are_structurally_adjacent(self, left_graph_node_id: str, right_graph_node_id: str) -> bool:
        left = self.passages.get(left_graph_node_id)
        right = self.passages.get(right_graph_node_id)
        if not left or not right:
            return False
        if left.doc_id != right.doc_id:
            return False
        neighbors = self.get_neighbor_nodes(left.doc_id, left.node_id, {"parent", "child", "sibling"})
        return (right.doc_id, right.node_id) in neighbors

    def _add_entity_link(self, entity_text: str, graph_node_id: str, relation_id: str) -> None:
        entity_id = _entity_id(entity_text)
        existing = self.entities.get(entity_id)
        if existing:
            node_ids = tuple(sorted(set(existing.node_ids) | {graph_node_id}))
            self.entities[entity_id] = GraphEntity(entity_id=entity_id, text=existing.text, node_ids=node_ids)
        else:
            self.entities[entity_id] = GraphEntity(entity_id=entity_id, text=entity_text, node_ids=(graph_node_id,))
        self.entity_to_relation_ids[entity_id].add(relation_id)


def _entity_id(entity_text: str) -> str:
    normalized = " ".join(entity_text.casefold().split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()
