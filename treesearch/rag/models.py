# -*- coding: utf-8 -*-
"""Data models for TreeSearch-guided GraphRAG."""

from dataclasses import dataclass
from typing import Literal


def make_graph_node_id(doc_id: str, node_id: str) -> str:
    """Return a graph-global node key derived from TreeSearch document identity."""
    return f"{doc_id}::{node_id}"


@dataclass(frozen=True)
class GraphNodePassage:
    node_id: str
    doc_id: str
    doc_name: str
    source_path: str
    source_type: str
    title: str
    text: str
    path_titles: tuple[str, ...]
    line_start: int | None
    line_end: int | None
    parent_node_id: str | None = None
    child_node_ids: tuple[str, ...] = ()
    sibling_node_ids: tuple[str, ...] = ()

    @property
    def graph_node_id(self) -> str:
        return make_graph_node_id(self.doc_id, self.node_id)


@dataclass(frozen=True)
class GraphEntity:
    entity_id: str
    text: str
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphRelation:
    relation_id: str
    subject: str
    predicate: str
    object: str
    text: str
    node_id: str
    doc_id: str
    source_type: str

    @property
    def graph_node_id(self) -> str:
        return make_graph_node_id(self.doc_id, self.node_id)


@dataclass(frozen=True)
class StructuralEdge:
    doc_id: str
    src_node_id: str
    dst_node_id: str
    edge_type: Literal["parent", "child", "sibling", "same_doc"]


@dataclass(frozen=True)
class GraphSeed:
    node_id: str
    doc_id: str
    score: float
    source: Literal["fts5", "grep", "tree"]

    @property
    def graph_node_id(self) -> str:
        return make_graph_node_id(self.doc_id, self.node_id)


@dataclass(frozen=True)
class CandidateRelation:
    relation: GraphRelation
    semantic_score: float = 0.0
    sparse_seed_score: float = 0.0
    structure_score: float = 0.0
    source_type_score: float = 0.0
    grounding_score: float = 0.0

    @property
    def total_score(self) -> float:
        return (
            self.semantic_score
            + self.sparse_seed_score
            + self.structure_score
            + self.source_type_score
            + self.grounding_score
        )


@dataclass(frozen=True)
class EvidenceCitation:
    node_id: str
    doc_id: str
    source_path: str
    line_start: int | None
    line_end: int | None
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceChain:
    query: str
    bridge_entities: tuple[str, ...]
    selected_relation_ids: tuple[str, ...]
    selected_node_ids: tuple[str, ...]
    reasoning_chain: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    evidence_sufficiency: bool


@dataclass(frozen=True)
class GraphBuildStats:
    documents: int
    passages: int
    relations: int
    structural_edges: int


@dataclass(frozen=True)
class ExpansionConfig:
    max_relations: int = 50
    max_hops: int = 1
    structure_weight: float = 1.0
    seed_weight: float = 1.0
    source_type_weight: float = 0.5
    grounding_weight: float = 0.5


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundedAnswer:
    query: str
    answer: str
    evidence_chain: EvidenceChain
    verification: VerificationResult
