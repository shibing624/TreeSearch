# -*- coding: utf-8 -*-
"""Build a structure-preserving graph from TreeSearch documents."""

from treesearch.tree import Document
from treesearch.rag.extractors import TripletExtractor
from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.models import GraphBuildStats
from treesearch.rag.node_graph import document_to_node_passages, document_to_structural_edges


class NodeGraphBuilder:
    """Build node-level graph passages, relations, and structural edges."""

    def __init__(self, extractor: TripletExtractor, store: InMemoryGraphStore):
        self.extractor = extractor
        self.store = store

    def build(self, documents: list[Document]) -> GraphBuildStats:
        self.store.clear()

        passage_count = 0
        relation_count = 0
        edge_count = 0

        for document in documents:
            passages = document_to_node_passages(document)
            edges = document_to_structural_edges(document)
            self.store.add_passages(passages)
            self.store.add_structural_edges(edges)
            passage_count += len(passages)
            edge_count += len(edges)

            for passage in passages:
                relations = self.extractor.extract(passage)
                self.store.add_relations(relations)
                relation_count += len(relations)

        return GraphBuildStats(
            documents=len(documents),
            passages=passage_count,
            relations=relation_count,
            structural_edges=edge_count,
        )
