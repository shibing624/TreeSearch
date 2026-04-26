# -*- coding: utf-8 -*-
"""User-facing TreeSearch-guided GraphRAG pipeline."""

from treesearch.treesearch import TreeSearch
from treesearch.rag.answer import TemplateAnswerGenerator
from treesearch.rag.evidence import EvidenceSelector, HeuristicEvidenceSelector
from treesearch.rag.expansion import StructureConstrainedExpander
from treesearch.rag.extractors import RuleBasedTripletExtractor, TripletExtractor
from treesearch.rag.graph_builder import NodeGraphBuilder
from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.models import EvidenceChain, ExpansionConfig, GraphBuildStats, GroundedAnswer
from treesearch.rag.seed import retrieve_seed_nodes
from treesearch.rag.verifier import EvidenceVerifier


class TreeSearchGraphRAG:
    """GraphRAG layer that uses TreeSearch nodes as grounded graph substrate."""

    def __init__(
        self,
        tree_search: TreeSearch,
        store: InMemoryGraphStore,
        graph_builder: NodeGraphBuilder,
        expander: StructureConstrainedExpander,
        selector: EvidenceSelector,
        verifier: EvidenceVerifier,
        answer_generator: TemplateAnswerGenerator,
    ):
        self.tree_search = tree_search
        self.store = store
        self.graph_builder = graph_builder
        self.expander = expander
        self.selector = selector
        self.verifier = verifier
        self.answer_generator = answer_generator
        self._last_build_stats: GraphBuildStats | None = None

    @classmethod
    def from_tree_search(
        cls,
        tree_search: TreeSearch,
        extractor: TripletExtractor | None = None,
        expansion_config: ExpansionConfig | None = None,
        selector: EvidenceSelector | None = None,
    ) -> "TreeSearchGraphRAG":
        store = InMemoryGraphStore()
        graph_builder = NodeGraphBuilder(
            extractor=extractor or RuleBasedTripletExtractor(),
            store=store,
        )
        expander = StructureConstrainedExpander(store, expansion_config or ExpansionConfig())
        verifier = EvidenceVerifier(store)
        return cls(
            tree_search=tree_search,
            store=store,
            graph_builder=graph_builder,
            expander=expander,
            selector=selector or HeuristicEvidenceSelector(),
            verifier=verifier,
            answer_generator=TemplateAnswerGenerator(),
        )

    def build_graph(self) -> GraphBuildStats:
        if not self.tree_search.documents and self.tree_search._pending_paths:
            self.tree_search.index(*self.tree_search._pending_paths)
        stats = self.graph_builder.build(self.tree_search.documents)
        self._last_build_stats = stats
        return stats

    def get_stats(self) -> dict[str, int]:
        """Return graph counts, mirroring the VectorGraphRAG stats API."""
        return {
            "passages": len(self.store.passages),
            "entities": len(self.store.entities),
            "relations": len(self.store.relations),
            "structural_edges": len(self.store.structural_edges),
        }

    def retrieve(self, query: str) -> EvidenceChain:
        """Retrieve and verify a compact evidence chain without answer generation."""
        if self._last_build_stats is None:
            self.build_graph()
        seeds = retrieve_seed_nodes(query, self.tree_search.documents)
        candidates = self.expander.expand(query, seeds)
        chain = self.selector.select(query, candidates, self.store)
        return chain

    def query(self, query: str) -> GroundedAnswer:
        chain = self.retrieve(query)
        verification = self.verifier.verify(chain)
        return self.answer_generator.generate(query, chain, verification)
