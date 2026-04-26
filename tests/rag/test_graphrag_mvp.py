# -*- coding: utf-8 -*-
"""Tests for the TreeSearch-guided GraphRAG MVP."""

from treesearch import TreeSearch
from treesearch.tree import Document
from treesearch.rag import TreeSearchGraphRAG
from treesearch.rag.extractors import PreExtractedTripletExtractor, RuleBasedTripletExtractor
from treesearch.rag.expansion import ExpansionConfig, StructureConstrainedExpander
from treesearch.rag.graph_builder import NodeGraphBuilder
from treesearch.rag.graph_store import InMemoryGraphStore
from treesearch.rag.models import EvidenceChain, EvidenceCitation, GraphSeed, make_graph_node_id
from treesearch.rag.node_graph import document_to_node_passages, document_to_structural_edges
from treesearch.rag.verifier import EvidenceVerifier


def _repo_docs() -> list[Document]:
    code_doc = Document(
        doc_id="code",
        doc_name="config.py",
        source_type="code",
        metadata={"source_path": "treesearch/config.py"},
        structure=[
            {
                "title": "Configuration",
                "node_id": "cfg-root",
                "text": "Configuration module.",
                "line_start": 1,
                "line_end": 20,
                "nodes": [
                    {
                        "title": "TreeSearchConfig",
                        "node_id": "cfg-class",
                        "text": "max_concurrency is defined_in TreeSearchConfig.",
                        "line_start": 10,
                        "line_end": 14,
                    }
                ],
            }
        ],
    )
    docs_doc = Document(
        doc_id="docs",
        doc_name="runtime.md",
        source_type="markdown",
        metadata={"source_path": "docs/runtime.md"},
        structure=[
            {
                "title": "Runtime Settings",
                "node_id": "runtime",
                "text": (
                    "TreeSearchConfig is documented_in Runtime Settings. "
                    "max_concurrency controls worker parallelism."
                ),
                "line_start": 42,
                "line_end": 58,
            }
        ],
    )
    return [code_doc, docs_doc]


def test_document_to_node_passages_preserves_structure_metadata():
    passages = document_to_node_passages(_repo_docs()[0])
    by_id = {passage.node_id: passage for passage in passages}

    passage = by_id["cfg-class"]
    assert passage.doc_id == "code"
    assert passage.source_path == "treesearch/config.py"
    assert passage.source_type == "code"
    assert passage.path_titles == ("Configuration", "TreeSearchConfig")
    assert passage.line_start == 10
    assert passage.line_end == 14
    assert passage.parent_node_id == "cfg-root"

    edges = document_to_structural_edges(_repo_docs()[0])
    assert any(edge.src_node_id == "cfg-root" and edge.dst_node_id == "cfg-class" for edge in edges)
    assert any(edge.src_node_id == "cfg-class" and edge.dst_node_id == "cfg-root" for edge in edges)


def test_graph_builder_links_relations_to_treesearch_nodes():
    store = InMemoryGraphStore()
    builder = NodeGraphBuilder(extractor=RuleBasedTripletExtractor(), store=store)

    stats = builder.build(_repo_docs())

    assert stats.documents == 2
    assert stats.passages == 3
    assert stats.relations == 2
    relation_texts = {relation.text for relation in store.relations.values()}
    assert "max_concurrency defined_in TreeSearchConfig" in relation_texts
    assert "TreeSearchConfig documented_in Runtime Settings" in relation_texts
    assert store.get_relations_by_node("code", "cfg-class")[0].doc_id == "code"


def test_graph_store_uses_document_scoped_node_keys():
    docs = [
        Document(
            doc_id="a",
            doc_name="a.md",
            source_type="markdown",
            metadata={"source_path": "a.md"},
            structure=[
                {
                    "title": "Same",
                    "node_id": "same-node",
                    "text": "Alpha is related_to Shared.",
                    "line_start": 1,
                    "line_end": 2,
                }
            ],
        ),
        Document(
            doc_id="b",
            doc_name="b.md",
            source_type="markdown",
            metadata={"source_path": "b.md"},
            structure=[
                {
                    "title": "Same",
                    "node_id": "same-node",
                    "text": "Beta is related_to Shared.",
                    "line_start": 1,
                    "line_end": 2,
                }
            ],
        ),
    ]
    store = InMemoryGraphStore()

    stats = NodeGraphBuilder(extractor=RuleBasedTripletExtractor(), store=store).build(docs)

    assert stats.passages == 2
    assert len(store.passages) == 2
    assert make_graph_node_id("a", "same-node") in store.passages
    assert make_graph_node_id("b", "same-node") in store.passages
    assert len(store.relations) == 2


def test_pre_extracted_triplet_extractor_builds_graph_without_text_patterns():
    docs = _repo_docs()
    docs[0].structure[0]["nodes"][0]["text"] = "This node intentionally has no extractor-friendly syntax."
    store = InMemoryGraphStore()
    extractor = PreExtractedTripletExtractor(
        {
            make_graph_node_id("code", "cfg-class"): [
                ("max_concurrency", "defined_in", "TreeSearchConfig"),
                {
                    "subject": "TreeSearchConfig",
                    "predicate": "documented_in",
                    "object": "Runtime Settings",
                },
            ]
        }
    )

    stats = NodeGraphBuilder(extractor=extractor, store=store).build(docs)

    assert stats.relations == 2
    relation_texts = {relation.text for relation in store.relations.values()}
    assert "max_concurrency defined_in TreeSearchConfig" in relation_texts
    assert "TreeSearchConfig documented_in Runtime Settings" in relation_texts


def test_structure_constrained_expansion_prefers_seed_node_relations():
    store = InMemoryGraphStore()
    NodeGraphBuilder(extractor=RuleBasedTripletExtractor(), store=store).build(_repo_docs())
    expander = StructureConstrainedExpander(store, ExpansionConfig(max_relations=2))

    candidates = expander.expand(
        "max_concurrency TreeSearchConfig Runtime Settings",
        seeds=[GraphSeed(node_id="cfg-class", doc_id="code", score=1.0, source="tree")],
    )

    assert len(candidates) == 2
    assert candidates[0].relation.text == "max_concurrency defined_in TreeSearchConfig"
    assert candidates[0].sparse_seed_score > candidates[1].sparse_seed_score
    assert candidates[0].structure_score >= candidates[1].structure_score


def test_evidence_verifier_rejects_forged_relation_and_node_ids():
    store = InMemoryGraphStore()
    NodeGraphBuilder(extractor=RuleBasedTripletExtractor(), store=store).build(_repo_docs())
    verifier = EvidenceVerifier(store)

    forged = EvidenceChain(
        query="max_concurrency",
        bridge_entities=("TreeSearchConfig",),
        selected_relation_ids=("missing-relation",),
        selected_node_ids=("missing-node",),
        reasoning_chain=("missing relation",),
        citations=(
            EvidenceCitation(
                node_id="missing-node",
                doc_id="code",
                source_path="treesearch/config.py",
                line_start=1,
                line_end=2,
            ),
        ),
        evidence_sufficiency=True,
    )

    result = verifier.verify(forged)
    assert not result.ok
    assert any("relation" in error for error in result.errors)
    assert any("node" in error for error in result.errors)


def test_pipeline_returns_verified_grounded_answer():
    ts = TreeSearch(db_path=None)
    ts.documents = _repo_docs()
    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=RuleBasedTripletExtractor(),
        expansion_config=ExpansionConfig(max_relations=5),
    )

    stats = rag.build_graph()
    answer = rag.query("max_concurrency TreeSearchConfig Runtime Settings")

    assert stats.relations == 2
    assert answer.verification.ok
    assert "max_concurrency defined_in TreeSearchConfig" in answer.answer
    citation_paths = {citation.source_path for citation in answer.evidence_chain.citations}
    assert "treesearch/config.py" in citation_paths
    assert "docs/runtime.md" in citation_paths


def test_pipeline_exposes_stats_and_retrieval_only_chain():
    ts = TreeSearch(db_path=None)
    ts.documents = _repo_docs()
    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=RuleBasedTripletExtractor(),
        expansion_config=ExpansionConfig(max_relations=5),
    )

    rag.build_graph()
    stats = rag.get_stats()
    chain = rag.retrieve("max_concurrency TreeSearchConfig Runtime Settings")

    assert stats["passages"] == 3
    assert stats["relations"] == 2
    assert chain.evidence_sufficiency
    assert "max_concurrency defined_in TreeSearchConfig" in chain.reasoning_chain
