# -*- coding: utf-8 -*-
"""Tests for CodeSearchNet GraphRAG benchmark adapter."""

import asyncio

from examples.benchmark.codesearchnet_benchmark import (
    CodeTripletExtractor,
    CodeCorpus,
    CodeSample,
    TreeSearchCodeGraphRAGIndex,
    TreeSearchCodeIndex,
)
from treesearch.rag.models import GraphNodePassage


def test_codesearchnet_graphrag_index_retrieves_matching_function(tmp_path):
    corpus = CodeCorpus(
        language="python",
        samples=[
            CodeSample(
                query="add two numbers",
                code="def add(a, b):\n    return a + b\n",
                func_name="add",
                language="python",
                idx=0,
            ),
            CodeSample(
                query="multiply two numbers",
                code="def multiply(a, b):\n    return a * b\n",
                func_name="multiply",
                language="python",
                idx=1,
            ),
        ],
    )
    base = TreeSearchCodeIndex()
    asyncio.run(base.index(corpus, str(tmp_path / "index")))

    graph_index = TreeSearchCodeGraphRAGIndex(base)
    results = graph_index.search("add two numbers", top_k=2)

    assert results
    assert results[0][0] == 0


def test_code_triplet_extractor_emits_function_and_call_relations():
    passage = GraphNodePassage(
        node_id="n1",
        doc_id="d1",
        doc_name="code.py",
        source_path="code.py",
        source_type="code",
        title="normalize query",
        text='def normalize_query(text):\n    """Normalize a search query."""\n    return clean_text(text)\n',
        path_titles=("normalize_query",),
        line_start=1,
        line_end=3,
    )

    relations = CodeTripletExtractor().extract(passage)
    relation_texts = {relation.text for relation in relations}

    assert "normalize_query defines function normalize_query" in relation_texts
    assert "normalize_query calls clean_text" in relation_texts
