# -*- coding: utf-8 -*-
"""Tests for CodeSearchNet GraphRAG benchmark adapter."""

import asyncio

from examples.benchmark.codesearchnet_benchmark import (
    CodeCorpus,
    CodeSample,
    TreeSearchCodeGraphRAGIndex,
    TreeSearchCodeIndex,
)


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
