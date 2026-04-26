# -*- coding: utf-8 -*-
"""TreeSearch seed retrieval for GraphRAG."""

from treesearch.search import search, search_sync
from treesearch.tree import Document
from treesearch.rag.models import GraphSeed


def retrieve_seed_nodes(
    query: str,
    documents: list[Document],
    top_k: int = 10,
) -> list[GraphSeed]:
    """Retrieve high-precision structural seed nodes with existing TreeSearch."""
    result = search_sync(
        query,
        documents,
        top_k_docs=5,
        max_nodes_per_doc=top_k,
        include_ancestors=True,
        merge_strategy="global_score",
    )
    return _seeds_from_search_result(result, top_k)


async def aretrieve_seed_nodes(
    query: str,
    documents: list[Document],
    top_k: int = 10,
) -> list[GraphSeed]:
    """Async variant for callers that already run inside an event loop."""
    result = await search(
        query,
        documents,
        top_k_docs=5,
        max_nodes_per_doc=top_k,
        include_ancestors=True,
        merge_strategy="global_score",
    )
    return _seeds_from_search_result(result, top_k)


def _seeds_from_search_result(result: dict, top_k: int) -> list[GraphSeed]:
    source = "tree" if result.get("mode") == "tree" else "fts5"
    seeds: list[GraphSeed] = []
    for node in result.get("flat_nodes", [])[:top_k]:
        node_id = str(node.get("node_id", ""))
        doc_id = str(node.get("doc_id", ""))
        if not node_id or not doc_id:
            continue
        seeds.append(
            GraphSeed(
                node_id=node_id,
                doc_id=doc_id,
                score=float(node.get("score", 0.0)),
                source=source,
            )
        )
    return seeds
