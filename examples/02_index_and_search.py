# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Indexing + search demo with TreeSearch and lower-level APIs.

Demonstrates:
  - TreeSearch: high-level API for indexing + search (recommended)
  - md_to_tree / text_to_tree: lower-level tree building APIs
  - FTS5Index: standalone keyword search over tree nodes

Usage:
    python examples/02_index_and_search.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import TreeSearch, md_to_tree, text_to_tree, Document, save_index, print_toc, FTS5Index

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "markdowns")
MD_FILE = os.path.join(DATA_DIR, "voice-call.md")

SAMPLE_TEXT = """\
1. Introduction

This chapter introduces the fundamental concepts of distributed systems.

1.1 Definition and Goals

The primary goals of distributed systems include:
- Resource sharing across networked computers
- Transparency in distribution

1.2 Types of Distributed Systems

There are several categories:
- Distributed computing systems (cluster, grid, cloud)
- Distributed information systems (transaction processing)

2. Architecture

This chapter covers architectural styles for distributed systems.

2.1 Layered Architecture

Software is organized into layers where each layer provides services
to the layer above and uses services from the layer below.

2.2 Peer-to-Peer Architecture

In P2P systems, all nodes are equal and can act as both client and server.
"""


def demo_treesearch():
    """Demo 1: TreeSearch high-level API (recommended)."""
    print("=" * 60)
    print("Demo 1: TreeSearch High-Level API (Recommended)")
    print("=" * 60)

    ts = TreeSearch(MD_FILE)
    result = ts.search("How to configure Twilio?")
    for doc in result["documents"]:
        print(f"\n[{doc['doc_name']}]")
        for node in doc["nodes"]:
            print(f"  [{node['score']:.4f}] {node['title']}")

    # Search more queries
    for query in ["What TTS providers are supported?", "语音通话配置"]:
        print(f"\nQuery: {query}")
        result = ts.search(query)
        for doc in result["documents"]:
            for node in doc["nodes"]:
                print(f"  [{node['score']:.4f}] {node['title']}")


async def demo_lower_level_apis():
    """Demo 2: Lower-level APIs — md_to_tree + FTS5Index."""
    print("\n" + "=" * 60)
    print("Demo 2: Lower-Level APIs (md_to_tree + FTS5Index)")
    print("=" * 60)

    result = await md_to_tree(md_path=MD_FILE, if_add_node_summary=True, if_add_node_text=True)

    print("\nTable of Contents:")
    print_toc(result["structure"])

    output_path = "indexes/voice-call.db"
    save_index(result, output_path)
    print(f"\nIndex saved to: {output_path}")

    doc = Document(doc_id="voice-call", doc_name=result["doc_name"], structure=result["structure"])

    fts = FTS5Index()
    fts.index_documents([doc])

    for query in ["How to configure Twilio?", "What TTS providers are supported?"]:
        print(f"\n--- Query: '{query}' ---")
        results = fts.search(query, top_k=3)
        for r in results:
            print(f"  [{r['fts_score']:.4f}] [{r['node_id']}] {r['title']}")


async def demo_plain_text():
    """Demo 3: Build tree from plain text with auto heading detection."""
    print("\n" + "=" * 60)
    print("Demo 3: Plain Text Indexing (auto heading detection)")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SAMPLE_TEXT)
        text_path = f.name

    try:
        result = await text_to_tree(text_path=text_path, fallback_to_llm="no", if_add_node_summary=True)
        print("\nTable of Contents:")
        print_toc(result["structure"])
    finally:
        os.unlink(text_path)


async def main():
    await demo_lower_level_apis()
    await demo_plain_text()


if __name__ == "__main__":
    demo_treesearch()
    asyncio.run(main())
