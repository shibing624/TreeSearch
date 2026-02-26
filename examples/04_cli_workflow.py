# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI workflow demo -- index documents then search via command line.

This script demonstrates the typical TreeSearch workflow:
  1. Build indexes for multiple documents using build_index (returns Documents directly)
  2. Search across all indexed documents with Best-First strategy (default)

Usage:
    python examples/04_cli_workflow.py
"""
import asyncio
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import build_index, search, load_documents

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "markdowns")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "indexes", "cli_demo")


async def main():
    # Clean up from previous runs
    if os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Step 1: Build indexes (returns list[Document] directly)
    print("=== Step 1: Building indexes ===\n")

    md_files = [
        os.path.join(DATA_DIR, "voice-call.md"),
        os.path.join(DATA_DIR, "agent-tools.md"),
    ]
    documents = await build_index(
        paths=md_files,
        output_dir=INDEX_DIR,
        if_add_node_summary=True,
        if_add_node_text=True,
        if_add_doc_description=True,
    )

    for doc in documents:
        print(f"Indexed: {doc.doc_name}")
        print(f"  Description: {doc.doc_description or 'N/A'}")

    # Step 2: Search directly with returned documents
    print(f"\n=== Step 2: Multi-document search (Best-First + BM25) ===\n")
    print(f"Using {len(documents)} documents\n")

    queries = [
        "How to configure the voice call webhook URL?",
        "How to register an optional agent tool?",
        "What TTS providers are supported for calls?",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = await search(
            query=query,
            documents=documents,
            strategy="best_first",
            max_llm_calls=15,
            use_bm25=True,
        )
        print(f"  Strategy: {result.strategy}, LLM calls: {result.total_llm_calls}")
        for doc_result in result.documents:
            for node in doc_result["nodes"]:
                print(f"  [{node.get('score', 0):.2f}] [{doc_result['doc_name']}] {node['title']}")
        print()

    # Alternative: load documents from disk later
    print("=== Alternative: Load from disk ===")
    loaded_docs = load_documents(INDEX_DIR)
    print(f"Loaded {len(loaded_docs)} documents from {INDEX_DIR}")

    # Equivalent CLI commands
    print("\n=== Equivalent CLI commands ===")
    print(f'treesearch index --paths {DATA_DIR}/voice-call.md {DATA_DIR}/agent-tools.md -o {INDEX_DIR} --add-description')
    print(f'treesearch search --index_dir {INDEX_DIR} --query "How to configure the voice call webhook URL?"')
    print(f'treesearch search --index_dir {INDEX_DIR} --query "deployment" --strategy mcts')
    print(f'treesearch search --index_dir {INDEX_DIR} --query "tools" --no-bm25')


if __name__ == "__main__":
    asyncio.run(main())
