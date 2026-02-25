# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Build tree index from a real Markdown file and search.

Demonstrates:
  - md_to_tree: parse Markdown into a hierarchical tree structure
  - BestFirstTreeSearch: find relevant sections via best-first tree search (default)
  - MCTSTreeSearch: alternative MCTS strategy
  - No vector embeddings or chunk splitting needed

Usage:
    python examples/01_index_and_search.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import md_to_tree, BestFirstTreeSearch, Document, save_index, print_toc

# Path to a real markdown file shipped with the repo
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "markdowns")
MD_FILE = os.path.join(DATA_DIR, "voice-call.md")


async def main():
    # Step 1: Build tree index from a real markdown document
    print("=== Building tree index from:", os.path.basename(MD_FILE), "===\n")
    result = await md_to_tree(
        md_path=MD_FILE,
        if_add_node_summary=True,
        if_add_node_id=True,
        if_add_node_text=True,
    )

    print("Table of Contents:")
    print_toc(result["structure"])

    output_path = "indexes/voice-call_structure.json"
    save_index(result, output_path)
    print(f"\nIndex saved to: {output_path}")

    # Step 2: Best-First search — find relevant sections by LLM reasoning
    doc = Document(
        doc_id="voice-call",
        doc_name=result["doc_name"],
        structure=result["structure"],
    )

    queries = [
        "How to configure Twilio for voice calls?",
        "What TTS providers are supported?",
        "How to enable inbound calls?",
    ]

    for query in queries:
        print(f"\n--- Query: '{query}' ---")
        searcher = BestFirstTreeSearch(
            document=doc,
            query=query,
            max_results=5,
            max_llm_calls=15,
        )
        results = await searcher.run()
        for r in results:
            print(f"  [{r['score']:.2f}] {r['title']}")


if __name__ == "__main__":
    asyncio.run(main())
