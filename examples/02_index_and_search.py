# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Build tree index from a real Markdown file and search.

Demonstrates:
  - md_to_tree: parse Markdown into a hierarchical tree structure
  - BestFirstTreeSearch: find relevant sections via best-first tree search (default)
  - No vector embeddings or chunk splitting needed

Usage:
    python examples/02_index_and_search.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import md_to_tree, BestFirstTreeSearch, Document, save_index, print_toc

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

    # Step 2: Build Document and search
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
            node_id = r.get("node_id", "")
            full = doc.get_node_by_id(node_id)
            line_start = full.get("line_start", "") if full else ""
            line_end = full.get("line_end", "") if full else ""
            summary = (full.get("summary", full.get("prefix_summary", "")) if full else "")[:80]
            text_preview = (full.get("text", "") if full else "").replace("\n", " ")[:120]
            print(f"  [{r['score']:.2f}] [{node_id}] {r['title']}  L{line_start}-{line_end}")
            if summary:
                print(f"         summary: {summary}...")
            if text_preview:
                print(f"         text: {text_preview}...")


if __name__ == "__main__":
    asyncio.run(main())
