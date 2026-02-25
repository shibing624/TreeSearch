# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Best-First Tree Search demo with BM25 pre-scoring.

Demonstrates the three-layer search architecture:
  Layer 1: BM25 node-level pre-scoring (structure-aware, Chinese/English)
  Layer 2: Best-First tree search with LLM relevance evaluation
  Layer 3: Results with budget control and early stopping

Compares: best_first (default) vs mcts vs llm strategies.

Usage:
    python examples/04_multi_doc_search.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import (
    Document,
    NodeBM25Index,
    BestFirstTreeSearch,
    build_index,
    load_index,
    search,
    tokenize,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "markdowns")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "indexes", "best_first_demo")


async def ensure_indexes() -> None:
    """Build indexes if not already present."""
    if os.path.isdir(INDEX_DIR) and any(f.endswith(".json") for f in os.listdir(INDEX_DIR)):
        print(f"Using existing indexes in {INDEX_DIR}/")
        return
    print("Building indexes from markdown files...")
    pattern = os.path.join(DATA_DIR, "*.md")
    results = await build_index(
        paths=[pattern],
        output_dir=INDEX_DIR,
        if_add_node_summary=True,
        if_add_node_text=True,
        if_add_doc_description=True,
        if_add_node_id=True,
    )
    print(f"Indexed {len(results)} file(s) to {INDEX_DIR}/\n")


def load_documents() -> list[Document]:
    """Load all indexed documents."""
    documents = []
    for fp in sorted(os.listdir(INDEX_DIR)):
        if not fp.endswith(".json") or fp.startswith("_"):
            continue
        data = load_index(os.path.join(INDEX_DIR, fp))
        doc = Document(
            doc_id=os.path.splitext(fp)[0].replace("_structure", ""),
            doc_name=data["doc_name"],
            structure=data["structure"],
            doc_description=data.get("doc_description", ""),
        )
        documents.append(doc)
    return documents


async def demo_bm25_standalone(documents: list[Document]):
    """Demo: BM25 node-level search (no LLM, instant results)."""
    print("=" * 60)
    print("Demo 1: BM25 Node-Level Search (no LLM needed)")
    print("=" * 60)

    index = NodeBM25Index(documents)

    queries = [
        "How to configure Twilio voice calls?",
        "agent tool registration",
        "memory workspace recall architecture",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print(f"Tokens: {tokenize(query)}")
        results = index.search(query, top_k=5)
        for r in results:
            print(f"  [{r['bm25_score']:.4f}] [{r['doc_id'][:20]}] {r['title']}")
        if not results:
            print("  (no BM25 matches)")


async def demo_best_first_search(documents: list[Document]):
    """Demo: Best-First tree search with BM25 + LLM."""
    print("\n" + "=" * 60)
    print("Demo 2: Best-First Tree Search (BM25 + LLM)")
    print("=" * 60)

    queries = [
        "How to configure Twilio for voice calls?",
        "What is the memory architecture for workspace recall?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = await search(
            query=query,
            documents=documents,
            strategy="best_first",
            top_k_docs=2,
            max_nodes_per_doc=3,
            max_llm_calls=15,
            use_bm25=True,
        )

        print(f"Strategy: {result.strategy}, LLM calls: {result.total_llm_calls}")
        for doc_result in result.documents:
            print(f"  [{doc_result['doc_name']}]")
            for node in doc_result["nodes"]:
                score = node.get("score", 0)
                print(f"    [{score:.2f}] {node['title']}")
                text = node.get("text", "")
                if text:
                    preview = text[:120].replace("\n", " ")
                    print(f"           {preview}...")


async def demo_strategy_comparison(documents: list[Document]):
    """Demo: Compare best_first vs mcts vs llm strategies."""
    print("\n" + "=" * 60)
    print("Demo 3: Strategy Comparison")
    print("=" * 60)

    query = "How to configure Twilio for voice calls?"
    print(f"\nQuery: {query}\n")

    for strategy in ["best_first", "mcts", "llm"]:
        kwargs = {"strategy": strategy, "top_k_docs": 2, "max_nodes_per_doc": 3}
        if strategy == "mcts":
            kwargs["mcts_iterations"] = 5
        if strategy == "best_first":
            kwargs["max_llm_calls"] = 15

        result = await search(query=query, documents=documents, **kwargs)

        print(f"--- {strategy.upper()} ---")
        print(f"  LLM calls: {result.total_llm_calls}")
        for doc_result in result.documents:
            for node in doc_result["nodes"][:3]:
                score = node.get("score", 0)
                print(f"  [{score:.2f}] {node['title']}")
        print()


async def demo_chinese_query(documents: list[Document]):
    """Demo: Chinese query support."""
    print("=" * 60)
    print("Demo 4: Chinese Query Support")
    print("=" * 60)

    index = NodeBM25Index(documents)
    query = "语音通话配置"

    print(f"\nQuery: {query}")
    print(f"Tokens: {tokenize(query)}")
    results = index.search(query, top_k=3)
    for r in results:
        print(f"  [{r['bm25_score']:.4f}] [{r['doc_id'][:20]}] {r['title']}")
    if not results:
        print("  (no matches)")


async def main():
    await ensure_indexes()

    documents = load_documents()
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        desc = doc.doc_description[:60] if doc.doc_description else "N/A"
        print(f"  - [{doc.doc_name}] {desc}")
    print()

    # BM25 demo (no LLM needed)
    await demo_bm25_standalone(documents)

    # Best-First search (needs LLM API key)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        await demo_best_first_search(documents)
        await demo_strategy_comparison(documents)
    else:
        print("\n(Skipping LLM demos - set OPENAI_API_KEY to enable)")

    # Chinese query demo (no LLM needed)
    await demo_chinese_query(documents)


if __name__ == "__main__":
    asyncio.run(main())
