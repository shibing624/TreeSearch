# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-document search with strategy comparison and GrepFilter demo.

Demonstrates:
  - TreeSearch multi-doc search (default FTS5, zero LLM calls)
  - BM25 standalone search (advanced)
  - GrepFilter for exact literal matching (advanced)
  - Strategy comparison: fts5_only vs best_first

Usage:
    python examples/04_multi_doc_search.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from treesearch import TreeSearch, NodeBM25Index, GrepFilter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "markdowns")
INDEX_DB = os.path.join(os.path.dirname(__file__), "indexes", "multi_doc_demo", "index.db")

# 1. Create engine and index
ts = TreeSearch(db_path=INDEX_DB)
ts.index(f"{DATA_DIR}/*.md")
print(f"Indexed {len(ts.documents)} documents: {[d.doc_name for d in ts.documents]}\n")

# --- Demo 1: FTS5 multi-doc search (default, no LLM) ---
print("=" * 60)
print("Demo 1: FTS5 Multi-Document Search (no LLM)")
print("=" * 60)
for query in ["voice call configuration", "agent tool registration"]:
    result = ts.search(query, max_nodes_per_doc=3)
    print(f"\nQuery: {query}")
    for doc in result["documents"]:
        for node in doc["nodes"]:
            print(f"  [{node['score']:.4f}] {node['title']}")

# --- Demo 2: BM25 standalone (advanced) ---
print("\n" + "=" * 60)
print("Demo 2: BM25 Standalone Search (no LLM)")
print("=" * 60)
bm25 = NodeBM25Index(ts.documents)
for query in ["Twilio voice calls", "语音通话配置"]:
    print(f"\nQuery: {query}")
    results = bm25.search(query, top_k=3)
    for r in results:
        print(f"  [{r['bm25_score']:.4f}] {r['title']}")
    if not results:
        print("  (no matches)")

# --- Demo 3: GrepFilter for exact matching (advanced) ---
print("\n" + "=" * 60)
print("Demo 3: GrepFilter Exact Matching")
print("=" * 60)
grep = GrepFilter(ts.documents)
for query in ["Twilio", "white_list"]:
    print(f"\nGrep: '{query}'")
    for doc in ts.documents:
        hits = grep.score_nodes(query, doc.doc_id)
        if hits:
            print(f"  [{doc.doc_name}] {len(hits)} node(s) matched")
            for nid in list(hits)[:3]:
                node = doc.get_node_by_id(nid)
                print(f"    [{nid}] {node.get('title', '') if node else ''}")

# --- Demo 4: Strategy comparison (needs LLM) ---
api_key = os.getenv("TREESEARCH_LLM_API_KEY")
if api_key:
    print("\n" + "=" * 60)
    print("Demo 4: Strategy Comparison")
    print("=" * 60)
    query = "How to configure Twilio for voice calls?"
    for strategy in ["fts5_only", "best_first"]:
        kwargs = {"strategy": strategy, "max_nodes_per_doc": 3}
        if strategy == "best_first":
            kwargs["max_llm_calls"] = 10
        result = ts.search(query, **kwargs)
        print(f"\n{strategy}: LLM calls={result.get('llm_calls', 0)}")
        for doc in result["documents"]:
            for node in doc["nodes"][:3]:
                print(f"  [{node['score']:.2f}] {node['title']}")
else:
    print("\n(Skipping LLM demos — set OPENAI_API_KEY to enable)")
