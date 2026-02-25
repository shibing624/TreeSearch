# TreeSearch

**Structure-aware document retrieval without embeddings.**

No vector embeddings. No chunk splitting. BM25 + LLM reasoning over document tree structures.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/shibing624/TreeSearch/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/treesearch.svg)](https://pypi.org/project/treesearch/)

## Why TreeSearch?

Traditional RAG systems split documents into fixed-size chunks and retrieve by vector similarity. This **destroys document structure**, loses heading hierarchy, and misses reasoning-dependent queries.

TreeSearch takes a fundamentally different approach — parse documents into **tree structures** based on their natural heading hierarchy, then use **BM25 + LLM reasoning** to navigate the tree and find the most relevant sections.

| | Traditional RAG | TreeSearch |
|---|---|---|
| **Preprocessing** | Chunk splitting + embedding | Parse headings → build tree |
| **Retrieval** | Vector similarity search | BM25 pre-scoring + LLM tree search |
| **Multi-doc** | Needs vector DB for routing | LLM routes by document descriptions |
| **Structure** | Lost after chunking | Fully preserved as tree hierarchy |
| **Dependencies** | Vector DB + embedding model | LLM only (no embedding, no vector DB) |
| **Zero-cost baseline** | N/A | BM25-only search (no LLM needed) |

### Key Advantages

- **No vector embeddings** — No embedding model to train, deploy, or pay for
- **No chunk splitting** — Documents retain their natural heading structure
- **No vector DB** — No Pinecone, Milvus, or Chroma to manage
- **Tree-aware retrieval** — Heading hierarchy guides search, not arbitrary chunk boundaries
- **BM25 zero-cost baseline** — Instant keyword search with no API calls, useful as standalone or pre-filter
- **Budget-controlled LLM calls** — Set max LLM calls per query, with early stopping when confidence is high

## Features

- **Three-layer search** — BM25 pre-scoring → Best-First tree search → LLM relevance evaluation
- **Tree-structured indexing** — Markdown and plain text documents are parsed into hierarchical trees
- **BM25 node-level index** — Structure-aware scoring with hierarchical field weighting (title > summary > body) and ancestor propagation
- **Best-First search** (default) — Priority queue driven, deterministic, with early stopping and budget control
- **MCTS search** — Monte Carlo Tree Search with LLM as value function
- **LLM single-pass** — One LLM call per document for minimal cost
- **Multi-document search** — Route queries across document collections via LLM reasoning
- **Chinese + English** — jieba tokenization for Chinese, regex for English (jieba is optional)
- **Batch indexing** — `build_index()` supports glob patterns for concurrent multi-file processing
- **Evaluation metrics** — Built-in Precision@K, Recall@K, MRR, NDCG@K, Hit@K, F1@K
- **Async-first** — All core functions are async with sync wrappers available
- **CLI included** — `treesearch index` and `treesearch search` commands

## Installation

```bash
pip install treesearch
```

With Chinese support:
```bash
pip install "treesearch[cn]"
```

From source:

```bash
git clone https://github.com/shibing624/TreeSearch.git
cd TreeSearch
pip install -e ".[dev]"
```

## Quick Start

### 1. Set up API key

```bash
export OPENAI_API_KEY="sk-..."
# Optional: custom endpoint
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

### 2. Build index and search

```python
import asyncio
from treesearch import build_index, load_index, Document, search

async def main():
    # Build indexes for multiple files (supports glob patterns)
    await build_index(
        paths=["docs/*.md"],
        output_dir="./indexes",
        if_add_doc_description=True,
    )

    # Load indexed documents
    import os
    documents = []
    for fp in sorted(os.listdir("./indexes")):
        if not fp.endswith(".json"):
            continue
        data = load_index(os.path.join("./indexes", fp))
        documents.append(Document(
            doc_id=fp,
            doc_name=data["doc_name"],
            structure=data["structure"],
            doc_description=data.get("doc_description", ""),
        ))

    # Search with Best-First strategy (default: BM25 + LLM)
    result = await search(
        query="How does the authentication system work?",
        documents=documents,
    )

    for doc_result in result.documents:
        for node in doc_result["nodes"]:
            print(f"[{node['score']:.2f}] {node['title']}")
            print(f"  {node.get('text', '')[:200]}")

asyncio.run(main())
```

### 3. BM25 standalone (no LLM needed)

```python
from treesearch import NodeBM25Index, Document, load_index

# Load documents
data = load_index("indexes/my_doc.json")
doc = Document(doc_id="doc1", doc_name=data["doc_name"], structure=data["structure"])

# BM25 node-level search — instant results, no API key needed
index = NodeBM25Index([doc])
results = index.search("authentication config", top_k=5)
for r in results:
    print(f"[{r['bm25_score']:.4f}] {r['title']}")
```

### CLI Usage

```bash
# Build indexes from glob pattern
treesearch index --paths "docs/*.md" --add-description

# Build index from specific files
treesearch index --paths doc1.md doc2.txt -o ./indexes

# Search with Best-First (default, BM25 + LLM)
treesearch search --index_dir ./indexes/ --query "How does auth work?"

# Search with MCTS strategy
treesearch search --index_dir ./indexes/ --query "deployment" --strategy mcts

# Search without BM25 pre-scoring
treesearch search --index_dir ./indexes/ --query "config" --no-bm25

# Control LLM budget
treesearch search --index_dir ./indexes/ --query "auth" --max-llm-calls 10
```

## How It Works

```
Input Documents (MD/TXT)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  Parse headings → build tree → generate summaries
   └────┬─────┘    (build_index supports glob for batch processing)
        │  JSON index files
        ▼
   ┌──────────┐
   │  search   │  BM25 pre-score → route to docs → tree search
   └────┬─────┘
        │  SearchResult
        ▼
  Ranked nodes with scores and text
```

**Layer 1 — BM25 Pre-Scoring**: `NodeBM25Index` scores all tree nodes using structure-aware BM25 with hierarchical field weighting (title > summary > body) and ancestor score propagation. Instant, no LLM needed.

**Layer 2 — Best-First Tree Search**: `BestFirstTreeSearch` uses a priority queue to expand the most promising nodes. LLM evaluates each node's relevance (title + summary only). Early stopping when top score drops below threshold.

**Layer 3 — Results**: Budget-controlled LLM calls with subtree caching for reuse across similar queries.

### Search Strategies

| Strategy | Description | LLM Calls | Best For |
|----------|-------------|-----------|----------|
| `best_first` (default) | BM25 pre-scoring + priority queue + LLM evaluation | Moderate (budget-controlled) | General-purpose, best accuracy |
| `mcts` | Monte Carlo Tree Search with LLM as value function | High | Complex reasoning queries |
| `llm` | Single LLM call per document | Minimal | Low-cost, simple queries |
| BM25-only | `NodeBM25Index.search()` standalone | Zero | Instant keyword search, no API key |

## Examples

| Example | Description |
|---------|-------------|
| [`01_index_and_search.py`](examples/01_index_and_search.py) | Single document indexing + BestFirst search |
| [`02_text_indexing.py`](examples/02_text_indexing.py) | Plain text → tree index with auto heading detection |
| [`03_cli_workflow.py`](examples/03_cli_workflow.py) | CLI workflow: build indexes + search with strategies |
| [`04_multi_doc_search.py`](examples/04_multi_doc_search.py) | Multi-doc BM25 + BestFirst + strategy comparison + Chinese |
| [`05_benchmark.py`](examples/05_benchmark.py) | Benchmark: BM25 / BestFirst / MCTS / LLM with metrics |

## Project Structure

```
treesearch/
├── llm.py            # Async LLM client with retry and JSON extraction
├── tree.py           # Document dataclass, tree operations, persistence
├── indexer.py        # Markdown / plain text → tree structure, batch build_index()
├── search.py         # Best-First, MCTS, LLM search, document routing, unified search() API
├── rank_bm25.py      # BM25Okapi, NodeBM25Index, Chinese/English tokenizer
├── metrics.py        # Evaluation: Precision@K, Recall@K, MRR, NDCG@K, Hit@K, F1@K
└── cli.py            # CLI entry point (index / search)
```

## Documentation

- [Architecture](https://github.com/shibing624/TreeSearch/blob/main/docs/architecture.md) — Design principles and three-layer architecture
- [API Reference](https://github.com/shibing624/TreeSearch/blob/main/docs/api.md) — Complete API documentation

## License

Apache License 2.0. See [LICENSE](https://github.com/shibing624/TreeSearch/blob/main/LICENSE) for details.
