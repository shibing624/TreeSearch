[**🌐English**](https://github.com/shibing624/TreeSearch/blob/main/README.md) | [**🇨🇳中文**](https://github.com/shibing624/TreeSearch/blob/main/README_ZH.md)

<div align="center">
  <a href="https://github.com/shibing624/TreeSearch">
    <img src="https://raw.githubusercontent.com/shibing624/TreeSearch/main/docs/logo.svg" height="150" alt="Logo">
  </a>
</div>

-----------------

# TreeSearch: Structure-Aware Document Retrieval
[![PyPI version](https://badge.fury.io/py/pytreesearch.svg)](https://badge.fury.io/py/pytreesearch)
[![Downloads](https://static.pepy.tech/badge/pytreesearch)](https://pepy.tech/project/pytreesearch)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/TreeSearch.svg)](https://github.com/shibing624/TreeSearch/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Community)

**TreeSearch** is a structure-aware document retrieval library. No vector embeddings. No chunk splitting. BM25 + LLM reasoning over document tree structures.

## Installation

```bash
pip install -U pytreesearch
```

## Quick Start

```python
import asyncio
from treesearch import build_index, load_index, Document, search

async def main():
    # Build indexes for markdown files
    await build_index(paths=["docs/*.md"], output_dir="./indexes")

    # Load indexed documents
    import os
    documents = []
    for fp in sorted(os.listdir("./indexes")):
        if not fp.endswith(".json"):
            continue
        data = load_index(os.path.join("./indexes", fp))
        documents.append(Document(
            doc_id=fp, doc_name=data["doc_name"],
            structure=data["structure"],
            doc_description=data.get("doc_description", ""),
        ))

    # Search with Best-First strategy (BM25 + LLM)
    result = await search(query="How does auth work?", documents=documents)
    for doc_result in result.documents:
        for node in doc_result["nodes"]:
            print(f"[{node['score']:.2f}] {node['title']}")

asyncio.run(main())
```

Set up API key first:

```bash
export OPENAI_API_KEY="sk-..."
# Optional: custom endpoint
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

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
- **Chinese + English** — Built-in jieba tokenization for Chinese and regex tokenization for English
- **Batch indexing** — `build_index()` supports glob patterns for concurrent multi-file processing
- **Evaluation metrics** — Built-in Precision@K, Recall@K, MRR, NDCG@K, Hit@K, F1@K
- **Async-first** — All core functions are async with sync wrappers available
- **CLI included** — `treesearch index` and `treesearch search` commands

## BM25 Standalone (No LLM Needed)

```python
from treesearch import NodeBM25Index, Document, load_index

data = load_index("indexes/my_doc.json")
doc = Document(doc_id="doc1", doc_name=data["doc_name"], structure=data["structure"])

index = NodeBM25Index([doc])
results = index.search("authentication config", top_k=5)
for r in results:
    print(f"[{r['bm25_score']:.4f}] {r['title']}")
```

## CLI

```bash
# Build indexes from glob pattern
treesearch index --paths "docs/*.md" --add-description

# Search with Best-First (default, BM25 + LLM)
treesearch search --index_dir ./indexes/ --query "How does auth work?"

# Search with MCTS strategy
treesearch search --index_dir ./indexes/ --query "deployment" --strategy mcts

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

## Community

- **GitHub Issues** — [Submit an issue](https://github.com/shibing624/TreeSearch/issues)
- **WeChat Group** — Add WeChat ID `xuming624`, note "llm", to join the tech group

<img src="https://github.com/shibing624/TreeSearch/blob/main/docs/wechat.jpeg" width="200" />

## Citation

If you use TreeSearch in your research, please cite:

```bibtex
@software{xu2026treesearch,
  author = {Xu, Ming},
  title = {TreeSearch: Structure-Aware Document Retrieval Without Embeddings},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/shibing624/TreeSearch}
}
```

## License

[Apache License 2.0](LICENSE)

## Contributing

Contributions are welcome! Please submit a [Pull Request](https://github.com/shibing624/TreeSearch/pulls).

## Acknowledgements

- [BM25 (Okapi BM25)](https://en.wikipedia.org/wiki/Okapi_BM25) — The classic probabilistic ranking function
- [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) — Inspiration for structure-aware indexing and retrieval