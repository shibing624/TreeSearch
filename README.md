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

**TreeSearch** is a structure-aware document retrieval library. No vector embeddings. No chunk splitting. SQLite FTS5 + BM25 + LLM reasoning over document tree structures. Supports Markdown, plain text, code files (Python AST + regex, Java/Go/JS/C++ etc.), HTML, XML, JSON, CSV, PDF, and DOCX.

Millisecond-latency search over tens of thousands of documents and large codebases, with structure preservation.

## Installation

```bash
pip install -U pytreesearch
```

## Quick Start

```python
from treesearch import TreeSearch

# Lazy indexing — auto-builds index on first search
ts = TreeSearch("docs/*.md", "src/*.py")
results = ts.search("How does auth work?")
for doc in results["documents"]:
    for node in doc["nodes"]:
        print(f"[{node['score']:.2f}] {node['title']}")
```

## Why TreeSearch?

Traditional RAG systems split documents into fixed-size chunks and retrieve by vector similarity. This **destroys document structure**, loses heading hierarchy, and misses reasoning-dependent queries.

TreeSearch takes a fundamentally different approach — parse documents into **tree structures** based on their natural heading hierarchy, then search with **FTS5/BM25 keyword matching** (zero-cost, no API key) or **LLM reasoning** for enhanced accuracy.

| | Traditional RAG | TreeSearch |
|---|---|---|
| **Preprocessing** | Chunk splitting + embedding | Parse headings → build tree |
| **Retrieval** | Vector similarity search | FTS5/BM25 pre-scoring + LLM tree search |
| **Multi-doc** | Needs vector DB for routing | LLM routes by document descriptions |
| **Structure** | Lost after chunking | Fully preserved as tree hierarchy |
| **Dependencies** | Vector DB + embedding model | SQLite only (LLM optional, no embedding, no vector DB) |

### Key Advantages

- **No vector embeddings** — No embedding model to train, deploy, or pay for
- **No chunk splitting** — Documents retain their natural heading structure
- **No vector DB** — No Pinecone, Milvus, or Chroma to manage
- **Tree-aware retrieval** — Heading hierarchy guides search, not arbitrary chunk boundaries
- **SQLite FTS5 pre-filter** (default) — Persistent inverted index with WAL mode, incremental updates, CJK support, and SQL aggregation
- **BM25 zero-cost baseline** — Instant keyword search with no API calls, useful as standalone or pre-filter

## Features

- **FTS5-only search** (default) — Zero LLM calls, millisecond-level FTS5/BM25 keyword matching, no API key needed
- **SQLite FTS5 engine** — Persistent inverted index, WAL mode, incremental updates, MD structure-aware columns (title/summary/body/code/front_matter), column weighting, CJK tokenization
- **Tree-structured indexing** — Markdown, plain text, code files (Python AST + regex, Java/Go/JS/C++/PHP), HTML, XML, JSON, CSV, PDF, and DOCX are parsed into hierarchical trees
- **Parser registry** — Extensible `ParserRegistry` with built-in parsers auto-registered; custom parsers via `ParserRegistry.register()`
- **Python AST parsing** — `ast` module extracts classes/functions with full signatures (parameters, return types); regex fallback for syntax errors
- **PDF/DOCX/HTML parsers** — Optional parsers via `pageindex`, `python-docx`, `beautifulsoup4` (install with `pip install pytreesearch[all]`)
- **GrepFilter** — Exact literal/regex matching for precise symbol and keyword search across tree nodes
- **BM25 node-level index** — Structure-aware scoring with hierarchical field weighting (title > summary > body) and ancestor propagation
- **Best-First search** (optional) — Priority queue driven, FTS5 pre-scoring + LLM evaluation, early stopping and budget control
- **Multi-document search** — Route queries across document collections via LLM reasoning
- **Chinese + English** — Built-in jieba tokenization for Chinese and regex tokenization for English
- **Batch indexing** — `build_index()` supports glob patterns for concurrent multi-file processing
- **Async-first** — All core functions are async with sync wrappers available
- **Config-driven defaults** — `search()` and `build_index()` read defaults from `get_config()`, overridable per-call
- **CLI included** — `treesearch index` and `treesearch search` commands

## FTS5 Standalone (No LLM Needed)

```python
from treesearch import FTS5Index, Document, load_index

data = load_index("indexes/my_doc.json")
doc = Document(doc_id="doc1", doc_name=data["doc_name"], structure=data["structure"])

fts = FTS5Index(db_path="indexes/fts.db")  # persistent, or omit for in-memory
fts.index_documents([doc])

# Simple keyword search
results = fts.search("authentication config", top_k=5)
for r in results:
    print(f"[{r['fts_score']:.4f}] {r['title']}")

# Advanced FTS5 query syntax
results = fts.search("auth", fts_expression='title:auth AND body:config', top_k=5)

# Per-document aggregation
agg = fts.search_with_aggregation("authentication", group_by_doc=True)
for doc_agg in agg:
    print(f"{doc_agg['doc_name']}: {doc_agg['hit_count']} hits, best={doc_agg['best_score']:.4f}")
```

## CLI

```bash
# Build indexes from glob pattern
treesearch index --paths "docs/*.md" --add-description

# Search with Best-First + FTS5 (default pre-filter)
treesearch search --index_dir ./indexes/ --query "How does auth work?" --fts

# Search with persistent FTS5 database
treesearch search --index_dir ./indexes/ --query "auth" --fts --fts-db ./indexes/fts.db

# Control LLM budget
treesearch search --index_dir ./indexes/ --query "auth" --max-llm-calls 10
```

## How It Works

```
Input Documents (MD/TXT/Code/JSON/CSV/HTML/XML/PDF/DOCX)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  ParserRegistry dispatch → parse structure → build tree → generate summaries
   └────┬─────┘    (build_index supports glob for batch processing)
        │  JSON index files
        ▼
   ┌──────────┐
   │  search   │  FTS5/Grep match → (optional) route to docs → tree search
   └────┬─────┘
        │  dict result
        ▼
  Ranked nodes with scores and text
```

**Layer 1 — FTS5/BM25 Pre-Scoring**: `FTS5Index` (default) uses SQLite FTS5 inverted index with MD structure-aware columns and column weighting for fast pre-filtering. Alternatively, `NodeBM25Index` provides in-memory BM25 scoring. Both are instant, no LLM needed.

**Layer 2 — Tree Search** (optional): `TreeSearch` uses a priority queue to expand the most promising nodes. LLM evaluates each node's relevance (title + summary only). Early stopping when top score drops below threshold.

**Layer 3 — Results**: Budget-controlled LLM calls with subtree caching for reuse across similar queries.

### Search Strategies

| Strategy | Description | LLM Calls | Best For |
|----------|-------------|-----------|----------|
| `fts5_only` (default) | Pure FTS5/BM25 scoring | Zero | Fast keyword search, no API key needed |
| `best_first` | FTS5/BM25 pre-scoring + priority queue + LLM evaluation | Moderate (budget-controlled) | Best accuracy |
| `auto` | Per-document strategy based on `source_type` (code → GrepFilter + FTS5) | Varies | Mixed file types |
| FTS5 standalone | `FTS5Index.search()` | Zero | Persistent inverted index, no API key |


FTS5/BM25 strategies work out of the box with no API key. For LLM-enhanced strategy (`best_first`), set up API key:

```bash
# Recommended: TreeSearch-specific environment variables (highest priority)
export TREESEARCH_LLM_API_KEY="sk-..."
export TREESEARCH_LLM_BASE_URL="https://api.openai.com/v1"
export TREESEARCH_MODEL="gpt-4o"

# Alternative: OpenAI-compatible environment variables (fallback)
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## Use Cases

### Use Case 1: Technical Documentation QA (Best Scenario)

**Problem**: Your company has 100+ technical docs (API docs, design docs, RFCs), and traditional search can't find the right answers.

```python
from treesearch import build_index, search

# 1. Build index (run once)
docs = await build_index(
    paths=["docs/*.md", "specs/*.txt"],
    output_dir="./indexes"
)

# 2. Search
result = search(
    query="How to configure Redis cluster?",
    documents=docs,
    strategy="fts5_only"  # Millisecond response
)

# 3. Results — complete sections, not fragments
for doc in result["documents"]:
    print(f"Doc: {doc['doc_name']}")
    for node in doc["nodes"]:
        print(f"  Section: {node['title']}")
        print(f"  Content: {node['text'][:200]}...")
```

**Why better than traditional RAG?**
- ✅ Finds **complete sections**, not fragments
- ✅ Includes **section titles** as context anchors
- ✅ Supports hierarchical navigation (parent/child sections)

### Use Case 2: Codebase Search

**Problem**: Want to search for "login-related classes and methods" in a large codebase, but grep only finds lines without structure.

```python
# Index codebase
docs = await build_index(
    paths=["src/**/*.py", "lib/**/*.java"],
    output_dir="./code_indexes"
)

# Search
result = search(
    query="user login authentication",
    documents=docs,
    strategy="auto"  # Auto-detects code files, uses AST parsing
)

# Results example:
# Doc: auth_service.py
#   class UserAuthenticator
#     def login(username, password)
#     def verify_token(token)
```

**Why better than grep/IDE search?**
- ✅ **Semantic understanding**: Not just keyword matching, understands "login" = "authentication"
- ✅ **Structure-aware**: Finds complete classes/methods with docstrings
- ✅ **Precise location**: Directly locates to code line numbers

### Use Case 3: Long Document QA (Papers/Books)

**Problem**: Have a 50-page paper, want to ask "What experimental methods are mentioned in Chapter 3?"

```python
docs = await build_index(paths=["paper.pdf"])

result = search(
    query="experimental methodology",
    documents=docs,
    strategy="fts5_only"
)

# Automatically finds "3.2 Experimental Design" section content
```

**Why better than Ctrl+F?**
- ✅ **Semantic matching**: Finds synonymous paragraphs for "experimental methods"
- ✅ **Section location**: Tells you which chapter and section
- ✅ **Scalable to multi-doc**: Search 10 papers simultaneously

### Real Case Comparison

**Case**: Find "How to request GPU machines" in company docs

**Traditional way (Ctrl+F)**:
```
Search "GPU" → Found 47 matches → Manual review → 10 minutes
```

**TreeSearch way**:
```python
result = search("How to request GPU machines", docs, strategy="fts5_only")
# Directly returns "Resource Guide > GPU Request Process" section
# Time: < 100ms
```

**Efficiency gain**: **100x**

### Comparison with Other Solutions

| Solution | Pros | Cons | Best For |
|----------|------|------|----------|
| **Ctrl+F** | Simple | No semantic understanding, fragmented results | Known keywords |
| **Traditional RAG** | Good semantic understanding | Chunking destroys context, slow response | Plain text QA |
| **Vector DB** | Similarity search | Requires embedding preprocessing, high cost | Large-scale semantic retrieval |
| **TreeSearch** | Preserves structure + Fast + Zero cost | Requires structured documents | Tech docs/Codebase |

## Benchmark

### Document Retrieval (QASPER)

Evaluated on [QASPER](https://huggingface.co/datasets/allenai/qasper) dataset (50 QA samples, 18 academic papers):

| Metric | Embedding (text-embedding-3-small) | TreeSearch FTS5 |
|--------|-----------------------------------|-----------------|
| **MRR** | 0.5403 | 0.4596 |
| **Precision@1** | 0.3830 | 0.1915 |
| **Recall@5** | 0.5139 | **0.6613** |
| **Index Time** | 118.7s | **0.0s** |
| **Query Time** | 573ms | **0.7ms** |

**Key Findings**:
- ✅ **Embedding MRR +18%** — Better semantic understanding
- ✅ **TreeSearch Recall@5 +29%** — Structure preservation helps recall more relevant content
- ✅ **TreeSearch 780x faster queries** — Milliseconds vs seconds
- ✅ **TreeSearch instant indexing** — No embedding API calls needed

### Code Retrieval (CodeSearchNet)

Evaluated on [CodeSearchNet](https://huggingface.co/datasets/code_search_net) dataset (50 queries, 500 Python corpus):

| Metric | Embedding (text-embedding-3-small) | TreeSearch FTS5 |
|--------|-----------------------------------|-----------------|
| **MRR** | 0.9567 | 0.8469 |
| **Hit@1** | 0.9200 | 0.8000 |
| **Recall@5** | 1.0000 | 0.9200 |
| **Index Time** | 73.7s | **3.3s** |
| **Query Time** | 620ms | **0.8ms** |

**Key Findings**:
- ✅ **Embedding MRR +13%** — Better code semantic understanding
- ✅ **TreeSearch MRR 84.7%** — Strong performance for keyword-based code search
- ✅ **TreeSearch 800x faster queries** — Milliseconds vs seconds
- ✅ **TreeSearch 22x faster indexing** — No embedding API calls needed

### Summary

> TreeSearch is not meant to replace embedding-based retrieval, but to provide a **zero-cost, ultra-fast** alternative. For scenarios prioritizing speed and recall over precision, TreeSearch is the better choice.

Run the benchmarks yourself:
```bash
# Document retrieval (QASPER)
python examples/benchmark/qasper_benchmark.py --max-samples 50 --max-papers 20 --with-embedding

# Code retrieval (CodeSearchNet)
python examples/benchmark/codesearchnet_benchmark.py --max-samples 50 --max-corpus 500 --with-embedding
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
