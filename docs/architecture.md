# Architecture

**TreeSearch: Structure-aware document retrieval without embeddings.**

TreeSearch is a **reasoning-based RAG** framework. Unlike traditional RAG systems that rely on vector embeddings and chunk splitting, TreeSearch preserves document structure as trees and uses BM25 + LLM reasoning for retrieval.

## Core Concepts

### Document Tree

Every document is converted into a **hierarchical tree** where:
- Each node represents a section (heading + content)
- Parent-child relationships reflect the document's logical structure
- Nodes carry metadata: `title`, `summary`, `text`, `node_id`

```
Document
├── Chapter 1: Introduction
│   ├── 1.1 Background
│   └── 1.2 Motivation
├── Chapter 2: Methods
│   ├── 2.1 Data Collection
│   └── 2.2 Model Design
└── Chapter 3: Results
```

### Three-Layer Search Architecture

```
Query
  │
  ▼
[Layer 1] BM25 Node-Level Pre-Scoring (millions → thousands)
  │       - Hierarchical field weighting: title > summary > body
  │       - Ancestor score propagation: parent inherits max(child scores)
  │       - Chinese + English tokenization (jieba / regex)
  │       - No LLM needed — instant results
  ▼
[Layer 2] Best-First Tree Search (thousands → tens)
  │       - Priority queue driven: always expands most promising node
  │       - BM25 scores as initial priority (warm start)
  │       - LLM evaluates relevance: title + summary only (no full text)
  │       - Early stopping: queue top score < threshold → stop
  │       - Budget control: max_llm_calls limit
  ▼
[Layer 3] Ranked Results (tens → 2~4)
          - Subtree cache: (query_fingerprint, node_id) for reuse
          - Results with scores, paths, and source text
```

**Stage 0 - Document Routing**: Given a query and multiple documents, the LLM reasons about which documents are most likely to contain the answer, based on document descriptions. No vector search needed. (Skipped automatically for single-document queries.)

**Stage 1 - BM25 Pre-Scoring**: `NodeBM25Index` scores all tree nodes using structure-aware BM25 with hierarchical field weighting (`title × 1.0 + summary × 0.7 + body × 0.3`) and ancestor score propagation. This is instant (no LLM needed) and provides a warm start for tree search.

**Stage 2 - Best-First Tree Search** (default): `BestFirstTreeSearch` uses a priority queue to expand the most promising nodes first. Each node is evaluated by LLM (title + summary only) and scored for relevance. BM25 scores serve as initial priority. Early stopping and budget control keep LLM usage efficient.

**Alternative - MCTS Tree Search**: `MCTSTreeSearch` uses Monte Carlo Tree Search with UCB1 selection. Preserved as an optional strategy via `strategy="mcts"`.

**Alternative - LLM Single-Pass**: `llm_tree_search` sends the full tree structure to LLM in one call. Fastest but least thorough. Use via `strategy="llm"`.

## Module Overview

| Module | Responsibility |
|---|---|
| `llm.py` | Async OpenAI client with retry, token counting, JSON extraction |
| `tree.py` | `Document` dataclass, tree traversal, persistence (save/load) |
| `indexer.py` | Markdown / plain text → tree structure conversion, batch `build_index()` |
| `search.py` | Best-First tree search (default), MCTS, single-pass LLM, document routing, unified `search()` API |
| `rank_bm25.py` | BM25Okapi (pure Python), `NodeBM25Index` with hierarchical weighting, Chinese/English tokenization |
| `cli.py` | CLI with `index` (glob support) and `search` subcommands |

## Why No Vector Embeddings?

Traditional RAG splits documents into fixed-size chunks and retrieves by vector similarity. This has known limitations:

1. **Lost structure**: Chunking destroys document hierarchy
2. **Boundary problems**: Important context may span chunk boundaries
3. **Semantic gaps**: Vector similarity misses reasoning-dependent queries

TreeSearch takes a different approach:

1. **Preserve structure**: Documents keep their natural hierarchy
2. **BM25 pre-scoring**: Structure-aware keyword matching with hierarchical weighting — no embedding model needed
3. **Reasoning over structure**: LLM reads section titles and summaries to navigate
4. **Best-First exploration**: Priority queue expands the most promising nodes first, with early stopping

## Why Best-First Over MCTS?

MCTS was designed for **non-deterministic** problems (e.g., Go) where multiple simulations are needed. In TreeSearch:

- LLM uses `temperature=0` → evaluations are **deterministic**
- Same query + node always produces the same relevance score
- MCTS's `visit_count/total_value` statistics add overhead for no benefit
- UCB1 exploration term decays to zero, making MCTS effectively a slower Best-First

Best-First is the default because it's simpler, faster, and natively supports BM25 integration, early stopping, and budget control. MCTS is preserved as `strategy="mcts"` for comparison.

## Data Flow

```
Input Documents (MD/TXT)
        │
        ▼
   ┌──────────┐
   │  Indexer  │  Parse headings → build tree → generate summaries
   └────┬─────┘    (build_index supports glob patterns for batch processing)
        │  JSON index files
        ▼
   ┌──────────┐
   │  search  │  BM25 pre-score → route to docs → Best-First tree search
   └────┬─────┘
        │  SearchResult
        ▼
  Ranked nodes with text content
```

## Comparison with GraphRAG

| Dimension | GraphRAG | TreeSearch |
|-----------|----------|------------|
| Offline build cost | High (knowledge graph construction) | Low (trees are natural) |
| Update complexity | High (graph incremental update) | Low (rebuild single doc tree) |
| Explainability | Moderate (complex graph paths) | Strong (tree paths are intuitive) |
| Query latency | High | Controllable (budget mechanism) |
| Engineering complexity | High (graph DB + embedding + LLM) | Medium (pure LLM + BM25) |
| Chinese / vertical domains | Unstable (embedding quality dependent) | Customizable (LLM prompts) |
| Vector dependency | Required | Not needed |

**Key insight**: GraphRAG solves "how to build structure from unstructured data". TreeSearch solves "how to truly leverage existing structure". For documents that are naturally trees, TreeSearch is the more efficient, cheaper, and controllable approach.
