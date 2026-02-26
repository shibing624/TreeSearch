# TreeSearch Retrieve-Rerank Pipeline Design

## Core Insight

Tree search 的价值不在"遍历"，而在"提供结构化上下文给 LLM 做精排"。
用廉价 signal（Embedding / BM25）解决 Recall，用昂贵 signal（LLM）解决 Precision。

## Benchmark Baseline (QASPER, 9 samples)

| Metric | BM25 | Embedding | Hybrid | BestFirst |
|--------|------|-----------|--------|-----------|
| MRR | 0.606 | **0.726** | 0.481 | 0.272 |
| Hit@1 | 0.556 | **0.667** | 0.333 | 0.222 |
| R@5 | 0.806 | **0.917** | 0.833 | 0.444 |
| NDCG@5 | 0.646 | **0.753** | 0.600 | 0.313 |
| LLM calls | 0 | 0 | 0 | 19/query |
| Latency | **0.04s** | 7.8s | 14.2s | 3.7s |

## Problems Identified

1. **BestFirst**: LLM 只看 title+summary（~20% 信息量），19 次 pointwise 打分效率极低
2. **Hybrid**: per-doc min-max 归一化破坏跨文档可比性
3. **BestFirst routing**: route_documents() 选错论文后，后续全废

## Three-Stage Pipeline Architecture

```
Stage 1: Recall Maximizer (cheap, parallel, 0 LLM calls)
  ├── Stage 1a: Embedding top-K_emb (semantic recall)
  └── Stage 1b: BM25 top-K_bm25 (keyword recall, parallel)
  → Union(K_emb ∪ K_bm25) → ~30-35 unique candidates

Stage 2: Signal Enrichment & Soft Prior (0 LLM calls)
  → Candidate-pool normalization (NOT per-doc)
  → score = alpha * bm25_norm + (1-alpha) * emb_norm
  → Soft sort, NO hard pruning
  → Take top-N for Stage 3

Stage 3: Precision Arbiter (1 LLM call, listwise)
  → LLM sees: title + text excerpt + ancestor path
  → Listwise ranking (all candidates in one prompt)
  → Tree-aware context: ancestor path, sibling titles
```

### Stage Roles

| Stage | Role | Cost |
|-------|------|------|
| Stage 1 | Recall maximizer (cheap, parallel) | 0 LLM |
| Stage 2 | Signal enrichment & soft prior | 0 LLM |
| Stage 3 | Precision arbiter (LLM as judge) | 1 LLM |

### Key Design: Stage 2 is NOT a Hard Filter

Stage 2 的职责是「扩展候选 + 提供更强排序 prior」，而不是 pruning。
BM25 和 Embedding 并行召回 → Union 候选池 → 避免过早丢掉 recall。

## Configuration (config-first)

所有参数通过 `RetrieveRerankConfig` 配置，支持环境变量覆盖：

```python
@dataclass
class RetrieveRerankConfig:
    # Stage 1: parallel recall
    embedding_topk: int = 20    # TREESEARCH_RR_EMB_TOPK
    bm25_topk: int = 20         # TREESEARCH_RR_BM25_TOPK

    # Stage 2: score fusion
    bm25_weight: float = 0.5    # TREESEARCH_RR_BM25_WEIGHT
    normalize: str = "candidate_pool"

    # Stage 3: LLM rerank
    rerank_top_n: int = 8       # TREESEARCH_RR_RERANK_N
    rerank_mode: str = "listwise"
    text_excerpt_len: int = 500 # TREESEARCH_RR_EXCERPT_LEN
    include_ancestors: bool = True
    include_sibling_titles: bool = False

    # Adaptive
    query_length_threshold: int = 8
    short_query_bm25_weight: float = 0.7
    long_query_bm25_weight: float = 0.3
```

## Why This Over Simple Hybrid

| Dimension | Simple Emb+BestFirst | Three-Stage Pipeline |
|-----------|---------------------|---------------------|
| Recall | BestFirst relies on LLM routing, may miss | Parallel Emb+BM25 global search, won't miss |
| Precision | LLM sees only summary | LLM sees text + tree context |
| LLM calls | 19 (wasted on irrelevant nodes) | 1 (listwise over top candidates) |
| Latency | ~4s (serial LLM calls) | ~2-3s (embedding fast + 1 LLM call) |
| Tree usage | Only traversal order | Ancestor path as context enhancement |
