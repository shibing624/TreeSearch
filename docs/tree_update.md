# TreeSearch Retrieve-Rerank Pipeline Design

## Core Insight

Embedding is the strongest single retrieval signal on academic text. BM25 catches
keyword-match nodes that embedding misses. LLM is unreliable for full reranking
but effective as a **comparative judge** between specific candidates.

## Benchmark (QASPER, 9 valid samples)

| Metric | best_first | embedding | bm25 | **retrieve_rerank** |
|--------|-----------|-----------|------|---------------------|
| MRR | 0.472 | 0.726 | 0.611 | **0.744** |
| P@3 | 0.259 | 0.296 | 0.259 | **0.296** |
| R@3 | 0.417 | 0.583 | 0.556 | **0.583** |
| NDCG@3 | 0.444 | 0.598 | 0.540 | **0.605** |
| Hit@1 | 0.444 | 0.667 | 0.556 | **0.667** |
| LLM calls | 21 | 0 | 0 | **1** |

## Evolution Summary

### v1 (broken): RRF + LLM full rerank → MRR=0.333
- Alpha-weighted normalization killed single-channel recall
- LLM rerank fully replaced fusion scores, introducing noise

### v2: RRF + LLM boost mode → MRR=0.402
- RRF fixed single-channel penalty, but still shuffled embedding order
- LLM blend (60/40) still degraded embedding's strong ranking

### v3 (current): Embedding-first + comparative swap → MRR=0.744
- Key lesson: **never reorder embedding's top-k**
- BM25 supplements provide candidates embedding may have missed
- LLM only does A-vs-B comparison between supplements and embedding tail

## Architecture

```
Stage 1: Embedding Recall (backbone, order is sacred)
  → Embedding top-20, take top-5 as initial result
  (0 LLM calls)

Stage 2: BM25 Supplement Discovery
  → BM25 top-20, filter to nodes NOT in embedding top-5
  → These are candidates embedding may have missed
  (0 LLM calls)

Stage 3: LLM Comparative Judge (1 LLM call)
  → LLM rates both: embedding tail (weakest 2) + BM25 supplements
  → If supplement scores > embedding_tail + 0.15 margin:
    swap the weakest embedding result with the strongest supplement
  → At most 1 swap per query (conservative)
```

## Key Design Principles

1. **Embedding order is sacred**: Never reorder embedding's top positions.
   All previous failures came from reordering (RRF, LLM blend, fusion).

2. **LLM as comparative judge, not absolute scorer**: LLM is unreliable for
   absolute relevance scoring but can reliably judge "is A or B more relevant?"

3. **Conservative swap**: At most 1 position swapped per query, and only when
   LLM confidence margin > 0.15. This limits downside risk.

4. **BM25 for recall expansion only**: BM25 never influences ranking within
   embedding's candidates. It only surfaces missed candidates.

## Configuration

All parameters in `RetrieveRerankConfig`, env var overridable:

| Env Var | Default | Description |
|---------|---------|-------------|
| TREESEARCH_RR_EMB_TOPK | 20 | Embedding recall pool size |
| TREESEARCH_RR_BM25_TOPK | 20 | BM25 supplement pool size |
| TREESEARCH_RR_RERANK_N | 15 | Candidate window for comparison |
| TREESEARCH_RR_LLM_WEIGHT | 0.3 | LLM weight (used in config) |
| TREESEARCH_RR_EXCERPT_LEN | 500 | Text excerpt chars for LLM |
