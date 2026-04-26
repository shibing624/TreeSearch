# Public QA Evaluation

This directory was copied from `/Users/xuming/Documents/Codes/vector-graph-rag/evaluation` and pruned for this repository.

Kept files:

- `data/test_sample*.json`
- `data/hotpotqa*.json`
- `data/musique*.json`
- `data/2wikimultihopqa*.json`
- `evaluate.py`, rewritten as a TreeSearch-native retrieval benchmark

Removed files:

- GPT OpenIE cache files (`openie_*`)
- NER cache files (`ner_cache/`)
- Milvus/vector-graph-rag-specific runner logic

## Datasets


| Dataset | Purpose |
|---|---|
| `test_sample` | Tiny sanity fixture for fast local checks |
| `hotpotqa` | 2-hop Wikipedia QA |
| `musique` | 2-4 hop compositional QA |
| `2wikimultihopqa` | Cross-document multi-hop QA |

## Run

```bash
python evaluation/evaluate.py --dataset test_sample --max-samples 10 --embedding-cache-path output/zhipu_embeddings_test_sample.json
python evaluation/evaluate.py --dataset hotpotqa --max-samples 50 --embedding-cache-path output/zhipu_embeddings_hotpotqa.json
python evaluation/evaluate.py --dataset musique --max-samples 50 --embedding-cache-path output/zhipu_embeddings_musique.json
python evaluation/evaluate.py --dataset 2wikimultihopqa --max-samples 50 --embedding-cache-path output/zhipu_embeddings_2wikimultihopqa.json
```

## GraphRAG-Benchmark / RepoQA / CodeSearchNet

External benchmark checkouts and large caches live locally under `evaluation/data/` and are ignored by git:

- `evaluation/data/GraphRAG-Benchmark/`
- `evaluation/data/repoqa/`
- `evaluation/data/repoqa-2024-06-23.json`
- `evaluation/data/codesearchnet_cache/`

Run TreeSearch-node GraphRAG on the newer GraphRAG-Benchmark:

```bash
python -m evaluation.graphrag_bench --subset medical --limit 5
python -m evaluation.graphrag_bench --subset novel --limit 5
```

The adapter writes official-compatible prediction rows. Add `--official-eval` to call the benchmark's `Evaluation.retrieval_eval` and `Evaluation.generation_eval`; this requires the official evaluation dependencies plus valid LLM/Zhipu API keys.
The local adapter now uses Zhipu `embedding-3` for official evaluation, so it does not initialize or download HuggingFace BGE embeddings:

```bash
python -m evaluation.graphrag_bench --subset medical --limit 1 --official-eval
```

Run RepoQA SNF retrieval and official `repoqa.compute_score`:

```bash
python -m evaluation.repoqa_snf --dataset-path evaluation/data/repoqa-2024-06-23.json --limit 20
```

Run CodeSearchNet from the evaluation-scoped wrapper:

```bash
python -m evaluation.codesearchnet_eval --language python --max-samples 50 --max-corpus 1000 --with-embedding
python -m evaluation.codesearchnet_eval --language javascript --max-samples 50 --max-corpus 1000 --with-embedding
```

Write external, not locally reproduced, baseline references:

```bash
python -m evaluation.external_baselines
```

Default outputs:

- `output/public_qa_{dataset}_results.json`
- `output/public_qa_{dataset}_results.md`

## Methods

| Method | Meaning |
|---|---|
| `treesearch` | TreeSearch public API with structure-aware retrieval |
| `fts5` | Direct SQLite FTS5 BM25 retrieval over the full corpus |
| `dense` | Zhipu `embedding-3` dense retrieval with JSON embedding cache |
| `hybrid` | Reciprocal-rank merge of TreeSearch and Zhipu dense retrieval |
| `graphrag` | TreeSearch GraphRAG path using a lightweight public-QA title/entity extractor |
| `graphrag_no_structure` | GraphRAG ablation without structural expansion/scoring |
| `graphrag_no_entity` | GraphRAG ablation without entity mention bridges |

The current public QA runner is still a pilot. It reports supporting-passage `Recall@K`, `Hit@K`, `MRR`, extractive answer exact match/accuracy/F1, and latency. It does not yet evaluate generated answers, LLM reranking, IRCoT, HippoRAG, or Vector Graph RAG under matched settings.
