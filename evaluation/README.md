# Public QA Evaluation

files in this directory:

- `data/test_sample*.json`
- `data/hotpotqa*.json`
- `data/musique*.json`
- `data/2wikimultihopqa*.json`
- `evaluate.py`, rewritten as a TreeSearch-native retrieval benchmark

## Datasets


| Dataset | Purpose |
|---|---|
| `test_sample` | Tiny sanity fixture for fast local checks |
| `hotpotqa` | 2-hop Wikipedia QA |
| `musique` | 2-4 hop compositional QA |
| `2wikimultihopqa` | Cross-document multi-hop QA |

## Run

```bash
python evaluation/evaluate.py --dataset test_sample --max-samples 10
python evaluation/evaluate.py --dataset hotpotqa --max-samples 50
python evaluation/evaluate.py --dataset musique --max-samples 50
python evaluation/evaluate.py --dataset 2wikimultihopqa --max-samples 50
```

Default outputs:

- `output/public_qa_{dataset}_results.json`
- `output/public_qa_{dataset}_results.md`

## Methods

| Method | Meaning |
|---|---|
| `treesearch` | TreeSearch public API with structure-aware retrieval |
| `fts5` | Direct SQLite FTS5 BM25 retrieval over the full corpus |
| `dense` | Dependency-free lexical cosine proxy used only for smoke/pilot experiments |
| `hybrid` | Reciprocal-rank merge of TreeSearch and lexical dense proxy |

The current public QA runner is a retrieval-only pilot. It reports supporting-passage `Recall@K`, `Hit@K`, `MRR`, and latency. It does not yet evaluate generated answers, LLM reranking, IRCoT, HippoRAG, or Vector Graph RAG under matched settings.
