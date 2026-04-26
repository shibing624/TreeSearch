# -*- coding: utf-8 -*-
"""Public multi-hop QA retrieval evaluation for TreeSearch.

The copied vector-graph-rag evaluation data is kept under ``evaluation/data``.
This runner evaluates TreeSearch-style retrieval methods against supporting
passage titles from HotpotQA, MuSiQue, 2WikiMultiHopQA, and the included
test_sample fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Protocol

from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.fts import FTS5Index
from treesearch.rag import ExpansionConfig
from treesearch.rag.llm import load_env_file
from treesearch.rag.models import GraphNodePassage, GraphRelation
from treesearch.tree import Document


DEFAULT_METHODS = ("treesearch", "fts5", "dense", "hybrid", "graphrag")
K_VALUES = (1, 2, 5, 10)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "evaluation" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_EMBEDDING_DIMENSIONS = 512
ZHIPU_API_KEY_ENV = "ZHIPU_API_KEY"


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        ...


class ZhipuEmbeddingClient:
    """Zhipu BigModel embedding-3 API client."""

    API_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __init__(
        self,
        api_key: str = "",
        model: str = "embedding-3",
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        env_path: str | Path | None = None,
    ):
        env = load_env_file(env_path or REPO_ROOT / ".env")
        self.api_key = api_key or env.get(ZHIPU_API_KEY_ENV) or os.getenv(ZHIPU_API_KEY_ENV, "")
        self.model = model
        self.dimensions = dimensions
        if not self.api_key:
            raise ValueError(
                f"Zhipu API key not set. Please set {ZHIPU_API_KEY_ENV} in environment or repo .env."
            )

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        if not texts:
            return []
        max_chars = 8000
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text[:max_chars].strip() if text.strip() else " " for text in batch]
            payload = json.dumps(
                {
                    "model": self.model,
                    "input": batch,
                    "dimensions": self.dimensions,
                }
            ).encode("utf-8")
            request = urllib.request.Request(
                self.API_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            embeddings.extend([item["embedding"] for item in result["data"]])
        return embeddings


class EmbeddingCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.values: dict[str, list[float]] = {}
        if self.path is not None and self.path.exists():
            self.values = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> list[float] | None:
        return self.values.get(key)

    def set(self, key: str, vector: list[float]) -> None:
        self.values[key] = vector

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.values), encoding="utf-8")


class PublicQATripletExtractor:
    """Lightweight graph extractor for title/entity public-QA passages."""

    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        title = passage.title.strip()
        if not title:
            return []
        entities = [title, *_entity_mentions(passage.text)]
        relations = []
        for entity in dict.fromkeys(entities):
            text = f"{title} mentions {entity}"
            relations.append(
                GraphRelation(
                    relation_id=_public_relation_id(passage.graph_node_id, text),
                    subject=title,
                    predicate="mentions",
                    object=entity,
                    text=text,
                    node_id=passage.node_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                )
            )
        return relations


def load_dataset(dataset_name: str, data_dir: Path = DEFAULT_DATA_DIR) -> tuple[list[dict], object]:
    questions_path = data_dir / f"{dataset_name}.json"
    corpus_path = data_dir / f"{dataset_name}_corpus.json"

    with questions_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    with corpus_path.open("r", encoding="utf-8") as f:
        corpus = json.load(f)
    return questions, corpus


def evaluate_public_qa(
    dataset_name: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    methods: Iterable[str] = DEFAULT_METHODS,
    max_samples: int | None = 50,
    top_k: int = 10,
    output_path: Path | None = None,
    markdown_path: Path | None = None,
    embedding_client: EmbeddingClient | None = None,
    embedding_cache_path: Path | None = None,
    zhipu_api_key: str = "",
    zhipu_model: str = "embedding-3",
    zhipu_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> dict:
    questions, corpus = load_dataset(dataset_name, data_dir)
    selected_methods = tuple(methods)
    selected_questions = questions[:max_samples] if max_samples is not None else questions
    documents, corpus_rows, fts_index = build_corpus_index(dataset_name, corpus)
    corpus_lookup = {row["retrieval_key"]: row["text"] for row in corpus_rows}
    tree_search = TreeSearch(db_path=None)
    tree_search.documents = documents
    dense_index = []
    if _needs_embedding(selected_methods):
        client = embedding_client or ZhipuEmbeddingClient(
            api_key=zhipu_api_key,
            model=zhipu_model,
            dimensions=zhipu_dimensions,
        )
        cache_path = embedding_cache_path or DEFAULT_OUTPUT_DIR / f"zhipu_embeddings_{dataset_name}.json"
        dense_index = build_dense_index(corpus_rows, client, EmbeddingCache(cache_path))
    graphrag = build_public_graphrag(tree_search, top_k) if "graphrag" in selected_methods else None

    rows = []
    summary = {}
    for method in selected_methods:
        method_rows, elapsed = evaluate_method(
            method=method,
            questions=selected_questions,
            dataset_name=dataset_name,
            tree_search=tree_search,
            dense_index=dense_index,
            fts_index=fts_index,
            corpus_lookup=corpus_lookup,
            graphrag=graphrag,
            top_k=top_k,
        )
        rows.extend(method_rows)
        summary[method] = summarize_rows(method_rows, elapsed)

    report = {
        "dataset": dataset_name,
        "num_questions": len(selected_questions),
        "num_corpus_docs": len(documents),
        "methods": list(selected_methods),
        "summary": summary,
        "rows": rows,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_markdown(report), encoding="utf-8")
    return report


def build_corpus_index(dataset_name: str, corpus: object) -> tuple[list[Document], list[dict], FTS5Index]:
    rows = list(iter_corpus_rows(dataset_name, corpus))
    documents = [
        Document(
            doc_id=f"{dataset_name}:{idx}",
            doc_name=row["title"],
            structure=[
                {
                    "node_id": "root",
                    "title": row["title"],
                    "summary": "",
                    "text": row["text"],
                    "line_start": 1,
                    "line_end": max(1, row["text"].count("\n") + 1),
                }
            ],
            metadata={
                "source_path": row["retrieval_key"],
                "title": row["title"],
            },
            source_type="text",
        )
        for idx, row in enumerate(rows)
    ]
    fts_index = FTS5Index(db_path=None)
    for document in documents:
        fts_index.index_document(document)
    return documents, rows, fts_index


def build_dense_index(
    corpus_rows: list[dict],
    embedding_client: EmbeddingClient,
    embedding_cache: EmbeddingCache,
    batch_size: int = 64,
) -> list[dict]:
    texts = [" ".join([row["title"], row["text"]]) for row in corpus_rows]
    vectors = embed_with_cache(texts, embedding_client, embedding_cache, batch_size=batch_size)
    return [
        {
            "retrieval_key": row["retrieval_key"],
            "title": row["title"],
            "text": row["text"],
            "vector": vector,
            "embedding_client": embedding_client,
            "embedding_cache": embedding_cache,
        }
        for row, vector in zip(corpus_rows, vectors)
    ]


def embed_with_cache(
    texts: list[str],
    embedding_client: EmbeddingClient,
    embedding_cache: EmbeddingCache,
    batch_size: int = 10,
    max_retries: int = 2,
) -> list[list[float]]:
    keys = [_embedding_key(text) for text in texts]
    vectors: list[list[float] | None] = [embedding_cache.get(key) for key in keys]
    missing_indexes = [idx for idx, vector in enumerate(vectors) if vector is None]
    for start in range(0, len(missing_indexes), batch_size):
        batch_indexes = missing_indexes[start:start + batch_size]
        batch_texts = [texts[idx] for idx in batch_indexes]
        embedded = embed_batch_with_retry(
            embedding_client,
            batch_texts,
            batch_size=batch_size,
            max_retries=max_retries,
        )
        for idx, vector in zip(batch_indexes, embedded):
            vectors[idx] = vector
            embedding_cache.set(keys[idx], vector)
        embedding_cache.save()
    return [vector for vector in vectors if vector is not None]


def embed_batch_with_retry(
    embedding_client: EmbeddingClient,
    texts: list[str],
    batch_size: int,
    max_retries: int,
) -> list[list[float]]:
    for attempt in range(max_retries + 1):
        try:
            return embedding_client.embed(texts, batch_size=batch_size)
        except (TimeoutError, urllib.error.URLError):
            if attempt >= max_retries:
                raise
            time.sleep(min(2 ** attempt, 5))
    raise RuntimeError("unreachable embedding retry state")


def build_public_graphrag(tree_search: TreeSearch, top_k: int) -> TreeSearchGraphRAG:
    rag = TreeSearchGraphRAG.from_tree_search(
        tree_search,
        extractor=PublicQATripletExtractor(),
        expansion_config=ExpansionConfig(max_relations=max(top_k * 3, 30), max_hops=1),
    )
    rag.build_graph()
    return rag


def iter_corpus_rows(dataset_name: str, corpus: object) -> Iterable[dict]:
    if isinstance(corpus, dict):
        for title, value in corpus.items():
            text = " ".join(value) if isinstance(value, list) else str(value)
            yield {
                "title": str(title),
                "text": text,
                "retrieval_key": str(title),
            }
        return

    if not isinstance(corpus, list):
        raise TypeError(f"unsupported corpus format for {dataset_name}: {type(corpus)!r}")

    for item in corpus:
        title = str(item["title"])
        text = str(item.get("text") or item.get("paragraph_text") or "")
        retrieval_key = title if dataset_name != "musique" else f"{title}\n{text}"
        yield {
            "title": title,
            "text": text,
            "retrieval_key": retrieval_key,
        }


def evaluate_method(
    method: str,
    questions: list[dict],
    dataset_name: str,
    tree_search: TreeSearch,
    dense_index: list[dict],
    fts_index: FTS5Index,
    corpus_lookup: dict[str, str],
    graphrag: TreeSearchGraphRAG | None,
    top_k: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    rows = []
    for sample in questions:
        query = str(sample["question"])
        retrieved = retrieve(method, query, tree_search, dense_index, fts_index, graphrag, top_k)
        gold = gold_items(sample, dataset_name)
        metrics = retrieval_metrics(gold, retrieved)
        predicted_answer = select_predicted_answer(query, retrieved, corpus_lookup)
        answer_side = answer_metrics(str(sample.get("answer", "")), predicted_answer)
        rows.append(
            {
                "method": method,
                "query_id": str(sample.get("_id") or sample.get("id") or len(rows)),
                "query": query,
                "gold_answer": str(sample.get("answer", "")),
                "predicted_answer": predicted_answer,
                "gold": sorted(gold),
                "retrieved": retrieved,
                **metrics,
                **answer_side,
            }
        )
    return rows, time.perf_counter() - started


def retrieve(
    method: str,
    query: str,
    tree_search: TreeSearch,
    dense_index: list[dict],
    fts_index: FTS5Index,
    graphrag: TreeSearchGraphRAG | None,
    top_k: int,
) -> list[str]:
    if method == "treesearch":
        return sparse_retrieve(tree_search, query, top_k, search_mode="auto")
    if method == "fts5":
        return fts_retrieve(fts_index, tree_search.documents, query, top_k)
    if method == "dense":
        return dense_retrieve(dense_index, query, top_k)
    if method == "hybrid":
        sparse = sparse_retrieve(tree_search, query, top_k, search_mode="auto")
        dense = dense_retrieve(dense_index, query, top_k)
        return rrf_merge(sparse, dense, top_k)
    if method == "graphrag":
        if graphrag is None:
            raise ValueError("graphrag method requires a TreeSearchGraphRAG instance")
        return graphrag_retrieve(graphrag, tree_search, query, top_k)
    raise ValueError(f"unsupported method: {method}")


def sparse_retrieve(tree_search: TreeSearch, query: str, top_k: int, search_mode: str) -> list[str]:
    routing_k = len(tree_search.documents) if search_mode == "flat" else top_k
    result = tree_search.search(
        query,
        top_k_docs=routing_k,
        max_nodes_per_doc=1,
        search_mode=search_mode,
        merge_strategy="global_score",
        text_mode="none",
    )
    retrieved = []
    for node in result.get("flat_nodes", []):
        source_path = str(node.get("source_path") or "")
        if source_path and source_path not in retrieved:
            retrieved.append(source_path)
        if len(retrieved) >= top_k:
            break
    return retrieved


def fts_retrieve(fts_index: FTS5Index, documents: list[Document], query: str, top_k: int) -> list[str]:
    doc_lookup = {document.doc_id: document for document in documents}
    rows = fts_index.search(query, top_k=top_k * 50)
    retrieved = []
    for row in rows:
        doc_id = str(row["doc_id"])
        if doc_id in doc_lookup:
            source_path = str(doc_lookup[doc_id].metadata["source_path"])
            if source_path not in retrieved:
                retrieved.append(source_path)
    return retrieved[:top_k]


def dense_retrieve(dense_index: list[dict], query: str, top_k: int) -> list[str]:
    if not dense_index:
        raise ValueError("dense retrieval requires an embedding index")
    embedding_client = dense_index[0]["embedding_client"]
    embedding_cache = dense_index[0]["embedding_cache"]
    query_vector = embed_with_cache([query], embedding_client, embedding_cache)[0]
    scored = sorted(
        dense_index,
        key=lambda item: (-cosine(query_vector, item["vector"]), item["retrieval_key"]),
    )
    return [str(item["retrieval_key"]) for item in scored[:top_k]]


def graphrag_retrieve(
    graphrag: TreeSearchGraphRAG,
    tree_search: TreeSearch,
    query: str,
    top_k: int,
) -> list[str]:
    answer = graphrag.query(query)
    retrieved = []
    for citation in answer.evidence_chain.citations:
        if citation.source_path and citation.source_path not in retrieved:
            retrieved.append(citation.source_path)
    if not retrieved:
        return sparse_retrieve(tree_search, query, top_k, search_mode="auto")
    return retrieved[:top_k]


def rrf_merge(left: list[str], right: list[str], top_k: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in (left, right):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [
        item
        for item, _score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    ][:top_k]


def gold_items(sample: dict, dataset_name: str) -> set[str]:
    if dataset_name in {"hotpotqa", "2wikimultihopqa", "test_sample"}:
        return {str(item[0]) for item in sample.get("supporting_facts", [])}

    if dataset_name == "musique":
        return {
            f"{paragraph['title']}\n{paragraph['paragraph_text']}"
            for paragraph in sample.get("paragraphs", [])
            if paragraph.get("is_supporting")
        }

    return {
        str(paragraph.get("title", ""))
        for paragraph in sample.get("paragraphs", [])
        if paragraph.get("is_supporting")
    }


def retrieval_metrics(gold: set[str], retrieved: list[str]) -> dict:
    metrics = {}
    for k in K_VALUES:
        metrics[f"recall@{k}"] = recall_at_k(gold, retrieved, k)
        metrics[f"hit@{k}"] = 1.0 if gold & set(retrieved[:k]) else 0.0
    metrics["mrr"] = reciprocal_rank(gold, retrieved)
    return metrics


def answer_metrics(gold_answer: str, predicted_answer: str) -> dict:
    normalized_gold = normalize_answer(gold_answer)
    normalized_predicted = normalize_answer(predicted_answer)
    exact_match = 1.0 if normalized_gold and normalized_gold == normalized_predicted else 0.0
    accuracy = 1.0 if normalized_gold and normalized_gold in normalized_predicted else exact_match
    return {
        "answer_exact_match": exact_match,
        "answer_accuracy": accuracy,
        "answer_f1": token_f1(normalized_gold, normalized_predicted),
    }


def select_predicted_answer(query: str, retrieved: list[str], corpus_lookup: dict[str, str]) -> str:
    query_terms = set(tokenize(normalize_answer(query)))
    best_sentence = ""
    best_score = -1.0
    for retrieval_key in retrieved:
        for sentence in split_sentences(corpus_lookup.get(retrieval_key, "")):
            sentence_terms = set(tokenize(normalize_answer(sentence)))
            if not sentence_terms:
                continue
            score = len(query_terms & sentence_terms) / max(len(query_terms), 1)
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return best_sentence


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def reciprocal_rank(gold: set[str], retrieved: list[str]) -> float:
    if not gold:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def summarize_rows(rows: list[dict], elapsed: float) -> dict:
    count = len(rows)
    summary = {
        "count": count,
        "latency_seconds": elapsed,
        "avg_latency_seconds": elapsed / count if count else 0.0,
    }
    metric_keys = (
        [f"recall@{k}" for k in K_VALUES]
        + [f"hit@{k}" for k in K_VALUES]
        + ["mrr", "answer_exact_match", "answer_accuracy", "answer_f1"]
    )
    for key in metric_keys:
        summary[key] = sum(float(row[key]) for row in rows) / count if count else 0.0
    return summary


def format_markdown(report: dict) -> str:
    lines = [
        f"# Public QA Retrieval Results: {report['dataset']}",
        "",
        f"- Questions: {report['num_questions']}",
        f"- Corpus docs: {report['num_corpus_docs']}",
        "",
        "| Method | Recall@1 | Recall@2 | Recall@5 | Recall@10 | MRR | Acc | F1 | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["summary"].items():
        lines.append(
            "| {method} | {r1:.3f} | {r2:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {acc:.3f} | {f1:.3f} | {lat:.4f} |".format(
                method=method,
                r1=row["recall@1"],
                r2=row["recall@2"],
                r5=row["recall@5"],
                r10=row["recall@10"],
                mrr=row["mrr"],
                acc=row["answer_accuracy"],
                f1=row["answer_f1"],
                lat=row["avg_latency_seconds"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def token_f1(normalized_gold: str, normalized_predicted: str) -> float:
    gold_tokens = tokenize(normalized_gold)
    predicted_tokens = tokenize(normalized_predicted)
    if not gold_tokens or not predicted_tokens:
        return 0.0
    gold_counts = token_counts(gold_tokens)
    predicted_counts = token_counts(predicted_tokens)
    overlap = sum(min(count, predicted_counts.get(token, 0)) for token, count in gold_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def normalize_answer(text: str) -> str:
    return " ".join(tokenize(text.casefold()))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _needs_embedding(methods: tuple[str, ...]) -> bool:
    return any(method in {"dense", "hybrid"} for method in methods)


def _embedding_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entity_mentions(text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}\b", text)
    blocked = {"The", "This", "It", "He", "She", "They", "In", "On", "At", "A", "An"}
    return [match for match in matches[:30] if match not in blocked]


def _public_relation_id(graph_node_id: str, text: str) -> str:
    payload = f"{graph_node_id}:{text}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TreeSearch retrieval on public multi-hop QA datasets.")
    parser.add_argument("--dataset", default="test_sample", choices=["test_sample", "hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--embedding-cache-path", type=Path)
    parser.add_argument("--zhipu-api-key", default="")
    parser.add_argument("--zhipu-model", default="embedding-3")
    parser.add_argument("--zhipu-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or DEFAULT_OUTPUT_DIR / f"public_qa_{args.dataset}_results.json"
    markdown = args.markdown_output or DEFAULT_OUTPUT_DIR / f"public_qa_{args.dataset}_results.md"
    report = evaluate_public_qa(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        methods=tuple(args.methods),
        max_samples=args.max_samples,
        top_k=args.top_k,
        output_path=output,
        markdown_path=markdown,
        embedding_cache_path=args.embedding_cache_path,
        zhipu_api_key=args.zhipu_api_key,
        zhipu_model=args.zhipu_model,
        zhipu_dimensions=args.zhipu_dimensions,
    )
    print(format_markdown(report))
    print(f"Saved JSON: {output}")
    print(f"Saved Markdown: {markdown}")


if __name__ == "__main__":
    main()
