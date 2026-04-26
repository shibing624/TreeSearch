# -*- coding: utf-8 -*-
"""Public multi-hop QA retrieval evaluation for TreeSearch.

The copied vector-graph-rag evaluation data is kept under ``evaluation/data``.
This runner evaluates TreeSearch-style retrieval methods against supporting
passage titles from HotpotQA, MuSiQue, 2WikiMultiHopQA, and the included
test_sample fixture.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from treesearch import TreeSearch
from treesearch.fts import FTS5Index
from treesearch.tree import Document


DEFAULT_METHODS = ("treesearch", "fts5", "dense", "hybrid")
K_VALUES = (1, 2, 5, 10)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "evaluation" / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"


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
) -> dict:
    questions, corpus = load_dataset(dataset_name, data_dir)
    selected_questions = questions[:max_samples] if max_samples is not None else questions
    documents, dense_index, fts_index = build_corpus_index(dataset_name, corpus)
    tree_search = TreeSearch(db_path=None)
    tree_search.documents = documents

    rows = []
    summary = {}
    for method in methods:
        method_rows, elapsed = evaluate_method(
            method=method,
            questions=selected_questions,
            dataset_name=dataset_name,
            tree_search=tree_search,
            dense_index=dense_index,
            fts_index=fts_index,
            top_k=top_k,
        )
        rows.extend(method_rows)
        summary[method] = summarize_rows(method_rows, elapsed)

    report = {
        "dataset": dataset_name,
        "num_questions": len(selected_questions),
        "num_corpus_docs": len(documents),
        "methods": list(methods),
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
    dense_index = [
        {
            "retrieval_key": row["retrieval_key"],
            "title": row["title"],
            "text": row["text"],
            "vector": term_vector(" ".join([row["title"], row["text"]])),
        }
        for row in rows
    ]
    fts_index = FTS5Index(db_path=None)
    for document in documents:
        fts_index.index_document(document)
    return documents, dense_index, fts_index


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
    top_k: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    rows = []
    for sample in questions:
        query = str(sample["question"])
        retrieved = retrieve(method, query, tree_search, dense_index, fts_index, top_k)
        gold = gold_items(sample, dataset_name)
        metrics = retrieval_metrics(gold, retrieved)
        rows.append(
            {
                "method": method,
                "query_id": str(sample.get("_id") or sample.get("id") or len(rows)),
                "query": query,
                "gold": sorted(gold),
                "retrieved": retrieved,
                **metrics,
            }
        )
    return rows, time.perf_counter() - started


def retrieve(
    method: str,
    query: str,
    tree_search: TreeSearch,
    dense_index: list[dict],
    fts_index: FTS5Index,
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
    query_vector = term_vector(query)
    scored = sorted(
        dense_index,
        key=lambda item: (-cosine(query_vector, item["vector"]), item["retrieval_key"]),
    )
    return [str(item["retrieval_key"]) for item in scored[:top_k]]


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
    for key in [f"recall@{k}" for k in K_VALUES] + [f"hit@{k}" for k in K_VALUES] + ["mrr"]:
        summary[key] = sum(float(row[key]) for row in rows) / count if count else 0.0
    return summary


def format_markdown(report: dict) -> str:
    lines = [
        f"# Public QA Retrieval Results: {report['dataset']}",
        "",
        f"- Questions: {report['num_questions']}",
        f"- Corpus docs: {report['num_corpus_docs']}",
        "",
        "| Method | Recall@1 | Recall@2 | Recall@5 | Recall@10 | MRR | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in report["summary"].items():
        lines.append(
            "| {method} | {r1:.3f} | {r2:.3f} | {r5:.3f} | {r10:.3f} | {mrr:.3f} | {lat:.4f} |".format(
                method=method,
                r1=row["recall@1"],
                r2=row["recall@2"],
                r5=row["recall@5"],
                r10=row["recall@10"],
                mrr=row["mrr"],
                lat=row["avg_latency_seconds"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def term_vector(text: str) -> Counter:
    return Counter(term.casefold() for term in re.findall(r"[\w_]+", text))


def cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right[token] for token in left.keys() & right.keys())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TreeSearch retrieval on public multi-hop QA datasets.")
    parser.add_argument("--dataset", default="test_sample", choices=["test_sample", "hotpotqa", "musique", "2wikimultihopqa"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
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
    )
    print(format_markdown(report))
    print(f"Saved JSON: {output}")
    print(f"Saved Markdown: {markdown}")


if __name__ == "__main__":
    main()
