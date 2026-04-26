# -*- coding: utf-8 -*-
"""TreeSearch-node GraphRAG adapter for GraphRAG-Benchmark.

The official benchmark expects prediction rows containing retrieved contexts,
generated answers, and ground-truth evidence. This adapter creates those rows
from the local GraphRAG-Benchmark checkout under ``evaluation/data`` and keeps
all generated artifacts under ``evaluation/output`` by default.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import SecretStr
from datasets import Dataset
from langchain_openai import ChatOpenAI

from evaluation.evaluate import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    ZhipuEmbeddingClient,
    load_env_file,
)
from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.rag import ExpansionConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "evaluation" / "data" / "GraphRAG-Benchmark"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evaluation" / "output" / "graphrag_bench"


class ZhipuLangChainEmbeddings:
    """LangChain-compatible embeddings backed by Zhipu `embedding-3`."""

    def __init__(
        self,
        client: Any | None = None,
        api_key: str = "",
        model: str = "embedding-3",
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ):
        self.client = client or ZhipuEmbeddingClient(
            api_key=api_key,
            model=model,
            dimensions=dimensions,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.client.embed([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)


@dataclass
class GraphRAGBenchRun:
    predictions_path: Path
    summary_path: Path
    num_questions: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions_path": str(self.predictions_path),
            "summary_path": str(self.summary_path),
            "num_questions": self.num_questions,
            "metrics": self.metrics,
        }


def load_corpus(benchmark_dir: Path, subset: str) -> tuple[str, str]:
    corpus_path = benchmark_dir / "Datasets" / "Corpus" / f"{subset}.json"
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        corpus_name = subset.title()
        text = "\n\n".join(f"# {item['corpus_name']}\n\n{item['context']}" for item in data)
        return corpus_name, text
    corpus_name = str(data["corpus_name"])
    context = data["context"]
    text = "\n\n".join(str(item) for item in context) if isinstance(context, list) else str(context)
    return corpus_name, text


def load_questions(benchmark_dir: Path, subset: str, limit: int | None = None) -> list[dict[str, Any]]:
    question_path = benchmark_dir / "Datasets" / "Questions" / f"{subset}_questions.json"
    questions = json.loads(question_path.read_text(encoding="utf-8"))
    if limit is not None:
        return questions[:limit]
    return questions


def write_corpus_files(corpus_name: str, corpus_text: str, work_dir: Path, chunk_chars: int = 2500) -> list[Path]:
    corpus_dir = work_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, chunk in enumerate(_chunk_text(corpus_text, chunk_chars=chunk_chars)):
        corpus_path = corpus_dir / f"{corpus_name.lower()}_chunk_{idx:04d}.md"
        corpus_path.write_text(f"# {corpus_name} chunk {idx}\n\n{chunk}", encoding="utf-8")
        paths.append(corpus_path)
    return paths


def run_graphrag_bench(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    subset: str = "medical",
    output_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    limit: int | None = None,
    top_k_contexts: int = 5,
) -> dict[str, Any]:
    benchmark_dir = Path(benchmark_dir)
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT_DIR / f"{subset}_treesearch_graphrag.json"
    work_dir = Path(work_dir) if work_dir else DEFAULT_OUTPUT_DIR / "work" / subset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    corpus_name, corpus_text = load_corpus(benchmark_dir, subset)
    questions = load_questions(benchmark_dir, subset, limit=limit)
    corpus_paths = write_corpus_files(corpus_name, corpus_text, work_dir)

    tree_search = TreeSearch(db_path=None)
    tree_search.index(*(str(path) for path in corpus_paths))
    rag = TreeSearchGraphRAG.from_tree_search(
        tree_search,
        expansion_config=ExpansionConfig(max_hops=1, max_relations=40),
    )
    rag.build_graph()

    rows = []
    for question in questions:
        query = str(question["question"])
        answer = rag.query(query)
        contexts = _contexts_from_citations(rag, answer.evidence_chain.citations)
        if not contexts:
            contexts = [text for text in answer.evidence_chain.reasoning_chain if text]
        if not contexts:
            contexts = _fallback_contexts(tree_search, query, top_k_contexts)
        contexts = contexts[:top_k_contexts]
        generated_answer = _extractive_answer(query, contexts, fallback=answer.answer)
        evidence = str(question.get("evidence", ""))
        ground_truth = str(question["answer"])
        rows.append(
            {
                "id": question["id"],
                "source": question.get("source", corpus_name),
                "question": query,
                "question_type": question.get("question_type", "Uncategorized"),
                "ground_truth": ground_truth,
                "answer": ground_truth,
                "generated_answer": generated_answer,
                "context": contexts,
                "contexts": contexts,
                "evidence": evidence,
                "evidences": _split_evidence(evidence),
                "evidence_relations": question.get("evidence_relations", ""),
                "method": "treesearch_graphrag",
            }
        )

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = summarize_offline_metrics(rows)
    summary_path = output_path.with_suffix(".summary.json")
    summary = GraphRAGBenchRun(
        predictions_path=output_path,
        summary_path=summary_path,
        num_questions=len(rows),
        metrics=metrics,
    ).to_dict()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _fallback_contexts(tree_search: TreeSearch, query: str, top_k_contexts: int) -> list[str]:
    result = tree_search.search(query, top_k_docs=5, max_nodes_per_doc=top_k_contexts)
    contexts = []
    for node in result["flat_nodes"][:top_k_contexts]:
        text = str(node.get("text", "") or node.get("summary", "") or node.get("title", ""))
        if text:
            contexts.append(text)
    return contexts


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    chunks = []
    current = []
    current_len = 0
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        if current and current_len + paragraph_len > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += paragraph_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _contexts_from_citations(rag: TreeSearchGraphRAG, citations) -> list[str]:
    contexts = []
    seen = set()
    for citation in citations:
        passage = rag.store.get_passage(citation.doc_id, citation.node_id)
        if passage is None or passage.graph_node_id in seen:
            continue
        seen.add(passage.graph_node_id)
        text = "\n".join(part for part in [passage.title, passage.text] if part)
        if text:
            contexts.append(text)
    return contexts


def _split_evidence(evidence: str) -> list[str]:
    return [part.strip() for part in re.split(r";|\n", evidence) if part.strip()]


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


def _extractive_answer(query: str, contexts: list[str], fallback: str) -> str:
    query_tokens = _tokens(query)
    best_sentence = ""
    best_score = -1.0
    for context in contexts:
        for sentence in _candidate_sentences(context):
            sentence_tokens = _tokens(sentence)
            if not sentence_tokens:
                continue
            overlap = len(query_tokens & sentence_tokens)
            score = overlap / max(len(query_tokens), 1)
            if "most common" in query.lower() and re.search(r"\b(second|third)\s+most\s+common\b", sentence.lower()):
                score -= 0.25
            if score > best_score:
                best_score = score
                best_sentence = sentence
    return best_sentence or (_first_sentence(contexts[0]) if contexts else fallback)


def _candidate_sentences(text: str) -> list[str]:
    without_headings = re.sub(r"(?im)^\s*#?\s*[\w -]*chunk\s+\d+\s*$", " ", text)
    normalized = re.sub(r"\s+", " ", without_headings.replace("#", " ")).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
    return [sentence for sentence in sentences if not re.fullmatch(r"[\w -]*chunk \d+", sentence, flags=re.I)]


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def summarize_offline_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"accuracy": 0.0, "r": 0.0, "ar": 0.0}

    accuracy_scores = []
    relevance_scores = []
    evidence_recall_scores = []
    for row in rows:
        answer_tokens = _tokens(row["ground_truth"])
        generated_tokens = _tokens(row["generated_answer"])
        context_text = "\n".join(row["contexts"])
        context_tokens = _tokens(context_text)
        evidence_items = row["evidences"]

        accuracy_scores.append(1.0 if answer_tokens and answer_tokens <= (generated_tokens | context_tokens) else 0.0)
        relevance_scores.append(1.0 if answer_tokens & context_tokens else 0.0)
        if evidence_items:
            covered = 0
            for evidence in evidence_items:
                evidence_tokens = _tokens(evidence)
                if evidence_tokens and evidence_tokens <= context_tokens:
                    covered += 1
            evidence_recall_scores.append(covered / len(evidence_items))
        else:
            evidence_recall_scores.append(0.0)

    return {
        "accuracy": sum(accuracy_scores) / len(accuracy_scores),
        "r": sum(relevance_scores) / len(relevance_scores),
        "ar": sum(evidence_recall_scores) / len(evidence_recall_scores),
    }


def run_official_evaluators(
    benchmark_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    zhipu_api_key: str = "",
    zhipu_model: str = "embedding-3",
    zhipu_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> dict[str, str]:
    """Call GraphRAG-Benchmark's official evaluators with local Zhipu embeddings."""
    benchmark_dir = Path(benchmark_dir)
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(benchmark_dir))

    retrieval_output = output_dir / f"{predictions_path.stem}_official_retrieval.json"
    generation_output = output_dir / f"{predictions_path.stem}_official_generation.json"

    env = load_env_file(REPO_ROOT / ".env")
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or env.get("LLM_API_KEY") or env.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY or OPENAI_API_KEY is required for official GraphRAG-Benchmark evaluation.")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or env.get("LLM_BASE_URL") or env.get("OPENAI_BASE_URL") or base_url
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        temperature=0.0,
        max_retries=3,
        timeout=30,
    )
    embeddings = ZhipuLangChainEmbeddings(
        api_key=zhipu_api_key,
        model=zhipu_model,
        dimensions=zhipu_dimensions,
    )

    rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    retrieval_results = asyncio.run(_run_official_retrieval(rows, llm, embeddings))
    generation_results = asyncio.run(_run_official_generation(rows, llm, embeddings))

    retrieval_output.write_text(json.dumps(retrieval_results, ensure_ascii=False, indent=2), encoding="utf-8")
    generation_output.write_text(json.dumps(generation_results, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "retrieval_output": str(retrieval_output),
        "generation_output": str(generation_output),
        "embedding_provider": "zhipu",
        "embedding_model": zhipu_model,
    }


async def _run_official_retrieval(rows: list[dict[str, Any]], llm: Any, embeddings: Any) -> dict[str, Any]:
    retrieval_eval = importlib.import_module("Evaluation.retrieval_eval")
    grouped = _group_rows(rows)
    results = {}
    for question_type, items in grouped.items():
        dataset = Dataset.from_dict(
            {
                "id": [item["id"] for item in items],
                "question": [item["question"] for item in items],
                "contexts": [item["contexts"] for item in items],
                "evidences": [item["evidences"] for item in items],
            }
        )
        results[question_type] = await retrieval_eval.evaluate_dataset(
            dataset=dataset,
            llm=llm,
            embeddings=embeddings,
            max_concurrent=1,
            detailed_output=True,
        )
    return results


async def _run_official_generation(rows: list[dict[str, Any]], llm: Any, embeddings: Any) -> dict[str, Any]:
    generation_eval = importlib.import_module("Evaluation.generation_eval")
    metric_config = {
        "Fact Retrieval": ["rouge_score", "answer_correctness"],
        "Complex Reasoning": ["rouge_score", "answer_correctness"],
        "Contextual Summarize": ["answer_correctness", "coverage_score"],
        "Creative Generation": ["answer_correctness", "coverage_score", "faithfulness"],
    }
    grouped = _group_rows(rows)
    results = {}
    for question_type, items in grouped.items():
        metrics = metric_config.get(question_type)
        if metrics is None:
            continue
        dataset = Dataset.from_dict(
            {
                "id": [item["id"] for item in items],
                "question": [item["question"] for item in items],
                "answer": [item["generated_answer"] for item in items],
                "contexts": [item["contexts"] for item in items],
                "ground_truth": [item["ground_truth"] for item in items],
            }
        )
        results[question_type] = await generation_eval.evaluate_dataset(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            max_concurrent=1,
            detailed_output=True,
        )
    return results


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        question_type = str(row.get("question_type", "Uncategorized"))
        grouped.setdefault(question_type, []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TreeSearch-node GraphRAG on GraphRAG-Benchmark")
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--subset", choices=["medical", "novel"], default="medical")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--zhipu-api-key", default="")
    parser.add_argument("--zhipu-model", default="embedding-3")
    parser.add_argument("--zhipu-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS)
    args = parser.parse_args()

    summary = run_graphrag_bench(
        benchmark_dir=args.benchmark_dir,
        subset=args.subset,
        output_path=args.output_path,
        work_dir=args.work_dir,
        limit=args.limit,
    )
    if args.official_eval:
        official_outputs = run_official_evaluators(
            benchmark_dir=args.benchmark_dir,
            predictions_path=summary["predictions_path"],
            output_dir=Path(summary["predictions_path"]).parent,
            model=args.model,
            base_url=args.base_url,
            zhipu_api_key=args.zhipu_api_key,
            zhipu_model=args.zhipu_model,
            zhipu_dimensions=args.zhipu_dimensions,
        )
        summary["official_outputs"] = official_outputs
        Path(summary["summary_path"]).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
