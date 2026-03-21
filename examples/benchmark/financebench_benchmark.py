# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: FinanceBench benchmark script.

Evaluates TreeSearch on financial document QA using the FinanceBench dataset.
Compares FTS5 structure-aware search against embedding-based retrieval.

FinanceBench Dataset Overview:
    FinanceBench (PatronusAI) is a benchmark for evaluating LLM performance on
    open-book financial QA. It contains 150 question-answer pairs about publicly
    traded companies' SEC filings (10-K, 10-Q, 8-K).

    Each sample has:
        - financebench_id: unique ID
        - company: company name (e.g. 3M, Amazon, Apple)
        - doc_name: document reference (e.g. AMAZON_2017_10K)
        - question: financial question
        - answer: ground truth answer
        - justification: explanation / calculation steps
        - evidence: list of dicts with evidence_text and page numbers
        - question_type: metrics-generated, domain-relevant, novel-generated
        - question_reasoning: Information extraction, Numerical reasoning, etc.
        - doc_link: URL to original SEC filing PDF
        - doc_type: 10k, 10q, 8k
        - doc_period: fiscal year
        - gics_sector: industry sector

    The benchmark tests retrieval + QA accuracy on complex financial reports.

Usage:
    # Evaluate on all 150 samples with FTS5 (downloads PDFs automatically, skips existing):
    python examples/benchmark/financebench_benchmark.py --max-samples 150

    # Quick test with 20 samples:
    python examples/benchmark/financebench_benchmark.py --max-samples 20

    # Compare with embedding retrieval:
    python examples/benchmark/financebench_benchmark.py --max-samples 50 --with-embedding

  📊 SUMMARY (FinanceBench (SEC Filings))
  ──────────────────────────────────────────────────
  TREESEARCH_FTS5          MRR=0.3969  R@5=0.2773  ⏱ 0.0158s/q  (50/50 valid)
  
"""
import asyncio
import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from treesearch import build_index, flatten_tree
from treesearch.fts import FTS5Index
from treesearch.tree import Document
from treesearch.parsers.pdf_parser import extract_pdf_text

from benchmark_utils import (
    BenchmarkSample,
    resolve_relevant_nodes,
)
from metrics import (
    CostTracker, aggregate_cost_stats,
    evaluate_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_ENV_ZHIPU_API_KEY = "ZHIPU_API_KEY"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FinanceBenchSample:
    """A single FinanceBench question-answer pair."""
    financebench_id: str = ""
    company: str = ""
    doc_name: str = ""
    question: str = ""
    answer: str = ""
    justification: str = ""
    evidence_texts: list[str] = field(default_factory=list)
    evidence_pages: list[int] = field(default_factory=list)
    question_type: str = ""
    question_reasoning: str = ""
    doc_link: str = ""
    doc_type: str = ""
    doc_period: int = 0
    gics_sector: str = ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_financebench_from_hf(
    max_samples: int = 150,
) -> tuple[list[FinanceBenchSample], list[BenchmarkSample]]:
    """Load FinanceBench from HuggingFace (PatronusAI/financebench).

    Uses local pkl cache to avoid slow HuggingFace loading on repeated runs.

    Returns:
        (raw_samples, benchmark_samples) - raw data and BenchmarkSample wrappers
    """
    import pickle

    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"financebench_{max_samples}.pkl")

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            raw_samples, benchmark_samples = pickle.load(f)
        logger.info("Loaded %d FinanceBench samples from cache (%s)",
                    len(raw_samples), cache_path)
        return raw_samples, benchmark_samples

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading FinanceBench from HuggingFace...")
    ds = hf_load("PatronusAI/financebench", split="train")

    raw_samples = []
    benchmark_samples = []

    for idx, row in enumerate(ds):
        if idx >= max_samples:
            break

        # Parse evidence
        evidence_list = row.get("evidence", [])
        evidence_texts = []
        evidence_pages = []
        if isinstance(evidence_list, list):
            for ev in evidence_list:
                if isinstance(ev, dict):
                    ev_text = ev.get("evidence_text", "")
                    ev_page = ev.get("evidence_page_num", 0)
                    if ev_text:
                        evidence_texts.append(ev_text)
                    if ev_page:
                        evidence_pages.append(int(ev_page))
                elif isinstance(ev, str) and ev.strip():
                    evidence_texts.append(ev)

        doc_name = row.get("doc_name", "")
        question = row.get("question", "")
        answer = row.get("answer", "")

        if not question or not answer:
            continue

        raw_sample = FinanceBenchSample(
            financebench_id=row.get("financebench_id", f"fb_{idx}"),
            company=row.get("company", ""),
            doc_name=doc_name,
            question=question,
            answer=answer,
            justification=row.get("justification", ""),
            evidence_texts=evidence_texts,
            evidence_pages=evidence_pages,
            question_type=row.get("question_type", ""),
            question_reasoning=row.get("question_reasoning", ""),
            doc_link=row.get("doc_link", ""),
            doc_type=row.get("doc_type", ""),
            doc_period=int(row.get("doc_period", 0) or 0),
            gics_sector=row.get("gics_sector", ""),
        )
        raw_samples.append(raw_sample)

        benchmark_samples.append(BenchmarkSample(
            question=question,
            answer=answer,
            evidence_texts=evidence_texts,
            relevant_section_titles=[],
            question_type=raw_sample.question_reasoning,
            doc_id=doc_name,
            metadata={
                "company": raw_sample.company,
                "doc_type": raw_sample.doc_type,
                "doc_period": raw_sample.doc_period,
                "gics_sector": raw_sample.gics_sector,
                "justification": raw_sample.justification,
                "evidence_pages": evidence_pages,
            },
        ))

    logger.info("Loaded %d FinanceBench samples", len(raw_samples))

    # Save to pkl cache for fast subsequent loads
    with open(cache_path, "wb") as f:
        pickle.dump((raw_samples, benchmark_samples), f)
    logger.info("Saved FinanceBench cache to %s", cache_path)

    return raw_samples, benchmark_samples


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------

_GITHUB_PDF_BASE = "https://github.com/patronus-ai/financebench/raw/main/pdfs"


def _download_one_pdf(url: str, pdf_path: str, timeout: int = 300) -> bool:
    """Download a single PDF from *url* to *pdf_path*. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (TreeSearch Benchmark)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(pdf_path, "wb") as f:
                f.write(resp.read())
        if os.path.getsize(pdf_path) > 1024:  # sanity: >1KB
            return True
        logger.warning("Downloaded file too small (%d B), treating as failure", os.path.getsize(pdf_path))
        return False
    except Exception as e:
        logger.warning("Download failed (%s): %s", url, e)
        return False


def download_pdfs(
    raw_samples: list[FinanceBenchSample],
    pdf_dir: str,
    skip_existing: bool = True,
    max_retries: int = 2,
    timeout: int = 300,
) -> dict[str, str]:
    """Download SEC filing PDFs for FinanceBench samples.

    Strategy:
      1. Try original SEC doc_link (with retries)
      2. If SEC link fails, fallback to GitHub mirror:
         https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc_name}.pdf

    Args:
        raw_samples: list of FinanceBenchSample with doc_link
        pdf_dir: local directory for PDFs
        skip_existing: skip already-downloaded PDFs
        max_retries: max download retries per source
        timeout: HTTP request timeout in seconds

    Returns:
        {doc_name: local_pdf_path}
    """
    os.makedirs(pdf_dir, exist_ok=True)

    # Deduplicate by doc_name
    doc_links: dict[str, str] = {}
    for s in raw_samples:
        if s.doc_name and s.doc_link:
            doc_links[s.doc_name] = s.doc_link

    pdf_paths: dict[str, str] = {}
    for doc_name, doc_link in doc_links.items():
        safe_name = doc_name.replace("/", "_").replace("\\", "_")
        pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

        if skip_existing and os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 1024:
            logger.info("PDF cached: %s", pdf_path)
            pdf_paths[doc_name] = pdf_path
            continue

        # Build candidate URLs: original SEC link + GitHub mirror
        github_url = f"{_GITHUB_PDF_BASE}/{safe_name}.pdf"
        candidate_urls = [
            ("GitHub", github_url),
            ("SEC", doc_link),
        ]

        downloaded = False
        for source_name, url in candidate_urls:
            for attempt in range(1, max_retries + 1):
                logger.info("Downloading [%s %d/%d]: %s -> %s", source_name, attempt, max_retries, url, pdf_path)
                if _download_one_pdf(url, pdf_path, timeout=timeout):
                    pdf_paths[doc_name] = pdf_path
                    logger.info("Downloaded: %s from %s (%.1f KB)", doc_name, source_name, os.path.getsize(pdf_path) / 1024)
                    downloaded = True
                    break
            if downloaded:
                break

        if not downloaded:
            logger.error("All download attempts failed for %s, skipping", doc_name)

    logger.info("Downloaded %d / %d PDFs", len(pdf_paths), len(doc_links))
    return pdf_paths


# ---------------------------------------------------------------------------
# Zhipu Embedding (same as QASPER/CodeSearchNet benchmarks)
# ---------------------------------------------------------------------------

class ZhipuEmbeddingClient:
    """智谱 BigModel embedding-3 API 客户端。"""

    API_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __init__(self, api_key: str = "", model: str = "embedding-3", dimensions: int = 512):
        self.api_key = api_key or os.getenv(_ENV_ZHIPU_API_KEY, "")
        self.model = model
        self.dimensions = dimensions
        if not self.api_key:
            raise ValueError(
                f"Zhipu API key not set. Please set {_ENV_ZHIPU_API_KEY} environment variable."
            )

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """批量获取 embedding 向量。"""
        if not texts:
            return []
        # embedding-3: max 3072 tokens per text, max 64 texts per batch
        # Truncate long texts to ~8000 chars (~3000 tokens for English)
        max_chars = 8000
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:max_chars].strip() if t.strip() else " " for t in batch]
            payload = json.dumps({
                "model": self.model,
                "input": batch,
                "dimensions": self.dimensions,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.API_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            batch_embs = [item["embedding"] for item in result["data"]]
            all_embeddings.extend(batch_embs)
            if i > 0 and i % 100 == 0:
                logger.info("Embedded %d / %d texts", i + len(batch), len(texts))
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """获取单条文本的 embedding。"""
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Chunk-based document representation (for embedding baseline)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A chunk of text from a document."""
    chunk_id: str
    doc_id: str
    text: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)


def chunk_pdf_text(
    doc_name: str,
    text: str,
    max_chunk_size: int = 1000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split extracted PDF text into overlapping chunks (traditional RAG approach).

    PyMuPDF page-text often lacks ``\\n\\n`` paragraph separators, so we
    progressively fall back: ``\\n\\n`` -> ``\\n`` -> fixed-size character windows.
    """
    chunks: list[Chunk] = []
    if not text or not text.strip():
        return chunks

    # Try paragraph split: \n\n first, fall back to \n
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 3:
        # PDF text has almost no double-newlines; split by single newline
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Further split any paragraph that is still too long (>2x max_chunk_size)
    split_paras: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chunk_size * 2:
            split_paras.append(para)
        else:
            # Force-split by sentence-ish boundaries or fixed window
            pos = 0
            while pos < len(para):
                end = min(pos + max_chunk_size, len(para))
                # Try to break at a sentence boundary ('. ')
                if end < len(para):
                    last_dot = para.rfind(". ", pos + max_chunk_size // 2, end)
                    if last_dot > pos:
                        end = last_dot + 2
                split_paras.append(para[pos:end].strip())
                pos = end

    chunk_idx = 0
    current_text = ""

    for para in split_paras:
        if len(current_text) + len(para) > max_chunk_size and current_text:
            chunks.append(Chunk(
                chunk_id=f"{doc_name}_chunk_{chunk_idx}",
                doc_id=doc_name,
                text=current_text.strip(),
            ))
            chunk_idx += 1
            # Keep overlap
            words = current_text.split()
            overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else []
            current_text = " ".join(overlap_words) + "\n" if overlap_words else ""

        current_text += para + "\n"

    if current_text.strip():
        chunks.append(Chunk(
            chunk_id=f"{doc_name}_chunk_{chunk_idx}",
            doc_id=doc_name,
            text=current_text.strip(),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Embedding index
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """In-memory embedding index with cosine similarity search and disk cache."""

    def __init__(self, emb_client: ZhipuEmbeddingClient, cache_dir: str = ""):
        self.client = emb_client
        self.cache_dir = cache_dir
        self.chunks: list[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._query_cache: dict[str, np.ndarray] = {}  # in-memory cache for query embeddings

    def _cache_key(self, texts: list[str]) -> str:
        content = json.dumps(texts, ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(f"{self.client.model}:{content}".encode()).hexdigest()[:16]
        return h

    def _load_cache(self, cache_key: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"emb_cache_{cache_key}.pkl")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            logger.info("Loaded embedding cache from %s (%d vectors)", path, len(data))
            return np.array(data, dtype=np.float32)
        return None

    def _save_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"emb_cache_{cache_key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(embeddings.tolist(), f)
        logger.info("Saved embedding cache to %s (%d vectors)", path, len(embeddings))

    def index_chunks(self, chunks: list[Chunk]) -> float:
        """Build embedding index for chunks. Returns indexing time."""
        t0 = time.time()
        self.chunks = chunks
        texts = [c.text for c in chunks]
        cache_key = self._cache_key(texts)

        cached = self._load_cache(cache_key)
        if cached is not None:
            self.embeddings = cached
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            self.embeddings = self.embeddings / (norms + 1e-9)
            logger.info("Embedding index loaded from cache: %d chunks", len(chunks))
            return time.time() - t0

        logger.info("Generating embeddings for %d chunks...", len(chunks))
        embeddings = self.client.embed(texts)
        self.embeddings = np.array(embeddings, dtype=np.float32)
        self._save_cache(cache_key, self.embeddings)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-9)

        logger.info("Embedding index built: %d chunks, dim=%d", len(chunks), self.embeddings.shape[1])
        return time.time() - t0

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_id: Optional[str] = None,
    ) -> list[dict]:
        """Search for top-k similar chunks."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        if query in self._query_cache:
            query_embedding = self._query_cache[query]
        else:
            query_embedding = np.array(self.client.embed_single(query), dtype=np.float32)
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
            self._query_cache[query] = query_embedding

        similarities = self.embeddings @ query_embedding

        if doc_id:
            mask = np.array([c.doc_id == doc_id or doc_id in c.doc_id for c in self.chunks])
            similarities = np.where(mask, similarities, -np.inf)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "score": float(similarities[idx]),
                })

        return results


# ---------------------------------------------------------------------------
# Ground truth matching for chunks
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: lowercase, collapse whitespace, strip table pipes."""
    import re
    text = text.lower()
    text = text.replace('|', ' ')  # Markdown table cell separators -> space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def _chunk_contains_evidence(chunk_text: str, evidence_texts: list[str]) -> bool:
    """Check whether a chunk's text contains any of the evidence texts.

    Uses progressive prefix matching (80→50→30 chars) with normalization.
    This is used for text-based relevance judgment in embedding evaluation,
    which avoids the chunk_id mismatch problem caused by overlapping chunks.
    """
    chunk_norm = _normalize_text(chunk_text)
    for evidence in evidence_texts:
        if not evidence:
            continue
        for prefix_len in [80, 50, 30]:
            evidence_prefix = _normalize_text(evidence[:prefix_len])
            if len(evidence_prefix) < 10:
                evidence_prefix = _normalize_text(evidence)
            if evidence_prefix in chunk_norm:
                return True
    return False


def resolve_relevant_chunks(
    sample: BenchmarkSample,
    retrieved_results: list[dict],
) -> list[str]:
    """Identify which of the *retrieved* chunks contain evidence text.

    Instead of pre-computing a fixed set of relevant chunk_ids (which suffers
    from chunk-overlap mismatch — the same evidence can appear in multiple
    overlapping chunks, and the retriever may return a different one than the
    first in order), we directly inspect the retrieved results' text.

    This gives a fair evaluation: a retrieved chunk is "relevant" iff its text
    actually contains the evidence, regardless of its chunk_id.

    Args:
        sample: benchmark sample with evidence_texts.
        retrieved_results: list of dicts from EmbeddingIndex.search(),
                           each having at least {"chunk_id", "text"}.

    Returns:
        List of chunk_ids from *retrieved_results* that contain evidence.
    """
    relevant_ids: list[str] = []
    for r in retrieved_results:
        if _chunk_contains_evidence(r["text"], sample.evidence_texts):
            if r["chunk_id"] not in relevant_ids:
                relevant_ids.append(r["chunk_id"])
    return relevant_ids


# ---------------------------------------------------------------------------
# Build tree indexes from PDFs (delegates to treesearch.build_index)
# ---------------------------------------------------------------------------

async def build_pdf_indexes(
    pdf_paths: dict[str, str],
    output_dir: str,
    max_concurrency: int = 5,
    force: bool = False,
) -> list[Document]:
    """Build tree indexes from PDF files using treesearch.build_index.

    treesearch's parser registry auto-dispatches .pdf -> pdf_to_tree -> text_to_tree,
    which handles PDF text extraction and structure detection internally.

    Args:
        pdf_paths: {doc_name: pdf_path}
        output_dir: directory for indexes
        max_concurrency: max concurrent indexing tasks
        force: force re-index even if file unchanged

    Returns:
        list of Document objects
    """
    if not pdf_paths:
        logger.warning("No PDF files to index")
        return []

    paths_list = list(pdf_paths.values())
    # Map filename (without extension) -> original doc_name
    # build_index uses os.path.splitext(os.path.basename(fp))[0] as doc_id
    filename_to_docname = {}
    for doc_name, pdf_path in pdf_paths.items():
        fname = os.path.splitext(os.path.basename(pdf_path))[0]
        filename_to_docname[fname] = doc_name

    logger.info("Building indexes for %d PDF documents...", len(paths_list))
    documents = await build_index(
        paths=paths_list,
        output_dir=output_dir,
        if_add_node_summary=True,
        if_add_doc_description=False,
        if_add_node_text=True,
        if_add_node_id=True,
        max_concurrency=max_concurrency,
        force=force,
    )

    # Map doc_id back to original doc_name (e.g. AMAZON_2017_10K)
    for doc in documents:
        original_name = filename_to_docname.get(doc.doc_id)
        if original_name:
            doc.doc_id = original_name
            doc.doc_name = original_name

    return documents


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_fts5(
    samples: list[BenchmarkSample],
    documents: list[Document],
    k_values: list[int],
) -> tuple[dict, list[dict]]:
    """Evaluate TreeSearch FTS5 retrieval on FinanceBench samples.

    Returns:
        (avg_metrics, per_sample_results)
    """
    fts_index = FTS5Index()
    fts_index.index_documents(documents)

    all_results = []
    cost_stats = []

    for i, sample in enumerate(samples):
        tracker = CostTracker()
        with tracker:
            all_scored: list[tuple[str, float]] = []
            target_docs = documents
            if sample.doc_id:
                matched = [d for d in documents if sample.doc_id in d.doc_id or sample.doc_id in d.doc_name]
                if matched:
                    target_docs = matched
            for doc in target_docs:
                node_scores = fts_index.score_nodes(sample.question, doc.doc_id)
                all_scored.extend(node_scores.items())
            all_scored.sort(key=lambda x: -x[1])
            retrieved_node_ids = [nid for nid, _ in all_scored[:max(k_values)]]

        relevant_node_ids = resolve_relevant_nodes(sample, documents)
        has_evidence = bool(sample.evidence_texts and any(e for e in sample.evidence_texts))
        if has_evidence:
            # Always evaluate: no relevant match → 0 score (miss), not skip
            metrics = evaluate_query(retrieved_node_ids, relevant_node_ids, k_values) if relevant_node_ids else evaluate_query(retrieved_node_ids, ["__no_match__"], k_values)
        else:
            metrics = {}

        if not has_evidence:
            print(f"  [{i + 1}/{len(samples)}] Skipped (no evidence)")
        else:
            hit = "HIT" if metrics.get("hit@5", 0) > 0 else "miss"
            print(f"  [{i + 1}/{len(samples)}] MRR={metrics.get('mrr', 0):.2f} "
                  f"P@3={metrics.get('precision@3', 0):.2f} R@5={metrics.get('recall@5', 0):.2f} "
                  f"[{hit}] {tracker.stats.latency_seconds:.3f}s")

        all_results.append({
            "metrics": metrics,
            "cost": tracker.stats,
            "sample": sample,
        })
        cost_stats.append(tracker.stats)

    # Aggregate
    valid = [r for r in all_results if r["metrics"]]
    if valid:
        avg_metrics = {}
        for key in valid[0]["metrics"]:
            avg_metrics[key] = sum(r["metrics"].get(key, 0) for r in valid) / len(valid)
        avg_cost = aggregate_cost_stats([r["cost"] for r in valid])
        avg_metrics["avg_query_time"] = avg_cost.latency_seconds
    else:
        avg_metrics = {}

    avg_metrics["total_queries"] = len(samples)
    avg_metrics["valid_queries"] = len(valid)

    return avg_metrics, all_results


def evaluate_tree(
    samples: list[BenchmarkSample],
    documents: list[Document],
    k_values: list[int],
) -> tuple[dict, list[dict]]:
    """Evaluate Tree mode (Best-First Search) retrieval on FinanceBench samples.

    Returns:
        (avg_metrics, per_sample_results)
    """
    from treesearch.tree_searcher import TreeSearcher
    from treesearch.config import set_config as _set_cfg, get_config as _get_cfg, TreeSearchConfig as _TSCfg

    fts_index = FTS5Index()
    fts_index.index_documents(documents)

    # Use wider config for benchmark
    _old = _get_cfg()
    _set_cfg(_TSCfg(
        path_top_k=max(max(k_values), _old.path_top_k),
        anchor_top_k=max(max(k_values), _old.anchor_top_k),
        max_expansions=max(60, _old.max_expansions),
    ))

    all_results = []
    cost_stats = []

    for i, sample in enumerate(samples):
        tracker = CostTracker()
        with tracker:
            target_docs = documents
            if sample.doc_id:
                matched = [d for d in documents if sample.doc_id in d.doc_id or sample.doc_id in d.doc_name]
                if matched:
                    target_docs = matched
            fts_score_map: dict[str, dict[str, float]] = {}
            for doc in target_docs:
                scores = fts_index.score_nodes(sample.question, doc.doc_id)
                if scores:
                    fts_score_map[doc.doc_id] = scores
            searcher = TreeSearcher()
            paths, flat_nodes = searcher.search(sample.question, target_docs, fts_score_map)
            retrieved_node_ids = [fn["node_id"] for fn in flat_nodes[:max(k_values)]]

        relevant_node_ids = resolve_relevant_nodes(sample, documents)
        has_evidence = bool(sample.evidence_texts and any(e for e in sample.evidence_texts))
        if has_evidence:
            metrics = evaluate_query(retrieved_node_ids, relevant_node_ids, k_values) if relevant_node_ids else evaluate_query(retrieved_node_ids, ["__no_match__"], k_values)
        else:
            metrics = {}

        if not has_evidence:
            print(f"  [{i + 1}/{len(samples)}] Skipped (no evidence)")
        else:
            hit = "HIT" if metrics.get("hit@5", 0) > 0 else "miss"
            print(f"  [{i + 1}/{len(samples)}] MRR={metrics.get('mrr', 0):.2f} "
                  f"P@3={metrics.get('precision@3', 0):.2f} R@5={metrics.get('recall@5', 0):.2f} "
                  f"[{hit}] {tracker.stats.latency_seconds:.3f}s")

        all_results.append({
            "metrics": metrics,
            "cost": tracker.stats,
            "sample": sample,
        })
        cost_stats.append(tracker.stats)

    _set_cfg(_old)  # restore config

    valid = [r for r in all_results if r["metrics"]]
    if valid:
        avg_metrics = {}
        for key in valid[0]["metrics"]:
            avg_metrics[key] = sum(r["metrics"].get(key, 0) for r in valid) / len(valid)
        avg_cost = aggregate_cost_stats([r["cost"] for r in valid])
        avg_metrics["avg_query_time"] = avg_cost.latency_seconds
    else:
        avg_metrics = {}

    avg_metrics["total_queries"] = len(samples)
    avg_metrics["valid_queries"] = len(valid)

    return avg_metrics, all_results


def evaluate_embedding(
    samples: list[BenchmarkSample],
    embedding_index: EmbeddingIndex,
    k_values: list[int],
) -> tuple[dict, list[dict]]:
    """Evaluate embedding retrieval on FinanceBench samples.

    Uses text-based relevance: a retrieved chunk is relevant if its text
    contains any evidence prefix. This avoids chunk-id mismatch caused by
    overlapping chunks.

    Returns:
        (avg_metrics, per_sample_results)
    """
    all_results = []

    for i, sample in enumerate(samples):
        tracker = CostTracker()
        with tracker:
            results = embedding_index.search(
                sample.question,
                top_k=max(k_values),
                doc_id=sample.doc_id,
            )
            retrieved_chunk_ids = [r["chunk_id"] for r in results]

        # Text-based relevance: check which retrieved chunks contain evidence
        relevant_chunk_ids = resolve_relevant_chunks(sample, results)
        has_evidence = bool(sample.evidence_texts and any(e for e in sample.evidence_texts))
        if has_evidence:
            # Always evaluate (even if no relevant found — that means a miss, score=0)
            metrics = evaluate_query(retrieved_chunk_ids, relevant_chunk_ids, k_values) if relevant_chunk_ids else evaluate_query(retrieved_chunk_ids, ["__no_match__"], k_values)
        else:
            metrics = {}

        if not has_evidence:
            print(f"  [{i + 1}/{len(samples)}] Skipped (no evidence)")
        else:
            hit = "HIT" if metrics.get("hit@5", 0) > 0 else "miss"
            print(f"  [{i + 1}/{len(samples)}] MRR={metrics.get('mrr', 0):.2f} "
                  f"P@3={metrics.get('precision@3', 0):.2f} R@5={metrics.get('recall@5', 0):.2f} "
                  f"[{hit}] {tracker.stats.latency_seconds:.3f}s")

        all_results.append({
            "metrics": metrics,
            "cost": tracker.stats,
        })

    # Aggregate
    valid = [r for r in all_results if r["metrics"]]
    if valid:
        avg_metrics = {}
        for key in valid[0]["metrics"]:
            avg_metrics[key] = sum(r["metrics"].get(key, 0) for r in valid) / len(valid)
        avg_cost = aggregate_cost_stats([r["cost"] for r in valid])
        avg_metrics["avg_query_time"] = avg_cost.latency_seconds
    else:
        avg_metrics = {}

    avg_metrics["total_queries"] = len(samples)
    avg_metrics["valid_queries"] = len(valid)

    return avg_metrics, all_results


# ---------------------------------------------------------------------------
# Per-type breakdown
# ---------------------------------------------------------------------------

def compute_per_type_metrics(results: list[dict]) -> dict:
    """Compute per question_reasoning type metrics."""
    per_type: dict[str, dict] = {}

    for r in results:
        metrics = r.get("metrics", {})
        if not metrics:
            continue
        sample = r.get("sample")
        if not sample:
            continue
        qtype = sample.question_type or "unknown"

        if qtype not in per_type:
            per_type[qtype] = {"count": 0, "metrics_sum": {}}
        per_type[qtype]["count"] += 1
        for key, val in metrics.items():
            per_type[qtype]["metrics_sum"].setdefault(key, 0.0)
            per_type[qtype]["metrics_sum"][key] += val

    # Average
    for qtype in per_type:
        cnt = per_type[qtype]["count"]
        per_type[qtype]["metrics"] = {
            k: v / cnt for k, v in per_type[qtype]["metrics_sum"].items()
        }
        del per_type[qtype]["metrics_sum"]

    return per_type


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_results(
    dataset_label: str,
    results: dict[str, dict],
    fts_per_type: Optional[dict] = None,
    extra_info: Optional[dict] = None,
):
    """Print formatted benchmark results (aligned with QASPER/CodeSearchNet style)."""
    w = 80
    col_w = 16
    col_names = list(results.keys())
    num_cols = len(results)

    print(f"\n{'=' * w}")
    print(f"  📋 BENCHMARK: {dataset_label}")
    print(f"{'=' * w}")

    def _hdr():
        h = f"  {'Metric':<22}"
        for name in col_names:
            h += f"{name.upper():>{col_w}}"
        return h

    def _row_key(label, key, fmt=".4f"):
        r = f"  {label:<22}"
        for name in col_names:
            val = results[name].get(key, 0)
            r += f"{val:>{col_w}{fmt}}"
        return r

    sep = "  " + "-" * (22 + col_w * num_cols)

    # --- Ranking Quality ---
    print(f"\n  ▸ Ranking Quality")
    print(_hdr())
    print(sep)
    print(_row_key("MRR", "mrr"))
    for k in [1, 3, 5, 10]:
        print(_row_key(f"NDCG@{k}", f"ndcg@{k}"))

    # --- Precision & Recall ---
    print(f"\n  ▸ Precision / Recall / F1")
    print(_hdr())
    print(sep)
    for k in [1, 3, 5, 10]:
        print(_row_key(f"P@{k}", f"precision@{k}"))
        print(_row_key(f"R@{k}", f"recall@{k}"))
        print(_row_key(f"F1@{k}", f"f1@{k}"))
        if k != 10:
            print()

    # --- Hit Rate ---
    print(f"\n  ▸ Hit Rate")
    print(_hdr())
    print(sep)
    for k in [1, 3, 5, 10]:
        print(_row_key(f"Hit@{k}", f"hit@{k}"))

    # --- Cost & Efficiency ---
    if extra_info:
        print(f"\n  ▸ Cost & Efficiency")
        print(_hdr())
        print(sep)
        for key, label in [("index_time", "Index time (s)"), ("num_items", "Num chunks/nodes"), ("avg_query_time", "Avg query time (s)")]:
            r = f"  {label:<22}"
            for name in col_names:
                val = extra_info.get(name, {}).get(key, 0)
                if key == "num_items":
                    r += f"{val:>{col_w}}"
                elif key == "index_time":
                    r += f"{val:>{col_w}.2f}"
                else:
                    r += f"{val:>{col_w}.4f}"
            print(r)

    print(f"\n{'=' * w}")

    # Summary
    print(f"\n  📊 SUMMARY ({dataset_label})")
    print(f"  {'─' * 50}")

    for name, m in results.items():
        mrr_val = m.get("mrr", 0)
        r5_val = m.get("recall@5", 0)
        qt_val = m.get("avg_query_time", 0)
        valid = m.get("valid_queries", 0)
        total = m.get("total_queries", 0)
        print(f"  {name.upper():<24} MRR={mrr_val:.4f}  R@5={r5_val:.4f}  "
              f"⏱ {qt_val:.4f}s/q  ({valid}/{total} valid)")

    # Cross comparison
    if len(results) == 2:
        names = list(results.keys())
        m0, m1 = results[names[0]], results[names[1]]
        mrr0, mrr1 = m0.get("mrr", 0), m1.get("mrr", 0)
        r5_0, r5_1 = m0.get("recall@5", 0), m1.get("recall@5", 0)
        qt0, qt1 = m0.get("avg_query_time", 0), m1.get("avg_query_time", 0)

        print()
        if mrr1 > mrr0:
            print(f"  ✅ {names[1].upper()} outperforms {names[0].upper()} by {(mrr1 - mrr0) / mrr0 * 100:.1f}% on MRR")
        elif mrr0 > mrr1:
            print(f"  📊 {names[0].upper()} outperforms {names[1].upper()} by {(mrr0 - mrr1) / mrr1 * 100:.1f}% on MRR")

        if r5_1 > r5_0:
            print(f"  ✅ {names[1].upper()} outperforms {names[0].upper()} by {(r5_1 - r5_0) / r5_0 * 100:.1f}% on Recall@5")
        elif r5_0 > r5_1:
            print(f"  📊 {names[0].upper()} outperforms {names[1].upper()} by {(r5_0 - r5_1) / r5_1 * 100:.1f}% on Recall@5")

        if qt0 > 0 and qt1 > 0:
            if qt0 > qt1:
                print(f"  ⚡ {names[1].upper()} is {qt0 / qt1:.1f}x faster per query")
            else:
                print(f"  ⚡ {names[0].upper()} is {qt1 / qt0:.1f}x faster per query")

    # Per-type breakdown
    if fts_per_type:
        print(f"\n  ▸ Per Question-Reasoning Type (FTS5)")
        print(f"  {'Type':<28}{'Count':>6}{'MRR':>10}{'R@5':>10}{'Hit@5':>10}")
        print(f"  {'─' * 64}")
        for qtype, info in sorted(fts_per_type.items(), key=lambda x: -x[1]["count"]):
            m = info["metrics"]
            print(f"  {qtype:<28}{info['count']:>6}"
                  f"{m.get('mrr', 0):>10.4f}"
                  f"{m.get('recall@5', 0):>10.4f}"
                  f"{m.get('hit@5', 0):>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="FinanceBench benchmark: evaluate TreeSearch on financial document QA"
    )
    parser.add_argument("--max-samples", type=int, default=150, help="Max QA samples to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results per query")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results/financebench", help="Output directory")
    parser.add_argument("--index-dir", type=str, default="./indexes/financebench", help="Index directory")
    parser.add_argument("--pdf-dir", type=str, default=os.path.join(_PROJECT_ROOT, "data/financebench_pdfs"), help="Directory for downloaded PDFs")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent indexing tasks")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing documents")
    parser.add_argument("--force-download", action="store_true", help="Force re-download PDFs (default: skip existing)")
    parser.add_argument("--with-embedding", action="store_true", help="Also run embedding comparison")
    parser.add_argument(
        "--embedding-model", type=str, default="embedding-3",
        help="Zhipu embedding model for comparison (default: embedding-3)"
    )
    args = parser.parse_args()

    # Step 1: Load dataset
    print("Loading FinanceBench dataset from HuggingFace...")
    raw_samples, benchmark_samples = load_financebench_from_hf(max_samples=args.max_samples)
    print(f"Dataset loaded: {len(raw_samples)} QA samples")

    # Print dataset statistics
    companies = {s.company for s in raw_samples}
    doc_names = {s.doc_name for s in raw_samples}
    doc_types = {s.doc_type for s in raw_samples}
    reasoning_types = {s.question_reasoning for s in raw_samples}
    print(f"  Companies: {len(companies)} | Documents: {len(doc_names)} | "
          f"Doc types: {doc_types} | Reasoning types: {len(reasoning_types)}")

    # Step 2: Download PDFs (default: skip existing, --force-download to re-download)
    print(f"\nPreparing SEC filing PDFs in {args.pdf_dir}...")
    pdf_paths = download_pdfs(raw_samples, args.pdf_dir, skip_existing=not args.force_download)

    if not pdf_paths:
        print("ERROR: No PDF files available. Check network and retry.")
        return

    # Filter samples to only those with available PDFs
    available_docs = set(pdf_paths.keys())
    benchmark_samples = [s for s in benchmark_samples if s.doc_id in available_docs]
    print(f"Samples with available PDFs: {len(benchmark_samples)}")

    # Step 3: Build tree indexes
    print(f"\n{'=' * 60}")
    print(f"Building TreeSearch tree indexes for {len(pdf_paths)} documents...")
    print(f"{'=' * 60}")

    t0 = time.time()
    documents = await build_pdf_indexes(
        pdf_paths=pdf_paths,
        output_dir=args.index_dir,
        max_concurrency=args.concurrency,
        force=args.force_reindex,
    )
    index_time = time.time() - t0
    total_nodes = sum(len(flatten_tree(d.structure)) for d in documents)
    print(f"Indexing completed in {index_time:.1f}s ({len(documents)} documents, {total_nodes} nodes)")

    k_values = [1, 3, 5, 10]

    # Step 4: Evaluate FTS5
    print(f"\n{'=' * 60}")
    print(f"Strategy: TREESEARCH FTS5 | Samples: {len(benchmark_samples)}")
    print(f"{'=' * 60}")

    fts_metrics, fts_results = evaluate_fts5(benchmark_samples, documents, k_values)
    fts_metrics["index_time"] = index_time
    fts_per_type = compute_per_type_metrics(fts_results)

    # Step 4b: Evaluate Tree mode
    print(f"\n{'=' * 60}")
    print(f"Strategy: TREESEARCH TREE (Best-First Search) | Samples: {len(benchmark_samples)}")
    print(f"{'=' * 60}")

    tree_metrics, tree_results = evaluate_tree(benchmark_samples, documents, k_values)
    tree_metrics["index_time"] = index_time
    tree_per_type = compute_per_type_metrics(tree_results)

    all_results = {
        "treesearch_fts5": fts_metrics,
        "treesearch_tree": tree_metrics,
    }
    extra_info = {
        "treesearch_fts5": {
            "index_time": index_time,
            "num_items": total_nodes,
            "avg_query_time": fts_metrics.get("avg_query_time", 0),
        },
        "treesearch_tree": {
            "index_time": index_time,
            "num_items": total_nodes,
            "avg_query_time": tree_metrics.get("avg_query_time", 0),
        },
    }

    # Step 5: Evaluate embedding (optional)
    if args.with_embedding:
        api_key = os.environ.get(_ENV_ZHIPU_API_KEY, "")
        if not api_key:
            print("ZHIPU_API_KEY not set, skipping embedding evaluation")
            print("Set it in .env or: export ZHIPU_API_KEY=your_api_key")
        else:
            print(f"\n{'=' * 60}")
            print("Building EMBEDDING index (traditional RAG approach)...")
            print(f"{'=' * 60}")

            # Build chunks from PDFs using plain text for embedding
            t0 = time.time()
            all_chunks = []
            for doc_name, pdf_path in pdf_paths.items():
                text = extract_pdf_text(pdf_path)
                chunks = chunk_pdf_text(doc_name, text)
                all_chunks.extend(chunks)

            print(f"Created {len(all_chunks)} chunks from {len(pdf_paths)} documents")

            emb_client = ZhipuEmbeddingClient(model=args.embedding_model)
            emb_idx = EmbeddingIndex(
                emb_client,
                cache_dir=os.path.join(args.output_dir, "embedding_cache"),
            )
            emb_idx.index_chunks(all_chunks)
            total_embed_time = time.time() - t0
            print(f"Embedding index built in {total_embed_time:.1f}s ({len(all_chunks)} chunks)")

            print(f"\n{'=' * 60}")
            print(f"Strategy: EMBEDDING ({args.embedding_model}) | Samples: {len(benchmark_samples)}")
            print(f"{'=' * 60}")

            emb_metrics, _ = evaluate_embedding(
                benchmark_samples, emb_idx, k_values,
            )
            emb_metrics["index_time"] = total_embed_time

            all_results["embedding"] = emb_metrics
            extra_info["embedding"] = {
                "index_time": total_embed_time,
                "num_items": len(all_chunks),
                "avg_query_time": emb_metrics.get("avg_query_time", 0),
            }

    # Step 6: Print formatted report
    print_results(
        dataset_label="FinanceBench (SEC Filings)",
        results=all_results,
        fts_per_type=fts_per_type,
        extra_info=extra_info,
    )

    # Print tree per-type breakdown
    if tree_per_type:
        print(f"\n  Per Question-Reasoning Type (TREE)")
        print(f"  {'Type':<28}{'Count':>6}{'MRR':>10}{'R@5':>10}{'Hit@5':>10}")
        print(f"  {'_' * 64}")
        for qtype, info in sorted(tree_per_type.items(), key=lambda x: -x[1]["count"]):
            m = info["metrics"]
            print(f"  {qtype:<28}{info['count']:>6}"
                  f"{m.get('mrr', 0):>10.4f}"
                  f"{m.get('recall@5', 0):>10.4f}"
                  f"{m.get('hit@5', 0):>10.4f}")

    # Step 7: Save results
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "financebench_report.json")
    save_data = {
        "dataset": "financebench",
        "num_samples": len(benchmark_samples),
        "num_documents": len(documents),
        "num_nodes": total_nodes,
        "results": {},
    }
    for name, m in all_results.items():
        save_data["results"][name] = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}
    if fts_per_type:
        save_data["per_type_metrics"] = fts_per_type

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
