# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CodeSearchNet benchmark script.

Evaluates TreeSearch on code retrieval using the CodeSearchNet dataset.
Compares FTS5 structure-aware search against embedding-based retrieval.

CodeSearchNet Dataset Overview:
    CodeSearchNet is a large-scale code search benchmark with 6 programming languages.
    Each sample has a natural language query (docstring) paired with code.

    Languages: Python, Java, JavaScript, Go, Ruby, PHP
    Task: Given a natural language query, retrieve the relevant code snippet.

    Data format: Each sample has:
        - func_code_string: the code
        - func_documentation_string: natural language docstring (used as query)
        - func_name: function name
        - language: programming language

Usage:
    # Evaluate on 50 samples
    python examples/benchmark/codesearchnet_benchmark.py --max-samples 50

    # Evaluate specific language
    python examples/benchmark/codesearchnet_benchmark.py --language python --max-samples 100

    # Compare with embedding
    python examples/benchmark/codesearchnet_benchmark.py --with-embedding --max-samples 50


  📊 SUMMARY (CodeSearchNet (PYTHON))
  ──────────────────────────────────────────────────
  TREESEARCH_FTS5          MRR=0.9050  R@5=0.9200  ⏱ 0.0045s/q
  TREESEARCH_TREE          MRR=0.2833  R@5=0.3000  ⏱ 0.0302s/q
  TREESEARCH_AUTO          MRR=0.9100  R@5=0.9200  ⏱ 0.0045s/q
  EMBEDDING                MRR=0.8483  R@5=0.9400  ⏱ 0.1660s/q
"""
import asyncio
import argparse
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from treesearch import TreeSearch, TreeSearchGraphRAG, build_index
from treesearch.fts import FTS5Index
from treesearch.rag import ExpansionConfig
from treesearch.rag.models import GraphNodePassage, GraphRelation
from treesearch.tree import Document, flatten_tree

from metrics import (
    CostTracker, aggregate_cost_stats,
    evaluate_query,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeSample:
    """A single code search sample."""
    query: str = ""  # Natural language query (docstring)
    code: str = ""  # Code snippet
    func_name: str = ""
    language: str = ""
    repo: str = ""
    path: str = ""
    url: str = ""
    idx: int = 0  # Index in the corpus


@dataclass
class CodeCorpus:
    """Code corpus for retrieval."""
    samples: list[CodeSample] = field(default_factory=list)
    language: str = ""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_codesearchnet_from_hf(
    language: str = "python",
    split: str = "test",
    max_samples: int = 100,
    max_corpus: int = 1000,
) -> tuple[list[CodeSample], CodeCorpus]:
    """Load CodeSearchNet from HuggingFace.

    Uses local pkl cache to avoid slow HuggingFace loading on repeated runs.

    Args:
        language: Programming language (python, java, javascript, go, ruby, php)
        split: Dataset split (train, validation, test)
        max_samples: Max query samples for evaluation
        max_corpus: Max code samples for corpus (retrieval candidates)

    Returns:
        (query_samples, corpus) - queries to evaluate, corpus to search in
    """
    import pickle

    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"codesearchnet_{language}_{split}_{max_samples}_{max_corpus}.pkl")

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            query_samples, corpus = pickle.load(f)
        logger.info("Loaded %d query samples, %d corpus samples from cache (%s)",
                    len(query_samples), len(corpus.samples), cache_path)
        return query_samples, corpus

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info(f"Loading CodeSearchNet ({language}, {split}) from HuggingFace...")
    ds = hf_load("code_search_net", language, split=split, trust_remote_code=True)

    # Build corpus from all samples
    corpus_samples = []
    for idx, row in enumerate(ds):
        if idx >= max_corpus:
            break

        code = row.get("func_code_string", "") or row.get("whole_func_string", "")
        docstring = row.get("func_documentation_string", "")

        if not code or not docstring:
            continue

        corpus_samples.append(CodeSample(
            query=docstring.strip(),
            code=code.strip(),
            func_name=row.get("func_name", ""),
            language=language,
            repo=row.get("repository_name", ""),
            path=row.get("path", ""),
            url=row.get("url", ""),
            idx=len(corpus_samples),
        ))

    corpus = CodeCorpus(samples=corpus_samples, language=language)

    # Use first max_samples as queries (with known ground truth = their own code)
    query_samples = corpus_samples[:max_samples]

    logger.info(f"Loaded {len(query_samples)} query samples, {len(corpus_samples)} corpus samples")

    # Save to pkl cache for fast subsequent loads
    with open(cache_path, "wb") as f:
        pickle.dump((query_samples, corpus), f)
    logger.info("Saved CodeSearchNet cache to %s", cache_path)

    return query_samples, corpus


# ---------------------------------------------------------------------------
# Embedding-based retrieval
# ---------------------------------------------------------------------------

class ZhipuEmbeddingClient:
    """智谱 BigModel embedding-3 API 客户端。"""

    API_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __init__(self, api_key: str, model: str = "embedding-3", dimensions: int = 512):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """批量获取 embedding 向量。"""
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t if t.strip() else " " for t in batch]
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


class EmbeddingIndex:
    """Embedding-based retrieval index with disk cache (using Zhipu embedding-3)."""

    def __init__(self, emb_client: ZhipuEmbeddingClient, cache_dir: str = ""):
        self.client = emb_client
        self.cache_dir = cache_dir
        self.embeddings: np.ndarray | None = None
        self.samples: list[CodeSample] = []

    def _cache_key(self, texts: list[str]) -> str:
        """Generate a stable cache key from text list and model name."""
        content = json.dumps(texts, ensure_ascii=False, sort_keys=True)
        h = hashlib.sha256(f"{self.client.model}:{content}".encode()).hexdigest()[:16]
        return h

    def _load_cache(self, cache_key: str) -> np.ndarray | None:
        """Load cached embeddings from disk."""
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
        """Save embeddings to disk cache."""
        if not self.cache_dir:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"emb_cache_{cache_key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(embeddings.tolist(), f)
        logger.info("Saved embedding cache to %s (%d vectors)", path, len(embeddings))

    def index(self, corpus: CodeCorpus) -> float:
        """Index corpus and return indexing time."""
        t0 = time.time()
        self.samples = corpus.samples

        texts = [s.code for s in corpus.samples]
        cache_key = self._cache_key(texts)

        # Try loading from cache
        cached = self._load_cache(cache_key)
        if cached is not None:
            self.embeddings = cached
            elapsed = time.time() - t0
            logger.info("Embedding index loaded from cache in %.2fs", elapsed)
            return elapsed

        # Compute embeddings via API
        logger.info("Generating embeddings for %d code snippets...", len(texts))
        self.embeddings = np.array(self.client.embed(texts), dtype=np.float32)

        # Save to cache
        self._save_cache(cache_key, self.embeddings)

        return time.time() - t0

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for query, return [(idx, score), ...]."""
        query_emb = np.array(self.client.embed_single(query), dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        scores = np.dot(self.embeddings, query_emb) / (norms + 1e-10)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


# ---------------------------------------------------------------------------
# TreeSearch FTS5-based retrieval
# ---------------------------------------------------------------------------

class TreeSearchCodeIndex:
    """TreeSearch FTS5-based code retrieval."""

    def __init__(self):
        self.fts_index: FTS5Index | None = None
        self.documents: list[Document] = []
        self.idx_to_node_id: dict[int, str] = {}  # Map corpus idx to node_id

    async def index(self, corpus: CodeCorpus, output_dir: str) -> float:
        """Index corpus and return indexing time."""
        t0 = time.time()

        # Create temporary code files for indexing
        os.makedirs(output_dir, exist_ok=True)
        code_dir = os.path.join(output_dir, "code_files")
        os.makedirs(code_dir, exist_ok=True)

        # Write each code sample as a file
        ext_map = {
            "python": ".py",
            "java": ".java",
            "javascript": ".js",
            "go": ".go",
            "ruby": ".rb",
            "php": ".php",
        }
        ext = ext_map.get(corpus.language, ".txt")

        code_paths = []
        for sample in corpus.samples:
            # Add docstring as comment for better searchability
            if corpus.language == "python":
                content = f'"""{sample.query}"""\n{sample.code}'
            elif corpus.language in ("java", "javascript", "go"):
                content = f"/* {sample.query} */\n{sample.code}"
            else:
                content = f"# {sample.query}\n{sample.code}"

            safe_name = f"code_{sample.idx}{ext}"
            path = os.path.join(code_dir, safe_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            code_paths.append(path)

        # Build TreeSearch index
        logger.info(f"Building TreeSearch index for {len(code_paths)} code files...")
        self.documents = await build_index(
            paths=code_paths,
            output_dir=output_dir,
            if_add_node_summary=False,
            if_add_doc_description=False,
            if_add_node_text=True,
            if_add_node_id=True,
            max_concurrency=10,
            force=True,
        )

        # Build FTS5 index
        self.fts_index = FTS5Index()
        self.fts_index.index_documents(self.documents)

        # Build mapping from corpus idx to node_id
        for doc in self.documents:
            # Extract idx from filename (code_0.py -> 0)
            try:
                basename = os.path.basename(doc.doc_name)
                idx = int(basename.split("_")[1].split(".")[0])
                # Get root node id
                nodes = flatten_tree(doc.structure)
                if nodes:
                    self.idx_to_node_id[idx] = nodes[0].get("node_id", "")
            except (IndexError, ValueError):
                pass

        return time.time() - t0

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for query, return [(idx, score), ...]."""
        results = self.fts_index.search(query, top_k=top_k * 2)  # Get more, then dedupe by doc

        # Map node_id back to corpus idx
        seen_idx = set()
        output = []
        for r in results:
            doc_id = r.get("doc_id", "")
            # Extract idx from doc_id (code_0.py -> 0)
            try:
                basename = os.path.basename(doc_id)
                idx = int(basename.split("_")[1].split(".")[0])
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    output.append((idx, r.get("fts_score", 0.0)))
                    if len(output) >= top_k:
                        break
            except (IndexError, ValueError):
                pass

        return output


class TreeSearchCodeTreeIndex:
    """Tree mode (Best-First Search) code retrieval -- uses TreeSearcher."""

    def __init__(self, base_index: TreeSearchCodeIndex):
        self.base: TreeSearchCodeIndex = base_index

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Tree search: anchor retrieval -> tree walk -> path aggregation."""
        from treesearch.tree_searcher import TreeSearcher
        from treesearch.config import set_config, get_config, TreeSearchConfig

        old_cfg = get_config()
        set_config(TreeSearchConfig(
            path_top_k=max(top_k, old_cfg.path_top_k),
            anchor_top_k=max(top_k, old_cfg.anchor_top_k),
            max_expansions=max(60, old_cfg.max_expansions),
        ))
        searcher = TreeSearcher(fts_index=self.base.fts_index)
        _, flat_nodes = searcher.search(query, self.base.documents)
        set_config(old_cfg)

        # Map flat_nodes back to corpus idx
        seen_idx = set()
        output = []
        for fn in flat_nodes:
            doc_id = fn.get("doc_id", "")
            try:
                basename = os.path.basename(doc_id)
                idx = int(basename.split("_")[1].split(".")[0])
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    output.append((idx, fn.get("score", 0.0)))
                    if len(output) >= top_k:
                        break
            except (IndexError, ValueError):
                pass

        return output


class TreeSearchCodeAutoIndex:
    """Auto mode: uses _resolve_auto_mode to pick flat vs tree,
    then dispatches to the appropriate retriever. Avoids the overhead
    of asyncio.run() per query and redundant FTS5 routing queries."""

    def __init__(self, base_index: TreeSearchCodeIndex):
        self.base: TreeSearchCodeIndex = base_index
        # Resolve mode once (all docs have the same source_type)
        from treesearch.search import _resolve_auto_mode
        self._mode = _resolve_auto_mode(base_index.documents)
        logger.info(f"Auto mode resolved to: {self._mode}")

        if self._mode == "tree":
            self._tree_index = TreeSearchCodeTreeIndex(base_index)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self._mode == "tree":
            return self._tree_index.search(query, top_k)
        else:
            return self.base.search(query, top_k)


class CodeTripletExtractor:
    """Lightweight code relation extractor for CodeSearchNet GraphRAG smoke runs."""

    def extract(self, passage: GraphNodePassage) -> list[GraphRelation]:
        subject = passage.title or passage.doc_name
        if not subject:
            return []
        identifiers = _code_identifiers(" ".join([passage.title, passage.text]))
        relations = []
        for identifier in identifiers:
            text = f"{subject} mentions {identifier}"
            relations.append(
                GraphRelation(
                    relation_id=_code_relation_id(passage.graph_node_id, text),
                    subject=subject,
                    predicate="mentions",
                    object=identifier,
                    text=text,
                    node_id=passage.node_id,
                    doc_id=passage.doc_id,
                    source_type=passage.source_type,
                )
            )
        return relations


class TreeSearchCodeGraphRAGIndex:
    """TreeSearch-node GraphRAG adapter with the same interface as code retrievers."""

    def __init__(self, base_index: TreeSearchCodeIndex):
        self.base = base_index
        self.tree_search = TreeSearch(db_path=None)
        self.tree_search.documents = list(base_index.documents)
        self.rag = TreeSearchGraphRAG.from_tree_search(
            self.tree_search,
            extractor=CodeTripletExtractor(),
            expansion_config=ExpansionConfig(max_relations=30, max_hops=1),
        )
        self.rag.build_graph()

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        answer = self.rag.query(query)
        output = self._rank_from_answer(answer, top_k)
        if output:
            return output
        return TreeSearchCodeAutoIndex(self.base).search(query, top_k=top_k)

    async def asearch(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        answer = await self.rag.aquery(query)
        output = self._rank_from_answer(answer, top_k)
        if output:
            return output
        return TreeSearchCodeAutoIndex(self.base).search(query, top_k=top_k)

    def _rank_from_answer(self, answer, top_k: int) -> list[tuple[int, float]]:
        output = []
        seen_idx = set()
        for citation in answer.evidence_chain.citations:
            idx = _idx_from_doc_reference(citation.source_path or citation.doc_id)
            if idx is None or idx in seen_idx:
                continue
            seen_idx.add(idx)
            output.append((idx, 1.0 / (len(output) + 1)))
            if len(output) >= top_k:
                break
        return output


def _code_identifiers(text: str, max_identifiers: int = 40) -> list[str]:
    identifiers = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        if len(token) < 2:
            continue
        if token in {"def", "return", "class", "self", "None", "True", "False"}:
            continue
        identifiers.append(token)
    return list(dict.fromkeys(identifiers))[:max_identifiers]


def _code_relation_id(graph_node_id: str, text: str) -> str:
    payload = f"{graph_node_id}:{text}"
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _idx_from_doc_reference(value: str) -> int | None:
    try:
        basename = os.path.basename(value)
        return int(basename.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    query_samples: list[CodeSample],
    index,
    k_values: list[int] = None,
) -> dict:
    """Evaluate retrieval performance.

    For each query, the ground truth is its own code (idx matches).
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_results = []
    cost_stats = []

    for sample in query_samples:
        tracker = CostTracker()
        with tracker:
            results = index.search(sample.query, top_k=max(k_values))

        retrieved_ids = [idx for idx, _ in results]
        relevant_ids = [sample.idx]  # Ground truth is the sample's own idx

        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values)

        all_results.append(metrics)
        cost_stats.append(tracker.stats)

        hit_str = "HIT" if sample.idx in retrieved_ids[:1] else "miss"
        logger.info(
            f"[{len(all_results)}/{len(query_samples)}] MRR={metrics['mrr']:.2f} "
            f"P@1={metrics['precision@1']:.2f} [{hit_str}] {tracker.stats.latency_seconds:.3f}s"
        )

    # Aggregate
    avg_metrics = {}
    if all_results:
        for key in all_results[0]:
            avg_metrics[key] = np.mean([r[key] for r in all_results])

    avg_cost = aggregate_cost_stats(cost_stats)
    avg_metrics["avg_query_time"] = avg_cost.latency_seconds
    avg_metrics["total_queries"] = len(query_samples)

    return avg_metrics


async def aevaluate_retrieval(
    query_samples: list[CodeSample],
    index,
    k_values: list[int] = None,
) -> dict:
    """Async retrieval evaluation for indexes that expose an async search API."""
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_results = []
    cost_stats = []

    for sample in query_samples:
        tracker = CostTracker()
        with tracker:
            results = await index.asearch(sample.query, top_k=max(k_values))

        retrieved_ids = [idx for idx, _ in results]
        relevant_ids = [sample.idx]
        metrics = evaluate_query(retrieved_ids, relevant_ids, k_values)

        all_results.append(metrics)
        cost_stats.append(tracker.stats)

        hit_str = "HIT" if sample.idx in retrieved_ids[:1] else "miss"
        logger.info(
            f"[{len(all_results)}/{len(query_samples)}] MRR={metrics['mrr']:.2f} "
            f"P@1={metrics['precision@1']:.2f} [{hit_str}] {tracker.stats.latency_seconds:.3f}s"
        )

    avg_metrics = {}
    if all_results:
        for key in all_results[0]:
            avg_metrics[key] = np.mean([r[key] for r in all_results])

    avg_cost = aggregate_cost_stats(cost_stats)
    avg_metrics["avg_query_time"] = avg_cost.latency_seconds
    avg_metrics["total_queries"] = len(query_samples)

    return avg_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="CodeSearchNet benchmark: evaluate TreeSearch on code retrieval"
    )
    parser.add_argument(
        "--language", type=str, default="python",
        choices=["python", "java", "javascript", "go", "ruby", "php"],
        help="Programming language (default: python)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split (default: test)"
    )
    parser.add_argument("--max-samples", type=int, default=100, help="Max query samples to evaluate")
    parser.add_argument("--max-corpus", type=int, default=1000, help="Max corpus size")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results per query")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output-dir", type=str, default=os.path.join(_script_dir, "benchmark_results", "codesearchnet"), help="Output directory")
    parser.add_argument("--index-dir", type=str, default=os.path.join(_script_dir, "indexes", "codesearchnet"), help="Index directory")
    parser.add_argument("--with-embedding", action="store_true", help="Also evaluate embedding-based retrieval")
    parser.add_argument("--with-graphrag", action="store_true", help="Also evaluate TreeSearch-node GraphRAG retrieval")
    args = parser.parse_args()

    # Load dataset
    query_samples, corpus = load_codesearchnet_from_hf(
        language=args.language,
        split=args.split,
        max_samples=args.max_samples,
        max_corpus=args.max_corpus,
    )

    print(f"\n{'=' * 70}")
    print(f"CodeSearchNet Benchmark: {args.language.upper()}")
    print(f"Queries: {len(query_samples)} | Corpus: {len(corpus.samples)} | Top-K: {args.top_k}")
    print(f"{'=' * 70}\n")

    results = {}

    # TreeSearch FTS5 evaluation
    print(f"\n{'=' * 60}")
    print("Strategy: TREESEARCH FTS5")
    print(f"{'=' * 60}")

    ts_index = TreeSearchCodeIndex()
    ts_index_time = await ts_index.index(corpus, args.index_dir)
    print(f"Index time: {ts_index_time:.2f}s")

    ts_metrics = evaluate_retrieval(query_samples, ts_index)
    ts_metrics["index_time"] = ts_index_time
    ts_metrics["num_nodes"] = sum(len(flatten_tree(d.structure)) for d in ts_index.documents)
    results["treesearch_fts5"] = ts_metrics

    # Tree mode evaluation (reuses the same index)
    print(f"\n{'=' * 60}")
    print("Strategy: TREESEARCH TREE (Best-First Search)")
    print(f"{'=' * 60}")

    tree_index = TreeSearchCodeTreeIndex(ts_index)
    tree_metrics = evaluate_retrieval(query_samples, tree_index)
    tree_metrics["index_time"] = ts_index_time  # same index
    tree_metrics["num_nodes"] = ts_metrics["num_nodes"]
    results["treesearch_tree"] = tree_metrics

    # Auto mode evaluation (uses treesearch.search with search_mode="auto")
    print(f"\n{'=' * 60}")
    print("Strategy: TREESEARCH AUTO (auto routing)")
    print(f"{'=' * 60}")

    auto_index = TreeSearchCodeAutoIndex(ts_index)
    auto_metrics = evaluate_retrieval(query_samples, auto_index)
    auto_metrics["index_time"] = ts_index_time
    auto_metrics["num_nodes"] = ts_metrics["num_nodes"]
    results["treesearch_auto"] = auto_metrics

    if args.with_graphrag:
        print(f"\n{'=' * 60}")
        print("Strategy: TREESEARCH-NODE GRAPHRAG")
        print(f"{'=' * 60}")

        graph_index = TreeSearchCodeGraphRAGIndex(ts_index)
        graph_metrics = await aevaluate_retrieval(query_samples, graph_index)
        graph_metrics["index_time"] = ts_index_time
        graph_metrics["num_nodes"] = ts_metrics["num_nodes"]
        results["treesearch_graphrag"] = graph_metrics

    # Embedding evaluation (optional)
    if args.with_embedding:
        api_key = os.environ.get("ZHIPU_API_KEY", "")

        if not api_key:
            print("ZHIPU_API_KEY not set, skipping embedding evaluation")
            print("Set it in .env or: export ZHIPU_API_KEY=your_api_key")
        else:
            print(f"\n{'=' * 60}")
            print("Strategy: EMBEDDING (Zhipu embedding-3)")
            print(f"{'=' * 60}")

            emb_client = ZhipuEmbeddingClient(api_key=api_key)
            emb_index = EmbeddingIndex(
                emb_client=emb_client,
                cache_dir=os.path.join(args.output_dir, "embedding_cache"),
            )
            emb_index_time = emb_index.index(corpus)
            print(f"Index time: {emb_index_time:.2f}s")

            emb_metrics = evaluate_retrieval(query_samples, emb_index)
            emb_metrics["index_time"] = emb_index_time
            emb_metrics["num_embeddings"] = len(corpus.samples)
            results["embedding"] = emb_metrics

    # Print comparison
    w = 80
    dataset_label = f"CodeSearchNet ({args.language.upper()})"
    col_w = 16
    num_cols = len(results)

    print(f"\n{'=' * w}")
    print(f"  📋 BENCHMARK: {dataset_label}")
    print(f"{'=' * w}")

    col_names = list(results.keys())

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
    print(f"\n  ▸ Cost & Efficiency")
    print(_hdr())
    print(sep)
    r = f"  {'Index time (s)':<22}"
    for name in col_names:
        r += f"{results[name].get('index_time', 0):>{col_w}.2f}"
    print(r)
    r = f"  {'Avg query time (s)':<22}"
    for name in col_names:
        r += f"{results[name].get('avg_query_time', 0):>{col_w}.4f}"
    print(r)

    print(f"\n{'=' * w}")

    # Summary
    print(f"\n  📊 SUMMARY ({dataset_label})")
    print(f"  {'─' * 50}")

    for name, m in results.items():
        mrr_val = m.get("mrr", 0)
        r5_val = m.get("recall@5", 0)
        qt_val = m.get("avg_query_time", 0)
        print(f"  {name.upper():<24} MRR={mrr_val:.4f}  R@5={r5_val:.4f}  ⏱ {qt_val:.4f}s/q")

    if "embedding" in results and "treesearch_fts5" in results:
        emb_mrr = results["embedding"]["mrr"]
        ts_mrr = results["treesearch_fts5"]["mrr"]
        emb_r5 = results["embedding"].get("recall@5", 0)
        ts_r5 = results["treesearch_fts5"].get("recall@5", 0)
        ts_query_time = results["treesearch_fts5"]["avg_query_time"]
        emb_query_time = results["embedding"]["avg_query_time"]

        print()
        if ts_mrr > emb_mrr:
            print(f"  ✅ TreeSearch FTS5 outperforms Embedding by {(ts_mrr - emb_mrr) / emb_mrr * 100:.1f}% on MRR")
        else:
            print(f"  📊 Embedding outperforms TreeSearch FTS5 by {(emb_mrr - ts_mrr) / ts_mrr * 100:.1f}% on MRR")

        if ts_r5 > emb_r5:
            print(f"  ✅ TreeSearch FTS5 outperforms Embedding by {(ts_r5 - emb_r5) / emb_r5 * 100:.1f}% on Recall@5")
        else:
            print(f"  📊 Embedding outperforms TreeSearch FTS5 by {(emb_r5 - ts_r5) / ts_r5 * 100:.1f}% on Recall@5")

        speedup = emb_query_time / ts_query_time if ts_query_time > 0 else 0
        print(f"  ⚡ TreeSearch FTS5 is {speedup:.1f}x faster per query")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"{args.language}_benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
