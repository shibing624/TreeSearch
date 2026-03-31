# -*- coding: utf-8 -*-
"""
FinanceBench End-to-End RAG Evaluation.

Evaluates the full RAG pipeline (retrieval → LLM generation) on FinanceBench,
comparing different retrieval strategies using a fixed LLM backbone.

Retrieval strategies compared:
  naive_rag           BM25 flat top-k (standard baseline)
  treesearch_flat     TreeSearch FTS5 flat mode
  treesearch_tree     TreeSearch tree-walk mode

LLM config loaded from environment variables (.env):
  TREESEARCH_LLM_API_KEY   API key for OpenAI-compatible LLM
  TREESEARCH_LLM_BASE_URL  Base URL (e.g. https://ark.cn-beijing.volces.com/api/v3)
  TREESEARCH_MODEL         Model name (e.g. ep-20260226110728-p6ntt)

Metrics:
  Exact Match (EM)   Whether predicted answer matches ground truth exactly
                     (after normalization: lowercase, strip punctuation)
  Token F1           Token-level overlap between prediction and ground truth
  Latency            Retrieval-only latency (ms) — LLM latency excluded

Usage:
  # Quick test (10 samples)
  python examples/rag/financebench_rag_eval.py --max-samples 10

  # Full evaluation (150 samples, all strategies)
  python examples/rag/financebench_rag_eval.py \\
      --max-samples 150 \\
      --strategies naive_rag treesearch_flat treesearch_tree

  # Override model via CLI
  python examples/rag/financebench_rag_eval.py \\
      --max-samples 50 --model gpt-4o-mini --api-key sk-xxx --base-url https://api.openai.com/v1
"""
import argparse
import asyncio
import json
import logging
import os
import pickle
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
benchmark_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark")
sys.path.insert(0, benchmark_dir)

from dotenv import load_dotenv
load_dotenv()

from treesearch.fts import FTS5Index
from treesearch.tree import Document, flatten_tree
from treesearch.tree_searcher import TreeSearcher
from treesearch.config import set_config, TreeSearchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Maximum context tokens sent to LLM (fair comparison across all strategies)
_MAX_CONTEXT_TOKENS = 2048

# RAG prompt template
_RAG_PROMPT = """\
You are a financial analyst assistant. Answer the question based ONLY on the provided context.
Be concise. If the answer requires a calculation, show the formula and result.
If the context does not contain enough information, try your best to answer based on available information.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RagSample:
    """One FinanceBench QA pair with retrieval context."""
    financebench_id: str
    question: str
    ground_truth: str
    justification: str
    question_reasoning: str
    doc_name: str
    # Filled during retrieval
    context: str = ""
    retrieval_latency: float = 0.0
    num_nodes_retrieved: int = 0


@dataclass
class RagResult:
    """End-to-end evaluation result for one sample."""
    sample: RagSample
    strategy: str
    prediction: str = ""
    exact_match: float = 0.0
    token_f1: float = 0.0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Text normalization and metrics
# ---------------------------------------------------------------------------

def _normalize_answer(s: str) -> str:
    """Normalize answer for exact match: lowercase, remove articles/punctuation."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    s = re.sub(r"\$\s*", "", s)
    s = re.sub(r",", "", s)
    s = re.sub(r"billion|million|thousand", lambda m: {
        "billion": "000000000", "million": "000000", "thousand": "000"
    }[m.group()], s)
    return s.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

def _truncate_context(text: str, max_tokens: int = _MAX_CONTEXT_TOKENS) -> str:
    """Rough token truncation: 1 token ≈ 4 chars."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------

def retrieve_naive_rag(
    question: str,
    documents: list[Document],
    fts_index: FTS5Index,
    top_k: int = 5,
) -> tuple[str, float, int]:
    """BM25 top-k flat retrieval — standard Naive RAG baseline."""
    t0 = time.time()
    all_scored: list[tuple[str, str, float]] = []
    for doc in documents:
        scores = fts_index.score_nodes(question, doc.doc_id)
        for nid, score in scores.items():
            all_scored.append((doc.doc_id, nid, score))
    all_scored.sort(key=lambda x: -x[2])
    latency = time.time() - t0

    doc_map = {doc.doc_id: doc for doc in documents}
    texts = []
    for doc_id, nid, _ in all_scored[:top_k]:
        doc = doc_map.get(doc_id)
        if doc is None:
            continue
        all_nodes = flatten_tree(doc.structure)
        node_map = {n.get("node_id"): n for n in all_nodes}
        node = node_map.get(nid, {})
        title = node.get("title", "")
        text = node.get("text", "") or ""
        if title:
            texts.append(f"[{title}]\n{text}")
        else:
            texts.append(text)

    context = _truncate_context("\n\n".join(texts))
    return context, latency, len(all_scored[:top_k])


def retrieve_treesearch(
    question: str,
    documents: list[Document],
    fts_index: FTS5Index,
    searcher: TreeSearcher,
    top_k: int = 5,
) -> tuple[str, float, int]:
    """TreeSearch tree mode retrieval."""
    t0 = time.time()
    fts_score_map: dict[str, dict[str, float]] = {}
    for doc in documents:
        scores = fts_index.score_nodes(question, doc.doc_id)
        if scores:
            fts_score_map[doc.doc_id] = scores

    _, flat_nodes = searcher.search(question, documents, fts_score_map)
    latency = time.time() - t0

    texts = []
    for fn in flat_nodes[:top_k]:
        title = fn.get("title", "")
        text = fn.get("text", "") or ""
        if title:
            texts.append(f"[{title}]\n{text}")
        else:
            texts.append(text)

    context = _truncate_context("\n\n".join(texts))
    return context, latency, len(flat_nodes[:top_k])


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def build_prompt(context: str, question: str) -> str:
    return _RAG_PROMPT.format(context=context, question=question)


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def load_financebench_samples(max_samples: int = 150) -> list[RagSample]:
    """Load FinanceBench from HuggingFace with local pkl cache."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"financebench_rag_{max_samples}.pkl")

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            samples = pickle.load(f)
        logger.info("Loaded %d FinanceBench samples from cache", len(samples))
        return samples

    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("datasets required: pip install datasets")

    logger.info("Loading FinanceBench from HuggingFace...")
    ds = hf_load("PatronusAI/financebench", split="train")

    samples = []
    for idx, row in enumerate(ds):
        if idx >= max_samples:
            break
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue
        samples.append(RagSample(
            financebench_id=row.get("financebench_id", f"fb_{idx}"),
            question=question,
            ground_truth=answer,
            justification=row.get("justification", ""),
            question_reasoning=row.get("question_reasoning", ""),
            doc_name=row.get("doc_name", ""),
        ))

    with open(cache_path, "wb") as f:
        pickle.dump(samples, f)
    logger.info("Cached %d FinanceBench samples", len(samples))
    return samples


def load_documents_for_samples(
    samples: list[RagSample],
) -> tuple[list[Document], FTS5Index]:
    """Load documents from existing FinanceBench TreeSearch index.

    Looks for the index built by financebench_benchmark.py in these locations
    (in priority order):
      1. examples/benchmark/indexes/financebench/index.db  (default benchmark output)
      2. examples/benchmark/indexes/financebench.db        (legacy flat path)

    If not found, raises FileNotFoundError with instructions.
    """
    from treesearch.tree import load_documents

    fb_benchmark_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benchmark",
    )

    candidates = [
        os.path.join(fb_benchmark_dir, "indexes", "financebench", "index.db"),
        os.path.join(fb_benchmark_dir, "indexes", "financebench.db"),
    ]

    for index_path in candidates:
        if os.path.isfile(index_path):
            logger.info("Loading FinanceBench index from: %s", index_path)
            documents = load_documents(index_path)
            fts = FTS5Index(db_path=index_path)
            logger.info("Loaded %d documents, %d total nodes",
                        len(documents),
                        sum(len(d.structure) for d in documents))
            return documents, fts

    raise FileNotFoundError(
        f"FinanceBench index not found. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
        + "\n\nPlease build the index first:\n"
        "  python examples/benchmark/financebench_benchmark.py --max-samples 150\n"
        "Then re-run this script."
    )


def run_retrieval_for_strategy(
    strategy: str,
    samples: list[RagSample],
    documents: list[Document],
    fts_index: FTS5Index,
    searcher: Optional[TreeSearcher],
    top_k: int = 5,
) -> list[RagSample]:
    """Run retrieval for all samples under one strategy, return samples with context filled."""
    import copy
    filled_samples = []

    for sample in samples:
        target_docs = [d for d in documents if sample.doc_name in d.doc_name or d.doc_name in sample.doc_name]
        if not target_docs:
            target_docs = documents

        if strategy == "naive_rag" or strategy == "treesearch_flat":
            context, latency, n = retrieve_naive_rag(sample.question, target_docs, fts_index, top_k)
        elif strategy == "treesearch_tree" and searcher is not None:
            context, latency, n = retrieve_treesearch(sample.question, target_docs, fts_index, searcher, top_k)
        else:
            context, latency, n = retrieve_naive_rag(sample.question, target_docs, fts_index, top_k)

        s = copy.copy(sample)
        s.context = context
        s.retrieval_latency = latency
        s.num_nodes_retrieved = n
        filled_samples.append(s)

    return filled_samples


def evaluate_sync(
    samples: list[RagSample],
    strategy: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    max_concurrency: int = 10,
) -> list[RagResult]:
    """Evaluate samples with concurrent LLM calls."""
    async def _run():
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        sem = asyncio.Semaphore(max_concurrency)

        async def _call_one(idx: int, sample: RagSample) -> RagResult:
            prompt = build_prompt(sample.context, sample.question)
            async with sem:
                try:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=256,
                    )
                    answer = (response.choices[0].message.content or "").strip()
                    usage = response.usage
                    p_tok = usage.prompt_tokens if usage else 0
                    c_tok = usage.completion_tokens if usage else 0
                    error = ""
                except Exception as e:
                    logger.warning("LLM call failed for %s: %s", sample.financebench_id, e)
                    answer, p_tok, c_tok = "", 0, 0
                    error = str(e)

            em = exact_match(answer, sample.ground_truth)
            f1 = token_f1(answer, sample.ground_truth)
            return RagResult(
                sample=sample,
                strategy=strategy,
                prediction=answer,
                exact_match=em,
                token_f1=f1,
                llm_prompt_tokens=p_tok,
                llm_completion_tokens=c_tok,
                error=error,
            )

        tasks = [_call_one(i, s) for i, s in enumerate(samples)]
        results = await asyncio.gather(*tasks)
        return list(results)

    results = asyncio.run(_run())

    valid = [r for r in results if not r.error]
    if valid:
        avg_em = sum(r.exact_match for r in valid) / len(valid)
        avg_f1 = sum(r.token_f1 for r in valid) / len(valid)
        print(f"  [{len(results)}/{len(samples)}] avg_EM={avg_em:.3f}  avg_F1={avg_f1:.3f}  "
              f"(concurrency={max_concurrency})")

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def aggregate_results(results: list[RagResult]) -> dict:
    """Aggregate evaluation results into summary metrics."""
    if not results:
        return {}
    valid = [r for r in results if not r.error]
    n = len(valid)
    if n == 0:
        return {"error": "all samples failed"}

    avg_em = sum(r.exact_match for r in valid) / n
    avg_f1 = sum(r.token_f1 for r in valid) / n
    avg_ret_lat = sum(r.sample.retrieval_latency for r in valid) / n
    avg_p_tok = sum(r.llm_prompt_tokens for r in valid) / n
    avg_c_tok = sum(r.llm_completion_tokens for r in valid) / n

    by_type: dict[str, dict] = {}
    for r in valid:
        rtype = r.sample.question_reasoning or "unknown"
        by_type.setdefault(rtype, {"count": 0, "em": 0.0, "f1": 0.0})
        by_type[rtype]["count"] += 1
        by_type[rtype]["em"] += r.exact_match
        by_type[rtype]["f1"] += r.token_f1
    for rtype in by_type:
        cnt = by_type[rtype]["count"]
        by_type[rtype]["em"] /= cnt
        by_type[rtype]["f1"] /= cnt

    return {
        "num_samples": n,
        "exact_match": avg_em,
        "token_f1": avg_f1,
        "avg_retrieval_latency_s": avg_ret_lat,
        "avg_prompt_tokens": avg_p_tok,
        "avg_completion_tokens": avg_c_tok,
        "by_reasoning_type": by_type,
    }


def print_summary(all_aggregated: dict[str, dict]) -> None:
    """Print comparison table across strategies."""
    print(f"\n{'='*70}")
    print("FINANCEBENCH END-TO-END RAG EVALUATION")
    print(f"{'='*70}")
    header = f"{'Strategy':<25}  {'EM':>8}  {'F1':>8}  {'Ret.Lat':>10}  {'PromptTok':>10}"
    print(header)
    print("-" * 70)
    for strategy, agg in all_aggregated.items():
        if not agg or "error" in agg:
            print(f"{strategy:<25}  [failed]")
            continue
        row = (
            f"{strategy:<25}  "
            f"{agg['exact_match']:>8.4f}  "
            f"{agg['token_f1']:>8.4f}  "
            f"{agg['avg_retrieval_latency_s']:>9.3f}s  "
            f"{agg['avg_prompt_tokens']:>10.0f}"
        )
        print(row)
    print(f"{'='*70}")

    print("\nPer reasoning type breakdown:")
    for strategy, agg in all_aggregated.items():
        if not agg or "error" in agg:
            continue
        print(f"\n  {strategy}:")
        for rtype, m in sorted(agg.get("by_reasoning_type", {}).items()):
            print(f"    {rtype[:50]:<50}  n={m['count']:3d}  EM={m['em']:.3f}  F1={m['f1']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FinanceBench end-to-end RAG evaluation"
    )
    parser.add_argument("--max-samples", type=int, default=150,
                        help="Number of FinanceBench samples (max 150)")
    parser.add_argument("--strategies", nargs="+",
                        default=["naive_rag", "treesearch_flat", "treesearch_tree"],
                        choices=["naive_rag", "treesearch_flat", "treesearch_tree"],
                        help="Retrieval strategies to compare")
    parser.add_argument("--model", default=None,
                        help="LLM model name (default: from TREESEARCH_MODEL env var)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (default: from TREESEARCH_LLM_BASE_URL env var)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: from TREESEARCH_LLM_API_KEY env var)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of nodes to retrieve per query")
    parser.add_argument("--output-dir", default=_RESULTS_DIR,
                        help="Directory to save results")
    args = parser.parse_args()

    # --- Resolve LLM config from env vars ---
    api_key = args.api_key or os.environ.get("TREESEARCH_LLM_API_KEY", "")
    base_url = args.base_url or os.environ.get("TREESEARCH_LLM_BASE_URL", "")
    model = args.model or os.environ.get("TREESEARCH_MODEL", "")

    if not api_key:
        raise ValueError(
            "LLM API key required. Set TREESEARCH_LLM_API_KEY in .env or use --api-key."
        )
    if not model:
        raise ValueError(
            "LLM model name required. Set TREESEARCH_MODEL in .env or use --model."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"LLM: {model}  base_url={base_url or 'openai default'}")

    # Load dataset
    print(f"Loading FinanceBench ({args.max_samples} samples)...")
    samples = load_financebench_samples(args.max_samples)
    print(f"Loaded {len(samples)} samples.")

    # Load documents and FTS index
    print("Loading document index...")
    try:
        documents, fts_index = load_documents_for_samples(samples)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print(f"Loaded {len(documents)} documents.")

    # Build TreeSearcher for tree mode
    searcher = None
    if "treesearch_tree" in args.strategies:
        set_config(TreeSearchConfig(path_top_k=10, anchor_top_k=10, max_expansions=60))
        searcher = TreeSearcher()

    all_results: dict[str, list[RagResult]] = {}
    all_aggregated: dict[str, dict] = {}

    for strategy in args.strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.upper()} | Model: {model}")
        print(f"{'='*60}")

        # Step 1: Retrieve context
        print(f"Running retrieval ({strategy})...")
        filled_samples = run_retrieval_for_strategy(
            strategy, samples, documents, fts_index, searcher, args.top_k
        )
        avg_ret_lat = sum(s.retrieval_latency for s in filled_samples) / len(filled_samples)
        print(f"Retrieval done. Avg latency: {avg_ret_lat*1000:.1f}ms/query")

        # Step 2: LLM generation
        print(f"Running LLM generation ({model})...")
        results = evaluate_sync(filled_samples, strategy, model, api_key, base_url or None)

        all_results[strategy] = results

        # Aggregate
        agg = aggregate_results(results)
        all_aggregated[strategy] = agg

        # Save per-strategy results
        out_path = os.path.join(args.output_dir, f"financebench_rag_{strategy}_{model}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": strategy,
                "model": model,
                "num_samples": len(results),
                "aggregated": agg,
                "samples": [
                    {
                        "id": r.sample.financebench_id,
                        "question": r.sample.question,
                        "ground_truth": r.sample.ground_truth,
                        "prediction": r.prediction,
                        "exact_match": r.exact_match,
                        "token_f1": r.token_f1,
                        "question_reasoning": r.sample.question_reasoning,
                    }
                    for r in results
                ],
            }, f, indent=2, ensure_ascii=False)
        print(f"Results saved: {out_path}")

    fts_index.close()

    # Print final comparison
    print_summary(all_aggregated)

    # Save combined report
    combined_path = os.path.join(args.output_dir, f"financebench_rag_combined_{model}.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_aggregated, f, indent=2, ensure_ascii=False)
    print(f"\nCombined report: {combined_path}")


if __name__ == "__main__":
    main()
