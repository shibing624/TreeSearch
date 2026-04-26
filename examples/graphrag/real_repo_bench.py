# -*- coding: utf-8 -*-
"""Run a RealRepoBench-style evaluation for TreeSearch GraphRAG.

Example:
    python examples/graphrag/real_repo_bench.py \
        --paths examples/graphrag/fixtures/repo \
        --queries examples/graphrag/fixtures/queries.json \
        --triplets examples/graphrag/fixtures/triplets.json \
        --baseline both \
        --graph-store sqlite \
        --output tmp/graphrag_real_repo_results.json \
        --markdown-output tmp/graphrag_real_repo_results.md
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.rag import (
    ExpansionConfig,
    PreExtractedTripletExtractor,
    RealRepoSample,
    RuleBasedTripletExtractor,
    evaluate_grounded_answer,
)
from examples.graphrag.baselines import answer_row, baseline_methods, run_retrieval_baseline
from examples.graphrag.metrics import aggregate_method_summary
from examples.graphrag.report import format_markdown_summary, write_reports
from treesearch.rag.llm import LLMClient, OpenAIChatClient
from treesearch.rag.sqlite_graph_store import SQLiteGraphStore


DEFAULT_FIXTURE_REPO = REPO_ROOT / "examples" / "graphrag" / "fixtures" / "repo"
DEFAULT_QUERIES = REPO_ROOT / "examples" / "graphrag" / "fixtures" / "queries.json"
DEFAULT_TRIPLETS = REPO_ROOT / "examples" / "graphrag" / "fixtures" / "triplets.json"


def _load_samples(path: Path) -> list[RealRepoSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = []
    for item in data:
        samples.append(
            RealRepoSample(
                query_id=str(item["query_id"]),
                query=str(item["query"]),
                gold_node_ids=tuple(item.get("gold_node_ids", [])),
                gold_source_paths=tuple(item.get("gold_source_paths", [])),
                gold_answer=str(item.get("gold_answer", "")),
                query_type=str(item.get("query_type", "")),
                requires_cross_source=bool(item.get("requires_cross_source", False)),
                needs_line_grounding=bool(item.get("needs_line_grounding", False)),
            )
        )
    return samples


def _load_triplets(path: Path) -> dict[str, list]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("triplets file must be a JSON object keyed by doc_id::node_id")
    return {str(node_id): triplets for node_id, triplets in data.items()}


def run(
    paths: list[str],
    queries_path: Path,
    triplets_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    markdown_path: Optional[Path] = None,
    latex_path: Optional[Path] = None,
    baseline: str = "both",
    graph_store: str = "memory",
    graph_store_path: Optional[Path] = None,
    use_llm: bool = False,
    llm_model: Optional[str] = None,
) -> dict:
    samples = _load_samples(queries_path)
    ts = TreeSearch(db_path=None)
    ts.index(*paths)

    report = {
        "build_stats": {},
        "summary": {},
        "results": [],
    }
    method_results = {}

    methods = _selected_methods(baseline)
    if "graphrag" in methods:
        graph_rows, graph_metrics, build_stats, graph_runtime = _run_graphrag(
            ts=ts,
            samples=samples,
            triplets_path=triplets_path,
            graph_store=graph_store,
            graph_store_path=graph_store_path,
            use_llm=use_llm,
            llm_model=llm_model,
        )
        report["build_stats"]["graphrag"] = build_stats
        report["summary"]["graphrag"] = aggregate_method_summary(
            graph_metrics,
            latency_seconds=graph_runtime["latency_seconds"],
            llm_calls=graph_runtime["llm_calls"],
        )
        method_results["graphrag"] = graph_rows

    for method in methods:
        if method == "graphrag":
            continue
        rows, metrics, runtime = run_retrieval_baseline(method, ts, samples)
        report["summary"][method] = aggregate_method_summary(
            metrics,
            latency_seconds=runtime["latency_seconds"],
            llm_calls=runtime["llm_calls"],
        )
        method_results[method] = rows

    report["results"] = method_results
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(report, markdown_path=markdown_path, latex_path=latex_path)
    return report


def _run_graphrag(
    ts: TreeSearch,
    samples: list[RealRepoSample],
    triplets_path: Optional[Path],
    graph_store: str,
    graph_store_path: Optional[Path],
    use_llm: bool,
    llm_model: Optional[str],
) -> tuple[list[dict], list, dict, dict]:
    extractor = (
        PreExtractedTripletExtractor(_load_triplets(triplets_path))
        if triplets_path is not None
        else RuleBasedTripletExtractor()
    )
    llm_client = OpenAIChatClient(model=llm_model) if use_llm else None
    store = SQLiteGraphStore(graph_store_path or Path("tmp/graphrag_store.db")) if graph_store == "sqlite" else None
    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=extractor,
        expansion_config=ExpansionConfig(max_relations=50),
        store=store,
        use_llm=use_llm,
        llm_client=llm_client,
        llm_model=llm_model,
    )
    started = time.perf_counter()
    build_stats = rag.build_graph()
    if isinstance(store, SQLiteGraphStore):
        store.save()

    rows = []
    metrics = []
    for sample in samples:
        answer = rag.query(sample.query)
        result = evaluate_grounded_answer(sample, answer)
        metrics.append(result)
        rows.append(answer_row("graphrag", sample, answer, result))
    runtime = _runtime_summary(started, len(samples), llm_client)
    return rows, metrics, {
        "documents": build_stats.documents,
        "passages": build_stats.passages,
        "relations": build_stats.relations,
        "structural_edges": build_stats.structural_edges,
    }, runtime


def _runtime_summary(started: float, sample_count: int, llm_client: LLMClient | None) -> dict:
    latency = time.perf_counter() - started
    return {
        "latency_seconds": latency,
        "avg_latency_seconds": latency / sample_count if sample_count else 0.0,
        "llm_calls": llm_client.call_count if llm_client is not None else 0,
    }


def _selected_methods(baseline: str) -> tuple[str, ...]:
    if baseline == "both":
        return ("graphrag", "treesearch")
    if baseline == "all":
        return baseline_methods()
    if baseline in baseline_methods():
        return (baseline,)
    raise ValueError(f"unsupported baseline: {baseline}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=[str(DEFAULT_FIXTURE_REPO)])
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    parser.add_argument("--output", default="output/graphrag_real_repo_results.json")
    parser.add_argument("--markdown-output", default="output/graphrag_real_repo_results.md")
    parser.add_argument("--latex-output")
    parser.add_argument("--baseline", choices=["both", "all", *baseline_methods()], default="both")
    parser.add_argument("--graph-store", choices=["memory", "sqlite"], default="memory")
    parser.add_argument("--graph-store-path")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-model")
    args = parser.parse_args()

    report = run(
        paths=args.paths,
        queries_path=Path(args.queries),
        triplets_path=Path(args.triplets) if args.triplets else None,
        output_path=Path(args.output),
        markdown_path=Path(args.markdown_output) if args.markdown_output else None,
        latex_path=Path(args.latex_output) if args.latex_output else None,
        baseline=args.baseline,
        graph_store=args.graph_store,
        graph_store_path=Path(args.graph_store_path) if args.graph_store_path else None,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
    )
    print(f"Wrote {args.output}")
    if args.markdown_output:
        print(f"Wrote {args.markdown_output}")
    if args.latex_output:
        print(f"Wrote {args.latex_output}")
    print(format_markdown_summary(report))


if __name__ == "__main__":
    main()
