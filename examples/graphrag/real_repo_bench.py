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
from pathlib import Path
from typing import Optional

from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.rag import (
    ExpansionConfig,
    EvidenceChain,
    EvidenceCitation,
    GroundedAnswer,
    PreExtractedTripletExtractor,
    RealRepoSample,
    RuleBasedTripletExtractor,
    VerificationResult,
    aggregate_eval_results,
    evaluate_grounded_answer,
)
from treesearch.rag.models import make_graph_node_id
from treesearch.rag.sqlite_graph_store import SQLiteGraphStore
from treesearch.tree import Document


REPO_ROOT = Path(__file__).resolve().parents[2]
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

    if baseline in {"both", "graphrag"}:
        graph_rows, graph_metrics, build_stats = _run_graphrag(
            ts=ts,
            samples=samples,
            triplets_path=triplets_path,
            graph_store=graph_store,
            graph_store_path=graph_store_path,
            use_llm=use_llm,
            llm_model=llm_model,
        )
        report["build_stats"]["graphrag"] = build_stats
        report["summary"]["graphrag"] = aggregate_eval_results(graph_metrics)
        method_results["graphrag"] = graph_rows

    if baseline in {"both", "treesearch"}:
        tree_rows, tree_metrics = _run_treesearch_baseline(ts, samples)
        report["summary"]["treesearch"] = aggregate_eval_results(tree_metrics)
        method_results["treesearch"] = tree_rows

    report["results"] = method_results
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_format_markdown_summary(report), encoding="utf-8")
    return report


def _run_graphrag(
    ts: TreeSearch,
    samples: list[RealRepoSample],
    triplets_path: Optional[Path],
    graph_store: str,
    graph_store_path: Optional[Path],
    use_llm: bool,
    llm_model: Optional[str],
) -> tuple[list[dict], list, dict]:
    extractor = (
        PreExtractedTripletExtractor(_load_triplets(triplets_path))
        if triplets_path is not None
        else RuleBasedTripletExtractor()
    )
    store = SQLiteGraphStore(graph_store_path or Path("tmp/graphrag_store.db")) if graph_store == "sqlite" else None
    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=extractor,
        expansion_config=ExpansionConfig(max_relations=50),
        store=store,
        use_llm=use_llm,
        llm_model=llm_model,
    )
    build_stats = rag.build_graph()
    if isinstance(store, SQLiteGraphStore):
        store.save()

    rows = []
    metrics = []
    for sample in samples:
        answer = rag.query(sample.query)
        result = evaluate_grounded_answer(sample, answer)
        metrics.append(result)
        rows.append(_answer_row("graphrag", sample, answer, result))
    return rows, metrics, {
        "documents": build_stats.documents,
        "passages": build_stats.passages,
        "relations": build_stats.relations,
        "structural_edges": build_stats.structural_edges,
    }


def _run_treesearch_baseline(ts: TreeSearch, samples: list[RealRepoSample]) -> tuple[list[dict], list]:
    rows = []
    metrics = []
    doc_map = {document.doc_id: document for document in ts.documents}
    for sample in samples:
        result = ts.search(
            sample.query,
            top_k_docs=5,
            max_nodes_per_doc=5,
            include_ancestors=True,
            merge_strategy="global_score",
        )
        answer = _treesearch_result_to_answer(sample.query, result, doc_map)
        eval_result = evaluate_grounded_answer(sample, answer)
        metrics.append(eval_result)
        rows.append(_answer_row("treesearch", sample, answer, eval_result))
    return rows, metrics


def _treesearch_result_to_answer(query: str, result: dict, doc_map: dict[str, Document]) -> GroundedAnswer:
    citations = []
    selected_node_ids = []
    snippets = []
    for node in result.get("flat_nodes", [])[:5]:
        node_id = str(node.get("node_id", ""))
        doc_id = str(node.get("doc_id", ""))
        if not node_id or not doc_id:
            continue
        source_node = doc_map[doc_id].get_node_by_id(node_id) if doc_id in doc_map else None
        selected_node_ids.append(make_graph_node_id(doc_id, node_id))
        citations.append(
            EvidenceCitation(
                node_id=node_id,
                doc_id=doc_id,
                source_path=str(node.get("source_path", "")),
                line_start=node.get("line_start") or (source_node.get("line_start") if source_node else None),
                line_end=node.get("line_end") or (source_node.get("line_end") if source_node else None),
            )
        )
        snippets.append(str(node.get("text", "") or node.get("snippet", "")))

    chain = EvidenceChain(
        query=query,
        bridge_entities=(),
        selected_relation_ids=(),
        selected_node_ids=tuple(dict.fromkeys(selected_node_ids)),
        reasoning_chain=tuple(snippet for snippet in snippets if snippet),
        citations=tuple(citations),
        evidence_sufficiency=bool(citations),
    )
    return GroundedAnswer(
        query=query,
        answer="\n".join(chain.reasoning_chain),
        evidence_chain=chain,
        verification=VerificationResult(ok=bool(citations)),
    )


def _answer_row(method: str, sample: RealRepoSample, answer: GroundedAnswer, metrics) -> dict:
    return {
        "method": method,
        "query_id": sample.query_id,
        "query": sample.query,
        "answer": answer.answer,
        "verification_ok": answer.verification.ok,
        "verification_errors": list(answer.verification.errors),
        "node_recall": metrics.node_recall,
        "source_path_recall": metrics.source_path_recall,
        "citation_precision": metrics.citation_precision,
        "citation_recall": metrics.citation_recall,
        "line_grounding_accuracy": metrics.line_grounding_accuracy,
        "task_success": metrics.task_success,
        "citations": [
            {
                "node_id": citation.node_id,
                "doc_id": citation.doc_id,
                "source_path": citation.source_path,
                "line_start": citation.line_start,
                "line_end": citation.line_end,
            }
            for citation in answer.evidence_chain.citations
        ],
    }


def _format_markdown_summary(report: dict) -> str:
    lines = [
        "# GraphRAG RealRepoBench Summary",
        "",
        "| method | count | node_recall | source_path_recall | citation_precision | citation_recall | line_grounding_accuracy | task_success_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, summary in report["summary"].items():
        lines.append(
            "| {method} | {count} | {node_recall:.3f} | {source_path_recall:.3f} | "
            "{citation_precision:.3f} | {citation_recall:.3f} | {line_grounding_accuracy:.3f} | "
            "{task_success_rate:.3f} |".format(method=method, **summary)
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", default=[str(DEFAULT_FIXTURE_REPO)])
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--triplets", default=str(DEFAULT_TRIPLETS))
    parser.add_argument("--output", default="output/graphrag_real_repo_results.json")
    parser.add_argument("--markdown-output", default="output/graphrag_real_repo_results.md")
    parser.add_argument("--baseline", choices=["both", "graphrag", "treesearch"], default="both")
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
        baseline=args.baseline,
        graph_store=args.graph_store,
        graph_store_path=Path(args.graph_store_path) if args.graph_store_path else None,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
    )
    print(f"Wrote {args.output}")
    if args.markdown_output:
        print(f"Wrote {args.markdown_output}")
    print(_format_markdown_summary(report))


if __name__ == "__main__":
    main()
