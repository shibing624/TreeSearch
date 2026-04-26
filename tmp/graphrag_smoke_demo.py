# -*- coding: utf-8 -*-
"""End-to-end smoke demo for TreeSearch-guided GraphRAG.

Run from repository root:
    python tmp/graphrag_smoke_demo.py
"""

from pathlib import Path

from treesearch import TreeSearch, TreeSearchGraphRAG
from treesearch.rag import ExpansionConfig, RuleBasedTripletExtractor


ROOT = Path(__file__).resolve().parents[1]
DEMO_REPO = ROOT / "tmp" / "graphrag_demo_repo"


def prepare_demo_repo() -> tuple[Path, Path]:
    src_dir = DEMO_REPO / "treesearch"
    docs_dir = DEMO_REPO / "docs"
    src_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    config_file = src_dir / "config.py"
    config_file.write_text(
        '''class TreeSearchConfig:
    """Runtime configuration.

    max_concurrency is defined_in TreeSearchConfig.
    """

    max_concurrency = 5
''',
        encoding="utf-8",
    )

    runtime_doc = docs_dir / "runtime.md"
    runtime_doc.write_text(
        """# Runtime Settings

TreeSearchConfig is documented_in Runtime Settings.

max_concurrency controls worker parallelism for indexing and search.
""",
        encoding="utf-8",
    )

    return config_file, runtime_doc


def main() -> None:
    config_file, runtime_doc = prepare_demo_repo()

    ts = TreeSearch(db_path=None)
    ts.index(str(config_file), str(runtime_doc))

    rag = TreeSearchGraphRAG.from_tree_search(
        ts,
        extractor=RuleBasedTripletExtractor(),
        expansion_config=ExpansionConfig(max_relations=5),
    )
    stats = rag.build_graph()
    answer = rag.query("max_concurrency TreeSearchConfig Runtime Settings")

    print("Graph stats:")
    print(f"  documents={stats.documents} passages={stats.passages} relations={stats.relations}")
    print()
    print("Answer:")
    print(answer.answer)
    print()
    print("Verification:")
    print(f"  ok={answer.verification.ok}")
    if answer.verification.errors:
        for error in answer.verification.errors:
            print(f"  error={error}")
    print()
    print("Citations:")
    for citation in answer.evidence_chain.citations:
        print(f"  {citation.source_path}:{citation.line_start}-{citation.line_end}")

    if not answer.verification.ok:
        raise SystemExit("GraphRAG smoke demo verification failed")
    if stats.relations < 2:
        raise SystemExit("GraphRAG smoke demo expected at least two extracted relations")


if __name__ == "__main__":
    main()
