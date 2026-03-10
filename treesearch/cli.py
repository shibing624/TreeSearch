# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI entry point for TreeSearch.

Subcommands:
  index     - Build tree structure index from documents
  search    - Search across indexed documents

Usage:
    treesearch index --paths "docs/*.md"
    treesearch search --index_dir ./indexes/ --query "How does auth work?"
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time

from treesearch.indexer import build_index, md_to_tree, text_to_tree
from treesearch.tree import Document, load_documents, print_toc
from treesearch.search import search
from treesearch.config import get_config, set_config, TreeSearchConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand: index
# ---------------------------------------------------------------------------

def _add_index_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--paths", nargs="+", required=True,
                     help="File paths or glob patterns (e.g. 'docs/*.md' paper.txt)")

    sub.add_argument("-o", "--output_dir", type=str, default="./indexes",
                     help="Output directory for database file (default: ./indexes)")
    sub.add_argument("--db", type=str, default="",
                     help="Path to SQLite database file (default: {output_dir}/index.db)")
    sub.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="LLM model name")
    sub.add_argument("--api-key", type=str, default=None,
                     help="LLM API key (overrides TREESEARCH_LLM_API_KEY / OPENAI_API_KEY)")
    sub.add_argument("--no-summary", action="store_true", help="Skip node summary generation")
    sub.add_argument("--add-description", action="store_true", help="Generate doc description")
    sub.add_argument("--add-text", action="store_true", help="Include node text in output")
    sub.add_argument("--no-node-id", action="store_true", help="Skip node ID assignment")
    sub.add_argument("--thinning", action="store_true", help="Apply tree thinning")
    sub.add_argument("--thinning-threshold", type=int, default=5000,
                     help="Min token threshold for thinning (default: 5000)")
    sub.add_argument("--summary-threshold", type=int, default=200,
                     help="Token threshold for summary generation (default: 200)")
    sub.add_argument("--fallback-to-llm", type=str, default="auto",
                     choices=["auto", "yes", "no"],
                     help="LLM fallback for text heading detection (default: auto)")
    sub.add_argument("--max-concurrency", type=int, default=5,
                     help="Max concurrent indexing tasks (default: 5)")
    sub.add_argument("--force", action="store_true",
                     help="Force re-index even if files unchanged")


async def _run_index(args) -> None:
    if args.api_key:
        cfg = get_config()
        cfg.api_key = args.api_key
        set_config(cfg)

    start_time = time.time()
    print(f"Indexing {len(args.paths)} path pattern(s)...")

    results = await build_index(
        paths=args.paths,
        output_dir=args.output_dir,
        db_path=args.db,
        model=args.model,
        if_add_node_summary=not args.no_summary,
        if_add_doc_description=args.add_description,
        if_add_node_text=args.add_text,
        if_add_node_id=not args.no_node_id,
        if_thinning=args.thinning,
        min_token_threshold=args.thinning_threshold,
        summary_token_threshold=args.summary_threshold,
        max_concurrency=args.max_concurrency,
        force=args.force,
    )

    db_path = args.db or os.path.join(args.output_dir, "index.db")
    elapsed = time.time() - start_time
    print(f"\nIndexed {len(results)} file(s) to {db_path} ({elapsed:.1f}s)")
    for doc in results:
        print(f"  - {doc.doc_name}")
        print(f"    TOC:")
        print_toc(doc.structure)


# ---------------------------------------------------------------------------
# Subcommand: search
# ---------------------------------------------------------------------------

def _add_search_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--index_dir", type=str, default="./indexes",
                     help="Directory containing the database file (default: ./indexes)")
    sub.add_argument("--db", type=str, default="",
                     help="Path to SQLite database file (default: {index_dir}/index.db)")
    sub.add_argument("--query", type=str, required=True,
                     help="Search query")
    sub.add_argument("--model", type=str, default="gpt-4o-2024-11-20", help="LLM model name")
    sub.add_argument("--api-key", type=str, default=None,
                     help="LLM API key (overrides TREESEARCH_LLM_API_KEY / OPENAI_API_KEY)")
    sub.add_argument("--strategy", type=str, default="fts5_only",
                     choices=["fts5_only", "best_first", "auto"],
                     help="Search strategy (default: fts5_only)")
    sub.add_argument("--fts", action="store_true",
                     help="Enable SQLite FTS5 full-text search as pre-filter backend")
    sub.add_argument("--fts-db", type=str, default=None,
                     help="Path to FTS5 database file (default: in-memory)")
    sub.add_argument("--top-k-docs", type=int, default=3,
                     help="Max documents to search (default: 3)")
    sub.add_argument("--max-nodes", type=int, default=5,
                     help="Max result nodes per document (default: 5)")
    sub.add_argument("--threshold", type=float, default=0.3,
                     help="Minimum relevance score (default: 0.3)")
    sub.add_argument("--max-llm-calls", type=int, default=30,
                     help="Max LLM calls per document for best_first (default: 30)")
    sub.add_argument("--no-bm25", action="store_true",
                     help="Disable BM25 pre-scoring for best_first strategy")


def _load_documents_from_dir(index_dir: str, db: str = "") -> list[Document]:
    """Load all documents from a database file."""
    db_path = db or os.path.join(index_dir, "index.db")
    if not os.path.isfile(db_path):
        print(f"Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    documents = load_documents(db_path)
    if not documents:
        print(f"No documents found in database: {db_path}", file=sys.stderr)
        sys.exit(1)
    for doc in documents:
        logger.info("Loaded document: %s (%d root nodes)", doc.doc_name, len(doc.structure))
    return documents


async def _run_search(args) -> None:
    if args.api_key:
        cfg = get_config()
        cfg.api_key = args.api_key
        set_config(cfg)

    # Enable FTS5 if --fts flag is set
    if args.fts:
        cfg = get_config()
        cfg.fts_enabled = True
        if args.fts_db:
            cfg.fts_db_path = args.fts_db
        set_config(cfg)

    documents = _load_documents_from_dir(args.index_dir, db=args.db)
    db_path = args.db or os.path.join(args.index_dir, "index.db")
    print(f"Loaded {len(documents)} document(s) from {db_path}")
    for doc in documents:
        print(f"  - {doc.doc_name}")

    print(f"\nQuery: {args.query}")
    pre_filter = "FTS5" if args.fts else ("BM25" if not args.no_bm25 else "none")
    strategy_desc = {
        "best_first": f"Best-First (max_llm_calls={args.max_llm_calls}, pre_filter={pre_filter})",
        "fts5_only": "FTS5-only (no LLM)",
        "auto": "Auto (per-document strategy based on source_type)",
    }
    print("Search strategy:", strategy_desc.get(args.strategy, args.strategy))
    print("---")

    start_time = time.time()

    result = await search(
        query=args.query,
        documents=documents,
        model=args.model,
        top_k_docs=args.top_k_docs,
        max_nodes_per_doc=args.max_nodes,
        strategy=args.strategy,
        value_threshold=args.threshold,
        max_llm_calls=args.max_llm_calls,
        use_bm25=not args.no_bm25,
    )
    elapsed = time.time() - start_time

    if not result["documents"]:
        print("\nNo relevant results found.")
        return

    print(f"\nFound results in {len(result['documents'])} document(s) "
          f"(LLM calls: {result.get('llm_calls', 0)}, {elapsed:.1f}s):\n")
    for doc_result in result["documents"]:
        print(f"[{doc_result['doc_name']}]")
        for node in doc_result["nodes"]:
            score = node.get("score", 0)
            print(f"  [{score:.2f}] {node['title']}")
            text = node.get("text", "")
            if text:
                preview = text[:200].replace("\n", " ")
                print(f"         {preview}{'...' if len(text) > 200 else ''}")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="treesearch",
        description=(
            "TreeSearch: Structure-aware document retrieval without embeddings.\n"
            "No vector embeddings. No chunk splitting. Pure LLM reasoning."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    sub = p.add_subparsers(dest="command", help="Available commands")

    idx = sub.add_parser("index", help="Build tree structure index from documents (supports glob)")
    _add_index_args(idx)

    sch = sub.add_parser("search", help="Search across indexed documents using tree reasoning")
    _add_search_args(sch)

    return p


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")

    if args.command == "index":
        asyncio.run(_run_index(args))
    elif args.command == "search":
        asyncio.run(_run_search(args))


if __name__ == "__main__":
    main()
