# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI entry point for TreeSearch.

Default usage (lazy index + search):
    treesearch "How does auth work?" src/ docs/*.md
    treesearch "FTS5 search" treesearch/

Advanced subcommands:
    treesearch index --paths src/ docs/ --force
    treesearch search --db ./indexes/index.db --query "auth"
"""
import argparse
import asyncio
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)


class _DefaultArgumentParser(argparse.ArgumentParser):
    """Default parser with small normalization for explicit query flags."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if getattr(parsed, "fts_expression", None) and parsed.query:
            parsed.paths = [parsed.query, *parsed.paths]
            parsed.query = None
        return parsed


# ---------------------------------------------------------------------------
# Default command: lazy search (the simplest way to use TreeSearch)
# ---------------------------------------------------------------------------

def _run_default(args) -> None:
    """Lazy index + search: the simplest workflow."""
    from treesearch.treesearch import TreeSearch

    paths = args.paths
    query = args.query
    fts_expression = args.fts_expression
    db_path = args.db or "./index.db"
    max_nodes = args.max_nodes
    search_mode = args.search_mode
    show_path = args.show_path
    regex = args.regex

    if regex and fts_expression is not None:
        print("Error: --regex and --fts-expression cannot be used together.", file=sys.stderr)
        sys.exit(2)

    if not paths:
        print("Error: no paths specified. Usage: treesearch \"query\" path1 [path2 ...]",
              file=sys.stderr)
        sys.exit(1)
    if query is None and fts_expression is None:
        print("Error: query is required unless --fts-expression is provided.", file=sys.stderr)
        sys.exit(2)

    start_time = time.time()

    ts = TreeSearch(*paths, db_path=db_path)
    display_query = fts_expression or query
    try:
        result = ts.search(
            display_query,
            max_nodes_per_doc=max_nodes,
            search_mode=search_mode,
            fts_expression=fts_expression,
            regex=regex,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    elapsed = time.time() - start_time

    if not result["documents"] or not result["flat_nodes"]:
        print(f"No results found for: {display_query}")
        return

    mode = result.get("mode", "flat")
    total_nodes = sum(len(d["nodes"]) for d in result["documents"])
    print(f"Found {total_nodes} result(s) in {len(result['documents'])} doc(s) [{mode} mode] ({elapsed:.1f}s)\n")

    # Show path results if available and requested
    if show_path and "paths" in result:
        for i, path_info in enumerate(result["paths"], 1):
            score = path_info.get("score", 0)
            doc_name = path_info.get("doc_name", "")
            reasons = path_info.get("reasons", [])
            path_nodes = path_info.get("path", [])
            snippet = path_info.get("snippet", "")

            print(f"Path {i} ({score:.2f}) {doc_name}")
            for j, pn in enumerate(path_nodes):
                indent = "    " * j
                connector = "|-- " if j < len(path_nodes) - 1 else "`-> "
                print(f"  {indent}{connector}{pn.get('title', '')}")
            if snippet:
                preview = snippet[:300]
                if len(snippet) > 300:
                    preview += "..."
                for line in preview.split("\n"):
                    print(f"        {line}")
            if reasons:
                print(f"    reasons: {'; '.join(reasons[:5])}")
            print()

    # Show flat node results
    for doc_result in result["documents"]:
        doc_name = doc_result["doc_name"]
        for node in doc_result["nodes"]:
            score = node.get("score", 0)
            title = node.get("title", "")
            line_start = node.get("line_start")
            line_end = node.get("line_end")
            text = node.get("text", "")

            loc = f"  (lines {line_start}-{line_end})" if line_start and line_end else ""
            print(f"[{score:.2f}] {doc_name} > {title}{loc}")

            if text:
                preview = text[:500]
                if len(text) > 500:
                    preview += "..."
                for line in preview.split("\n"):
                    print(f"  {line}")
            print()


# ---------------------------------------------------------------------------
# Subcommand: index
# ---------------------------------------------------------------------------

def _add_index_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--paths", nargs="+", required=True,
                     help="File paths, glob patterns, or directories (e.g. src/ 'docs/*.md')")
    sub.add_argument("-o", "--output_dir", type=str, default="./indexes",
                     help="Output directory for database file (default: ./indexes)")
    sub.add_argument("--db", type=str, default="",
                     help="Path to SQLite database file (default: {output_dir}/index.db)")
    sub.add_argument("--no-summary", action="store_true", help="Skip node summary generation")
    sub.add_argument("--add-description", action="store_true", help="Generate doc description")
    sub.add_argument("--add-text", action="store_true", help="Include node text in output")
    sub.add_argument("--no-node-id", action="store_true", help="Skip node ID assignment")
    sub.add_argument("--thinning", action="store_true", help="Apply tree thinning")
    sub.add_argument("--thinning-threshold", type=int, default=15000,
                     help="Min chars threshold for thinning (default: 15000)")
    sub.add_argument("--summary-threshold", type=int, default=600,
                     help="Chars threshold for summary generation (default: 600)")
    sub.add_argument("--max-concurrency", type=int, default=None,
                     help="Max concurrent indexing tasks (default: auto based on CPU cores)")
    sub.add_argument("--force", action="store_true",
                     help="Force re-index even if files unchanged")
    sub.add_argument("--stats", action="store_true",
                     help="Show detailed indexing statistics after completion")


async def _run_index(args) -> None:
    from treesearch.indexer import build_index
    from treesearch.tree import print_toc

    start_time = time.time()
    print(f"Indexing {len(args.paths)} path(s)...")

    results = await build_index(
        paths=args.paths,
        output_dir=args.output_dir,
        db_path=args.db,
        if_add_node_summary=not args.no_summary,
        if_add_doc_description=args.add_description,
        if_add_node_text=args.add_text,
        if_add_node_id=not args.no_node_id,
        if_thinning=args.thinning,
        min_thinning_chars=args.thinning_threshold,
        summary_chars_threshold=args.summary_threshold,
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

    # Display stats if requested
    if args.stats and hasattr(results, 'stats') and results.stats:
        print(f"\n{results.stats.summary()}")


# ---------------------------------------------------------------------------
# Subcommand: search (over pre-built index)
# ---------------------------------------------------------------------------

def _add_search_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--index_dir", type=str, default="./indexes",
                     help="Directory containing the database file (default: ./indexes)")
    sub.add_argument("--db", type=str, default="",
                     help="Path to SQLite database file (default: {index_dir}/index.db)")
    query_group = sub.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        type=str,
        help="Search query. Supports auth* (prefix) and *auth* (contains regex)",
    )
    query_group.add_argument(
        "--fts-expression",
        type=str,
        dest="fts_expression",
        help="Raw FTS5 expression, e.g. auth* or \"auth NEAR/5 token\"",
    )
    sub.add_argument("--regex", action="store_true",
                     help="Treat --query as a raw regex pattern")
    sub.add_argument("--top-k-docs", type=int, default=3,
                     help="Max documents to search (default: 3)")
    sub.add_argument("--max-nodes", type=int, default=5,
                     help="Max result nodes per document (default: 5)")
    sub.add_argument("--search-mode", type=str, default="auto",
                     choices=["auto", "tree", "flat"],
                     help="Search mode: 'auto', 'tree' or 'flat' (default: auto)")
    sub.add_argument("--show-path", action="store_true",
                     help="Show path-based results (tree mode only)")


def _load_documents_from_dir(index_dir: str, db: str = ""):
    """Load all documents from a database file."""
    from treesearch.tree import Document, load_documents

    db_path = db or os.path.join(index_dir, "index.db")
    if not os.path.isfile(db_path):
        print(f"Database file not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    documents = load_documents(db_path)
    if not documents:
        print(f"No documents found in database: {db_path}", file=sys.stderr)
        sys.exit(1)
    return documents


async def _run_search(args) -> None:
    from treesearch.search import search

    documents = _load_documents_from_dir(args.index_dir, db=args.db)
    print(f"Loaded {len(documents)} document(s)\n")

    if args.regex and args.fts_expression is not None:
        print("Error: --regex and --fts-expression cannot be used together.", file=sys.stderr)
        sys.exit(2)

    display_query = args.fts_expression or args.query

    print(f"Query: {display_query}")
    print("---")

    start_time = time.time()
    try:
        result = await search(
            query=args.query or args.fts_expression,
            documents=documents,
            top_k_docs=args.top_k_docs,
            max_nodes_per_doc=args.max_nodes,
            search_mode=args.search_mode,
            fts_expression=args.fts_expression,
            regex=args.regex,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    elapsed = time.time() - start_time

    if not result["documents"]:
        print("\nNo relevant results found.")
        return

    mode = result.get("mode", "flat")
    total_nodes = sum(len(d["nodes"]) for d in result["documents"])
    print(f"\nFound {total_nodes} result(s) in {len(result['documents'])} doc(s) [{mode} mode] ({elapsed:.1f}s)\n")

    # Show paths if available
    if args.show_path and "paths" in result:
        for i, path_info in enumerate(result["paths"], 1):
            score = path_info.get("score", 0)
            doc_name = path_info.get("doc_name", "")
            path_nodes = path_info.get("path", [])
            reasons = path_info.get("reasons", [])
            snippet = path_info.get("snippet", "")

            print(f"Path {i} ({score:.2f}) {doc_name}")
            for j, pn in enumerate(path_nodes):
                indent = "    " * j
                connector = "|-- " if j < len(path_nodes) - 1 else "`-> "
                print(f"  {indent}{connector}{pn.get('title', '')}")
            if snippet:
                preview = snippet[:300]
                if len(snippet) > 300:
                    preview += "..."
                for line in preview.split("\n"):
                    print(f"        {line}")
            if reasons:
                print(f"    reasons: {'; '.join(reasons[:5])}")
            print()

    for doc_result in result["documents"]:
        doc_name = doc_result["doc_name"]
        for node in doc_result["nodes"]:
            score = node.get("score", 0)
            title = node.get("title", "")
            line_start = node.get("line_start")
            line_end = node.get("line_end")
            text = node.get("text", "")

            loc = f"  (lines {line_start}-{line_end})" if line_start and line_end else ""
            print(f"[{score:.2f}] {doc_name} > {title}{loc}")

            if text:
                preview = text[:500]
                if len(text) > 500:
                    preview += "..."
                for line in preview.split("\n"):
                    print(f"  {line}")
            print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_SUBCOMMANDS = {"index", "search"}


def _build_default_parser() -> argparse.ArgumentParser:
    """Parser for default mode: treesearch "query" path1 path2 ..."""
    p = _DefaultArgumentParser(
        prog="treesearch",
        description=(
            "TreeSearch: Structure-aware document retrieval.\n\n"
            "Quick usage:\n"
            '  treesearch "search query" src/ docs/\n'
            '  treesearch "How does auth work?" project/\n\n'
            "Advanced:\n"
            "  treesearch index --paths src/ docs/ --force\n"
            "  treesearch search --db ./index.db --query \"auth\"\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    p.add_argument("query", nargs="?", default=None,
                   help="Search query. Supports auth* (prefix) and *auth* (contains regex)")
    p.add_argument("paths", nargs="*", default=[],
                   help="Files, directories, or glob patterns to search")
    p.add_argument("--regex", action="store_true",
                   help="Treat the positional query as a raw regex pattern")
    p.add_argument("--fts-expression", type=str, default=None, dest="fts_expression",
                   help="Raw FTS5 expression, e.g. auth* or \"auth NEAR/5 token\"")
    p.add_argument("--db", type=str, default="",
                   help="Path to SQLite database file (default: ./index.db)")
    p.add_argument("--max-nodes", type=int, default=5,
                   help="Max result nodes per document (default: 5)")
    p.add_argument("--search-mode", type=str, default="tree",
                   choices=["tree", "flat"],
                   help="Search mode: 'tree' (Best-First Search) or 'flat' (original FTS5-only). Default: tree")
    p.add_argument("--show-path", action="store_true",
                   help="Show path-based results with traversal trace (tree mode only)")
    return p


def _build_index_parser() -> argparse.ArgumentParser:
    """Parser for: treesearch index --paths ..."""
    p = argparse.ArgumentParser(prog="treesearch index")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    _add_index_args(p)
    return p


def _build_search_parser() -> argparse.ArgumentParser:
    """Parser for: treesearch search --query ..."""
    p = argparse.ArgumentParser(prog="treesearch search")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    _add_search_args(p)
    return p


def _detect_subcommand(argv: list[str]) -> str | None:
    """Detect if argv contains a subcommand (index/search) as the first non-flag arg."""
    for arg in argv:
        if arg.startswith("-"):
            continue
        if arg in _SUBCOMMANDS:
            return arg
        break  # first positional arg is not a subcommand
    return None


def main(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    subcmd = _detect_subcommand(argv)

    if subcmd == "index":
        # Strip the subcommand word from argv
        idx_argv = []
        found = False
        for a in argv:
            if not found and a == "index":
                found = True
                continue
            idx_argv.append(a)
        parser = _build_index_parser()
        args = parser.parse_args(idx_argv)
        level = logging.DEBUG if args.verbose else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")
        asyncio.run(_run_index(args))

    elif subcmd == "search":
        sch_argv = []
        found = False
        for a in argv:
            if not found and a == "search":
                found = True
                continue
            sch_argv.append(a)
        parser = _build_search_parser()
        args = parser.parse_args(sch_argv)
        level = logging.DEBUG if args.verbose else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")
        asyncio.run(_run_search(args))

    else:
        parser = _build_default_parser()
        args = parser.parse_args(argv)
        level = logging.DEBUG if args.verbose else logging.WARNING
        logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")
        if args.query or args.fts_expression:
            _run_default(args)
        else:
            parser.print_help()
            sys.exit(0)


if __name__ == "__main__":
    main()
