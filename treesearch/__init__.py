# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: TreeSearch - Structure-aware document retrieval via tree-structured indexing.

No vector embeddings. No chunk splitting. FTS5 keyword matching over document trees.

Quick Start::

    from treesearch import TreeSearch

    # Lazy indexing -- auto-builds index on first search
    ts = TreeSearch("./docs/")
    results = ts.search("How to configure voice calls?")
"""
__version__ = "0.6.3"

# ============================================================================
# FTS5 Compatibility: Use pysqlite3 on systems without FTS5 support
# ============================================================================
import sys

def _ensure_fts5_support():
    """Ensure SQLite has FTS5 support, use pysqlite3 as fallback."""
    try:
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE _fts5_test USING fts5(content)")
        conn.close()
        return True
    except Exception:
        return False

if not _ensure_fts5_support():
    try:
        from pysqlite3 import dbapi2 as _sqlite3
        sys.modules["sqlite3"] = _sqlite3
        if "treesearch.fts" in sys.modules:
            sys.modules["treesearch.fts"].sqlite3 = _sqlite3
    except ImportError:
        import warnings
        warnings.warn(
            "SQLite FTS5 not available. Full-text search will use LIKE fallback "
            "(slower, no BM25 ranking). For best performance, install pysqlite3-binary:\n"
            "    pip install pysqlite3-binary",
            RuntimeWarning,
            stacklevel=2,
        )

# ============================================================================
# Public API
# ============================================================================

# -- Primary: the only class most users need --
from treesearch.treesearch import TreeSearch

# -- Core --
from treesearch.indexer import build_index, md_to_tree, text_to_tree, IndexStats
from treesearch.search import search, search_sync, GrepFilter
from treesearch.tree import Document, load_index, load_documents, save_index, flatten_tree, print_toc
from treesearch.config import TreeSearchConfig, get_config, set_config, reset_config
from treesearch.fts import FTS5Index

__all__ = [
    # Primary
    "TreeSearch",
    # Indexing
    "build_index", "md_to_tree", "text_to_tree", "IndexStats",
    # Search
    "search", "search_sync", "GrepFilter",
    # Document & tree
    "Document", "load_index", "load_documents", "save_index", "flatten_tree", "print_toc",
    # Config
    "TreeSearchConfig", "get_config", "set_config", "reset_config",
    # FTS5
    "FTS5Index",
]

# Advanced APIs — import from submodules directly:
#   from treesearch.indexer import code_to_tree, json_to_tree, jsonl_to_tree, csv_to_tree
#   from treesearch.search import PreFilter
#   from treesearch.fts import get_fts_index, set_fts_index, reset_fts_index
#   from treesearch.tokenizer import tokenize
#   from treesearch.utils import count_tokens
#   from treesearch.tree import assign_node_ids, find_node, get_leaf_nodes, ...
#   from treesearch.parsers import ParserRegistry, get_parser, extract_pdf_text
