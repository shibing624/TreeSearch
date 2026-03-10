# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified TreeSearch engine class for easy out-of-the-box usage.

This is the ONLY class most users need. It wraps indexing, searching,
saving, and loading into a single, minimal API.

All data (tree structures, FTS5 indexes, incremental metadata) is stored
in a single SQLite .db file — no more scattered JSON files.
"""
import asyncio
import glob
import logging
import os
from typing import Optional, List

from .tree import Document, load_documents
from .search import search
from .config import get_config

logger = logging.getLogger(__name__)


class TreeSearch:
    """Unified TreeSearch Engine — the only class you need.

    Usage::

        from treesearch import TreeSearch

        # Eager indexing
        ts = TreeSearch(model="gpt-4o")
        ts.index("docs/*.md", "src/*.py")
        results = ts.search("How to configure voice calls?")

        # Lazy indexing — auto-builds index on first search
        ts = TreeSearch("docs/*.md", "src/*.py")
        results = ts.search("How to configure voice calls?")

        # Save / load indexes via single DB file
        ts.save_index("./my_index.db")
        ts.load_index("./my_index.db")
    """

    def __init__(
        self,
        *paths: str,
        db_path: str = "./index.db",
        model: Optional[str] = None,
        strategy: str = "fts5_only",
        **kwargs
    ):
        """
        Initialize the TreeSearch engine.

        Args:
            *paths: File paths or glob patterns to index lazily on first search.
            db_path: Path to the SQLite database file for all data storage.
            model: LLM model name (for 'best_first' strategy).
            strategy: Default search strategy. Options: 'fts5_only', 'best_first', 'auto'.
            **kwargs: Additional default arguments for search().
        """
        self._pending_paths: List[str] = list(paths)
        self.db_path = db_path
        self.strategy = strategy
        self.documents: List[Document] = []
        self.config = get_config()
        self.model = model or self.config.model
        self.kwargs = kwargs

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    async def aindex(self, *paths: str, force: bool = False, **kwargs) -> List[Document]:
        """Async: Build tree indexes from files. Supports glob patterns."""
        from .indexer import build_index

        resolved_paths = []
        for p in paths:
            if "*" in p or "?" in p:
                resolved_paths.extend(glob.glob(p, recursive=True))
            else:
                resolved_paths.append(p)

        if not resolved_paths:
            logger.warning("No files found to index.")
            return self.documents

        self.documents = await build_index(
            resolved_paths,
            db_path=self.db_path,
            force=force,
            **kwargs
        )
        return self.documents

    def index(self, *paths: str, force: bool = False, **kwargs) -> List[Document]:
        """Sync: Build tree indexes from files. Supports glob patterns like 'docs/*.md'.

        Returns:
            List of indexed Document objects.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError("Event loop is already running. Please use `await aindex()` instead.")
        except RuntimeError as e:
            if "Event loop is already running" in str(e):
                raise
            pass

        return asyncio.run(self.aindex(*paths, force=force, **kwargs))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def asearch(self, query: str, **kwargs) -> dict:
        """Async: Search across indexed documents. Auto-builds index if pending paths exist.
        Search across one or more documents using tree-structured retrieval.

        All parameters default to ``get_config()`` values when not explicitly set.

        Args:
            query: user query
            documents: list of Document objects (single or multiple)
            model: LLM model name
            top_k_docs: max documents to search (routing stage)
            max_nodes_per_doc: max result nodes per document
            strategy: 'fts5_only' (default) | 'best_first' | 'auto'
                    'fts5_only' uses pure FTS5/BM25 scoring without any LLM calls (fastest)
                    'best_first' uses BM25 pre-scoring + LLM batch ranking (highest quality)
                    'auto' selects per-document strategy based on source_type (all default to fts5_only)
            value_threshold: minimum relevance score
            max_llm_calls: max LLM calls per document (only for best_first)
            use_bm25: enable built-in BM25 pre-scoring (ignored if pre_filter is set)
            pre_filter: custom PreFilter instance for node pre-scoring (overrides use_bm25)
            text_mode: 'full' (default) | 'summary' | 'none' - controls text in results
            include_ancestors: attach ancestor titles for context anchoring
            merge_strategy: 'interleave' (default) | 'per_doc' | 'global_score'
            
        Returns:
            dict with 'documents', 'query', and 'llm_calls'.
        """
        if not self.documents and self._pending_paths:
            await self.aindex(*self._pending_paths)
            self._pending_paths.clear()

        if not self.documents:
            if os.path.isfile(self.db_path):
                self.documents = load_documents(self.db_path)

        if not self.documents:
            raise ValueError(
                "No documents available. Pass file paths to TreeSearch() or call index() first."
            )

        search_kwargs = {
            "model": self.model,
            "strategy": self.strategy,
            **self.kwargs,
            **kwargs
        }
        return await search(query, self.documents, **search_kwargs)

    def search(self, query: str, **kwargs) -> dict:
        """Sync: Search across indexed documents.
        Search across one or more documents using tree-structured retrieval.

        All parameters default to ``get_config()`` values when not explicitly set.

        Args:
            query: user query
            documents: list of Document objects (single or multiple)
            model: LLM model name
            top_k_docs: max documents to search (routing stage)
            max_nodes_per_doc: max result nodes per document
            strategy: 'fts5_only' (default) | 'best_first' | 'auto'
                    'fts5_only' uses pure FTS5/BM25 scoring without any LLM calls (fastest)
                    'best_first' uses BM25 pre-scoring + LLM batch ranking (highest quality)
                    'auto' selects per-document strategy based on source_type (all default to fts5_only)
            value_threshold: minimum relevance score
            max_llm_calls: max LLM calls per document (only for best_first)
            use_bm25: enable built-in BM25 pre-scoring (ignored if pre_filter is set)
            pre_filter: custom PreFilter instance for node pre-scoring (overrides use_bm25)
            text_mode: 'full' (default) | 'summary' | 'none' - controls text in results
            include_ancestors: attach ancestor titles for context anchoring
            merge_strategy: 'interleave' (default) | 'per_doc' | 'global_score'

        Returns:
            dict with 'documents', 'query', and 'llm_calls'.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError("Event loop is already running. Please use `await asearch()` instead.")
        except RuntimeError as e:
            if "Event loop is already running" in str(e):
                raise
            pass

        return asyncio.run(self.asearch(query, **kwargs))

    # ------------------------------------------------------------------
    # Save / Load indexes
    # ------------------------------------------------------------------

    def save_index(self, db_path: Optional[str] = None) -> str:
        """Save current documents to a database file.

        Args:
            db_path: Target database file path. Defaults to self.db_path.

        Returns:
            Path to the database file.
        """
        from .fts import FTS5Index
        out = db_path or self.db_path
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

        fts = FTS5Index(db_path=out)
        for doc in self.documents:
            fts.save_document(doc)
            fts.index_document(doc)
        fts.close()

        logger.info("Saved %d documents to %s", len(self.documents), out)
        return out

    def load_index(self, db_path: Optional[str] = None) -> List[Document]:
        """Load documents from a database file.

        Args:
            db_path: Source database file path. Defaults to self.db_path.

        Returns:
            List of loaded Document objects.
        """
        src = db_path or self.db_path
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Database file not found: {src}")

        self.documents = load_documents(src)
        logger.info("Loaded %d documents from %s", len(self.documents), src)
        return self.documents
