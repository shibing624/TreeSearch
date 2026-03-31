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
import logging
import os
from typing import Optional, List

from .tree import Document, load_documents
from .search import search
from .config import get_config
from .pathutil import resolve_paths, DEFAULT_IGNORE_DIRS

logger = logging.getLogger(__name__)


class TreeSearch:
    """Unified TreeSearch Engine — the only class you need.

    Usage::

        from treesearch import TreeSearch

        # Eager indexing
        ts = TreeSearch()
        ts.index("docs/*.md", "src/*.py")
        results = ts.search("How to configure voice calls?")

        # Lazy indexing — auto-builds index on first search
        ts = TreeSearch("docs/*.md", "src/*.py")
        results = ts.search("How to configure voice calls?")

        # In-memory mode — no index.db file written to disk
        ts = TreeSearch("docs/", db_path=None)
        results = ts.search("voice calls")

        # Save / load indexes via single DB file
        ts.save_index("./my_index.db")
        ts.load_index("./my_index.db")

    In-memory mode (``db_path=None``):
        All FTS5 indexes are kept in memory using SQLite ``:memory:``.
        No ``.db`` file is created on disk. This is convenient for quick
        searches, scripts, and ephemeral use cases. Performance is excellent
        even with thousands of documents (5,000 docs < 10ms).
        The trade-off is that indexes are lost when the process exits —
        set ``db_path`` to a file path for persistent, incremental indexing.
    """

    def __init__(
        self,
        *paths: str,
        db_path: str | None = "./index.db",
        ignore_dirs: frozenset[str] = DEFAULT_IGNORE_DIRS,
        respect_gitignore: bool = True,
        max_files: int | None = None,
        **kwargs
    ):
        """
        Initialize the TreeSearch engine.

        Args:
            *paths: File paths, glob patterns, or directories to index lazily on first search.
            db_path: Path to the SQLite database file for all data storage.
                Set to ``None`` for in-memory mode (no file written to disk).
                In-memory mode is ideal for quick searches and scripts; indexes
                are lost when the process exits. Default: ``"./index.db"``.
            ignore_dirs: Directory names to skip when recursively walking directories.
            respect_gitignore: Honour .gitignore files when walking directories (requires ``pathspec``).
            max_files: Safety cap on files discovered per directory walk.
                Defaults to ``get_config().max_dir_files`` (10,000).
            **kwargs: Additional default arguments for search().
        """
        self._pending_paths: List[str] = list(paths)
        self.db_path = db_path
        self.documents: List[Document] = []
        self._last_index_stats = None  # IndexStats from last index() call
        self.config = get_config()
        self.kwargs = kwargs
        self._ignore_dirs = ignore_dirs
        self._respect_gitignore = respect_gitignore
        self._max_files = max_files if max_files is not None else self.config.max_dir_files
        # Ensure FTS5 scorer uses the same DB as tree storage
        if db_path and not self.config.fts_db_path:
            self.config.fts_db_path = db_path

    def _resolve_patterns(self, patterns: list[str]) -> list[str]:
        """Resolve glob patterns, files, and directories into a flat list of file paths."""
        return resolve_paths(
            patterns,
            ignore_dirs=self._ignore_dirs,
            respect_gitignore=self._respect_gitignore,
            max_files=self._max_files,
        )

    def _get_changed_files(self, stored_meta: dict = None) -> List[str]:
        """Return list of pending source files that changed since last index.

        Uses (mtime_ns, size) fingerprints stored in index_meta.
        Returns only the files whose fingerprint differs from the stored value.

        Args:
            stored_meta: Pre-loaded index metadata dict. If None, reads from DB.
        """
        from .indexer import _file_hash

        if stored_meta is None:
            from .fts import FTS5Index
            fts = FTS5Index(db_path=self.db_path)
            stored_meta = fts.get_all_index_meta()
            fts.close()

        changed = []
        for fp in self._resolve_patterns(self._pending_paths):
            abs_fp = os.path.abspath(fp)
            current_hash = _file_hash(abs_fp)
            if not current_hash:
                continue
            if stored_meta.get(abs_fp) != current_hash:
                changed.append(fp)
        return changed

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    async def aindex(self, *paths: str, force: bool = False, **kwargs) -> List[Document]:
        """Async: Build tree indexes from files, directories, or glob patterns."""
        from .indexer import build_index

        result = await build_index(
            list(paths),
            db_path=self.db_path or "",
            force=force,
            ignore_dirs=self._ignore_dirs,
            respect_gitignore=self._respect_gitignore,
            max_files=self._max_files,
            **kwargs
        )
        self.documents = list(result)
        # Capture IndexStats if available
        if hasattr(result, 'stats'):
            self._last_index_stats = result.stats
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

        Args:
            query: user query
            top_k_docs: max documents to search (routing stage)
            max_nodes_per_doc: max result nodes per document
            pre_filter: custom PreFilter instance for node pre-scoring
            text_mode: 'full' (default) | 'summary' | 'none' - controls text in results
            include_ancestors: attach ancestor titles for context anchoring
            merge_strategy: 'interleave' (default) | 'per_doc' | 'global_score'

        Returns:
            dict with 'documents', 'query', and 'flat_nodes'.
        """
        if not self.documents and self._pending_paths:
            if self.db_path and os.path.isfile(self.db_path):
                from .fts import FTS5Index, get_fts_index, set_fts_index
                fts = get_fts_index(db_path=self.db_path)
                cached_docs = fts.load_all_documents()
                stored_meta = fts.get_all_index_meta() if cached_docs else {}

                if cached_docs:
                    changed = self._get_changed_files(stored_meta=stored_meta)
                    if not changed:
                        self.documents = cached_docs
                        self._pending_paths.clear()
                    else:
                        logger.info("Incremental re-index: %d file(s) changed", len(changed))
                        await self.aindex(*changed)
                        self.documents = load_documents(self.db_path)
                        self._pending_paths.clear()
            if not self.documents and self._pending_paths:
                await self.aindex(*self._pending_paths)
                self._pending_paths.clear()

        if not self.documents:
            if self.db_path and os.path.isfile(self.db_path):
                self.documents = load_documents(self.db_path)

        if not self.documents:
            raise ValueError(
                "No documents available. Pass file paths to TreeSearch() or call index() first."
            )

        search_kwargs = {
            **self.kwargs,
            **kwargs
        }
        return await search(query, self.documents, **search_kwargs)

    def search(self, query: str, **kwargs) -> dict:
        """Sync: Search across indexed documents.

        Args:
            query: user query
            top_k_docs: max documents to search (routing stage)
            max_nodes_per_doc: max result nodes per document
            pre_filter: custom PreFilter instance for node pre-scoring
            text_mode: 'full' (default) | 'summary' | 'none' - controls text in results
            include_ancestors: attach ancestor titles for context anchoring
            merge_strategy: 'interleave' (default) | 'per_doc' | 'global_score'

        Returns:
            dict with 'documents', 'query', and 'flat_nodes'.
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
    # Batch search
    # ------------------------------------------------------------------

    async def abatch_search(self, queries: list[str], **kwargs) -> list[dict]:
        """Async: Search for multiple queries in one call, sharing index initialisation.

        Queries are executed concurrently via ``asyncio.gather``.  This is
        significantly faster than calling ``asearch()`` in a loop because
        document loading, FTS5 connection setup, and tree initialisation
        happen only once, not once per query.

        Args:
            queries:  List of query strings.
            **kwargs: Same keyword arguments accepted by ``asearch()``
                      (top_k_docs, max_nodes_per_doc, search_mode, etc.).

        Returns:
            List of result dicts, in the **same order** as ``queries``.
            Each element has the same shape as the dict returned by ``asearch()``.

        Example::

            ts = TreeSearch("docs/")
            results = await ts.abatch_search(
                ["authentication flow", "database schema", "rate limits"],
                search_mode="tree",
            )
            for query, result in zip(queries, results):
                print(query, len(result["flat_nodes"]))
        """
        if not queries:
            return []

        # Ensure documents are loaded once (not once per query)
        if not self.documents and self._pending_paths:
            if self.db_path and os.path.isfile(self.db_path):
                from .fts import get_fts_index
                fts = get_fts_index(db_path=self.db_path)
                cached_docs = fts.load_all_documents()
                stored_meta = fts.get_all_index_meta() if cached_docs else {}
                if cached_docs:
                    changed = self._get_changed_files(stored_meta=stored_meta)
                    if not changed:
                        self.documents = cached_docs
                        self._pending_paths.clear()
                    else:
                        logger.info("Batch search: incremental re-index, %d file(s) changed", len(changed))
                        await self.aindex(*changed)
                        from .tree import load_documents
                        self.documents = load_documents(self.db_path)
                        self._pending_paths.clear()
            if not self.documents and self._pending_paths:
                await self.aindex(*self._pending_paths)
                self._pending_paths.clear()

        if not self.documents and self.db_path and os.path.isfile(self.db_path):
            from .tree import load_documents
            self.documents = load_documents(self.db_path)

        if not self.documents:
            raise ValueError(
                "No documents available. Pass file paths to TreeSearch() or call index() first."
            )

        search_kwargs = {**self.kwargs, **kwargs}

        # Run all queries concurrently — documents and FTS index are already warm
        from .search import search as _search
        tasks = [_search(q, self.documents, **search_kwargs) for q in queries]
        results = await asyncio.gather(*tasks)
        return list(results)

    def batch_search(self, queries: list[str], **kwargs) -> list[dict]:
        """Sync: Search for multiple queries in one call, sharing index initialisation.

        This is the synchronous equivalent of ``abatch_search()``.  Use this
        when you are not inside an async context.  If you already have a
        running event loop (e.g. inside Jupyter), use ``await abatch_search()``
        instead.

        Args:
            queries:  List of query strings.
            **kwargs: Same keyword arguments accepted by ``search()``.

        Returns:
            List of result dicts, one per query, in the same order as ``queries``.

        Example::

            ts = TreeSearch("docs/")
            ts.index()
            all_results = ts.batch_search(
                ["auth flow", "db schema"],
                max_nodes_per_doc=5,
            )
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Event loop is already running. Please use `await abatch_search()` instead."
                )
        except RuntimeError as e:
            if "Event loop is already running" in str(e):
                raise
            pass

        return asyncio.run(self.abatch_search(queries, **kwargs))

    # ------------------------------------------------------------------
    # File listing
    # ------------------------------------------------------------------

    def resolve_glob_files(self, *paths: str) -> List[str]:
        """Resolve glob patterns and return the list of matched file paths.

        If no paths are given, resolves the pending paths passed to the constructor.

        Args:
            *paths: File paths or glob patterns (e.g. 'docs/**/*.md', 'src/*.py').
                    Supports recursive ``**`` patterns.

        Returns:
            Sorted list of resolved absolute file paths.
        """
        targets = list(paths) if paths else self._pending_paths
        resolved = self._resolve_patterns(targets)
        return sorted(set(os.path.abspath(f) for f in resolved))

    def get_indexed_files(self) -> List[dict]:
        """Return information about all files that have been indexed in the database.

        Each item contains:
            - source_path: absolute path of the source file
            - doc_id: document identifier in the index
            - doc_name: document name
            - source_type: file type (e.g. 'markdown', 'code', 'text')

        Returns:
            List of dicts with indexed file information, sorted by source_path.
        """
        if not self.db_path or not os.path.isfile(self.db_path):
            return []

        docs = load_documents(self.db_path)
        result = []
        for doc in docs:
            result.append({
                "source_path": doc.metadata.get("source_path", ""),
                "doc_id": doc.doc_id,
                "doc_name": doc.doc_name,
                "source_type": doc.source_type,
            })
        return sorted(result, key=lambda x: x["source_path"])

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, *paths_or_ids: str) -> int:
        """Delete documents from the index by source path or doc_id.

        Accepts file paths, absolute paths, or doc_ids (the internal identifier
        stored in the database).  Matching is attempted in this order:

        1. Exact ``source_path`` match in the database (most common use case —
           pass the same path you passed to ``index()``).
        2. Absolute path version of the given string.
        3. Direct ``doc_id`` match (for advanced callers who know the doc_id).

        The in-memory ``self.documents`` list is kept in sync so subsequent
        ``search()`` calls reflect the deletion immediately.

        Args:
            *paths_or_ids: one or more file paths / doc_ids to remove.

        Returns:
            Number of documents actually deleted (0 if none were found).

        Example::

            ts = TreeSearch("docs/", db_path="./index.db")
            ts.index()
            removed = ts.delete("docs/old_file.md")
            # removed == 1 if the file was indexed, 0 if not found
        """
        if not paths_or_ids:
            return 0

        if not self.db_path or not os.path.isfile(self.db_path):
            # In-memory mode or no DB yet — only remove from self.documents.
            removed = 0
            abs_targets = {os.path.abspath(p) for p in paths_or_ids}
            remaining = []
            for doc in self.documents:
                sp = doc.metadata.get("source_path", "")
                if sp in abs_targets or os.path.abspath(sp) in abs_targets or doc.doc_id in paths_or_ids:
                    removed += 1
                    logger.info("Deleted document (in-memory): doc_id=%r", doc.doc_id)
                else:
                    remaining.append(doc)
            self.documents = remaining
            return removed

        from .fts import FTS5Index
        fts = FTS5Index(db_path=self.db_path)
        removed = 0

        for target in paths_or_ids:
            # 1. Try source_path exact match.
            doc_id = fts.get_doc_id_by_source_path(target)

            # 2. Try absolute path.
            if doc_id is None:
                abs_target = os.path.abspath(target)
                if abs_target != target:
                    doc_id = fts.get_doc_id_by_source_path(abs_target)

            # 3. Try treating the argument itself as a doc_id.
            if doc_id is None:
                if fts.is_document_indexed(target):
                    doc_id = target

            if doc_id is None:
                logger.warning("delete: %r not found in index (checked source_path and doc_id)", target)
                continue

            if fts.delete_document(doc_id):
                removed += 1
                # Keep self.documents in sync.
                self.documents = [d for d in self.documents if d.doc_id != doc_id]

        fts.close()

        if removed:
            logger.info("Deleted %d document(s) from index %r", removed, self.db_path)
        return removed

    # ------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------

    def get_index_stats(self):
        """Return IndexStats from the most recent index() or aindex() call.

        Returns:
            IndexStats object, or None if no indexing has been performed yet.
        """
        return self._last_index_stats

    # ------------------------------------------------------------------
    # Save / Load indexes
    # ------------------------------------------------------------------

    def save_index(self, db_path: Optional[str] = None) -> str:
        """Save current documents to a database file.

        Args:
            db_path: Target database file path. Defaults to self.db_path.

        Returns:
            Path to the database file.

        Raises:
            ValueError: If no db_path is provided and instance is in-memory mode.
        """
        from .fts import FTS5Index
        out = db_path or self.db_path
        if not out:
            raise ValueError(
                "No db_path specified. Pass a path to save_index() or set db_path in constructor."
            )
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

        Raises:
            ValueError: If no db_path is provided and instance is in-memory mode.
        """
        src = db_path or self.db_path
        if not src:
            raise ValueError(
                "No db_path specified. Pass a path to load_index() or set db_path in constructor."
            )
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Database file not found: {src}")

        self.documents = load_documents(src)
        logger.info("Loaded %d documents from %s", len(self.documents), src)
        return self.documents
