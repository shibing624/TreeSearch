# -*- coding: utf-8 -*-
"""
@description: Tests for v1.1 incremental-indexing improvements.

Covers:
  - INDEX_SCHEMA_VERSION embedded in file fingerprint
  - content vs stat fingerprint modes
  - orphan-document pruning (prune=True for directories)
  - node-level diff in fts.index_document
  - file move/rename detection (no full reindex)
  - FTS5Index.verify_index / repair_index
  - FTS5Index.delete_documents (batched)
  - TreeSearch.delete batch transaction
  - watch.py debounced reindex
"""
import os
import threading
import time

import pytest

from treesearch import TreeSearch
from treesearch.config import (
    INDEX_SCHEMA_VERSION,
    TreeSearchConfig,
    set_config,
    reset_config,
)
from treesearch.fts import FTS5Index
from treesearch.indexer import _file_hash, build_index
from treesearch.tree import Document, assign_node_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md(tmp_path, name: str, body: str = "# Title\n\nBody.\n") -> str:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_includes_schema_version(self, tmp_path):
        f = _md(tmp_path, "a.md")
        h = _file_hash(f)
        assert h.startswith(f"v{INDEX_SCHEMA_VERSION}:")

    def test_stat_mode_default(self, tmp_path):
        f = _md(tmp_path, "a.md")
        h = _file_hash(f)
        assert ":stat:" in h

    def test_content_mode_changes_payload(self, tmp_path):
        f = _md(tmp_path, "a.md", "hello\n")
        h_stat = _file_hash(f, mode="stat")
        h_content = _file_hash(f, mode="content")
        assert h_stat != h_content
        assert ":content:" in h_content

    def test_content_mode_stable_across_touch(self, tmp_path):
        """Touch should NOT invalidate a content fingerprint."""
        f = _md(tmp_path, "a.md", "stable bytes\n")
        h1 = _file_hash(f, mode="content")
        # Bump mtime forward by 2s without touching bytes.
        new_t = time.time() + 2
        os.utime(f, (new_t, new_t))
        h2 = _file_hash(f, mode="content")
        assert h1 == h2

    def test_stat_mode_invalidated_by_touch(self, tmp_path):
        f = _md(tmp_path, "a.md")
        h1 = _file_hash(f, mode="stat")
        new_t = time.time() + 2
        os.utime(f, (new_t, new_t))
        h2 = _file_hash(f, mode="stat")
        assert h1 != h2

    def test_missing_file_returns_empty(self, tmp_path):
        assert _file_hash(str(tmp_path / "ghost.md")) == ""


# ---------------------------------------------------------------------------
# Orphan pruning
# ---------------------------------------------------------------------------

class TestOrphanPrune:
    @pytest.mark.asyncio
    async def test_directory_prunes_deleted_files(self, tmp_path):
        f1 = _md(tmp_path, "a.md", "# A\n\nAlpha.\n")
        f2 = _md(tmp_path, "b.md", "# B\n\nBeta.\n")
        db = str(tmp_path / "idx.db")

        docs = await build_index([str(tmp_path)], db_path=db)
        assert len(docs) == 2

        os.unlink(f1)
        docs2 = await build_index([str(tmp_path)], db_path=db)
        names = {d.doc_name for d in docs2}
        assert "b" in names
        assert "a" not in names
        assert docs2.stats.pruned_paths
        assert any("a.md" in p for p in docs2.stats.pruned_paths)

    @pytest.mark.asyncio
    async def test_single_file_does_not_prune(self, tmp_path):
        f1 = _md(tmp_path, "a.md")
        f2 = _md(tmp_path, "b.md")
        db = str(tmp_path / "idx.db")

        await build_index([f1, f2], db_path=db)
        # Re-index ONLY f2 — f1 must NOT be pruned (single-file scope).
        docs = await build_index([f2], db_path=db)
        assert not docs.stats.pruned_paths

    @pytest.mark.asyncio
    async def test_explicit_prune_true_on_files(self, tmp_path):
        f1 = _md(tmp_path, "a.md")
        f2 = _md(tmp_path, "b.md")
        db = str(tmp_path / "idx.db")

        await build_index([f1, f2], db_path=db)
        docs = await build_index([f2], db_path=db, prune=True)
        assert any("a.md" in p for p in docs.stats.pruned_paths)


# ---------------------------------------------------------------------------
# Node-level diff
# ---------------------------------------------------------------------------

def _doc_with(structure):
    assign_node_ids(structure)
    return Document(
        doc_id="d1", doc_name="d", structure=structure,
        metadata={"source_path": "/synthetic/d1"},
    )


class TestNodeLevelDiff:
    def test_first_index_all_added(self, tmp_path):
        fts = FTS5Index(db_path=str(tmp_path / "x.db"))
        doc = _doc_with([
            {"title": "S1", "text": "alpha", "nodes": []},
            {"title": "S2", "text": "beta", "nodes": []},
        ])
        n = fts.index_document(doc)
        assert n == 2
        assert fts.last_node_diff["added"] == 2
        assert fts.last_node_diff["changed"] == 0
        fts.close()

    def test_text_change_only_touches_one_node(self, tmp_path):
        fts = FTS5Index(db_path=str(tmp_path / "x.db"))
        s = [
            {"title": "S1", "text": "alpha", "nodes": []},
            {"title": "S2", "text": "beta", "nodes": []},
        ]
        fts.index_document(_doc_with(s))

        s2 = [
            {"title": "S1", "text": "alpha", "nodes": []},
            {"title": "S2", "text": "BETA-CHANGED", "nodes": []},
        ]
        fts.index_document(_doc_with(s2))
        diff = fts.last_node_diff
        assert diff["changed"] == 1
        assert diff["kept"] == 1
        assert diff["added"] == 0
        assert diff["removed"] == 0
        fts.close()

    def test_node_addition_detected_as_added(self, tmp_path):
        fts = FTS5Index(db_path=str(tmp_path / "x.db"))
        s = [{"title": "S1", "text": "a", "nodes": []}]
        fts.index_document(_doc_with(s))

        s2 = [
            {"title": "S1", "text": "a", "nodes": []},
            {"title": "S2", "text": "b", "nodes": []},
        ]
        fts.index_document(_doc_with(s2))
        d = fts.last_node_diff
        assert d["added"] == 1
        assert d["kept"] == 1
        fts.close()

    def test_no_change_skips_via_doc_hash(self, tmp_path):
        fts = FTS5Index(db_path=str(tmp_path / "x.db"))
        s = [{"title": "S1", "text": "a", "nodes": []}]
        fts.index_document(_doc_with(s))
        n = fts.index_document(_doc_with(s))
        assert n == 0  # short-circuited by document.index_hash match
        fts.close()


# ---------------------------------------------------------------------------
# File move detection
# ---------------------------------------------------------------------------

class TestMoveDetection:
    @pytest.mark.asyncio
    async def test_rename_does_not_full_reindex(self, tmp_path):
        f1 = _md(tmp_path, "old.md", "# Auth\n\nJWT tokens are great.\n")
        db = str(tmp_path / "idx.db")
        await build_index([f1], db_path=db)

        new_path = str(tmp_path / "renamed.md")
        os.rename(f1, new_path)

        docs = await build_index([str(tmp_path)], db_path=db)
        # No file should be in the indexed_files bucket — the rename was caught
        # by the move detector.
        assert docs.stats.indexed_files == 0
        assert docs.stats.skipped_files == 1

        fts = FTS5Index(db_path=db)
        # source_path was updated to the new location.
        assert fts.get_doc_id_by_source_path(os.path.abspath(new_path)) is not None
        # Old path is gone from index_meta.
        assert fts.get_index_meta(os.path.abspath(f1)) is None
        fts.close()


# ---------------------------------------------------------------------------
# Verify / repair
# ---------------------------------------------------------------------------

class TestVerifyRepair:
    def test_clean_db_is_healthy(self, tmp_path, sample_tree_structure):
        db = str(tmp_path / "x.db")
        doc = Document(doc_id="d1", doc_name="d", structure=sample_tree_structure,
                       metadata={"source_path": ""})
        fts = FTS5Index(db_path=db)
        fts.index_document(doc)
        rep = fts.verify_index()
        assert rep["healthy"] is True
        fts.close()

    def test_orphan_node_detected_and_repaired(self, tmp_path, sample_tree_structure):
        db = str(tmp_path / "x.db")
        doc = Document(doc_id="d1", doc_name="d", structure=sample_tree_structure)
        fts = FTS5Index(db_path=db)
        fts.index_document(doc)

        # Inject an orphan row by directly inserting into nodes.
        fts._conn.execute(
            "INSERT INTO nodes (node_id, doc_id, title, content_hash) VALUES (?, ?, ?, ?)",
            ("orphan-node", "missing-doc", "x", "deadbeef"),
        )
        fts._conn.commit()

        rep = fts.verify_index()
        assert "missing-doc" in rep["orphan_node_doc_ids"]

        removed = fts.repair_index()
        assert removed["orphan_nodes"] >= 1
        rep2 = fts.verify_index()
        assert rep2["healthy"] is True
        fts.close()


# ---------------------------------------------------------------------------
# Batch delete
# ---------------------------------------------------------------------------

class TestBatchDelete:
    def test_delete_documents_batch(self, tmp_path, sample_tree_structure):
        db = str(tmp_path / "x.db")
        fts = FTS5Index(db_path=db)
        ids = []
        for i in range(5):
            doc_id = f"doc{i}"
            doc = Document(doc_id=doc_id, doc_name=doc_id, structure=sample_tree_structure)
            fts.index_document(doc)
            ids.append(doc_id)

        n = fts.delete_documents(ids[:3])
        assert n == 3
        assert fts.is_document_indexed("doc0") is False
        assert fts.is_document_indexed("doc4") is True
        fts.close()

    def test_treesearch_delete_uses_batch(self, tmp_path):
        fa = _md(tmp_path, "a.md", "# A\n\nAlpha.\n")
        fb = _md(tmp_path, "b.md", "# B\n\nBeta.\n")
        fc = _md(tmp_path, "c.md", "# C\n\nGamma.\n")

        ts = TreeSearch(db_path=str(tmp_path / "idx.db"))
        ts.index(fa, fb, fc)
        n = ts.delete(fa, fb, fc)
        assert n == 3
        assert ts.documents == []


# ---------------------------------------------------------------------------
# Schema-version invalidation
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    @pytest.mark.asyncio
    async def test_old_meta_invalidated_by_version_bump(self, tmp_path):
        """Stored hashes lacking the current vX prefix MUST trigger reindex."""
        fa = _md(tmp_path, "a.md")
        db = str(tmp_path / "x.db")

        # First, build normally.
        await build_index([fa], db_path=db)

        # Now corrupt index_meta to simulate an old (v1) hash.
        fts = FTS5Index(db_path=db)
        fts.set_index_meta(os.path.abspath(fa), "1234:567")  # no v-prefix
        fts.close()

        docs = await build_index([fa], db_path=db)
        assert docs.stats.indexed_files == 1
        assert docs.stats.skipped_files == 0


# ---------------------------------------------------------------------------
# Content fingerprint via TreeSearchConfig (opt-in)
# ---------------------------------------------------------------------------

class TestContentFingerprintConfig:
    def test_config_opt_in_changes_default_mode(self, tmp_path):
        cfg = TreeSearchConfig(fingerprint_mode="content")
        set_config(cfg)
        try:
            f = _md(tmp_path, "a.md", "stable\n")
            h = _file_hash(f)
            assert ":content:" in h
        finally:
            reset_config()


# ---------------------------------------------------------------------------
# Watch mode (uses watchdog, optional dep)
# ---------------------------------------------------------------------------

watchdog = pytest.importorskip("watchdog")


class TestWatchMode:
    def test_watch_reindexes_on_modify(self, tmp_path):
        from treesearch.watch import watch as watch_fn

        f1 = _md(tmp_path, "a.md", "# A\n\nfirst.\n")
        db = str(tmp_path / "idx.db")

        events: list[tuple] = []
        cond = threading.Condition()
        ready = threading.Event()

        def _on_event(paths, action):
            with cond:
                events.append((set(paths), action))
                cond.notify_all()

        def _runner():
            try:
                watch_fn([str(tmp_path)], db_path=db, debounce_s=0.2,
                         extensions={".md"}, on_event=_on_event,
                         on_ready=ready.set)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=5.0), "watch failed to start"
            with open(f1, "w") as fh:
                fh.write("# A\n\nupdated.\n")

            with cond:
                cond.wait_for(lambda: any(a == "reindex" for _, a in events), timeout=5.0)
            assert any(a == "reindex" for _, a in events), \
                f"expected reindex event, got: {events}"
        finally:
            _stop_watch_thread(t)


# ---------------------------------------------------------------------------
# Regression: rename / move API consistency (fix 2)
# ---------------------------------------------------------------------------

class TestRenameApiConsistency:
    """After a rename, build_index() / TreeSearch.aindex() must return the
    moved document under its new identity — not silently drop it."""

    @pytest.mark.asyncio
    async def test_treesearch_aindex_keeps_doc_after_rename(self, tmp_path):
        f1 = _md(tmp_path, "old.md", "# Auth\n\nJWT tokens.\n")
        ts = TreeSearch(db_path=str(tmp_path / "idx.db"))
        await ts.aindex(f1)
        assert len(ts.documents) == 1

        new_path = str(tmp_path / "renamed.md")
        os.rename(f1, new_path)
        await ts.aindex(str(tmp_path))

        # Document must NOT disappear from ts.documents.
        assert len(ts.documents) == 1, f"doc dropped after rename: {ts.documents}"
        d = ts.documents[0]
        # Identity follows the file: doc_id and doc_name reflect the new basename.
        assert d.doc_id == "renamed"
        assert d.doc_name == "renamed"
        assert d.metadata["source_path"] == os.path.abspath(new_path)

    @pytest.mark.asyncio
    async def test_rename_updates_doc_id_and_doc_name_in_db(self, tmp_path):
        f1 = _md(tmp_path, "old.md", "# Auth\n\nJWT tokens.\n")
        db = str(tmp_path / "idx.db")
        await build_index([f1], db_path=db)

        os.rename(f1, str(tmp_path / "renamed.md"))
        await build_index([str(tmp_path)], db_path=db)

        fts = FTS5Index(db_path=db)
        try:
            assert fts.is_document_indexed("old") is False
            assert fts.is_document_indexed("renamed") is True
            row = fts._conn.execute(
                "SELECT doc_name, source_path FROM documents WHERE doc_id = ?",
                ("renamed",),
            ).fetchone()
            assert row[0] == "renamed"
            assert row[1] == os.path.abspath(str(tmp_path / "renamed.md"))
        finally:
            fts.close()

    def test_rename_document_collision_returns_false(self, tmp_path, sample_tree_structure):
        """If the target doc_id already belongs to a *different* doc, refuse
        to rename in place — the caller should fall back to a full reindex."""
        db = str(tmp_path / "x.db")
        fts = FTS5Index(db_path=db)
        try:
            doc_a = Document(doc_id="a", doc_name="a", structure=sample_tree_structure,
                             metadata={"source_path": "/tmp/a.md"})
            doc_b = Document(doc_id="b", doc_name="b", structure=sample_tree_structure,
                             metadata={"source_path": "/tmp/b.md"})
            fts.index_document(doc_a)
            fts.index_document(doc_b)
            ok = fts.rename_document("a", "b", "b", "/tmp/new_b.md")
            assert ok is False
            # State unchanged.
            assert fts.is_document_indexed("a") is True
            assert fts.is_document_indexed("b") is True
        finally:
            fts.close()


# ---------------------------------------------------------------------------
# Regression: watch self-loop on DB sidecar files (fix 1)
# ---------------------------------------------------------------------------

class TestWatchSelfLoop:
    def test_db_sidecar_set_includes_all_known_suffixes(self, tmp_path):
        from treesearch.watch import _db_sidecar_set
        db_abs = os.path.abspath(str(tmp_path / "idx.db"))
        excluded = _db_sidecar_set(str(tmp_path / "idx.db"))
        assert db_abs in excluded
        for suffix in ("-wal", "-shm", "-journal", ".lock"):
            assert (db_abs + suffix) in excluded

    def test_watch_does_not_loop_on_db_writes_with_default_extensions(self, tmp_path):
        """`treesearch watch .` (extensions=None, default db_path under cwd)
        used to feedback-loop because every commit touches index.db / -wal /
        -shm and re-triggers reindex. The fix excludes these paths upstream."""
        from treesearch.watch import watch as watch_fn

        f1 = _md(tmp_path, "a.md", "# A\n\nfirst.\n")
        db = str(tmp_path / "idx.db")

        events: list[tuple] = []
        cond = threading.Condition()
        ready = threading.Event()

        def _on_event(paths, action):
            with cond:
                events.append((set(paths), action))
                cond.notify_all()

        def _runner():
            try:
                # extensions=None so the handler would otherwise pick up *.db.
                watch_fn([str(tmp_path)], db_path=db, debounce_s=0.2,
                         extensions=None, on_event=_on_event, on_ready=ready.set)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=5.0), "watch failed to start"
            # Drain the "initial" event from startup-sync.
            with cond:
                cond.wait_for(lambda: any(a == "initial" for _, a in events),
                              timeout=5.0)
            initial_count = len([e for e in events if e[1] == "initial"])

            # Modify the markdown once. We expect EXACTLY one reindex.
            with open(f1, "w") as fh:
                fh.write("# A\n\nupdated.\n")
            with cond:
                cond.wait_for(lambda: any(a == "reindex" for _, a in events),
                              timeout=5.0)
            # Wait for any potential feedback-loop reindexes to surface.
            time.sleep(1.5)

            reindex_count = sum(1 for _, a in events if a == "reindex")
            assert reindex_count == 1, (
                f"expected 1 reindex (no DB-feedback loop), saw "
                f"{reindex_count}; events={events}"
            )
            assert initial_count == 1
        finally:
            _stop_watch_thread(t)


# ---------------------------------------------------------------------------
# Regression: watch initial sync (fix 4) and rename through watch (fix 3)
# ---------------------------------------------------------------------------

class TestWatchInitialSync:
    def test_initial_sync_indexes_existing_files_before_any_event(self, tmp_path):
        """Files present before `watch` starts must already be in the DB by
        the time `on_ready` fires — i.e. searchable without any modification."""
        from treesearch.watch import watch as watch_fn

        f1 = _md(tmp_path, "a.md", "# Alpha\n\nFirst.\n")
        f2 = _md(tmp_path, "b.md", "# Beta\n\nSecond.\n")
        db = str(tmp_path / "idx.db")
        ready = threading.Event()

        def _runner():
            try:
                watch_fn([str(tmp_path)], db_path=db, debounce_s=0.2,
                         extensions={".md"}, on_ready=ready.set)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=5.0), "watch failed to start"
            fts = FTS5Index(db_path=db)
            try:
                assert fts.is_document_indexed("a") is True, \
                    "initial-sync should have indexed pre-existing a.md"
                assert fts.is_document_indexed("b") is True
            finally:
                fts.close()
        finally:
            _stop_watch_thread(t)

    def test_initial_sync_can_be_disabled(self, tmp_path):
        """Power-users can opt out of initial sync via initial_sync=False."""
        from treesearch.watch import watch as watch_fn

        _md(tmp_path, "a.md")
        db = str(tmp_path / "idx.db")
        ready = threading.Event()

        def _runner():
            try:
                watch_fn([str(tmp_path)], db_path=db, debounce_s=0.2,
                         extensions={".md"}, initial_sync=False,
                         on_ready=ready.set)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=5.0)
            # DB should be empty (no initial build was performed).
            fts = FTS5Index(db_path=db)
            try:
                assert fts.is_document_indexed("a") is False
            finally:
                fts.close()
        finally:
            _stop_watch_thread(t)


class TestWatchRenameFastPath:
    def test_rename_through_watch_keeps_identity_consistent(self, tmp_path):
        """Renaming a file while `watch` is running must not duplicate the doc
        or leave it under the old name. With fix 3 (build existing before
        prune missing) the move-detection pre-pass in build_index claims the
        rename and the subsequent prune of the old path becomes a no-op."""
        from treesearch.watch import watch as watch_fn

        f1 = _md(tmp_path, "old.md", "# Auth\n\nJWT tokens are great.\n")
        db = str(tmp_path / "idx.db")
        ready = threading.Event()
        events: list[tuple] = []
        cond = threading.Condition()

        def _on_event(paths, action):
            with cond:
                events.append((set(paths), action))
                cond.notify_all()

        def _runner():
            try:
                watch_fn([str(tmp_path)], db_path=db, debounce_s=0.3,
                         extensions={".md"}, on_event=_on_event,
                         on_ready=ready.set)
            except KeyboardInterrupt:
                pass

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        try:
            assert ready.wait(timeout=5.0)
            # Wait for initial sync to complete.
            with cond:
                cond.wait_for(lambda: any(a == "initial" for _, a in events),
                              timeout=5.0)

            new_path = str(tmp_path / "renamed.md")
            os.rename(f1, new_path)

            # Wait until at least one reindex batch has fired.
            with cond:
                cond.wait_for(lambda: any(a == "reindex" for _, a in events),
                              timeout=5.0)
            # Allow any follow-up prune batch to settle.
            time.sleep(0.8)

            fts = FTS5Index(db_path=db)
            try:
                assert fts.is_document_indexed("old") is False, \
                    "old doc_id should be gone after rename"
                assert fts.is_document_indexed("renamed") is True, \
                    "new doc_id should exist after rename"
                row = fts._conn.execute(
                    "SELECT source_path FROM documents WHERE doc_id = ?",
                    ("renamed",),
                ).fetchone()
                assert row[0] == os.path.abspath(new_path)
            finally:
                fts.close()
        finally:
            _stop_watch_thread(t)


def _stop_watch_thread(t: threading.Thread) -> None:
    """Inject KeyboardInterrupt into a running watch() thread and join it."""
    if not t.is_alive():
        return
    import ctypes
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(t.ident), ctypes.py_object(KeyboardInterrupt)
    )
    t.join(timeout=5.0)
