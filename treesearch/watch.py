# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: File-system watch mode — push-based incremental indexing.

Optional dependency: ``pip install pytreesearch[watch]`` installs ``watchdog``.

Public entry point::

    from treesearch.watch import watch

    watch(["docs/", "src/"], db_path="./index.db", debounce_s=0.5)

The function blocks until ``KeyboardInterrupt``. On each batch of file system
events it re-runs ``build_index(...)`` for the affected paths.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Filesystem events worth reacting to. Move/rename are split by watchdog into
# (deleted_old, created_new); both are dispatched, and build_index handles
# move-detection by content fingerprint.
_RELEVANT_EVENTS = {"created", "modified", "deleted", "moved"}

# SQLite sidecar suffixes that may appear next to the watched DB.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal", ".lock")


def _ensure_watchdog():
    """Import watchdog lazily; raise an actionable error if missing."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
        return FileSystemEventHandler, Observer
    except ImportError as e:
        raise ImportError(
            "Watch mode requires the optional 'watchdog' dependency.\n"
            "Install it with:  pip install 'pytreesearch[watch]'"
        ) from e


def _db_sidecar_set(db_path: str) -> set[str]:
    """Absolute paths that should never be treated as indexable input.

    Includes the DB itself plus the standard SQLite sidecar files. Required
    so that watching ``--paths .`` with the default ``./index.db`` does not
    enter a self-triggering loop (every commit rewrites -wal/-shm).
    """
    db_abs = os.path.abspath(db_path)
    paths = {db_abs}
    for suffix in _DB_SIDECAR_SUFFIXES:
        paths.add(db_abs + suffix)
    return paths


class _DebouncedReindexer:
    """Coalesce a stream of file-system events into batched reindex calls.

    `notify(path)` is called from watchdog's I/O thread. We accumulate paths
    in a set and arm a single timer; when the timer fires after `debounce_s`
    of quiet, we hand the batch to the reindex callback (running it in a
    background asyncio loop dedicated to this watcher).
    """

    def __init__(
        self,
        on_batch,
        debounce_s: float = 0.5,
    ):
        self._on_batch = on_batch
        self._debounce_s = debounce_s
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._timer: Optional[threading.Timer] = None

    def notify(self, path: str) -> None:
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_s, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            batch = self._pending
            self._pending = set()
            self._timer = None
        if not batch:
            return
        try:
            self._on_batch(sorted(batch))
        except Exception:
            logger.exception("watch: reindex batch failed")

    def shutdown(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def watch(
    paths: list[str] | str,
    *,
    db_path: str = "./index.db",
    debounce_s: float = 0.5,
    extensions: Optional[Iterable[str]] = None,
    poll_seconds: Optional[float] = None,
    initial_sync: bool = True,
    on_event=None,
    on_ready=None,
    **build_index_kwargs,
) -> None:
    """Watch *paths* and rebuild the index incrementally on changes.

    Args:
        paths: directories or files to watch (a single string is accepted).
            Directories are watched recursively.
        db_path: SQLite DB path; defaults to ``./index.db``.
        debounce_s: coalesce events that arrive within this many seconds
            (default 0.5). Larger values trade latency for fewer rebuilds.
        extensions: optional whitelist (e.g. ``{".md", ".py"}``). If provided,
            events for other file types are ignored.
        poll_seconds: if set, use the polling observer instead of the OS-native
            backend. Useful on network filesystems where inotify/FSEvents
            does not fire reliably. ``None`` (default) → native backend.
        initial_sync: if True (default), run a full ``build_index(prune=True)``
            over ``paths`` once before the observer starts so existing files
            are searchable immediately and stale docs get pruned.
        on_event: optional callable ``(set[str], action: str) -> None`` invoked
            after each successful batch (mostly for tests). ``action`` is
            ``"reindex"``, ``"prune"``, or ``"initial"``.
        on_ready: optional zero-arg callable invoked once the observer is
            running and (if requested) the initial sync has completed. Useful
            for tests that need to wait until events will be picked up.
        **build_index_kwargs: forwarded to ``build_index`` (e.g. ``force=False``,
            ``if_add_node_summary=...``).

    Blocks until ``KeyboardInterrupt``.
    """
    if isinstance(paths, str):
        paths = [paths]
    paths = [os.path.abspath(p) for p in paths]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"watch: path does not exist: {p}")

    FileSystemEventHandler, Observer = _ensure_watchdog()

    if poll_seconds is not None:
        from watchdog.observers.polling import PollingObserver
        observer = PollingObserver(timeout=poll_seconds)
    else:
        observer = Observer()

    ext_set = {e.lower() for e in extensions} if extensions else None
    db_excluded = _db_sidecar_set(db_path)

    # Build_index requires an event loop. Spin up a dedicated one in a
    # background thread so the main thread can run the watchdog loop.
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(
        target=_run_loop_forever, args=(loop,), daemon=True
    )
    loop_thread.start()

    from .indexer import build_index
    from .fts import FTS5Index

    def _run_build(targets: list[str], *, prune: bool):
        future = asyncio.run_coroutine_threadsafe(
            build_index(targets, db_path=db_path, prune=prune, **build_index_kwargs),
            loop,
        )
        return future.result()

    # ---------------- Initial sync ----------------
    # Without this, a fresh DB started under `treesearch watch .` would only
    # learn about files that get modified *after* startup. We pay one full
    # build up-front so the index is consistent the moment we start watching.
    if initial_sync:
        try:
            t0 = time.monotonic()
            init_docs = _run_build(paths, prune=True)
            logger.info(
                "watch: initial sync indexed %d document(s) in %.2fs",
                len(init_docs), time.monotonic() - t0,
            )
            if on_event:
                on_event(set(paths), "initial")
        except Exception:
            logger.exception("watch: initial sync failed")
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5.0)
            loop.close()
            raise

    def _do_reindex(batch: list[str]) -> None:
        # Filter out our own DB sidecar files (defense in depth — _Handler
        # already drops them, but a manual `notify()` from a test might not).
        batch = [p for p in batch if p not in db_excluded]
        if not batch:
            return

        existing = [p for p in batch if os.path.exists(p)]
        missing = [p for p in batch if not os.path.exists(p)]

        # IMPORTANT: index existing files FIRST. build_index's pre-pass
        # uses content fingerprints from `index_meta` to reclaim docs whose
        # source_path moved (rename = deleted_old + created_new). If we
        # pruned `missing` first, the old fingerprint would be gone and the
        # rename would degrade to "delete old doc + create new doc".
        if existing:
            t0 = time.monotonic()
            try:
                docs = _run_build(existing, prune=False)
                logger.info(
                    "watch: reindexed %d file(s) in %.2fs",
                    len(docs), time.monotonic() - t0,
                )
            except Exception:
                logger.exception("watch: build_index failed for %s", existing)
                return
            if on_event:
                on_event(set(existing), "reindex")

        if missing:
            fts = FTS5Index(db_path=db_path)
            try:
                ids: list[str] = []
                # After the build above, any rename-source path will have been
                # remapped onto the new file by build_index.rename_document(),
                # so its lookup here returns None and we leave the doc alone.
                for mp in missing:
                    doc_id = fts.get_doc_id_by_source_path(mp)
                    if doc_id:
                        ids.append(doc_id)
                if ids:
                    fts.delete_documents(ids)
                    logger.info("watch: pruned %d missing file(s)", len(ids))
            finally:
                fts.close()
            if on_event:
                on_event(set(missing), "prune")

    debouncer = _DebouncedReindexer(_do_reindex, debounce_s=debounce_s)

    class _Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.event_type not in _RELEVANT_EVENTS:
                return
            if event.is_directory:
                return

            abs_src = os.path.abspath(event.src_path)
            if abs_src in db_excluded:
                return
            src_ok = ext_set is None or os.path.splitext(abs_src)[1].lower() in ext_set
            if src_ok:
                debouncer.notify(abs_src)

            # Move events also carry a destination path; treat it independently.
            if event.event_type == "moved":
                abs_dest = os.path.abspath(event.dest_path)
                if abs_dest in db_excluded:
                    return
                if ext_set is None or os.path.splitext(abs_dest)[1].lower() in ext_set:
                    debouncer.notify(abs_dest)

    handler = _Handler()
    for p in paths:
        observer.schedule(handler, p, recursive=os.path.isdir(p))

    observer.start()
    logger.info("watch: monitoring %s (debounce=%.2fs)", paths, debounce_s)
    if on_ready:
        on_ready()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("watch: shutting down")
    finally:
        observer.stop()
        observer.join(timeout=5.0)
        debouncer.shutdown()
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5.0)
        loop.close()


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()
