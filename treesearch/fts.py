# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SQLite FTS5 full-text search engine for tree-structured documents.

Single-file storage: tree structures, FTS5 indexes, and incremental metadata
are all stored in one SQLite database (.db file).

Architecture: "SQLite FTS5 + Producer-Consumer + LLM Query Generation"
  - Deferred indexing via WAL mode solves real-time freshness
  - Local SQL execution handles aggregation needs
  - FTS5 inverted index guarantees retrieval performance
  - Tree structure persistence in `documents.structure_json` column

Key features:
  - WAL mode for concurrent read/write
  - Lazy indexing: nodes are inserted on demand, not precomputed
  - MD-aware schema: front_matter, title, summary, body, code_blocks
  - CJK-aware tokenizer: jieba segmentation only when Chinese text is detected
  - Implements PreFilter protocol for seamless integration with search()
  - Hierarchical field boosting via FTS5 column weighting
  - LLM-generated FTS5 query expressions (AND/OR/NOT/NEAR)
"""
import hashlib
import json
import logging
import os
import re
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# FTS5 column weights: title > body > summary > code
# Used in bm25() ranking function
_DEFAULT_WEIGHTS = {
    "title": 5.0,
    "summary": 2.0,
    "body": 10.0,
    "code_blocks": 1.0,
    "front_matter": 2.0,
}


# ---------------------------------------------------------------------------
# Markdown structure parser
# ---------------------------------------------------------------------------

_RE_FRONT_MATTER = re.compile(r"^---\s*\n(.*?\n)---\s*\n", re.DOTALL)
_RE_CODE_BLOCK = re.compile(r"```[\w]*\n(.*?)```", re.DOTALL)
_RE_HEADING_LINE = re.compile(r"^#{1,6}\s+")


def parse_md_node_text(text: str) -> dict:
    """Parse a node's text into MD-aware structured fields.

    Returns:
        {
            "front_matter": str,  # YAML front matter (if present)
            "body": str,          # main text (headings, paragraphs)
            "code_blocks": str,   # concatenated code blocks
        }
    """
    if not text:
        return {"front_matter": "", "body": "", "code_blocks": ""}

    front_matter = ""
    remaining = text

    # Extract front matter
    fm_match = _RE_FRONT_MATTER.match(text)
    if fm_match:
        front_matter = fm_match.group(1).strip()
        remaining = text[fm_match.end():]

    # Extract code blocks
    code_parts = []
    def _replace_code(m):
        code_parts.append(m.group(1).strip())
        return ""  # remove from body
    body = _RE_CODE_BLOCK.sub(_replace_code, remaining)
    code_blocks = "\n".join(code_parts)

    # Clean body: collapse blank lines
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return {
        "front_matter": front_matter,
        "body": body,
        "code_blocks": code_blocks,
    }


# ---------------------------------------------------------------------------
# Tokenizer for FTS5 (Chinese/English)
# ---------------------------------------------------------------------------

_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _tokenize_for_fts(text: str) -> str:
    """Tokenize text for FTS5 indexing. Space-separated tokens.

    Only uses jieba segmentation when Chinese (CJK) characters are detected.
    For pure English/non-CJK text, relies on FTS5's built-in unicode61 tokenizer
    (no jieba overhead).
    """
    if not text or not text.strip():
        return ""
    if _RE_HAS_CJK.search(text):
        from .rank_bm25 import tokenize
        tokens = tokenize(text)
        return " ".join(tokens)
    # English / non-CJK: return as-is, FTS5 unicode61 handles tokenization
    return text


# FTS5 operators that should NOT be tokenized
_FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}

# Characters that have special meaning in FTS5 query syntax
_RE_FTS5_SPECIAL = re.compile(r'[?*"()\^{}]')


def _tokenize_fts_expression(expr: str) -> str:
    """Tokenize terms in an FTS5 expression while preserving operators.

    Raw FTS5 expressions like ``"machine AND learning"`` must have their
    terms tokenized to match the indexed content, but FTS5
    operators (AND, OR, NOT, NEAR) must remain untouched.

    Only applies jieba segmentation when CJK characters are detected.
    """
    parts = expr.split()
    result = []
    for part in parts:
        if part.upper() in _FTS5_OPERATORS:
            result.append(part.upper())
        else:
            tokenized = _tokenize_for_fts(part)
            if tokenized.strip():
                result.append(tokenized.strip())
    return " ".join(result)


# ---------------------------------------------------------------------------
# FTS5 Index Engine
# ---------------------------------------------------------------------------

class FTS5Index:
    """SQLite FTS5 full-text search index for tree-structured documents.

    Features:
      - WAL journal mode for concurrent read/write
      - MD-aware columns: title, summary, body, code_blocks, front_matter
      - Hierarchical column weighting via bm25() rank function
      - Deferred indexing: call index_document() when ready
      - Implements PreFilter protocol: score_nodes(query, doc_id)
      - Supports FTS5 query syntax: AND, OR, NOT, NEAR, phrase "..."
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        weights: Optional[dict] = None,
    ):
        """
        Args:
            db_path: path to SQLite database file. None = in-memory.
            weights: column weight overrides for bm25() ranking.
        """
        self._db_path = db_path or ":memory:"
        self._weights = {**_DEFAULT_WEIGHTS, **(weights or {})}
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with FTS5 virtual table."""
        if self._db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self._db_path)), exist_ok=True)

        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Metadata table for nodes (structured fields for filtering)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                depth INTEGER DEFAULT 0,
                line_start INTEGER,
                line_end INTEGER,
                parent_node_id TEXT,
                content_hash TEXT,
                PRIMARY KEY (doc_id, node_id)
            )
        """)

        # FTS5 virtual table with content sync
        # tokenize='unicode61' handles basic multi-language, but we pre-tokenize
        # Chinese text with jieba and store space-separated tokens
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_nodes USING fts5(
                node_id UNINDEXED,
                doc_id UNINDEXED,
                title,
                summary,
                body,
                code_blocks,
                front_matter,
                tokenize='unicode61 remove_diacritics 2'
            )
        """)

        # Document metadata table (also stores tree structure for persistence)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_name TEXT DEFAULT '',
                doc_description TEXT DEFAULT '',
                source_path TEXT DEFAULT '',
                source_type TEXT DEFAULT '',
                structure_json TEXT DEFAULT '',
                node_count INTEGER DEFAULT 0,
                index_hash TEXT
            )
        """)

        # Incremental index metadata (replaces _index_meta.json)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS index_meta (
                source_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL
            )
        """)

        # Performance indexes for large-scale document sets (10k+ docs)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_doc_id ON nodes (doc_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents (source_path)"
        )

        self._conn.commit()

    @property
    def db_path(self) -> str:
        return self._db_path

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()

    # -------------------------------------------------------------------
    # Indexing (Producer side)
    # -------------------------------------------------------------------

    def index_document(self, document, force: bool = False) -> int:
        """Index all nodes from a Document into FTS5.

        Args:
            document: Document object with structure tree
            force: re-index even if content hash matches

        Returns:
            number of nodes indexed
        """
        from .tree import flatten_tree

        # Compute content hash for incremental check
        content_str = json.dumps(document.structure, ensure_ascii=False, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()

        # Check if already indexed
        if not force:
            row = self._conn.execute(
                "SELECT index_hash FROM documents WHERE doc_id = ?",
                (document.doc_id,),
            ).fetchone()
            if row and row[0] == content_hash:
                logger.debug("Document %s already indexed (hash match), skipping", document.doc_id)
                return 0

        # Clear old entries for this document
        # FTS5 UNINDEXED columns can't be indexed, so use rowid-based delete for performance
        old_rowids = self._conn.execute(
            "SELECT rowid FROM fts_nodes WHERE doc_id = ?", (document.doc_id,)
        ).fetchall()
        if old_rowids:
            placeholders = ",".join("?" for _ in old_rowids)
            self._conn.execute(
                f"DELETE FROM fts_nodes WHERE rowid IN ({placeholders})",
                [r[0] for r in old_rowids],
            )
        self._conn.execute("DELETE FROM nodes WHERE doc_id = ?", (document.doc_id,))

        # Flatten tree and index each node
        all_nodes = flatten_tree(document.structure)
        count = 0

        # Build parent map for depth calculation
        parent_map: dict[str, Optional[str]] = {}
        depth_map: dict[str, int] = {}

        def _scan(structure, parent_id=None, depth=0):
            if isinstance(structure, list):
                for item in structure:
                    _scan(item, parent_id, depth)
            elif isinstance(structure, dict):
                nid = structure.get("node_id", "")
                parent_map[nid] = parent_id
                depth_map[nid] = depth
                for child in structure.get("nodes", []):
                    _scan(child, nid, depth + 1)

        _scan(document.structure)

        for node in all_nodes:
            nid = node.get("node_id", "")
            if not nid:
                continue

            title = node.get("title", "")
            summary = node.get("summary", node.get("prefix_summary", ""))
            text = node.get("text", "")
            depth = depth_map.get(nid, 0)

            # Parse MD structure
            parsed = parse_md_node_text(text)

            # Pre-tokenize for CJK support
            title_tok = _tokenize_for_fts(title)
            summary_tok = _tokenize_for_fts(summary)
            body_tok = _tokenize_for_fts(parsed["body"])
            code_tok = _tokenize_for_fts(parsed["code_blocks"])
            fm_tok = _tokenize_for_fts(parsed["front_matter"])

            # Insert into metadata table
            self._conn.execute(
                """INSERT OR REPLACE INTO nodes
                   (node_id, doc_id, title, summary, depth, line_start, line_end, parent_node_id, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    nid, document.doc_id, title, summary, depth,
                    node.get("line_start"), node.get("line_end"),
                    parent_map.get(nid), hashlib.md5(text.encode()).hexdigest()[:16],
                ),
            )

            # Insert into FTS5 index
            self._conn.execute(
                """INSERT INTO fts_nodes
                   (node_id, doc_id, title, summary, body, code_blocks, front_matter)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (nid, document.doc_id, title_tok, summary_tok, body_tok, code_tok, fm_tok),
            )
            count += 1

        # Update document metadata (including tree structure)
        structure_json = json.dumps(document.structure, ensure_ascii=False)
        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, doc_name, doc_description, source_path, source_type, structure_json, node_count, index_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document.doc_id, document.doc_name, document.doc_description,
                document.metadata.get("source_path", ""),
                document.source_type,
                structure_json,
                count, content_hash,
            ),
        )

        self._conn.commit()
        logger.info("FTS5 indexed document %s: %d nodes", document.doc_id, count)
        return count

    def index_documents(self, documents: list, force: bool = False) -> int:
        """Batch index multiple documents.

        Returns:
            total number of nodes indexed
        """
        total = 0
        for doc in documents:
            total += self.index_document(doc, force=force)
        return total

    # -------------------------------------------------------------------
    # Search (Consumer side)
    # -------------------------------------------------------------------

    def search(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 20,
        fts_expression: Optional[str] = None,
    ) -> list[dict]:
        """Search nodes using FTS5 BM25 ranking.

        Args:
            query: natural language query (will be tokenized)
            doc_id: optional filter by document
            top_k: max results
            fts_expression: raw FTS5 query expression (overrides query tokenization).
                            Supports AND, OR, NOT, NEAR, phrases.

        Returns:
            list of {node_id, doc_id, title, summary, fts_score, depth}
        """
        if fts_expression:
            # Tokenize terms in the expression while preserving FTS5 operators
            match_expr = _tokenize_fts_expression(fts_expression)
        else:
            # Tokenize query for FTS5 matching
            tokens = _tokenize_for_fts(query)
            if not tokens.strip():
                return []
            words = tokens.split()
            # Strip FTS5 special characters from each token
            clean_words = []
            for w in words:
                cleaned = _RE_FTS5_SPECIAL.sub("", w).strip()
                if cleaned:
                    clean_words.append(cleaned)
            if not clean_words:
                return []
            if len(clean_words) > 1:
                # Use OR for recall; FTS5 bm25() naturally ranks multi-term matches higher
                match_expr = " OR ".join(clean_words)
            else:
                match_expr = clean_words[0]

        # Phase 1: phrase boosting for multi-word queries
        # Run a separate phrase match query and record which nodes get a boost
        phrase_boost_nids: set[str] = set()
        if not fts_expression and len(query.split()) >= 2:
            # Build phrase expression from original (unstemmed) query words
            raw_words = [w.lower().strip() for w in re.split(r'\W+', query) if w.strip() and len(w.strip()) > 2]
            if len(raw_words) >= 2:
                # Try phrase match: "word1 word2 ..."
                phrase_expr = '"' + ' '.join(raw_words) + '"'
                try:
                    if doc_id:
                        phrase_rows = self._conn.execute(
                            f"SELECT f.node_id FROM fts_nodes f WHERE fts_nodes MATCH ? AND f.doc_id = ? LIMIT 50",
                            (phrase_expr, doc_id),
                        ).fetchall()
                    else:
                        phrase_rows = self._conn.execute(
                            f"SELECT f.node_id FROM fts_nodes f WHERE fts_nodes MATCH ? LIMIT 50",
                            (phrase_expr,),
                        ).fetchall()
                    phrase_boost_nids = {r[0] for r in phrase_rows}
                except sqlite3.OperationalError:
                    pass  # phrase query syntax error, skip boost

        # Build SQL with column weights for bm25()
        # bm25(fts_nodes, w1, w2, w3, w4, w5) where weights correspond to:
        # node_id(UNINDEXED), doc_id(UNINDEXED), title, summary, body, code_blocks, front_matter
        w = self._weights
        weight_args = f"{w['title']}, {w['summary']}, {w['body']}, {w['code_blocks']}, {w['front_matter']}"

        if doc_id:
            sql = f"""
                SELECT f.node_id, f.doc_id,
                       n.title, n.summary, n.depth,
                       bm25(fts_nodes, {weight_args}) AS rank_score
                FROM fts_nodes f
                JOIN nodes n ON f.node_id = n.node_id AND f.doc_id = n.doc_id
                WHERE fts_nodes MATCH ?
                  AND f.doc_id = ?
                ORDER BY rank_score
                LIMIT ?
            """
            params = (match_expr, doc_id, top_k)
        else:
            sql = f"""
                SELECT f.node_id, f.doc_id,
                       n.title, n.summary, n.depth,
                       bm25(fts_nodes, {weight_args}) AS rank_score
                FROM fts_nodes f
                JOIN nodes n ON f.node_id = n.node_id AND f.doc_id = n.doc_id
                WHERE fts_nodes MATCH ?
                ORDER BY rank_score
                LIMIT ?
            """
            params = (match_expr, top_k)

        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 query error: %s, query=%r", e, match_expr)
            rows = []

        results = []
        for row in rows:
            # bm25() returns negative values (lower = more relevant)
            fts_score = -row[5] if row[5] else 0.0
            # Apply phrase boost: nodes matching exact phrase get 50% score bonus
            if row[0] in phrase_boost_nids:
                fts_score *= 1.5
            results.append({
                "node_id": row[0],
                "doc_id": row[1],
                "title": row[2],
                "summary": row[3],
                "depth": row[4],
                "fts_score": round(fts_score, 6),
            })

        # Re-sort after phrase boosting
        if phrase_boost_nids:
            results.sort(key=lambda x: -x["fts_score"])

        return results

    def search_with_aggregation(
        self,
        query: str,
        group_by_doc: bool = True,
        top_k: int = 20,
    ) -> list[dict]:
        """Search with SQL aggregation capabilities.

        Returns per-document aggregated results: total hits, max score, avg score.
        """
        if not group_by_doc:
            return self.search(query, top_k=top_k)

        # Two-step: first get all matched nodes with scores, then aggregate in Python
        results = self.search(query, top_k=200)
        if not results:
            return []

        doc_agg: dict[str, dict] = {}
        for r in results:
            did = r["doc_id"]
            if did not in doc_agg:
                # Fetch doc_name
                row = self._conn.execute(
                    "SELECT doc_name FROM documents WHERE doc_id = ?", (did,)
                ).fetchone()
                doc_agg[did] = {
                    "doc_id": did,
                    "doc_name": row[0] if row else "",
                    "hit_count": 0,
                    "best_score": 0.0,
                    "total_score": 0.0,
                }
            doc_agg[did]["hit_count"] += 1
            doc_agg[did]["best_score"] = max(doc_agg[did]["best_score"], r["fts_score"])
            doc_agg[did]["total_score"] += r["fts_score"]

        agg_results = []
        for agg in doc_agg.values():
            agg["avg_score"] = round(agg["total_score"] / agg["hit_count"], 6)
            agg["best_score"] = round(agg["best_score"], 6)
            del agg["total_score"]
            agg_results.append(agg)

        agg_results.sort(key=lambda x: -x["best_score"])
        return agg_results[:top_k]

    def score_nodes(self, query: str, doc_id: str, ancestor_decay: float = 0.6) -> dict[str, float]:
        """PreFilter protocol: return {node_id: score} for search() integration.

        This allows FTS5Index to be used as a drop-in replacement for
        NodeBM25Index in the search pipeline.

        Includes:
        - Ancestor score propagation: parent nodes inherit child scores
        """
        results = self.search(query, doc_id=doc_id, top_k=200)

        if not results:
            return {}

        # Normalize scores to [0, 1] range
        max_score = max(r["fts_score"] for r in results) if results else 1.0
        if max_score <= 0:
            max_score = 1.0

        scores = {
            r["node_id"]: r["fts_score"] / max_score
            for r in results
        }

        # Ancestor propagation using parent_node_id from nodes table
        if ancestor_decay > 0:
            rows = self._conn.execute(
                "SELECT node_id, parent_node_id FROM nodes WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            parent_map = {r[0]: r[1] for r in rows if r[1]}
            children_map: dict[str, list[str]] = {}
            for nid, pid in parent_map.items():
                children_map.setdefault(pid, []).append(nid)

            # Bottom-up: propagate max child score to parent
            # Single pass is enough for tree structures
            changed = True
            while changed:
                changed = False
                for pid, cids in children_map.items():
                    child_scores = [scores.get(c, 0.0) for c in cids]
                    if not child_scores:
                        continue
                    bonus = ancestor_decay * max(child_scores)
                    old = scores.get(pid, 0.0)
                    new_score = old + bonus
                    if new_score > old + 1e-9:
                        scores[pid] = new_score
                        changed = False  # single pass is enough for tree

            # Re-normalize to [0, 1] after ancestor propagation
            final_max = max(scores.values()) if scores else 1.0
            if final_max > 1.0:
                scores = {nid: s / final_max for nid, s in scores.items()}

        return {nid: round(s, 6) for nid, s in scores.items()}

    # -------------------------------------------------------------------
    # Document persistence (tree structure storage)
    # -------------------------------------------------------------------

    def save_document(self, document) -> None:
        """Save/update a Document's tree structure into the DB.

        This persists the tree structure so that JSON files are no longer needed.
        FTS indexing is NOT performed here — call index_document() separately.
        """
        structure_json = json.dumps(document.structure, ensure_ascii=False)
        content_hash = hashlib.md5(structure_json.encode()).hexdigest()
        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (doc_id, doc_name, doc_description, source_path, source_type, structure_json, node_count, index_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document.doc_id, document.doc_name, document.doc_description,
                document.metadata.get("source_path", ""),
                document.source_type,
                structure_json,
                len(list(self._iter_flat_nodes(document.structure))),
                content_hash,
            ),
        )
        self._conn.commit()

    @staticmethod
    def _iter_flat_nodes(structure):
        """Iterate over all nodes in a tree structure."""
        if isinstance(structure, dict):
            yield structure
            for child in structure.get("nodes", []):
                yield from FTS5Index._iter_flat_nodes(child)
        elif isinstance(structure, list):
            for item in structure:
                yield from FTS5Index._iter_flat_nodes(item)

    def load_document(self, doc_id: str):
        """Load a single Document from the DB by doc_id.

        Returns:
            Document object, or None if not found.
        """
        from .tree import Document
        row = self._conn.execute(
            "SELECT doc_id, doc_name, doc_description, source_path, source_type, structure_json FROM documents WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        if not row:
            return None
        structure = json.loads(row[5]) if row[5] else []
        return Document(
            doc_id=row[0],
            doc_name=row[1],
            structure=structure,
            doc_description=row[2] or "",
            metadata={"source_path": row[3] or ""},
            source_type=row[4] or "",
        )

    def load_all_documents(self) -> list:
        """Load all Documents stored in the DB.

        Returns:
            List of Document objects.
        """
        from .tree import Document
        rows = self._conn.execute(
            "SELECT doc_id, doc_name, doc_description, source_path, source_type, structure_json FROM documents ORDER BY doc_id"
        ).fetchall()
        documents = []
        for row in rows:
            structure = json.loads(row[5]) if row[5] else []
            documents.append(Document(
                doc_id=row[0],
                doc_name=row[1],
                structure=structure,
                doc_description=row[2] or "",
                metadata={"source_path": row[3] or ""},
                source_type=row[4] or "",
            ))
        return documents

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and all its indexed nodes from the DB."""
        # FTS5 UNINDEXED columns can't be indexed, so use rowid-based delete for performance
        old_rowids = self._conn.execute(
            "SELECT rowid FROM fts_nodes WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        if old_rowids:
            placeholders = ",".join("?" for _ in old_rowids)
            self._conn.execute(
                f"DELETE FROM fts_nodes WHERE rowid IN ({placeholders})",
                [r[0] for r in old_rowids],
            )
        self._conn.execute("DELETE FROM nodes WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    # -------------------------------------------------------------------
    # Index metadata (replaces _index_meta.json)
    # -------------------------------------------------------------------

    def get_index_meta(self, source_path: str) -> Optional[str]:
        """Get the stored file hash for a source path.

        Returns:
            File hash string, or None if not tracked.
        """
        row = self._conn.execute(
            "SELECT file_hash FROM index_meta WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        return row[0] if row else None

    def set_index_meta(self, source_path: str, file_hash: str) -> None:
        """Store/update the file hash for a source path."""
        self._conn.execute(
            "INSERT OR REPLACE INTO index_meta (source_path, file_hash) VALUES (?, ?)",
            (source_path, file_hash),
        )
        self._conn.commit()

    def get_all_index_meta(self) -> dict[str, str]:
        """Get all stored file hashes.

        Returns:
            Dict mapping source_path -> file_hash.
        """
        rows = self._conn.execute("SELECT source_path, file_hash FROM index_meta").fetchall()
        return {r[0]: r[1] for r in rows}

    # -------------------------------------------------------------------
    # FTS5 query expression builder
    # -------------------------------------------------------------------

    @staticmethod
    def build_fts_expression(
        keywords: list[str],
        operator: str = "OR",
        column: Optional[str] = None,
        near_distance: Optional[int] = None,
    ) -> str:
        """Build FTS5 match expression from keyword list.

        Args:
            keywords: list of search terms
            operator: "AND" | "OR" | "NOT" (first keyword AND NOT others)
            column: optional column filter (e.g. "title", "body")
            near_distance: if set, uses NEAR(kw1 kw2, N) syntax

        Returns:
            FTS5 match expression string

        Examples:
            build_fts_expression(["python", "async"], "AND")
            -> "python AND async"

            build_fts_expression(["machine", "learning"], column="title")
            -> "title : (machine OR learning)"

            build_fts_expression(["deep", "learning"], near_distance=5)
            -> 'NEAR(deep learning, 5)'
        """
        if not keywords:
            return ""

        # Escape FTS5 special characters in keywords
        safe_kws = []
        for kw in keywords:
            # Remove FTS5 operators that could break syntax
            cleaned = re.sub(r'["\(\)\*]', '', kw.strip())
            if cleaned:
                # Tokenize for CJK
                tokenized = _tokenize_for_fts(cleaned)
                if tokenized.strip():
                    safe_kws.append(tokenized.strip())

        if not safe_kws:
            return ""

        if near_distance is not None and len(safe_kws) >= 2:
            all_tokens = " ".join(safe_kws)
            expr = f"NEAR({all_tokens}, {near_distance})"
        elif operator == "NOT" and len(safe_kws) >= 2:
            expr = f"{safe_kws[0]} NOT {' NOT '.join(safe_kws[1:])}"
        else:
            expr = f" {operator} ".join(safe_kws)

        if column:
            expr = f"{column} : ({expr})"

        return expr

    # -------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------

    def optimize(self) -> None:
        """Run FTS5 merge optimization for better query performance."""
        try:
            self._conn.execute("INSERT INTO fts_nodes(fts_nodes) VALUES('optimize')")
            self._conn.commit()
            logger.info("FTS5 index optimized")
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 optimize failed: %s", e)

    def rebuild(self) -> None:
        """Rebuild FTS5 index from scratch."""
        try:
            self._conn.execute("INSERT INTO fts_nodes(fts_nodes) VALUES('rebuild')")
            self._conn.commit()
            logger.info("FTS5 index rebuilt")
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 rebuild failed: %s", e)

    def get_stats(self) -> dict:
        """Get index statistics."""
        doc_count = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        node_count = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        return {
            "db_path": self._db_path,
            "document_count": doc_count,
            "node_count": node_count,
        }

    def clear(self) -> None:
        """Clear all indexed data."""
        self._conn.execute("DELETE FROM fts_nodes")
        self._conn.execute("DELETE FROM nodes")
        self._conn.execute("DELETE FROM documents")
        self._conn.execute("DELETE FROM index_meta")
        self._conn.commit()

    def is_document_indexed(self, doc_id: str) -> bool:
        """Check if a document is already indexed."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# Global FTS5 index singleton
# ---------------------------------------------------------------------------

_global_fts: Optional[FTS5Index] = None


def get_fts_index(db_path: Optional[str] = None) -> FTS5Index:
    """Get or create the global FTS5 index.

    Args:
        db_path: database path. If None, uses in-memory database.
                 Pass a file path for persistent indexing across sessions.
    """
    global _global_fts
    if _global_fts is None:
        _global_fts = FTS5Index(db_path=db_path)
    return _global_fts


def set_fts_index(index: FTS5Index) -> None:
    """Set the global FTS5 index instance."""
    global _global_fts
    _global_fts = index


def reset_fts_index() -> None:
    """Close and reset the global FTS5 index."""
    global _global_fts
    if _global_fts is not None:
        _global_fts.close()
        _global_fts = None
