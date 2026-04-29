# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified configuration for TreeSearch.

Priority (high -> low):
    1. set_config(TreeSearchConfig(...))
    2. Environment variables
    3. Built-in defaults

Environment variables:
    Tokenizer: TREESEARCH_CJK_TOKENIZER, TREESEARCH_JIEBA_USER_DICT
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Literal, Optional, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index schema version
# ---------------------------------------------------------------------------
# Bump whenever a change in tree builder, tokenizer, FTS schema, or node_id
# algorithm would invalidate previously-built indexes. The version is folded
# into every file's fingerprint so old index_meta entries automatically miss
# and the file is re-indexed on next run.
#
# History:
#   "1" — original (mtime_ns:size) fingerprint with sequential int node_ids.
#   "2" — stable hash node_ids + node-level diff + atomic per-doc transaction.
INDEX_SCHEMA_VERSION = "2"

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
_ENV_CJK_TOKENIZER = "TREESEARCH_CJK_TOKENIZER"
_ENV_FINGERPRINT_MODE = "TREESEARCH_FINGERPRINT_MODE"
_ENV_PRUNE = "TREESEARCH_PRUNE"
_ENV_JIEBA_USER_DICT = "TREESEARCH_JIEBA_USER_DICT"
_ENV_JIEBA_USER_WORDS = "TREESEARCH_JIEBA_USER_WORDS"
_ENV_JIEBA_DEL_WORDS = "TREESEARCH_JIEBA_DEL_WORDS"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TreeSearchConfig:
    """Single configuration class for TreeSearch.

    Priority: set_config() > env vars > defaults.
    """
    # Search
    max_nodes_per_doc: int = 5
    top_k_docs: int = 3

    # Index
    if_add_node_summary: bool = True
    if_add_doc_description: bool = False
    if_add_node_text: bool = True
    if_add_node_id: bool = True
    if_thinning: bool = False
    min_thinning_chars: int = 15000  # min chars to keep a sub-tree during thinning
    summary_chars_threshold: int = 600  # nodes shorter than this use full text as summary
    max_concurrency: int = field(default_factory=lambda: min(os.cpu_count() or 4, 256))
    max_dir_files: int = 10_000  # safety cap for directory walk

    # Text length limits
    max_node_chars: int = 8000  # max characters per node text when indexing into FTS5
    max_result_chars: int = 32000  # max total characters of returned search result texts

    # FTS
    fts_db_path: str = ""  # empty = same DB as tree storage (default: index.db)
    fts_title_weight: float = 5.0
    fts_summary_weight: float = 2.0
    fts_body_weight: float = 10.0
    fts_code_weight: float = 1.0
    fts_front_matter_weight: float = 2.0

    # Tree Search
    search_mode: str = "auto"  # "auto" | "flat" | "tree" | "auto" degrades to flat for code-only docs
    anchor_top_k: int = 5  # max anchor nodes per document
    max_anchor_per_doc: int = 3  # anchors to expand per document
    max_expansions: int = 40  # max total node expansions in tree walk
    max_hops: int = 3  # max depth offset from anchor
    max_siblings: int = 2  # max sibling nodes to expand per step
    min_frontier_score: float = 0.1  # stop if best frontier score below this
    early_stop_score: float = 0.95  # stop early if a path reaches this score
    path_top_k: int = 3  # top paths to return

    # Tokenizer
    cjk_tokenizer: str = "auto"  # "auto" | "jieba" | "bigram" | "char"

    # Jieba custom dictionary support (improves Chinese retrieval accuracy
    # for domain-specific terms, brand names, multi-word entities, etc.).
    # `jieba_user_dict_paths` are file paths in jieba dict format
    #   (one entry per line: "word [freq] [tag]"; freq/tag optional).
    # `jieba_user_words` are in-memory entries. Each string is parsed as
    #   "word [freq] [tag]" — `freq` and `tag` are both optional, so
    #   plain `"石墨烯"` works just like `jieba.add_word("石墨烯")`.
    # `jieba_del_words` are words to remove via `jieba.del_word(...)`.
    # All are applied lazily on first jieba use; changes trigger reload.
    jieba_user_dict_paths: List[str] = field(default_factory=list)
    jieba_user_words: List[str] = field(default_factory=list)
    jieba_del_words: List[str] = field(default_factory=list)

    # Incremental indexing
    fingerprint_mode: Literal["stat", "content"] = "stat"
    # "stat":    fast `(mtime_ns:size)` fingerprint. Re-indexes after a `touch`.
    # "content": samples first/middle/last 64KB of large files (full md5 for
    #            files <1MB) — robust against `touch`/CI replay; ~1ms cost on small
    #            files, dozens of ms on multi-GB files. Opt-in.
    content_fingerprint_size_threshold: int = 1_000_000  # bytes; full md5 below this
    content_fingerprint_sample_bytes: int = 64 * 1024     # head/mid/tail sample size

    # Auto FTS5 maintenance: run `optimize` every N reindexed documents
    # within a single build_index call. 0 disables.
    auto_optimize_threshold: int = 1000

    # Default prune policy when build_index sees a directory in `paths`.
    # Set to False to never auto-delete orphans even on directory walks.
    prune_orphans_on_directory: bool = True

    @classmethod
    def from_env(cls) -> "TreeSearchConfig":
        """Create config from environment variables, falling back to defaults."""
        config = cls()

        env_cjk = os.getenv(_ENV_CJK_TOKENIZER)
        if env_cjk:
            config.cjk_tokenizer = env_cjk

        env_fp = os.getenv(_ENV_FINGERPRINT_MODE)
        if env_fp in ("stat", "content"):
            config.fingerprint_mode = env_fp

        env_prune = os.getenv(_ENV_PRUNE)
        if env_prune is not None:
            config.prune_orphans_on_directory = env_prune.lower() in ("1", "true", "yes")

        env_dict = os.getenv(_ENV_JIEBA_USER_DICT)
        if env_dict:
            # Support both os.pathsep (':' on POSIX, ';' on Windows) and ','.
            parts = []
            for chunk in env_dict.split(os.pathsep):
                parts.extend(p.strip() for p in chunk.split(",") if p.strip())
            config.jieba_user_dict_paths = parts

        env_words = os.getenv(_ENV_JIEBA_USER_WORDS)
        if env_words:
            # Comma-separated entries; each entry may itself contain
            # whitespace-separated "word [freq] [tag]".
            config.jieba_user_words = [w.strip() for w in env_words.split(",") if w.strip()]

        env_del = os.getenv(_ENV_JIEBA_DEL_WORDS)
        if env_del:
            config.jieba_del_words = [w.strip() for w in env_del.split(",") if w.strip()]

        return config


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_default_config: Optional[TreeSearchConfig] = None


def get_config(reload: bool = False) -> TreeSearchConfig:
    """Get global configuration (lazy singleton).

    First call reads env vars + defaults. Subsequent calls return cached instance.
    """
    global _default_config
    if reload or _default_config is None:
        _default_config = TreeSearchConfig.from_env()
    return _default_config


def set_config(config: TreeSearchConfig) -> None:
    """Set global configuration (highest priority)."""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset global config. Next get_config() re-initializes from env."""
    global _default_config
    _default_config = None
