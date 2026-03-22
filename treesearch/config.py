# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified configuration for TreeSearch.

Priority (high -> low):
    1. set_config(TreeSearchConfig(...))
    2. Environment variables
    3. Built-in defaults

Environment variables:
    Tokenizer: TREESEARCH_CJK_TOKENIZER
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
_ENV_CJK_TOKENIZER = "TREESEARCH_CJK_TOKENIZER"


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

    @classmethod
    def from_env(cls) -> "TreeSearchConfig":
        """Create config from environment variables, falling back to defaults."""
        config = cls()

        env_cjk = os.getenv(_ENV_CJK_TOKENIZER)
        if env_cjk:
            config.cjk_tokenizer = env_cjk

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
