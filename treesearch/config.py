# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified configuration management for TreeSearch.

This is the SINGLE source of truth for all environment variables and defaults.
Other modules (llm.py, embeddings.py, etc.) should import from here instead of
calling os.getenv() directly.

Provides sensible defaults with environment variable overrides.
"""
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    strategy: str = "best_first"
    max_nodes_per_doc: int = 5
    top_k_docs: int = 3
    value_threshold: float = 0.3
    max_llm_calls: int = 30
    use_bm25: bool = True
    mcts_iterations: int = 10


@dataclass
class RetrieveRerankConfig:
    """Configuration for embedding-first retrieve-rerank pipeline.

    Stage 1: Embedding recall (primary backbone) + BM25 recall (supplements)
    Stage 2: Merge -- embedding order preserved, BM25-only nodes appended
    Stage 3: LLM rerank on uncertain zone only (blend, not replace)
    """
    # Stage 1: parallel recall
    embedding_topk: int = int(os.getenv("TREESEARCH_RR_EMB_TOPK", "20"))
    bm25_topk: int = int(os.getenv("TREESEARCH_RR_BM25_TOPK", "20"))

    # Stage 2: candidate window for LLM rerank
    rerank_top_n: int = int(os.getenv("TREESEARCH_RR_RERANK_N", "15"))

    # Stage 3: LLM rerank
    text_excerpt_len: int = int(os.getenv("TREESEARCH_RR_EXCERPT_LEN", "500"))
    include_ancestors: bool = True
    include_sibling_titles: bool = False
    llm_weight: float = float(os.getenv("TREESEARCH_RR_LLM_WEIGHT", "0.3"))

    # Adaptive (used by HybridPreFilter, not by RRF pipeline)
    query_length_threshold: int = 8
    short_query_bm25_weight: float = float(os.getenv("TREESEARCH_RR_SHORT_ALPHA", "0.4"))
    long_query_bm25_weight: float = float(os.getenv("TREESEARCH_RR_LONG_ALPHA", "0.3"))


@dataclass
class IndexConfig:
    """Configuration for index building."""
    if_add_node_summary: bool = True
    if_add_doc_description: bool = True
    if_add_node_text: bool = True
    if_add_node_id: bool = True
    if_thinning: bool = False
    min_token_threshold: int = 5000
    summary_token_threshold: int = 200
    max_concurrency: int = 5


@dataclass
class TreeSearchConfig:
    """Top-level configuration for TreeSearch.

    All environment variable reading is centralized here.
    Other modules should call get_config() to access these values.
    """
    # LLM settings
    model: str = os.getenv("TREESEARCH_MODEL", "gpt-4o-mini")
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

    # Embedding settings (optional)
    embedding_model: str = os.getenv("TREESEARCH_EMBEDDING_MODEL", "text-embedding-3-small")
    use_embedding: bool = False

    # Sub-configs
    search: SearchConfig = field(default_factory=SearchConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieve_rerank: RetrieveRerankConfig = field(default_factory=RetrieveRerankConfig)

    @classmethod
    def from_env(cls) -> "TreeSearchConfig":
        """Create config from environment variables with overrides."""
        config = cls()

        if os.getenv("TREESEARCH_STRATEGY"):
            config.search.strategy = os.getenv("TREESEARCH_STRATEGY")
        if os.getenv("TREESEARCH_MAX_LLM_CALLS"):
            config.search.max_llm_calls = int(os.getenv("TREESEARCH_MAX_LLM_CALLS"))
        if os.getenv("TREESEARCH_THRESHOLD"):
            config.search.value_threshold = float(os.getenv("TREESEARCH_THRESHOLD"))
        if os.getenv("TREESEARCH_USE_EMBEDDING"):
            config.use_embedding = os.getenv("TREESEARCH_USE_EMBEDDING").lower() in ("1", "true", "yes")

        return config


# Global default config (lazy singleton)
_default_config: Optional[TreeSearchConfig] = None


def get_config() -> TreeSearchConfig:
    """Get the global default configuration (lazy singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = TreeSearchConfig.from_env()
    return _default_config


def set_config(config: TreeSearchConfig) -> None:
    """Set the global default configuration."""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset global config to None, forcing re-initialization on next get_config() call."""
    global _default_config
    _default_config = None
