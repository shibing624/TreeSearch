# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified configuration for TreeSearch.

Priority (high -> low):
    1. set_config(TreeSearchConfig(...))
    2. Environment variables
    3. Built-in defaults

Environment variables:
    LLM secrets: TREESEARCH_LLM_API_KEY > OPENAI_API_KEY
                 TREESEARCH_LLM_BASE_URL > OPENAI_BASE_URL
    Model:       TREESEARCH_MODEL
    Search:      TREESEARCH_STRATEGY, TREESEARCH_MAX_LLM_CALLS, TREESEARCH_THRESHOLD
    Thinking:    TREESEARCH_THINKING_TYPE
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Auto-load .env (best-effort, python-dotenv is optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
_ENV_LLM_API_KEY = "TREESEARCH_LLM_API_KEY"
_ENV_LLM_BASE_URL = "TREESEARCH_LLM_BASE_URL"
_ENV_API_KEY_FALLBACK = "OPENAI_API_KEY"
_ENV_BASE_URL_FALLBACK = "OPENAI_BASE_URL"
_ENV_MODEL = "TREESEARCH_MODEL"
_ENV_STRATEGY = "TREESEARCH_STRATEGY"
_ENV_MAX_LLM_CALLS = "TREESEARCH_MAX_LLM_CALLS"
_ENV_THRESHOLD = "TREESEARCH_THRESHOLD"
_ENV_THINKING_TYPE = "TREESEARCH_THINKING_TYPE"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TreeSearchConfig:
    """Single configuration class for TreeSearch.

    Priority: set_config() > env vars > defaults.
    """
    # LLM (optional, leave empty to disable LLM features)
    model: str = ""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    thinking_type: str = "disabled"  # "disabled" | "enabled" | "auto"

    # Search
    strategy: str = "fts5_only"  # "fts5_only" | "best_first"
    max_nodes_per_doc: int = 5
    top_k_docs: int = 3
    value_threshold: float = 0.15
    max_llm_calls: int = 30
    use_bm25: bool = True

    # Best-first tuning
    bm25_weight: float = 0.5
    depth_penalty: float = 0.02
    text_excerpt_len: int = 800
    adaptive_depth_threshold: int = 0
    dynamic_threshold: bool = True
    min_threshold: float = 0.1
    max_prompt_tokens: int = 60000

    # Index
    if_add_node_summary: bool = True
    if_add_doc_description: bool = False
    if_add_node_text: bool = True
    if_add_node_id: bool = True
    if_thinning: bool = False
    min_token_threshold: int = 5000
    summary_token_threshold: int = 200
    max_concurrency: int = 5

    # FTS
    fts_enabled: bool = True
    fts_db_path: str = ""  # empty = same DB as tree storage (default: index.db)
    fts_title_weight: float = 5.0
    fts_summary_weight: float = 2.0
    fts_body_weight: float = 10.0
    fts_code_weight: float = 1.0
    fts_front_matter_weight: float = 2.0
    fts_auto_index: bool = True

    @classmethod
    def from_env(cls) -> "TreeSearchConfig":
        """Create config from environment variables, falling back to defaults."""
        config = cls()

        # Secrets: env only
        config.api_key = os.getenv(_ENV_LLM_API_KEY) or os.getenv(_ENV_API_KEY_FALLBACK)
        config.base_url = os.getenv(_ENV_LLM_BASE_URL) or os.getenv(_ENV_BASE_URL_FALLBACK)

        # Model: env > default
        env_model = os.getenv(_ENV_MODEL)
        if env_model:
            config.model = env_model

        # Thinking: env > default
        env_thinking = os.getenv(_ENV_THINKING_TYPE)
        if env_thinking:
            config.thinking_type = env_thinking.strip().lower()

        # Search overrides
        env_strategy = os.getenv(_ENV_STRATEGY)
        if env_strategy:
            config.strategy = env_strategy

        env_max_llm = os.getenv(_ENV_MAX_LLM_CALLS)
        if env_max_llm:
            try:
                config.max_llm_calls = int(env_max_llm)
            except ValueError:
                logger.warning("Invalid TREESEARCH_MAX_LLM_CALLS: %s", env_max_llm)

        env_threshold = os.getenv(_ENV_THRESHOLD)
        if env_threshold:
            try:
                config.value_threshold = float(env_threshold)
            except ValueError:
                logger.warning("Invalid TREESEARCH_THRESHOLD: %s", env_threshold)

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
