# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.config module.
"""
import os
from unittest.mock import patch

import pytest
from treesearch.config import (
    TreeSearchConfig,
    get_config,
    set_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# TreeSearchConfig defaults
# ---------------------------------------------------------------------------

class TestTreeSearchConfig:
    def test_defaults(self):
        c = TreeSearchConfig()
        # LLM
        assert c.api_key is None
        assert c.base_url is None
        assert c.thinking_type == "disabled"
        # Search
        assert c.strategy == "fts5_only"
        assert c.max_nodes_per_doc == 5
        assert c.top_k_docs == 3
        assert c.value_threshold == 0.15
        assert c.max_llm_calls == 30
        assert c.use_bm25 is True
        # Best-first tuning
        assert c.bm25_weight == 0.5
        assert c.depth_penalty == 0.02
        assert c.text_excerpt_len == 800
        assert c.adaptive_depth_threshold == 0
        assert c.dynamic_threshold is True
        assert c.min_threshold == 0.1
        assert c.max_prompt_tokens == 60000
        # Index
        assert c.max_concurrency == 5
        assert c.min_token_threshold == 5000
        # FTS
        assert c.fts_enabled is True
        assert c.fts_db_path == ""
        assert c.fts_title_weight == 5.0
        assert c.fts_summary_weight == 2.0
        assert c.fts_body_weight == 10.0
        assert c.fts_code_weight == 1.0
        assert c.fts_front_matter_weight == 2.0
        assert c.fts_auto_index is True


# ---------------------------------------------------------------------------
# from_env: env vars (TREESEARCH_LLM_* > OPENAI_* fallback)
# ---------------------------------------------------------------------------

class TestFromEnvVars:
    """Secrets and model names come from env vars."""

    def test_treesearch_llm_env_preferred(self):
        """TREESEARCH_LLM_API_KEY takes priority over OPENAI_API_KEY."""
        env = {
            "TREESEARCH_LLM_API_KEY": "sk-treesearch",
            "OPENAI_API_KEY": "sk-openai-fallback",
            "TREESEARCH_LLM_BASE_URL": "https://ts.api.com",
            "OPENAI_BASE_URL": "https://openai-fallback.api.com",
        }
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.api_key == "sk-treesearch"
        assert c.base_url == "https://ts.api.com"

    def test_openai_env_fallback(self):
        """Without TREESEARCH_LLM_*, falls back to OPENAI_*."""
        keys_to_clear = ["TREESEARCH_LLM_API_KEY", "TREESEARCH_LLM_BASE_URL"]
        env = {
            "OPENAI_API_KEY": "sk-openai",
            "OPENAI_BASE_URL": "https://openai.api.com",
        }
        with patch.dict(os.environ, env, clear=False):
            for k in keys_to_clear:
                os.environ.pop(k, None)
            c = TreeSearchConfig.from_env()
        assert c.api_key == "sk-openai"
        assert c.base_url == "https://openai.api.com"

    def test_model_from_env(self):
        env = {"TREESEARCH_MODEL": "gpt-4o"}
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.model == "gpt-4o"

    def test_search_overrides_from_env(self):
        env = {
            "TREESEARCH_STRATEGY": "best_first",
            "TREESEARCH_MAX_LLM_CALLS": "50",
            "TREESEARCH_THRESHOLD": "0.5",
        }
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.strategy == "best_first"
        assert c.max_llm_calls == 50
        assert c.value_threshold == 0.5

    def test_invalid_env_max_llm_calls_keeps_default(self):
        """Invalid env value for max_llm_calls falls back to default."""
        env = {"TREESEARCH_MAX_LLM_CALLS": "not_a_number"}
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.max_llm_calls == 30  # default

    def test_invalid_env_threshold_keeps_default(self):
        """Invalid env value for threshold falls back to default."""
        env = {"TREESEARCH_THRESHOLD": "invalid"}
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.value_threshold == 0.15  # default

    def test_no_env_uses_defaults(self):
        """Without env vars, defaults apply."""
        keys_to_clear = [k for k in os.environ if k.startswith(("TREESEARCH_", "OPENAI_"))]
        with patch.dict(os.environ, {}, clear=False):
            for k in keys_to_clear:
                os.environ.pop(k, None)
            c = TreeSearchConfig.from_env()
        assert c.api_key is None
        assert c.strategy == "fts5_only"
        assert c.max_llm_calls == 30

    def test_thinking_type_from_env(self):
        env = {"TREESEARCH_THINKING_TYPE": "Enabled"}
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.thinking_type == "enabled"


# ---------------------------------------------------------------------------
# get_config / set_config / reset_config / reload
# ---------------------------------------------------------------------------

class TestConfigSingleton:
    def test_get_config_returns_instance(self):
        reset_config()
        c = get_config()
        assert isinstance(c, TreeSearchConfig)
        reset_config()

    def test_set_config(self):
        custom = TreeSearchConfig(model="custom-model")
        set_config(custom)
        assert get_config().model == "custom-model"
        reset_config()

    def test_singleton_returns_same_instance(self):
        reset_config()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2
        reset_config()

    def test_reset_config(self):
        """reset_config() forces re-initialization on next get_config()."""
        c1 = get_config()
        reset_config()
        c2 = get_config()
        assert c1 is not c2
        reset_config()

    def test_reload_forces_reinit(self):
        """get_config(reload=True) re-reads env."""
        reset_config()
        c1 = get_config()
        c2 = get_config(reload=True)
        assert c1 is not c2
        reset_config()

    def test_reload_picks_up_new_env(self):
        """get_config(reload=True) picks up newly set env vars."""
        reset_config()
        keys_to_clear = [k for k in os.environ if k.startswith(("TREESEARCH_", "OPENAI_"))]
        with patch.dict(os.environ, {}, clear=False):
            for k in keys_to_clear:
                os.environ.pop(k, None)
            c1 = get_config()
            assert c1.api_key is None
            os.environ["TREESEARCH_LLM_API_KEY"] = "sk-new-key"
            c2 = get_config(reload=True)
            assert c2.api_key == "sk-new-key"
            os.environ.pop("TREESEARCH_LLM_API_KEY", None)
        reset_config()
