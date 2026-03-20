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
        # Search
        assert c.max_nodes_per_doc == 5
        assert c.top_k_docs == 3
        assert c.min_thinning_chars == 15000
        # FTS
        assert c.fts_db_path == ""
        assert c.fts_title_weight == 5.0
        assert c.fts_summary_weight == 2.0
        assert c.fts_body_weight == 10.0
        assert c.fts_code_weight == 1.0
        assert c.fts_front_matter_weight == 2.0
        # Tokenizer
        assert c.cjk_tokenizer == "auto"


# ---------------------------------------------------------------------------
# from_env: env vars
# ---------------------------------------------------------------------------

class TestFromEnvVars:
    def test_cjk_tokenizer_from_env(self):
        env = {"TREESEARCH_CJK_TOKENIZER": "bigram"}
        with patch.dict(os.environ, env, clear=False):
            c = TreeSearchConfig.from_env()
        assert c.cjk_tokenizer == "bigram"

    def test_no_env_uses_defaults(self):
        keys_to_clear = [k for k in os.environ if k.startswith("TREESEARCH_")]
        with patch.dict(os.environ, {}, clear=False):
            for k in keys_to_clear:
                os.environ.pop(k, None)
            c = TreeSearchConfig.from_env()
        assert c.max_nodes_per_doc == 5


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
        custom = TreeSearchConfig(max_nodes_per_doc=10)
        set_config(custom)
        assert get_config().max_nodes_per_doc == 10
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
