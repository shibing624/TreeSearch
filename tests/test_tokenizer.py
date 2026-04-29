# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.tokenizer module.
"""
import os
import tempfile

import pytest
from treesearch.config import reset_config, set_config, TreeSearchConfig
from treesearch.tokenizer import (
    tokenize,
    reset_jieba,
    _bigrams_from_chars,
    _tokenize_cjk_bigram,
)


class TestTokenize:
    def setup_method(self):
        reset_config()

    def teardown_method(self):
        reset_config()

    def test_english_basic(self):
        tokens = tokenize("Hello world, this is a test")
        assert "hello" in tokens or any("hello" in t for t in tokens)
        assert "world" in tokens or any("world" in t for t in tokens)
        assert "test" in tokens or any("test" in t for t in tokens)

    def test_empty_string(self):
        assert tokenize("") == []

    def test_whitespace_only(self):
        assert tokenize("   ") == []

    def test_chinese_text_auto(self):
        """Default 'auto' mode: uses jieba for CJK tokenization."""
        tokens = tokenize("这是一个测试文档")
        assert len(tokens) > 0
        assert any(len(t) >= 1 for t in tokens)

    def test_mixed_chinese_english(self):
        tokens = tokenize("使用FastAPI构建后端服务")
        assert len(tokens) > 0

    def test_english_stopword_removal(self):
        tokens = tokenize("this is a test", remove_stopwords=True)
        assert "this" not in tokens
        assert "is" not in tokens

    def test_no_stopword_removal(self):
        tokens = tokenize("this is a test", remove_stopwords=False)
        assert "test" in tokens or any("test" in t for t in tokens)


class TestBigramTokenizer:
    def setup_method(self):
        reset_config()
        set_config(TreeSearchConfig(cjk_tokenizer="bigram"))

    def teardown_method(self):
        reset_config()

    def test_bigram_basic(self):
        tokens = tokenize("机器学习")
        assert "机器" in tokens
        assert "器学" in tokens
        assert "学习" in tokens

    def test_bigram_single_char(self):
        """Single CJK char: no bigram possible, returned as-is."""
        result = _bigrams_from_chars(["中"])
        assert result == ["中"]

    def test_bigram_helper(self):
        result = _bigrams_from_chars(["你", "好", "吗"])
        assert result == ["你好", "好吗"]


class TestCharTokenizer:
    def setup_method(self):
        reset_config()
        set_config(TreeSearchConfig(cjk_tokenizer="char"))

    def teardown_method(self):
        reset_config()

    def test_char_mode(self):
        tokens = tokenize("你好世界")
        # char mode: individual characters
        assert "你" in tokens
        assert "好" in tokens
        assert "世" in tokens
        assert "界" in tokens


class TestCJKBigramFunction:
    def test_bigram_mixed(self):
        tokens = _tokenize_cjk_bigram("你好world")
        # "你好" bigram + "w","o","r","l","d" individual
        assert "你好" in tokens


class TestJiebaUserDict:
    """Custom jieba dictionary support — file paths and in-memory words."""

    def setup_method(self):
        reset_config()
        reset_jieba()

    def teardown_method(self):
        reset_config()
        reset_jieba()

    def test_baseline_splits_unknown_term(self):
        # A made-up multi-character term jieba doesn't know — typically
        # gets segmented into shorter pieces.
        tokens = tokenize("超级灵魂引擎是新框架")
        assert "超级灵魂引擎" not in tokens

    def test_user_words_keeps_term_intact(self):
        set_config(TreeSearchConfig(jieba_user_words=["超级灵魂引擎 5000 n"]))
        tokens = tokenize("超级灵魂引擎是新框架")
        assert "超级灵魂引擎" in tokens

    def test_user_dict_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("树搜索引擎 9000 n\n")
            f.write("结构感知检索 9000 n\n")
            path = f.name
        try:
            set_config(TreeSearchConfig(jieba_user_dict_paths=[path]))
            tokens = tokenize("树搜索引擎支持结构感知检索")
            assert "树搜索引擎" in tokens
            assert "结构感知检索" in tokens
        finally:
            os.unlink(path)

    def test_env_var_paths(self, monkeypatch):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("元宇宙引擎 8000 n\n")
            path = f.name
        try:
            monkeypatch.setenv("TREESEARCH_JIEBA_USER_DICT", path)
            reset_config()
            reset_jieba()
            tokens = tokenize("元宇宙引擎正在测试")
            assert "元宇宙引擎" in tokens
        finally:
            os.unlink(path)

    def test_env_var_words(self, monkeypatch):
        monkeypatch.setenv("TREESEARCH_JIEBA_USER_WORDS", "图神经网络 6000 n,知识蒸馏 6000 n")
        reset_config()
        reset_jieba()
        tokens = tokenize("图神经网络与知识蒸馏的结合")
        assert "图神经网络" in tokens
        assert "知识蒸馏" in tokens

    def test_pure_words_no_freq_no_tag(self):
        # Pure word entries (no freq, no tag) — like jieba.add_word("石墨烯").
        set_config(TreeSearchConfig(jieba_user_words=["石墨烯", "凱特琳"]))
        tokens = tokenize("mac上可分出石墨烯此時又可以分出來凱特琳了")
        assert "石墨烯" in tokens
        assert "凱特琳" in tokens

    def test_del_words(self):
        # Sanity: jieba keeps "计算机科学" as one token by default.
        reset_config()
        reset_jieba()
        baseline = tokenize("计算机科学很有趣")
        assert "计算机科学" in baseline
        # Remove it via del_words — jieba should no longer emit it.
        set_config(TreeSearchConfig(jieba_del_words=["计算机科学"]))
        reset_jieba()
        tokens = tokenize("计算机科学很有趣")
        assert "计算机科学" not in tokens

    def test_env_var_del_words(self, monkeypatch):
        monkeypatch.setenv("TREESEARCH_JIEBA_DEL_WORDS", "计算机科学")
        reset_config()
        reset_jieba()
        tokens = tokenize("计算机科学很有趣")
        assert "计算机科学" not in tokens

    def test_runtime_reload_on_config_change(self):
        # First call: no custom dict.
        tokens = tokenize("超新星协议是新协议")
        assert "超新星协议" not in tokens
        # Update config — next call must pick it up automatically.
        set_config(TreeSearchConfig(jieba_user_words=["超新星协议 9000 n"]))
        tokens = tokenize("超新星协议是新协议")
        assert "超新星协议" in tokens
