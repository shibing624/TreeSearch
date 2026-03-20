# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for treesearch.tokenizer module.
"""
import pytest
from treesearch.config import reset_config, set_config, TreeSearchConfig
from treesearch.tokenizer import tokenize, _bigrams_from_chars, _tokenize_cjk_bigram


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
