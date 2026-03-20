# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tokenizer for TreeSearch FTS5 and search indexing.

Supports Chinese (jieba / bigram / char) and English tokenization.
CJK tokenization mode is configurable via ``TreeSearchConfig.cjk_tokenizer``.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CJK detection
# ---------------------------------------------------------------------------

# CJK Unicode ranges
_RE_CJK_CHAR = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")
_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

_RE_SPLIT_EN = re.compile(r"\W+")

# English stopwords (compact set covering most frequent terms)
_EN_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "nor", "as", "if", "then", "than", "so", "such", "both",
    "each", "all", "any", "few", "more", "most", "other", "some", "only",
    "same", "also", "very", "just", "about", "above", "after", "again",
    "between", "into", "through", "during", "before", "under", "over",
})

# ---------------------------------------------------------------------------
# Lazy-loaded heavy dependencies (import cost optimization)
# ---------------------------------------------------------------------------

_JIEBA_LOADED = False
_jieba = None
_STEMMER = None


def _ensure_jieba():
    """Lazy-load jieba to avoid ~1s import cost when unused (no CJK text)."""
    global _JIEBA_LOADED, _jieba
    if not _JIEBA_LOADED:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
        _JIEBA_LOADED = True
    return _jieba


def _ensure_stemmer():
    """Lazy-load Porter stemmer to avoid import cost when unused."""
    global _STEMMER
    if _STEMMER is None:
        try:
            from nltk.stem import PorterStemmer
            _STEMMER = PorterStemmer()
        except ImportError:
            _STEMMER = False  # sentinel: tried but unavailable
    return _STEMMER if _STEMMER is not False else None


# ---------------------------------------------------------------------------
# CJK tokenization strategies
# ---------------------------------------------------------------------------

def _tokenize_cjk_jieba(text: str) -> list[str]:
    """Tokenize CJK text using jieba word segmentation."""
    jieba_mod = _ensure_jieba()
    return list(jieba_mod.cut(text))


def _tokenize_cjk_bigram(text: str) -> list[str]:
    """Tokenize CJK text using character bigrams.

    "机器学习" → ["机器", "器学", "学习"]
    Non-CJK characters are split by whitespace.
    """
    tokens = []
    cjk_buffer = []
    for char in text:
        if _RE_HAS_CJK.match(char):
            cjk_buffer.append(char)
        else:
            if cjk_buffer:
                tokens.extend(_bigrams_from_chars(cjk_buffer))
                cjk_buffer = []
            if char.strip():
                tokens.append(char.lower())
    if cjk_buffer:
        tokens.extend(_bigrams_from_chars(cjk_buffer))
    return tokens


def _bigrams_from_chars(chars: list[str]) -> list[str]:
    """Generate bigrams from a list of CJK characters."""
    if len(chars) <= 1:
        return list(chars)
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def _tokenize_cjk_char(text: str) -> list[str]:
    """Tokenize CJK text by splitting each character individually (legacy)."""
    tokens = []
    for char in text:
        if _RE_HAS_CJK.match(char):
            tokens.append(char)
        elif char.strip():
            tokens.append(char.lower())
    return tokens


# ---------------------------------------------------------------------------
# Public tokenize function
# ---------------------------------------------------------------------------

def tokenize(text: str, use_stemmer: bool = True, remove_stopwords: bool = True) -> list[str]:
    """Tokenize text for indexing / search. Supports Chinese and English.

    CJK tokenization mode is determined by ``get_config().cjk_tokenizer``:
      - ``"auto"``: jieba word segmentation (default)
      - ``"jieba"``: same as auto (explicit)
      - ``"bigram"``: CJK character 2-grams
      - ``"char"``: single-character splitting (legacy behaviour)
    """
    if not text:
        return []

    tokens: list[str]
    if _RE_HAS_CJK.search(text):
        from .config import get_config
        mode = get_config().cjk_tokenizer

        if mode == "jieba":
            tokens = _tokenize_cjk_jieba(text)
        elif mode == "bigram":
            tokens = _tokenize_cjk_bigram(text)
        elif mode == "char":
            tokens = _tokenize_cjk_char(text)
        else:
            # "auto": use jieba (always available as a required dependency)
            jieba_mod = _ensure_jieba()
            tokens = list(jieba_mod.cut(text))
    else:
        tokens = _RE_SPLIT_EN.split(text.lower())

    # Filter: keep CJK single chars, require len>1 for English tokens
    filtered = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if len(t) == 1 and _RE_HAS_CJK.match(t):
            filtered.append(t)
        elif len(t) > 1:
            filtered.append(t)
    tokens = filtered

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _EN_STOPWORDS]

    if use_stemmer and not _RE_HAS_CJK.search(text):
        stemmer = _ensure_stemmer()
        if stemmer is not None:
            tokens = [stemmer.stem(t) for t in tokens]

    return tokens
