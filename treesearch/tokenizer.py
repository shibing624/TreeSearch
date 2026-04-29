# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tokenizer for TreeSearch FTS5 and search indexing.

Supports Chinese (jieba / bigram / char) and English tokenization.
CJK tokenization mode is configurable via ``TreeSearchConfig.cjk_tokenizer``.
"""
import logging
import os
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
_jieba_tokenizer = None  # private jieba.Tokenizer instance
_JIEBA_USERDICT_FINGERPRINT: Optional[tuple] = None
_STEMMER = None


def _userdict_fingerprint(paths: list, words: list, del_words: list) -> tuple:
    """Build a fingerprint that changes whenever user dict config changes.

    For files we include (path, mtime_ns, size) so that edits to a dict file
    also force a reload. Missing files are recorded as (path, None, None).
    """
    file_sig = []
    for p in paths:
        try:
            st = os.stat(p)
            file_sig.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            file_sig.append((p, None, None))
    return (tuple(file_sig), tuple(words), tuple(del_words))


def _apply_jieba_user_dicts(tokenizer, paths: list, words: list, del_words: list) -> None:
    """Load user dict files, add user words, and remove deleted words.

    Word entries can be in any of these forms (matching jieba's API):
      "石墨烯"                — pure word, equivalent to ``add_word("石墨烯")``
      "石墨烯 9000"           — word + freq
      "石墨烯 9000 n"         — word + freq + part-of-speech tag
      "石墨烯 n"              — word + tag (when 2nd token is non-numeric)
    """
    for path in paths:
        if not path:
            continue
        try:
            tokenizer.load_userdict(path)
            logger.info("jieba: loaded user dict %s", path)
        except (OSError, FileNotFoundError) as e:
            logger.warning("jieba: failed to load user dict %s: %s", path, e)

    for entry in words:
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split()
        word = parts[0]
        freq = None
        tag = None
        if len(parts) >= 2:
            try:
                freq = int(parts[1])
            except ValueError:
                tag = parts[1]
        if len(parts) >= 3:
            tag = parts[2]
        tokenizer.add_word(word, freq=freq, tag=tag)

    for word in del_words:
        word = word.strip()
        if word:
            tokenizer.del_word(word)


def _ensure_jieba():
    """Lazy-load jieba and apply any configured user dictionaries.

    A private ``jieba.Tokenizer`` is used so we can rebuild it from scratch
    whenever user-dict configuration (paths, file mtimes, in-memory words)
    changes — without polluting ``jieba``'s module-level default tokenizer.
    """
    global _JIEBA_LOADED, _jieba, _jieba_tokenizer, _JIEBA_USERDICT_FINGERPRINT
    if not _JIEBA_LOADED:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        _jieba = jieba
        _jieba_tokenizer = jieba.Tokenizer()
        _JIEBA_LOADED = True

    from .config import get_config
    cfg = get_config()
    fp = _userdict_fingerprint(
        cfg.jieba_user_dict_paths, cfg.jieba_user_words, cfg.jieba_del_words,
    )
    if fp != _JIEBA_USERDICT_FINGERPRINT:
        if _JIEBA_USERDICT_FINGERPRINT is not None:
            # Rebuild from scratch so previously-loaded user entries don't
            # leak across reloads.
            _jieba_tokenizer = _jieba.Tokenizer()
        _apply_jieba_user_dicts(
            _jieba_tokenizer,
            cfg.jieba_user_dict_paths,
            cfg.jieba_user_words,
            cfg.jieba_del_words,
        )
        _JIEBA_USERDICT_FINGERPRINT = fp
    return _jieba_tokenizer


def reset_jieba() -> None:
    """Reset jieba state (used by tests / config reloads)."""
    global _JIEBA_LOADED, _jieba, _jieba_tokenizer, _JIEBA_USERDICT_FINGERPRINT
    _JIEBA_LOADED = False
    _jieba = None
    _jieba_tokenizer = None
    _JIEBA_USERDICT_FINGERPRINT = None


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
    tk = _ensure_jieba()
    return list(tk.cut(text))


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
            tk = _ensure_jieba()
            tokens = list(tk.cut(text))
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
