# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Node-level BM25 for tree-structured documents.

Supports Chinese (jieba) and English tokenization. Provides structure-aware
scoring with hierarchical field weighting and ancestor score propagation.
"""
import logging
import math
import re
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_JIEBA_LOADED = False
_jieba = None

# Chinese character range detection
_RE_HAS_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_RE_SPLIT_EN = re.compile(r"\W+")


def _ensure_jieba():
    """Lazy-load jieba to avoid import cost when unused."""
    global _JIEBA_LOADED, _jieba
    if not _JIEBA_LOADED:
        try:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            _jieba = jieba
        except ImportError:
            _jieba = None
        _JIEBA_LOADED = True
    return _jieba


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing. Supports Chinese and English.

    - Chinese text: jieba word segmentation (falls back to character-level if jieba not installed)
    - English text: lowercase + split on non-word characters
    - Mixed text: handled correctly
    """
    if not text:
        return []

    tokens = []
    if _RE_HAS_CJK.search(text):
        jieba_mod = _ensure_jieba()
        if jieba_mod is not None:
            tokens = list(jieba_mod.cut(text))
        else:
            # Fallback: split CJK chars individually, English by whitespace
            for char in text:
                if _RE_HAS_CJK.match(char):
                    tokens.append(char)
                elif char.strip():
                    tokens.append(char.lower())
    else:
        tokens = _RE_SPLIT_EN.split(text.lower())

    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 0]


# ---------------------------------------------------------------------------
# BM25 Okapi (pure Python, no numpy dependency)
# ---------------------------------------------------------------------------

class BM25Okapi:
    """
    BM25 Okapi ranking over a corpus of token lists.

    Pure Python implementation, no numpy dependency.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.avgdl = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.doc_len: list[int] = []
        self.idf: dict[str, float] = {}

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: list[list[str]]) -> dict[str, int]:
        nd: dict[str, int] = {}
        total_len = 0
        for document in corpus:
            self.doc_len.append(len(document))
            total_len += len(document)

            freq: dict[str, int] = {}
            for word in document:
                freq[word] = freq.get(word, 0) + 1
            self.doc_freqs.append(freq)

            for word in freq:
                nd[word] = nd.get(word, 0) + 1

        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 1.0
        return nd

    def _calc_idf(self, nd: dict[str, int]) -> None:
        idf_sum = 0.0
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * abs(average_idf) if average_idf != 0 else self.epsilon
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list[str]) -> list[float]:
        """Score all documents against a tokenized query."""
        scores = [0.0] * self.corpus_size
        for q in query:
            idf = self.idf.get(q, 0.0)
            for i in range(self.corpus_size):
                q_freq = self.doc_freqs[i].get(q, 0)
                dl = self.doc_len[i]
                denom = q_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (q_freq * (self.k1 + 1)) / denom if denom > 0 else 0.0
        return scores

    def get_top_n(self, query: list[str], n: int = 5) -> list[tuple[int, float]]:
        """Return top-n (index, score) pairs sorted by score descending."""
        scores = self.get_scores(query)
        indexed = [(i, s) for i, s in enumerate(scores)]
        indexed.sort(key=lambda x: -x[1])
        return indexed[:n]


# ---------------------------------------------------------------------------
# Node-level BM25 index for tree-structured documents
# ---------------------------------------------------------------------------

class NodeBM25Index:
    """
    Structure-aware BM25 index over document tree nodes.

    Features:
      - Hierarchical field weighting: title > summary > body
      - Ancestor score propagation: parent inherits max(child scores)
      - Supports Chinese + English via jieba tokenizer
    """

    def __init__(
        self,
        documents: list,
        title_weight: float = 1.0,
        summary_weight: float = 0.7,
        body_weight: float = 0.3,
        ancestor_decay: float = 0.5,
        tokenizer: Optional[Callable] = None,
    ):
        """
        Args:
            documents: list of Document objects
            title_weight: BM25 weight for title field
            summary_weight: BM25 weight for summary field
            body_weight: BM25 weight for body text
            ancestor_decay: decay factor for ancestor score propagation (alpha)
            tokenizer: custom tokenizer function, defaults to built-in tokenize()
        """
        self.title_weight = title_weight
        self.summary_weight = summary_weight
        self.body_weight = body_weight
        self.ancestor_decay = ancestor_decay
        self._tokenize = tokenizer or tokenize

        # Build index
        self._nodes: list[dict] = []  # flat list of {node_id, doc_id, node_ref, parent_idx, children_idxs}
        self._title_corpus: list[list[str]] = []
        self._summary_corpus: list[list[str]] = []
        self._body_corpus: list[list[str]] = []

        self._build_index(documents)

        self._bm25_title = BM25Okapi(self._title_corpus) if self._title_corpus else None
        self._bm25_summary = BM25Okapi(self._summary_corpus) if self._summary_corpus else None
        self._bm25_body = BM25Okapi(self._body_corpus) if self._body_corpus else None

    def _build_index(self, documents: list) -> None:
        """Flatten all document trees into indexed node entries."""
        for doc in documents:
            self._flatten_doc(doc.structure, doc.doc_id, parent_idx=None)

    def _flatten_doc(self, structure, doc_id: str, parent_idx: Optional[int]) -> None:
        if isinstance(structure, list):
            for item in structure:
                self._flatten_doc(item, doc_id, parent_idx)
        elif isinstance(structure, dict):
            idx = len(self._nodes)
            node_entry = {
                "idx": idx,
                "node_id": structure.get("node_id", ""),
                "doc_id": doc_id,
                "title": structure.get("title", ""),
                "summary": structure.get("summary", structure.get("prefix_summary", "")),
                "parent_idx": parent_idx,
                "children_idxs": [],
            }
            self._nodes.append(node_entry)

            if parent_idx is not None:
                self._nodes[parent_idx]["children_idxs"].append(idx)

            # Build per-field token corpus
            title_text = structure.get("title", "")
            summary_text = structure.get("summary", structure.get("prefix_summary", ""))
            body_text = structure.get("text", "")
            # Use first 200 chars of body as excerpt for efficiency
            body_excerpt = body_text[:500] if body_text else ""

            self._title_corpus.append(self._tokenize(title_text))
            self._summary_corpus.append(self._tokenize(summary_text))
            self._body_corpus.append(self._tokenize(body_excerpt))

            for child in structure.get("nodes", []):
                self._flatten_doc(child, doc_id, parent_idx=idx)

    def search(
        self,
        query: str,
        top_k: int = 20,
        propagate: bool = True,
    ) -> list[dict]:
        """
        Search nodes by BM25 with hierarchical weighting.

        Args:
            query: search query string
            top_k: max number of results
            propagate: enable ancestor score propagation

        Returns:
            list of {node_id, doc_id, title, summary, bm25_score}
        """
        if not self._nodes:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n = len(self._nodes)
        scores = [0.0] * n

        # Weighted BM25 across fields
        if self._bm25_title:
            title_scores = self._bm25_title.get_scores(query_tokens)
            for i in range(n):
                scores[i] += self.title_weight * title_scores[i]

        if self._bm25_summary:
            summary_scores = self._bm25_summary.get_scores(query_tokens)
            for i in range(n):
                scores[i] += self.summary_weight * summary_scores[i]

        if self._bm25_body:
            body_scores = self._bm25_body.get_scores(query_tokens)
            for i in range(n):
                scores[i] += self.body_weight * body_scores[i]

        # Ancestor score propagation: parent.score += alpha * max(child.score)
        if propagate:
            scores = self._propagate_scores(scores)

        # Collect and rank
        results = []
        for i in range(n):
            if scores[i] > 0:
                entry = self._nodes[i]
                results.append({
                    "node_id": entry["node_id"],
                    "doc_id": entry["doc_id"],
                    "title": entry["title"],
                    "summary": entry["summary"],
                    "bm25_score": round(scores[i], 6),
                })

        results.sort(key=lambda x: (-x["bm25_score"], x["node_id"]))
        return results[:top_k]

    def _propagate_scores(self, scores: list[float]) -> list[float]:
        """Bottom-up propagation: parent gets alpha * max(children scores)."""
        # Process bottom-up by traversing in reverse
        propagated = list(scores)
        # Find all leaf nodes first, then propagate up
        for i in range(len(self._nodes) - 1, -1, -1):
            children = self._nodes[i]["children_idxs"]
            if children:
                max_child = max(propagated[c] for c in children)
                propagated[i] += self.ancestor_decay * max_child
        return propagated

    def get_node_scores_for_doc(
        self,
        query: str,
        doc_id: str,
        top_k: int = 50,
    ) -> dict[str, float]:
        """
        Get BM25 scores for nodes within a specific document.

        Returns:
            dict mapping node_id -> bm25_score
        """
        results = self.search(query, top_k=top_k)
        return {r["node_id"]: r["bm25_score"] for r in results if r["doc_id"] == doc_id}

    def score_nodes(self, query: str, doc_id: str) -> dict[str, float]:
        """PreFilter protocol: return {node_id: score} for a given query and document."""
        return self.get_node_scores_for_doc(query, doc_id)
