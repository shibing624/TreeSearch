# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Heuristic scoring for tree search.

All scoring logic is centralized here so it can be tested, tuned, and
extended independently of the search pipeline.

No LLM or embedding dependencies. Pure rule-based scoring.

Scoring philosophy:
  - FTS5 lexical score (body text BM25) is the dominant signal
  - Title/phrase matches are bonuses, not requirements
  - Body text term overlap provides content-aware scoring even when titles are generic
  - Ancestor support propagates path-level relevance
"""
import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query Plan
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """Structured representation of a parsed query.

    Attributes:
        raw: original query string
        terms: cleaned individual terms
        phrases: exact phrase fragments (from quoted substrings or consecutive terms)
        is_code_query: whether the query likely targets code (function/class/import)
        is_structural_query: whether the query targets structural location (chapter/section)
    """
    raw: str = ""
    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    is_code_query: bool = False
    is_structural_query: bool = False


# Patterns for query intent detection
_CODE_SIGNALS = re.compile(
    r'\b(function|func|def|class|import|module|method|param|return|error|exception|api|endpoint)\b',
    re.IGNORECASE,
)
_STRUCT_SIGNALS = re.compile(
    r'\b(chapter|section|appendix|part|table of contents|toc)\b|'
    r'第[一二三四五六七八九十\d]+[章节篇部]|'
    r'\b[Qq]\d+\b|\bv\d+\.\d+',
    re.IGNORECASE,
)
_QUOTED_PHRASE = re.compile(r'"([^"]+)"')

# Stop words to ignore during term overlap computation
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "so", "if", "than", "too", "very", "just", "about",
    "also", "then", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "we", "they", "he", "she",
    "us", "our", "their", "your", "my", "i", "me", "you",
})


def build_query_plan(query: str) -> QueryPlan:
    """Parse a raw query string into a structured QueryPlan.

    Steps:
    1. Extract quoted phrases
    2. Clean and split remaining terms (filter stop words)
    3. Detect code / structural intent via regex
    """
    plan = QueryPlan(raw=query)

    # Extract quoted phrases
    for m in _QUOTED_PHRASE.finditer(query):
        plan.phrases.append(m.group(1).strip())
    remaining = _QUOTED_PHRASE.sub("", query)

    # Clean and split terms
    remaining = remaining.strip().lower()
    remaining = re.sub(r'[^\w\u4e00-\u9fff\s]', ' ', remaining)
    raw_terms = [t for t in remaining.split() if t]
    # Keep all raw terms for phrase matching, but filter stop words for scoring
    plan.terms = [t for t in raw_terms if t not in _STOP_WORDS and len(t) > 1]
    # Fallback: if all terms were stop words, keep original
    if not plan.terms and raw_terms:
        plan.terms = raw_terms

    # Build implicit phrase from consecutive terms (2-gram)
    if len(raw_terms) >= 2 and not plan.phrases:
        plan.phrases.append(" ".join(raw_terms))

    # Intent detection
    plan.is_code_query = bool(_CODE_SIGNALS.search(query))
    plan.is_structural_query = bool(_STRUCT_SIGNALS.search(query))

    return plan


# ---------------------------------------------------------------------------
# Term overlap ratio (content-aware scoring)
# ---------------------------------------------------------------------------

def compute_term_overlap(text: str, terms: list[str], idf: dict[str, float] | None = None) -> float:
    """Compute IDF-weighted fraction of query terms that appear in the text.

    When ``idf`` is provided, rare terms contribute more than common terms.
    Falls back to uniform weighting when ``idf`` is None.

    Returns a value in [0.0, 1.0].
    """
    if not text or not terms:
        return 0.0
    text_lower = text.lower()
    if idf:
        total_w = sum(idf.get(t, 1.0) for t in terms)
        if total_w <= 0:
            return 0.0
        hit_w = sum(idf.get(t, 1.0) for t in terms if t in text_lower)
        return hit_w / total_w
    # Uniform fallback
    matched = sum(1 for t in terms if t in text_lower)
    return matched / len(terms)


def estimate_idf(terms: list[str], corpus_texts: list[str]) -> dict[str, float]:
    """Estimate IDF weights for query terms from a corpus of node texts.

    Uses smooth IDF: log((N + 1) / (df + 1)) + 1 to avoid zero weights.
    Corpus is typically all node texts from a single document.

    Args:
        terms: query terms (lowercased)
        corpus_texts: list of node text strings

    Returns:
        {term: idf_weight} for each term
    """
    n = len(corpus_texts)
    if n == 0:
        return {t: 1.0 for t in terms}
    df: dict[str, int] = {t: 0 for t in terms}
    for text in corpus_texts:
        text_lower = text.lower()
        for t in terms:
            if t in text_lower:
                df[t] += 1
    idf = {}
    for t in terms:
        idf[t] = math.log((n + 1) / (df[t] + 1)) + 1.0
    return idf


# ---------------------------------------------------------------------------
# Anchor Scorer
# ---------------------------------------------------------------------------

def score_anchor(
    fts_score: float,
    depth: int,
    has_title_match: bool = False,
    has_phrase_match: bool = False,
    body_term_overlap: float = 0.0,
    max_depth: int = 6,
) -> float:
    """Score a candidate anchor node.

    Anchors should be high-level entry points, so deeper nodes get penalized.
    FTS5 score (which already incorporates body text BM25) is the primary signal.

    Args:
        fts_score: normalized FTS5 relevance score [0, 1]
        depth: node depth in tree (0 = root)
        has_title_match: whether query terms appear in node title
        has_phrase_match: whether exact phrase appears in node
        body_term_overlap: fraction of query terms found in node body [0, 1]
        max_depth: max depth for normalization

    Returns:
        anchor score in [0, 1] range
    """
    # Depth penalty: deeper nodes are less ideal as anchors
    depth_penalty = min(depth / max(max_depth, 1), 1.0) * 0.10

    # Title match bonus (smaller than before -- titles are often generic)
    title_bonus = 0.08 if has_title_match else 0.0

    # Phrase match bonus
    phrase_bonus = 0.07 if has_phrase_match else 0.0

    # Body content overlap bonus -- reward nodes whose text contains query terms
    body_bonus = 0.10 * body_term_overlap

    score = fts_score + title_bonus + phrase_bonus + body_bonus - depth_penalty
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Walk Scorer
# ---------------------------------------------------------------------------

def score_walk_node(
    lexical_score: float,
    *,
    has_title_match: bool = False,
    has_phrase_match: bool = False,
    body_term_overlap: float = 0.0,
    ancestor_support: float = 0.0,
    hop: int = 0,
    is_redundant: bool = False,
    max_hops: int = 3,
) -> float:
    """Score a node during tree walk expansion.

    Design: FTS5 lexical_score already captures BM25 relevance over body text.
    We weight it heavily so that content-rich nodes with query term hits rank
    high even when their title is generic (e.g. "Results", "Discussion").

    Args:
        lexical_score: FTS5 normalized score for this node [0, 1]
        has_title_match: query terms appear in title
        has_phrase_match: exact phrase appears in node
        body_term_overlap: fraction of query terms in node body [0, 1]
        ancestor_support: max score among ancestors on current path [0, 1]
        hop: number of hops from anchor
        is_redundant: whether this node overlaps with already-visited paths
        max_hops: max hops for normalization

    Returns:
        walk score (higher = more worth expanding)
    """
    # Base: FTS5 lexical relevance is the primary signal (heavy weight)
    score = 0.45 * lexical_score

    # Body text content overlap -- works even when FTS5 score is 0 (unexpanded nodes)
    score += 0.15 * body_term_overlap

    # Title match bonus (reduced from old 0.20 -> 0.08)
    if has_title_match:
        score += 0.08

    # Phrase match bonus
    if has_phrase_match:
        score += 0.07

    # Ancestor support: path consistency
    score += 0.12 * ancestor_support

    # Hop penalty: further from anchor = less relevant
    hop_ratio = min(hop / max(max_hops, 1), 1.0)
    score -= 0.08 * hop_ratio

    # Redundancy penalty
    if is_redundant:
        score -= 0.08

    return max(0.0, score)


# ---------------------------------------------------------------------------
# Path Scorer
# ---------------------------------------------------------------------------

def score_path(
    leaf_score: float,
    path_titles: list[str],
    path_texts: list[str],
    query_terms: list[str],
    path_length: int,
    leaf_fts_score: float = 0.0,
    max_path_length: int = 6,
) -> float:
    """Score a complete root-to-leaf path.

    Combines leaf node quality with path-level content coverage.
    Uses both title and body text for path scoring to avoid title-dependency.

    Args:
        leaf_score: walk score of the terminal node
        path_titles: titles along the path (root to leaf)
        path_texts: body texts along the path (root to leaf)
        query_terms: cleaned query terms from QueryPlan
        path_length: number of nodes in path
        leaf_fts_score: FTS5 score of the leaf node [0, 1]
        max_path_length: max path length for normalization

    Returns:
        path score (higher = better answer path)
    """
    # Leaf score dominates (walk-level quality)
    score = 0.35 * leaf_score

    # Leaf FTS5 score direct contribution (content relevance of the answer node)
    score += 0.20 * leaf_fts_score

    # Path content coverage: how many query terms appear in ANY node's text along the path
    if query_terms and path_texts:
        all_text = " ".join(path_texts).lower()
        covered = sum(1 for t in query_terms if t in all_text)
        text_coverage = covered / len(query_terms)
        score += 0.20 * text_coverage

    # Path title consistency: how many path titles contain query terms
    if path_titles and query_terms:
        match_count = 0
        for title in path_titles:
            title_lower = title.lower()
            if any(t in title_lower for t in query_terms):
                match_count += 1
        consistency = match_count / len(path_titles)
        score += 0.10 * consistency

    # Context coverage: how many query terms appear somewhere in path titles
    if query_terms and path_titles:
        all_titles_text = " ".join(path_titles).lower()
        covered = sum(1 for t in query_terms if t in all_titles_text)
        coverage = covered / len(query_terms)
        score += 0.08 * coverage

    # Readability bonus: shorter paths are easier to present
    length_ratio = min(path_length / max(max_path_length, 1), 1.0)
    readability = 1.0 - length_ratio * 0.5
    score += 0.07 * readability

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Utility: term matching helpers
# ---------------------------------------------------------------------------

def check_title_match(title: str, terms: list[str]) -> bool:
    """Check if any query term appears in the node title."""
    if not title or not terms:
        return False
    title_lower = title.lower()
    return any(t in title_lower for t in terms)


def check_phrase_match(text: str, phrases: list[str]) -> bool:
    """Check if any exact phrase appears in the text."""
    if not text or not phrases:
        return False
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in phrases)


# ---------------------------------------------------------------------------
# Generic Section Detection
# ---------------------------------------------------------------------------

# Sections that typically contain broad overview text rather than specific answers.
# These get high BM25 scores because they mention many terms, but rarely contain
# the precise information a question targets.
_GENERIC_SECTIONS = frozenset({
    "abstract", "introduction", "conclusion", "conclusions",
    "related work", "acknowledgments", "acknowledgements",
    "conclusion and outlook", "conclusions and outlook",
    "conclusion and future work", "conclusions and future work",
    "background", "overview",
})


def is_generic_section(title: str, depth: int) -> bool:
    """Check if a node is a generic overview section.

    Only applies to top-level sections (depth 0-1) whose base title
    (before ::: delimiter) is in the generic set.

    Args:
        title: node title string
        depth: node depth in tree (0 = root)

    Returns:
        True if the node is a generic section that should be demoted
    """
    if depth > 1:
        return False
    if not title:
        return False
    # For ::: delimited titles, only check the base (leftmost) part
    base_title = title.split(" ::: ")[0].strip().lower() if " ::: " in title else title.strip().lower()
    # Root node (depth=0, paper title) is almost never relevant
    if depth == 0:
        return True
    return base_title in _GENERIC_SECTIONS
