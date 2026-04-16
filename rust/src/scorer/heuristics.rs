//! Heuristic scoring for tree search.
//!
//! All scoring logic is centralized here so it can be tested, tuned, and
//! extended independently of the search pipeline.
//!
//! No LLM or embedding dependencies. Pure rule-based scoring.
//!
//! Scoring philosophy:
//!   - FTS5 lexical score (body text BM25) is the dominant signal
//!   - Title/phrase matches are bonuses, not requirements
//!   - Body text term overlap provides content-aware scoring even when titles are generic
//!   - Ancestor support propagates path-level relevance

use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::config::CjkTokenizerMode;
use crate::tokenizer;

// ---------------------------------------------------------------------------
// Stop Words
// ---------------------------------------------------------------------------

/// Full English + Chinese stop words list (ported from Python).
pub static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    // English
    for w in &[
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
    ] {
        s.insert(*w);
    }
    // Chinese
    for w in &[
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
        "吗", "吧", "呢", "啊", "呀", "哦", "嗯", "嘛", "哈",
        "怎样", "怎么", "什么", "哪", "哪个", "哪些", "为什么", "如何",
        "可以", "能", "把", "被", "让", "给", "对", "从", "向", "跟",
        "还", "又", "再", "已", "已经", "正在", "将", "将要",
    ] {
        s.insert(*w);
    }
    s
});

// ---------------------------------------------------------------------------
// Regex patterns for query intent detection
// ---------------------------------------------------------------------------

static CODE_SIGNALS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(function|func|def|class|import|module|method|param|return|error|exception|api|endpoint)\b"
    ).unwrap()
});

static STRUCT_SIGNALS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(chapter|section|appendix|part|table of contents|toc)\b|第[一二三四五六七八九十\d]+[章节篇部]|\b[Qq]\d+\b|\bv\d+\.\d+"
    ).unwrap()
});

static QUOTED_PHRASE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""([^"]+)""#).unwrap()
});

// ---------------------------------------------------------------------------
// Generic Sections
// ---------------------------------------------------------------------------

static GENERIC_SECTIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let mut s = HashSet::new();
    for w in &[
        "abstract",
        "introduction",
        "conclusion",
        "conclusions",
        "related work",
        "acknowledgments",
        "acknowledgements",
        "conclusion and outlook",
        "conclusions and outlook",
        "conclusion and future work",
        "conclusions and future work",
        "background",
        "overview",
    ] {
        s.insert(*w);
    }
    s
});

// ---------------------------------------------------------------------------
// Query Plan
// ---------------------------------------------------------------------------

/// Structured representation of a parsed query.
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Original query string.
    pub raw: String,
    /// Cleaned individual terms (deduplicated, stop-words removed).
    pub terms: Vec<String>,
    /// Exact phrase fragments (from quoted substrings or consecutive terms).
    pub phrases: Vec<String>,
    /// Whether the query likely targets code (function/class/import).
    pub is_code_query: bool,
    /// Whether the query targets structural location (chapter/section).
    pub is_structural_query: bool,
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self {
            raw: String::new(),
            terms: Vec::new(),
            phrases: Vec::new(),
            is_code_query: false,
            is_structural_query: false,
        }
    }
}

/// Parse a raw query string into a structured [`QueryPlan`].
///
/// Steps:
/// 1. Extract quoted phrases
/// 2. Tokenize remaining text (CJK-aware)
/// 3. Filter stop words
/// 4. Detect code / structural intent via regex
pub fn build_query_plan(query: &str) -> QueryPlan {
    let mut plan = QueryPlan {
        raw: query.to_string(),
        ..Default::default()
    };

    // 1. Extract quoted phrases
    for cap in QUOTED_PHRASE.captures_iter(query) {
        if let Some(m) = cap.get(1) {
            let phrase = m.as_str().trim().to_string();
            if !phrase.is_empty() {
                plan.phrases.push(phrase);
            }
        }
    }
    let remaining = QUOTED_PHRASE.replace_all(query, "").to_string();
    let remaining = remaining.trim();

    // 2. Tokenize with CJK support (no stopword removal — we do it ourselves)
    let tokens = tokenizer::tokenize(remaining, false, CjkTokenizerMode::Auto);
    let raw_terms: Vec<String> = tokens
        .into_iter()
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.to_lowercase())
        .collect();

    // 3. Filter stop words and single-char English tokens (keep CJK single chars)
    plan.terms = raw_terms
        .iter()
        .filter(|t| {
            !STOP_WORDS.contains(t.as_str())
                && (t.chars().count() > 1 || tokenizer::has_cjk(t))
        })
        .cloned()
        .collect();

    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    plan.terms.retain(|t| seen.insert(t.clone()));

    // Fallback: if all terms were stop words, keep original
    if plan.terms.is_empty() && !raw_terms.is_empty() {
        plan.terms = raw_terms.clone();
    }

    // Build implicit phrase from consecutive terms (2-gram)
    if raw_terms.len() >= 2 && plan.phrases.is_empty() {
        plan.phrases.push(raw_terms.join(" "));
    }

    // 4. Intent detection
    plan.is_code_query = CODE_SIGNALS.is_match(query);
    plan.is_structural_query = STRUCT_SIGNALS.is_match(query);

    plan
}

// ---------------------------------------------------------------------------
// Term overlap ratio (content-aware scoring)
// ---------------------------------------------------------------------------

/// Compute IDF-weighted fraction of query terms that appear in the text.
///
/// When `idf` is provided, rare terms contribute more than common terms.
/// Falls back to uniform weighting when `idf` is None.
///
/// Returns a value in `[0.0, 1.0]`.
pub fn compute_term_overlap(
    text: &str,
    terms: &[String],
    idf: Option<&HashMap<String, f64>>,
) -> f64 {
    if text.is_empty() || terms.is_empty() {
        return 0.0;
    }
    let text_lower = text.to_lowercase();

    if let Some(idf_map) = idf {
        let total_w: f64 = terms.iter().map(|t| idf_map.get(t).copied().unwrap_or(1.0)).sum();
        if total_w <= 0.0 {
            return 0.0;
        }
        let hit_w: f64 = terms
            .iter()
            .filter(|t| text_lower.contains(t.as_str()))
            .map(|t| idf_map.get(t).copied().unwrap_or(1.0))
            .sum();
        hit_w / total_w
    } else {
        // Uniform fallback
        let matched = terms.iter().filter(|t| text_lower.contains(t.as_str())).count();
        matched as f64 / terms.len() as f64
    }
}

/// Estimate IDF weights for query terms from a corpus of node texts.
///
/// Uses smooth IDF: `log((N + 1) / (df + 1)) + 1` to avoid zero weights.
pub fn estimate_idf(terms: &[String], corpus_texts: &[&str]) -> HashMap<String, f64> {
    let n = corpus_texts.len();
    if n == 0 {
        return terms.iter().map(|t| (t.clone(), 1.0)).collect();
    }

    // Pre-compute document frequency for each term
    let mut df: HashMap<String, usize> = terms.iter().map(|t| (t.clone(), 0)).collect();
    for text in corpus_texts {
        let text_lower = text.to_lowercase();
        for t in terms {
            if text_lower.contains(t.as_str()) {
                *df.get_mut(t).unwrap() += 1;
            }
        }
    }

    let mut idf = HashMap::with_capacity(terms.len());
    for t in terms {
        let doc_freq = *df.get(t).unwrap_or(&0);
        idf.insert(
            t.clone(),
            ((n as f64 + 1.0) / (doc_freq as f64 + 1.0)).ln() + 1.0,
        );
    }
    idf
}

// ---------------------------------------------------------------------------
// Anchor Scorer
// ---------------------------------------------------------------------------

/// Score a candidate anchor node.
///
/// Anchors should be high-level entry points, so deeper nodes get penalized.
/// FTS5 score (which already incorporates body text BM25) is the primary signal.
///
/// Weight breakdown:
///   - depth_penalty: 0.10 (deeper = less ideal as anchor)
///   - title_bonus:   0.15 (query terms in title)
///   - phrase_bonus:  0.07 (exact phrase match)
///   - body_bonus:    0.10 (body term overlap fraction)
pub fn score_anchor(
    fts_score: f64,
    depth: u32,
    has_title_match: bool,
    has_phrase_match: bool,
    body_term_overlap: f64,
    max_depth: u32,
) -> f64 {
    // Depth penalty: deeper nodes are less ideal as anchors
    let depth_penalty =
        (depth as f64 / (max_depth.max(1) as f64)).min(1.0) * 0.10;

    // Title match bonus
    let title_bonus = if has_title_match { 0.15 } else { 0.0 };

    // Phrase match bonus
    let phrase_bonus = if has_phrase_match { 0.07 } else { 0.0 };

    // Body content overlap bonus
    let body_bonus = 0.10 * body_term_overlap;

    let score = fts_score + title_bonus + phrase_bonus + body_bonus - depth_penalty;
    score.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Walk Scorer
// ---------------------------------------------------------------------------

/// Score a node during tree walk expansion.
///
/// FTS5 `lexical_score` already captures BM25 relevance over body text.
/// We weight it heavily so that content-rich nodes rank high even when
/// their title is generic.
///
/// Weight breakdown:
///   - 0.45 lexical_score
///   - 0.15 body_term_overlap
///   - 0.08 title match
///   - 0.07 phrase match
///   - 0.12 ancestor support
///   - -0.08 hop penalty
///   - -0.08 redundancy penalty
pub fn score_walk_node(
    lexical_score: f64,
    has_title_match: bool,
    has_phrase_match: bool,
    body_term_overlap: f64,
    ancestor_support: f64,
    hop: u32,
    is_redundant: bool,
    max_hops: u32,
) -> f64 {
    // Base: FTS5 lexical relevance is the primary signal
    let mut score = 0.45 * lexical_score;

    // Body text content overlap
    score += 0.15 * body_term_overlap;

    // Title match bonus
    if has_title_match {
        score += 0.08;
    }

    // Phrase match bonus
    if has_phrase_match {
        score += 0.07;
    }

    // Ancestor support: path consistency
    score += 0.12 * ancestor_support;

    // Hop penalty: further from anchor = less relevant
    let hop_ratio = (hop as f64 / max_hops.max(1) as f64).min(1.0);
    score -= 0.08 * hop_ratio;

    // Redundancy penalty
    if is_redundant {
        score -= 0.08;
    }

    score.max(0.0)
}

// ---------------------------------------------------------------------------
// Path Scorer
// ---------------------------------------------------------------------------

/// Score a complete root-to-leaf path.
///
/// Combines leaf node quality with path-level content coverage.
///
/// Weight breakdown:
///   - 0.30 leaf_score (walk-level quality)
///   - 0.30 leaf_fts_score (FTS5 content relevance)
///   - 0.20 text_coverage (query terms in path texts)
///   - 0.10 title_consistency (path titles with query terms)
///   - 0.08 title_coverage (query terms in all path titles)
///   - 0.07 readability (shorter paths preferred)
pub fn score_path(
    leaf_score: f64,
    path_titles: &[String],
    path_texts: &[String],
    query_terms: &[String],
    path_length: usize,
    leaf_fts_score: f64,
    max_path_length: usize,
) -> f64 {
    // Leaf score dominates (walk-level quality)
    let mut score = 0.30 * leaf_score;

    // Leaf FTS5 score direct contribution
    score += 0.30 * leaf_fts_score;

    // Path content coverage: how many query terms appear in ANY node's text
    if !query_terms.is_empty() && !path_texts.is_empty() {
        let all_text = path_texts.join(" ").to_lowercase();
        let covered = query_terms
            .iter()
            .filter(|t| all_text.contains(t.as_str()))
            .count();
        let text_coverage = covered as f64 / query_terms.len() as f64;
        score += 0.20 * text_coverage;
    }

    // Path title consistency: how many path titles contain query terms
    if !path_titles.is_empty() && !query_terms.is_empty() {
        let match_count = path_titles
            .iter()
            .filter(|title| {
                let title_lower = title.to_lowercase();
                query_terms.iter().any(|t| title_lower.contains(t.as_str()))
            })
            .count();
        let consistency = match_count as f64 / path_titles.len() as f64;
        score += 0.10 * consistency;
    }

    // Context coverage: how many query terms appear somewhere in path titles
    if !query_terms.is_empty() && !path_titles.is_empty() {
        let all_titles_text = path_titles.join(" ").to_lowercase();
        let covered = query_terms
            .iter()
            .filter(|t| all_titles_text.contains(t.as_str()))
            .count();
        let coverage = covered as f64 / query_terms.len() as f64;
        score += 0.08 * coverage;
    }

    // Readability bonus: shorter paths are easier to present
    let length_ratio = (path_length as f64 / max_path_length.max(1) as f64).min(1.0);
    let readability = 1.0 - length_ratio * 0.5;
    score += 0.07 * readability;

    score.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Utility: term matching helpers
// ---------------------------------------------------------------------------

/// Check if any query term appears in the node title.
pub fn check_title_match(title: &str, terms: &[String]) -> bool {
    if title.is_empty() || terms.is_empty() {
        return false;
    }
    let title_lower = title.to_lowercase();
    terms.iter().any(|t| title_lower.contains(t.as_str()))
}

/// Check if any exact phrase appears in the text.
pub fn check_phrase_match(text: &str, phrases: &[String]) -> bool {
    if text.is_empty() || phrases.is_empty() {
        return false;
    }
    let text_lower = text.to_lowercase();
    phrases.iter().any(|p| text_lower.contains(&p.to_lowercase()))
}

/// Check if a node is a generic overview section.
///
/// Only applies to top-level sections (depth 0-1) whose base title
/// (before `:::` delimiter) is in the generic set.
pub fn is_generic_section(title: &str, depth: u32) -> bool {
    if depth > 1 {
        return false;
    }
    if title.is_empty() {
        return false;
    }
    // For ::: delimited titles, only check the base (leftmost) part
    let base_title = if title.contains(" ::: ") {
        title.split(" ::: ").next().unwrap_or("").trim()
    } else {
        title.trim()
    };
    let base_lower = base_title.to_lowercase();

    // Root node (depth=0, paper title) is almost never relevant
    if depth == 0 {
        return true;
    }
    GENERIC_SECTIONS.contains(base_lower.as_str())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- QueryPlan --

    #[test]
    fn test_build_query_plan_simple() {
        let plan = build_query_plan("machine learning models");
        assert_eq!(plan.raw, "machine learning models");
        assert!(plan.terms.contains(&"machine".to_string()));
        assert!(plan.terms.contains(&"learning".to_string()));
        assert!(plan.terms.contains(&"models".to_string()));
        assert!(!plan.is_code_query);
        assert!(!plan.is_structural_query);
    }

    #[test]
    fn test_build_query_plan_quoted_phrases() {
        let plan = build_query_plan(r#""neural network" training"#);
        assert_eq!(plan.phrases, vec!["neural network"]);
        assert!(plan.terms.contains(&"training".to_string()));
    }

    #[test]
    fn test_build_query_plan_code_intent() {
        let plan = build_query_plan("import numpy function");
        assert!(plan.is_code_query);
    }

    #[test]
    fn test_build_query_plan_structural_intent() {
        let plan = build_query_plan("chapter 3 results");
        assert!(plan.is_structural_query);
    }

    #[test]
    fn test_build_query_plan_chinese_structural() {
        let plan = build_query_plan("第三章 实验结果");
        assert!(plan.is_structural_query);
    }

    #[test]
    fn test_build_query_plan_stop_words_filtered() {
        let plan = build_query_plan("the quick brown fox");
        // "the" is a stop word, should be filtered
        assert!(!plan.terms.contains(&"the".to_string()));
        assert!(plan.terms.contains(&"quick".to_string()));
    }

    #[test]
    fn test_build_query_plan_deduplication() {
        let plan = build_query_plan("test test test");
        // Should be deduplicated to a single "test"
        assert_eq!(plan.terms.iter().filter(|t| *t == "test").count(), 1);
    }

    #[test]
    fn test_build_query_plan_all_stop_words_fallback() {
        let plan = build_query_plan("the is are");
        // All stop words -> fallback keeps them
        assert!(!plan.terms.is_empty());
    }

    #[test]
    fn test_build_query_plan_implicit_phrase() {
        let plan = build_query_plan("machine learning");
        // Should generate implicit phrase from consecutive terms
        assert!(!plan.phrases.is_empty());
        assert!(plan.phrases[0].contains("machine"));
    }

    // -- Term overlap --

    #[test]
    fn test_compute_term_overlap_uniform() {
        let terms = vec!["hello".into(), "world".into()];
        assert_eq!(
            compute_term_overlap("hello world example", &terms, None),
            1.0
        );
        assert_eq!(
            compute_term_overlap("hello example", &terms, None),
            0.5
        );
        assert_eq!(
            compute_term_overlap("nothing here", &terms, None),
            0.0
        );
    }

    #[test]
    fn test_compute_term_overlap_empty() {
        let terms: Vec<String> = vec!["hello".into()];
        assert_eq!(compute_term_overlap("", &terms, None), 0.0);
        assert_eq!(compute_term_overlap("hello", &[], None), 0.0);
    }

    #[test]
    fn test_compute_term_overlap_idf() {
        let terms = vec!["rare".into(), "common".into()];
        let mut idf = HashMap::new();
        idf.insert("rare".to_string(), 3.0);
        idf.insert("common".to_string(), 1.0);
        // Only "rare" present -> 3.0 / 4.0 = 0.75
        let overlap = compute_term_overlap("this is rare", &terms, Some(&idf));
        assert!((overlap - 0.75).abs() < 1e-9);
    }

    // -- IDF estimation --

    #[test]
    fn test_estimate_idf_empty_corpus() {
        let terms = vec!["hello".into(), "world".into()];
        let idf = estimate_idf(&terms, &[]);
        assert_eq!(idf["hello"], 1.0);
        assert_eq!(idf["world"], 1.0);
    }

    #[test]
    fn test_estimate_idf_basic() {
        let terms = vec!["rare".into(), "common".into()];
        let corpus = vec!["common word here", "common and more", "rare item"];
        let idf = estimate_idf(&terms, &corpus);
        // "common" appears in 2/3 docs -> log(4/3) + 1
        // "rare" appears in 1/3 docs -> log(4/2) + 1
        assert!(idf["rare"] > idf["common"]);
    }

    // -- Anchor scoring --

    #[test]
    fn test_score_anchor_basic() {
        let score = score_anchor(0.8, 0, true, false, 0.5, 6);
        // 0.8 + 0.15 + 0.0 + 0.05 - 0.0 = 1.0 (clamped)
        assert!(score > 0.9);
    }

    #[test]
    fn test_score_anchor_deep_penalty() {
        let shallow = score_anchor(0.5, 0, false, false, 0.0, 6);
        let deep = score_anchor(0.5, 6, false, false, 0.0, 6);
        // Deep node should score lower due to depth penalty
        assert!(shallow > deep);
    }

    #[test]
    fn test_score_anchor_title_bonus() {
        let with_title = score_anchor(0.5, 1, true, false, 0.0, 6);
        let without_title = score_anchor(0.5, 1, false, false, 0.0, 6);
        assert!((with_title - without_title - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_score_anchor_phrase_bonus() {
        let with_phrase = score_anchor(0.5, 1, false, true, 0.0, 6);
        let without_phrase = score_anchor(0.5, 1, false, false, 0.0, 6);
        assert!((with_phrase - without_phrase - 0.07).abs() < 1e-9);
    }

    #[test]
    fn test_score_anchor_body_bonus() {
        let with_body = score_anchor(0.5, 1, false, false, 1.0, 6);
        let without_body = score_anchor(0.5, 1, false, false, 0.0, 6);
        assert!((with_body - without_body - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_score_anchor_clamped() {
        // Should clamp to [0, 1]
        let high = score_anchor(1.0, 0, true, true, 1.0, 6);
        assert!(high <= 1.0);
        let low = score_anchor(0.0, 6, false, false, 0.0, 6);
        assert!(low >= 0.0);
    }

    // -- Walk scoring --

    #[test]
    fn test_score_walk_node_basic() {
        let score = score_walk_node(0.8, true, true, 0.5, 0.6, 0, false, 3);
        // 0.45*0.8 + 0.15*0.5 + 0.08 + 0.07 + 0.12*0.6 - 0.0 - 0.0
        // = 0.36 + 0.075 + 0.08 + 0.07 + 0.072 = 0.657
        assert!((score - 0.657).abs() < 1e-9);
    }

    #[test]
    fn test_score_walk_node_hop_penalty() {
        let near = score_walk_node(0.5, false, false, 0.0, 0.0, 0, false, 3);
        let far = score_walk_node(0.5, false, false, 0.0, 0.0, 3, false, 3);
        assert!(near > far);
        // Difference should be 0.08
        assert!((near - far - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_score_walk_node_redundancy_penalty() {
        let fresh = score_walk_node(0.5, false, false, 0.0, 0.0, 0, false, 3);
        let redundant = score_walk_node(0.5, false, false, 0.0, 0.0, 0, true, 3);
        assert!((fresh - redundant - 0.08).abs() < 1e-9);
    }

    #[test]
    fn test_score_walk_node_floor() {
        // All penalties, no positive signals -> should floor at 0.0
        let score = score_walk_node(0.0, false, false, 0.0, 0.0, 3, true, 3);
        assert_eq!(score, 0.0);
    }

    // -- Path scoring --

    #[test]
    fn test_score_path_basic() {
        let titles = vec!["Machine Learning".into(), "Neural Networks".into()];
        let texts = vec![
            "Introduction to machine learning".into(),
            "Neural networks for classification".into(),
        ];
        let terms = vec!["machine".into(), "learning".into(), "neural".into()];
        let score = score_path(0.6, &titles, &texts, &terms, 2, 0.7, 6);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_score_path_empty_terms() {
        let titles = vec!["Title".into()];
        let texts = vec!["Some text".into()];
        let terms: Vec<String> = vec![];
        let score = score_path(0.5, &titles, &texts, &terms, 1, 0.5, 6);
        // 0.30 * 0.5 + 0.30 * 0.5 + 0.0 + 0.0 + 0.0 + 0.07 * readability
        // readability = 1.0 - (1/6)*0.5 = 0.9167
        // = 0.15 + 0.15 + 0.07 * 0.9167 ≈ 0.364
        assert!(score > 0.3);
    }

    #[test]
    fn test_score_path_readability() {
        let terms = vec!["test".into()];
        let titles: Vec<String> = vec![];
        let texts: Vec<String> = vec![];
        // Short path is more readable
        let short = score_path(0.5, &titles, &texts, &terms, 1, 0.5, 6);
        let long = score_path(0.5, &titles, &texts, &terms, 6, 0.5, 6);
        assert!(short > long);
    }

    #[test]
    fn test_score_path_clamped() {
        let titles = vec!["Test".into()];
        let texts = vec!["test content".into()];
        let terms = vec!["test".into()];
        let score = score_path(1.0, &titles, &texts, &terms, 1, 1.0, 6);
        assert!(score <= 1.0);
        assert!(score >= 0.0);
    }

    // -- Title / phrase matching --

    #[test]
    fn test_check_title_match() {
        let terms = vec!["neural".into(), "network".into()];
        assert!(check_title_match("Neural Networks", &terms));
        assert!(check_title_match("NEURAL NETS", &terms));
        assert!(!check_title_match("Machine Learning", &terms));
        assert!(!check_title_match("", &terms));
        assert!(!check_title_match("Neural", &[]));
    }

    #[test]
    fn test_check_phrase_match() {
        let phrases = vec!["neural network".into()];
        assert!(check_phrase_match(
            "The neural network architecture",
            &phrases
        ));
        assert!(check_phrase_match("NEURAL NETWORK", &phrases));
        assert!(!check_phrase_match("neural net", &phrases));
        assert!(!check_phrase_match("", &phrases));
        assert!(!check_phrase_match("some text", &[]));
    }

    // -- Generic section detection --

    #[test]
    fn test_is_generic_section_root() {
        // depth=0 is always generic (root / paper title)
        assert!(is_generic_section("My Paper Title", 0));
    }

    #[test]
    fn test_is_generic_section_known() {
        assert!(is_generic_section("Introduction", 1));
        assert!(is_generic_section("Abstract", 1));
        assert!(is_generic_section("Conclusion", 1));
        assert!(is_generic_section("Related Work", 1));
        assert!(is_generic_section("Background", 1));
        assert!(is_generic_section("Conclusion and Future Work", 1));
    }

    #[test]
    fn test_is_generic_section_not_generic() {
        assert!(!is_generic_section("Neural Architecture Search", 1));
        assert!(!is_generic_section("Experimental Results", 1));
    }

    #[test]
    fn test_is_generic_section_deep_nodes() {
        // depth > 1 is never generic
        assert!(!is_generic_section("Introduction", 2));
        assert!(!is_generic_section("Abstract", 3));
    }

    #[test]
    fn test_is_generic_section_delimited_title() {
        assert!(is_generic_section("Introduction ::: 1.1 Overview", 1));
        assert!(!is_generic_section("Results ::: Accuracy", 2));
    }

    #[test]
    fn test_is_generic_section_empty() {
        assert!(!is_generic_section("", 1));
    }

    // -- Stop words --

    #[test]
    fn test_stop_words_english() {
        assert!(STOP_WORDS.contains("the"));
        assert!(STOP_WORDS.contains("is"));
        assert!(STOP_WORDS.contains("you"));
        assert!(!STOP_WORDS.contains("neural"));
    }

    #[test]
    fn test_stop_words_chinese() {
        assert!(STOP_WORDS.contains("的"));
        assert!(STOP_WORDS.contains("是"));
        assert!(STOP_WORDS.contains("什么"));
    }
}
