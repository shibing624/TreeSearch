//! Tokenizer module for TreeSearch FTS5 and search indexing.
//!
//! Supports CJK (Chinese, Japanese, Korean) and English tokenization.
//! CJK tokenization mode is configurable via `CjkTokenizerMode` in config.
//!
//! Ported from Python `treesearch/tokenizer.py` and `treesearch/fts.py`.

pub mod cjk;

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::config::CjkTokenizerMode;

// ---------------------------------------------------------------------------
// Global regex patterns
// ---------------------------------------------------------------------------

/// Regex to detect CJK characters (Chinese, Japanese, Korean).
pub static RE_HAS_CJK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\uf900-\ufaff]")
        .unwrap()
});

/// Regex to split English text on non-word boundaries.
static RE_SPLIT_EN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\W+").unwrap());

/// Regex to strip FTS5 special characters.
/// Keeps only word characters (letters, digits, underscore) and CJK ranges.
static RE_FTS5_SPECIAL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[^\w\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
        .unwrap()
});

/// FTS5 operators that must NOT be tokenized.
const FTS5_OPERATORS: &[&str] = &["AND", "OR", "NOT", "NEAR"];

// ---------------------------------------------------------------------------
// Stop words (ported from Python treesearch/heuristics.py)
// ---------------------------------------------------------------------------

/// English stop words.
static EN_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
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
        "nor", "need", "dare", "him", "his", "her", "them",
        "any", "again", "under", "over",
    ]
    .into_iter()
    .collect()
});

/// Chinese stop words.
static ZH_STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
        "吗", "吧", "呢", "啊", "呀", "哦", "嗯", "嘛", "哈",
        "怎样", "怎么", "什么", "哪", "哪个", "哪些", "为什么", "如何",
        "可以", "能", "把", "被", "让", "给", "对", "从", "向", "跟",
        "还", "又", "再", "已", "已经", "正在", "将", "将要",
    ]
    .into_iter()
    .collect()
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether a string contains any CJK character.
pub fn has_cjk(text: &str) -> bool {
    RE_HAS_CJK.is_match(text)
}

/// Tokenize text for FTS5 indexing. Returns space-separated tokens.
///
/// Only uses jieba for CJK text; English text is returned as-is
/// (FTS5 unicode61 handles it natively).
///
/// Equivalent to Python's `_tokenize_for_fts()` in `fts.py`.
pub fn tokenize_for_fts(text: &str, cjk_mode: CjkTokenizerMode) -> String {
    if text.is_empty() || text.trim().is_empty() {
        return String::new();
    }
    if RE_HAS_CJK.is_match(text) {
        let tokens = tokenize(text, false, cjk_mode);
        tokens.join(" ")
    } else {
        // English / non-CJK: return as-is, FTS5 unicode61 handles tokenization
        text.to_string()
    }
}

/// Tokenize a query string, preserving FTS5 operators (AND, OR, NOT, NEAR).
///
/// Non-operator parts are tokenized via `tokenize_for_fts`. FTS5 special
/// characters are stripped from each token.
///
/// Equivalent to Python's `_tokenize_fts_expression()` in `fts.py`.
pub fn tokenize_fts_expression(expr: &str, cjk_mode: CjkTokenizerMode) -> String {
    let parts: Vec<&str> = expr.split_whitespace().collect();
    let mut result = Vec::new();

    for part in parts {
        if FTS5_OPERATORS.contains(&part.to_uppercase().as_str()) {
            result.push(part.to_uppercase());
        } else {
            // Strip FTS5 special characters
            let cleaned = RE_FTS5_SPECIAL.replace_all(part, "");
            if cleaned.is_empty() {
                continue;
            }
            let tokenized = tokenize_for_fts(&cleaned, cjk_mode);
            let trimmed = tokenized.trim().to_string();
            if !trimmed.is_empty() {
                result.push(trimmed);
            }
        }
    }

    result.join(" ")
}

/// General tokenization with optional stopword removal.
///
/// CJK tokenization mode is determined by `cjk_mode`:
///   - `Auto` / `Jieba`: jieba word segmentation
///   - `Bigram`: CJK character 2-grams
///   - `Char`: single-character splitting
///
/// Equivalent to Python's `tokenize()` in `tokenizer.py`.
pub fn tokenize(text: &str, remove_stopwords: bool, cjk_mode: CjkTokenizerMode) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let is_cjk = RE_HAS_CJK.is_match(text);

    let raw_tokens: Vec<String> = if is_cjk {
        match cjk_mode {
            CjkTokenizerMode::Jieba | CjkTokenizerMode::Auto => cjk::jieba_cut(text),
            CjkTokenizerMode::Bigram => cjk::bigrams(text),
            CjkTokenizerMode::Char => cjk::chars(text),
        }
    } else {
        RE_SPLIT_EN
            .split(text)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    };

    // Filter: keep CJK single chars, require len>1 for English tokens
    let filtered: Vec<String> = raw_tokens
        .into_iter()
        .filter(|t| {
            let trimmed = t.trim();
            if trimmed.is_empty() {
                return false;
            }
            if trimmed.chars().count() == 1 {
                // Single char: only keep if CJK
                let ch = trimmed.chars().next().unwrap();
                is_cjk_char(ch)
            } else {
                true
            }
        })
        .map(|t| t.trim().to_string())
        .collect();

    if !remove_stopwords {
        return filtered;
    }

    // Remove stop words
    filtered
        .into_iter()
        .filter(|t| {
            let lower = t.to_lowercase();
            !EN_STOPWORDS.contains(lower.as_str()) && !ZH_STOPWORDS.contains(t.as_str())
        })
        .collect()
}

/// Check if a character is in CJK ranges (inline, no regex).
fn is_cjk_char(ch: char) -> bool {
    matches!(ch,
        '\u{4e00}'..='\u{9fff}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4dbf}' // CJK Unified Ideographs Extension A
        | '\u{3040}'..='\u{309f}' // Hiragana
        | '\u{30a0}'..='\u{30ff}' // Katakana
        | '\u{ac00}'..='\u{d7af}' // Hangul Syllables
        | '\u{f900}'..='\u{faff}' // CJK Compatibility Ideographs
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- has_cjk ---

    #[test]
    fn test_has_cjk_chinese() {
        assert!(has_cjk("hello你好"));
        assert!(has_cjk("机器学习"));
    }

    #[test]
    fn test_has_cjk_japanese() {
        assert!(has_cjk("こんにちは"));
        assert!(has_cjk("カタカナ"));
    }

    #[test]
    fn test_has_cjk_korean() {
        assert!(has_cjk("한글"));
    }

    #[test]
    fn test_has_cjk_english_only() {
        assert!(!has_cjk("hello world"));
        assert!(!has_cjk("rust programming 123"));
    }

    #[test]
    fn test_has_cjk_empty() {
        assert!(!has_cjk(""));
    }

    // --- tokenize_for_fts ---

    #[test]
    fn test_tokenize_for_fts_english() {
        let result = tokenize_for_fts("hello world test", CjkTokenizerMode::Auto);
        // English text returned as-is for FTS5 unicode61
        assert_eq!(result, "hello world test");
    }

    #[test]
    fn test_tokenize_for_fts_chinese() {
        let result = tokenize_for_fts("机器学习是人工智能", CjkTokenizerMode::Auto);
        // Should be space-separated jieba tokens
        assert!(!result.is_empty());
        assert!(result.contains("机器"));
        assert!(result.contains("学习"));
    }

    #[test]
    fn test_tokenize_for_fts_empty() {
        assert_eq!(tokenize_for_fts("", CjkTokenizerMode::Auto), "");
        assert_eq!(tokenize_for_fts("   ", CjkTokenizerMode::Auto), "");
    }

    #[test]
    fn test_tokenize_for_fts_bigram_mode() {
        let result = tokenize_for_fts("机器学习", CjkTokenizerMode::Bigram);
        assert!(result.contains("机器"));
        assert!(result.contains("器学"));
        assert!(result.contains("学习"));
    }

    #[test]
    fn test_tokenize_for_fts_char_mode() {
        let result = tokenize_for_fts("机器学习", CjkTokenizerMode::Char);
        assert!(result.contains("机"));
        assert!(result.contains("器"));
        assert!(result.contains("学"));
        assert!(result.contains("习"));
    }

    // --- tokenize_fts_expression ---

    #[test]
    fn test_fts_expression_preserves_operators() {
        let result = tokenize_fts_expression(
            "machine AND learning",
            CjkTokenizerMode::Auto,
        );
        assert!(result.contains("AND"));
        assert!(result.contains("machine"));
        assert!(result.contains("learning"));
    }

    #[test]
    fn test_fts_expression_or_operator() {
        let result = tokenize_fts_expression("rust OR python", CjkTokenizerMode::Auto);
        assert!(result.contains("OR"));
    }

    #[test]
    fn test_fts_expression_not_operator() {
        let result = tokenize_fts_expression("search NOT vector", CjkTokenizerMode::Auto);
        assert!(result.contains("NOT"));
    }

    #[test]
    fn test_fts_expression_near_operator() {
        let result = tokenize_fts_expression("tree NEAR search", CjkTokenizerMode::Auto);
        assert!(result.contains("NEAR"));
    }

    #[test]
    fn test_fts_expression_case_insensitive_operators() {
        let result = tokenize_fts_expression("rust and python", CjkTokenizerMode::Auto);
        assert!(result.contains("AND"));
    }

    #[test]
    fn test_fts_expression_chinese() {
        let result = tokenize_fts_expression(
            "机器学习 AND 深度学习",
            CjkTokenizerMode::Auto,
        );
        assert!(result.contains("AND"));
        assert!(result.contains("机器"));
    }

    #[test]
    fn test_fts_expression_strips_special_chars() {
        let result = tokenize_fts_expression(
            "hello! AND world?",
            CjkTokenizerMode::Auto,
        );
        assert!(result.contains("AND"));
        // Special chars should be stripped
        assert!(!result.contains("!"));
        assert!(!result.contains("?"));
    }

    // --- tokenize (general) ---

    #[test]
    fn test_tokenize_english_no_stopwords() {
        let tokens = tokenize("hello world test", false, CjkTokenizerMode::Auto);
        assert_eq!(tokens, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_tokenize_english_with_stopword_removal() {
        let tokens = tokenize(
            "the quick brown fox is a test",
            true,
            CjkTokenizerMode::Auto,
        );
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Stop words removed
        assert!(!tokens.contains(&"the".to_string()));
        assert!(!tokens.contains(&"is".to_string()));
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_tokenize_chinese_no_stopwords() {
        let tokens = tokenize("机器学习是人工智能的分支", false, CjkTokenizerMode::Auto);
        assert!(!tokens.is_empty());
        let joined = tokens.join(" ");
        assert!(joined.contains("机器"));
        assert!(joined.contains("学习"));
    }

    #[test]
    fn test_tokenize_chinese_with_stopword_removal() {
        let tokens = tokenize("机器学习是人工智能的分支", true, CjkTokenizerMode::Auto);
        // "是" and "的" are Chinese stop words
        assert!(!tokens.contains(&"是".to_string()));
        assert!(!tokens.contains(&"的".to_string()));
    }

    #[test]
    fn test_tokenize_mixed_text() {
        let tokens = tokenize("Python机器学习", false, CjkTokenizerMode::Auto);
        assert!(!tokens.is_empty());
        // Should contain both English and CJK tokens
        let joined = tokens.join(" ");
        assert!(
            joined.contains("机器") || joined.contains("机"),
            "expected CJK tokens in: {joined}"
        );
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("", false, CjkTokenizerMode::Auto);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenize_filters_single_english_chars() {
        // Single English characters (non-CJK) should be filtered out
        let tokens = tokenize("a b c hello", false, CjkTokenizerMode::Auto);
        assert!(!tokens.contains(&"b".to_string()));
        assert!(!tokens.contains(&"c".to_string()));
        assert!(tokens.contains(&"hello".to_string()));
    }

    #[test]
    fn test_tokenize_keeps_single_cjk_chars() {
        let tokens = tokenize("我", false, CjkTokenizerMode::Auto);
        assert!(tokens.contains(&"我".to_string()));
    }

    // --- stopwords ---

    #[test]
    fn test_english_stopwords_coverage() {
        assert!(EN_STOPWORDS.contains("the"));
        assert!(EN_STOPWORDS.contains("is"));
        assert!(EN_STOPWORDS.contains("and"));
        assert!(EN_STOPWORDS.contains("or"));
        assert!(EN_STOPWORDS.contains("not"));
        assert!(EN_STOPWORDS.contains("i"));
        assert!(EN_STOPWORDS.contains("you"));
        assert!(EN_STOPWORDS.contains("we"));
    }

    #[test]
    fn test_chinese_stopwords_coverage() {
        assert!(ZH_STOPWORDS.contains("的"));
        assert!(ZH_STOPWORDS.contains("了"));
        assert!(ZH_STOPWORDS.contains("是"));
        assert!(ZH_STOPWORDS.contains("在"));
        assert!(ZH_STOPWORDS.contains("什么"));
        assert!(ZH_STOPWORDS.contains("为什么"));
    }

    #[test]
    fn test_non_stopword_not_present() {
        assert!(!EN_STOPWORDS.contains("machine"));
        assert!(!EN_STOPWORDS.contains("learning"));
        assert!(!ZH_STOPWORDS.contains("机器"));
        assert!(!ZH_STOPWORDS.contains("学习"));
    }
}
