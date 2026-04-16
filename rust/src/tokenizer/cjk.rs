//! CJK-specific tokenization strategies.
//!
//! Provides jieba word segmentation, bigram, and single-character tokenization
//! for Chinese/Japanese/Korean text. Ported from Python `treesearch/tokenizer.py`.

use jieba_rs::Jieba;
use std::sync::LazyLock;


/// Global jieba instance (lazy-loaded, thread-safe).
/// Equivalent to Python's `_ensure_jieba()` lazy-load pattern,
/// but zero-cost after first access thanks to `LazyLock`.
static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// Tokenize CJK text using jieba word segmentation.
///
/// Equivalent to Python's `_tokenize_cjk_jieba`. Filters out whitespace-only
/// tokens produced by jieba.
///
/// # Example
/// ```
/// let tokens = treesearch::tokenizer::cjk::jieba_cut("机器学习是人工智能的子领域");
/// assert!(tokens.contains(&"机器".to_string()));
/// assert!(tokens.contains(&"学习".to_string()));
/// ```
pub fn jieba_cut(text: &str) -> Vec<String> {
    JIEBA
        .cut(text, false)
        .into_iter()
        .map(|s| s.to_string())
        .filter(|s| !s.trim().is_empty())
        .collect()
}

/// Bigram tokenization for CJK text (fallback when jieba is not desired).
///
/// CJK characters are paired into overlapping bigrams:
/// "机器学习" -> ["机器", "器学", "学习"]
///
/// Non-CJK characters are emitted individually (lowercased) when non-whitespace.
///
/// # Example
/// ```
/// let tokens = treesearch::tokenizer::cjk::bigrams("机器学习");
/// assert_eq!(tokens, vec!["机器", "器学", "学习"]);
/// ```
pub fn bigrams(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cjk_buf: Vec<char> = Vec::new();

    for ch in text.chars() {
        if is_cjk_char(ch) {
            cjk_buf.push(ch);
        } else {
            if !cjk_buf.is_empty() {
                tokens.extend(bigrams_from_chars(&cjk_buf));
                cjk_buf.clear();
            }
            if !ch.is_whitespace() {
                tokens.push(ch.to_lowercase().to_string());
            }
        }
    }

    // Flush remaining CJK buffer
    if !cjk_buf.is_empty() {
        tokens.extend(bigrams_from_chars(&cjk_buf));
    }

    tokens
}

/// Generate bigrams from a slice of CJK characters.
/// Single-char input is returned as-is.
fn bigrams_from_chars(chars: &[char]) -> Vec<String> {
    if chars.len() <= 1 {
        return chars.iter().map(|c| c.to_string()).collect();
    }
    chars
        .windows(2)
        .map(|w| format!("{}{}", w[0], w[1]))
        .collect()
}

/// Single-character tokenization for CJK text.
///
/// Each CJK character becomes its own token. Non-CJK non-whitespace
/// characters are emitted individually (lowercased).
///
/// # Example
/// ```
/// let tokens = treesearch::tokenizer::cjk::chars("机器学习");
/// assert_eq!(tokens, vec!["机", "器", "学", "习"]);
/// ```
pub fn chars(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for ch in text.chars() {
        if is_cjk_char(ch) {
            tokens.push(ch.to_string());
        } else if !ch.is_whitespace() {
            tokens.push(ch.to_lowercase().to_string());
        }
    }
    tokens
}

/// Check if a character is in the CJK ranges (inline, avoids regex per-char).
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

    #[test]
    fn test_jieba_cut_chinese() {
        let tokens = jieba_cut("机器学习是人工智能的子领域");
        assert!(!tokens.is_empty());
        let joined = tokens.join(" ");
        assert!(joined.contains("机器"), "expected '机器' in: {joined}");
        assert!(joined.contains("学习"), "expected '学习' in: {joined}");
    }

    #[test]
    fn test_jieba_cut_filters_whitespace() {
        let tokens = jieba_cut("  你好  世界  ");
        for t in &tokens {
            assert!(!t.trim().is_empty(), "got whitespace token: '{t}'");
        }
    }

    #[test]
    fn test_jieba_cut_empty() {
        let tokens = jieba_cut("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_bigrams_basic() {
        let tokens = bigrams("机器学习");
        assert_eq!(tokens, vec!["机器", "器学", "学习"]);
    }

    #[test]
    fn test_bigrams_single_char() {
        let tokens = bigrams("学");
        assert_eq!(tokens, vec!["学"]);
    }

    #[test]
    fn test_bigrams_mixed() {
        let tokens = bigrams("机器AI学习");
        assert!(tokens.contains(&"机器".to_string()));
        assert!(tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"i".to_string()));
        assert!(tokens.contains(&"学习".to_string()));
    }

    #[test]
    fn test_chars_basic() {
        let tokens = chars("机器学习");
        assert_eq!(tokens, vec!["机", "器", "学", "习"]);
    }

    #[test]
    fn test_chars_mixed() {
        let tokens = chars("机器AI");
        assert_eq!(tokens, vec!["机", "器", "a", "i"]);
    }

    #[test]
    fn test_chars_empty() {
        let tokens = chars("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_chars_whitespace_only() {
        let tokens = chars("   ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_bigrams_japanese_hiragana() {
        let tokens = bigrams("あいう");
        assert_eq!(tokens, vec!["あい", "いう"]);
    }

    #[test]
    fn test_chars_korean() {
        let tokens = chars("한글");
        assert_eq!(tokens, vec!["한", "글"]);
    }
}
