//! CJK-specific tokenization strategies.
//!
//! Provides jieba word segmentation, bigram, and single-character tokenization
//! for Chinese/Japanese/Korean text. Ported from Python `treesearch/tokenizer.py`.

use jieba_rs::Jieba;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex, RwLock};
use std::time::SystemTime;

use tracing::{info, warn};

/// Global jieba instance (lazy-loaded, thread-safe, mutable).
///
/// Mutable so that user dictionaries can be injected via
/// [`configure_jieba`]. Read-heavy workloads use the `.read()` lock,
/// which has near-zero overhead in the absence of writers.
///
/// Mirrors the Python `_jieba_tokenizer` private Tokenizer instance.
static JIEBA: LazyLock<RwLock<Jieba>> = LazyLock::new(|| RwLock::new(Jieba::new()));

/// Fingerprint of the user-dict configuration last applied to [`JIEBA`].
///
/// Comparing this against the current config lets us skip rebuilding
/// jieba when nothing changed — and force a rebuild when a dict file's
/// mtime/size or the in-memory word lists differ.
type FileSig = (PathBuf, Option<u128>, Option<u64>);
type Fingerprint = (Vec<FileSig>, Vec<String>, Vec<String>);
static USER_DICT_FP: LazyLock<Mutex<Option<Fingerprint>>> = LazyLock::new(|| Mutex::new(None));

/// Compute the fingerprint of a user-dict configuration.
///
/// Files contribute `(path, mtime_ns, size)`; missing files contribute
/// `(path, None, None)` — that way an admin restoring a deleted file
/// also forces a reload.
fn compute_fingerprint(paths: &[PathBuf], words: &[String], del_words: &[String]) -> Fingerprint {
    let file_sigs: Vec<FileSig> = paths
        .iter()
        .map(|p| match std::fs::metadata(p) {
            Ok(meta) => {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos());
                (p.clone(), mtime, Some(meta.len()))
            }
            Err(_) => (p.clone(), None, None),
        })
        .collect();
    (file_sigs, words.to_vec(), del_words.to_vec())
}

/// Parse one in-memory user-word entry of the form `"word [freq] [tag]"`.
///
/// All of these are valid (matching jieba's own API):
///   `"石墨烯"`           -> word, no freq/tag
///   `"石墨烯 9000"`      -> word + freq
///   `"石墨烯 9000 n"`    -> word + freq + tag
///   `"石墨烯 n"`         -> word + tag (when 2nd token is non-numeric)
fn parse_user_word(entry: &str) -> Option<(String, Option<usize>, Option<String>)> {
    let entry = entry.trim();
    if entry.is_empty() {
        return None;
    }
    let parts: Vec<&str> = entry.split_whitespace().collect();
    let word = parts[0].to_string();
    let mut freq: Option<usize> = None;
    let mut tag: Option<String> = None;
    if parts.len() >= 2 {
        match parts[1].parse::<usize>() {
            Ok(n) => freq = Some(n),
            Err(_) => tag = Some(parts[1].to_string()),
        }
    }
    if parts.len() >= 3 {
        tag = Some(parts[2].to_string());
    }
    Some((word, freq, tag))
}

/// Apply user-dict files / words / del_words to a fresh jieba instance.
fn apply_user_dicts(
    jieba: &mut Jieba,
    paths: &[PathBuf],
    words: &[String],
    del_words: &[String],
) {
    for path in paths {
        match File::open(path) {
            Ok(file) => {
                let mut reader = BufReader::new(file);
                if let Err(e) = jieba.load_dict(&mut reader) {
                    warn!("jieba: failed to parse user dict {:?}: {}", path, e);
                } else {
                    info!("jieba: loaded user dict {:?}", path);
                }
            }
            Err(e) => warn!("jieba: failed to open user dict {:?}: {}", path, e),
        }
    }

    for entry in words {
        if let Some((word, freq, tag)) = parse_user_word(entry) {
            jieba.add_word(&word, freq, tag.as_deref());
        }
    }

    // jieba-rs has no `del_word`; emulate by inserting the word with freq=0
    // so it's present in the trie but never selected by the segmenter.
    for word in del_words {
        let w = word.trim();
        if !w.is_empty() {
            jieba.add_word(w, Some(0), None);
        }
    }
}

/// Configure the global jieba instance with custom user dictionaries.
///
/// Idempotent: if the resolved fingerprint matches what was last applied,
/// this is a cheap no-op (just one mutex lock + comparison). Otherwise
/// the global jieba is rebuilt from scratch and the user dicts/words/
/// del_words are re-applied in order.
///
/// Designed to be called once at process startup (e.g. from `main`)
/// after the [`crate::config::TreeSearchConfig`] is finalized. May be
/// called repeatedly — useful in tests and long-running processes.
pub fn configure_jieba(paths: &[PathBuf], words: &[String], del_words: &[String]) {
    let new_fp = compute_fingerprint(paths, words, del_words);

    let mut current_fp = USER_DICT_FP.lock().expect("USER_DICT_FP poisoned");
    if current_fp.as_ref() == Some(&new_fp) {
        return;
    }

    let mut jieba = JIEBA.write().expect("JIEBA RwLock poisoned");
    *jieba = Jieba::new();
    apply_user_dicts(&mut jieba, paths, words, del_words);
    *current_fp = Some(new_fp);
}

/// Convenience wrapper that pulls jieba user-dict fields off
/// `TreeSearchConfig` and forwards them to [`configure_jieba`].
pub fn configure_from(config: &crate::config::TreeSearchConfig) {
    configure_jieba(
        &config.jieba_user_dict_paths,
        &config.jieba_user_words,
        &config.jieba_del_words,
    );
}

/// Reset jieba state (used by tests).
#[cfg(test)]
pub fn reset_jieba() {
    let mut jieba = JIEBA.write().expect("JIEBA RwLock poisoned");
    *jieba = Jieba::new();
    let mut fp = USER_DICT_FP.lock().expect("USER_DICT_FP poisoned");
    *fp = None;
}

/// Tokenize CJK text using jieba word segmentation.
///
/// Equivalent to Python's `_tokenize_cjk_jieba`. Filters out whitespace-only
/// tokens produced by jieba. Honors any user dict installed via
/// [`configure_jieba`].
///
/// # Example
/// ```
/// let tokens = treesearch::tokenizer::cjk::jieba_cut("机器学习是人工智能的子领域");
/// assert!(tokens.contains(&"机器".to_string()));
/// assert!(tokens.contains(&"学习".to_string()));
/// ```
pub fn jieba_cut(text: &str) -> Vec<String> {
    let jieba = JIEBA.read().expect("JIEBA RwLock poisoned");
    jieba
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

    // ---------------------------------------------------------------
    // configure_jieba: custom user dictionary tests
    //
    // The global JIEBA / USER_DICT_FP statics are process-wide, so
    // these tests share state. We use a Mutex to serialize them and
    // call reset_jieba() in setup/teardown to keep things hermetic.
    // ---------------------------------------------------------------

    use std::io::Write;
    use std::sync::Mutex as StdMutex;
    static USERDICT_TEST_LOCK: StdMutex<()> = StdMutex::new(());

    fn with_clean_jieba<F: FnOnce()>(f: F) {
        let _guard = USERDICT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset_jieba();
        f();
        reset_jieba();
    }

    #[test]
    fn test_configure_jieba_inline_pure_word() {
        with_clean_jieba(|| {
            // Baseline: jieba doesn't know this made-up term.
            let baseline = jieba_cut("超级灵魂引擎是新框架");
            assert!(!baseline.iter().any(|t| t == "超级灵魂引擎"));

            // After configure: pure-word entry (no freq, no tag).
            configure_jieba(&[], &["超级灵魂引擎".to_string()], &[]);
            let tokens = jieba_cut("超级灵魂引擎是新框架");
            assert!(tokens.iter().any(|t| t == "超级灵魂引擎"),
                "expected '超级灵魂引擎' in {:?}", tokens);
        });
    }

    #[test]
    fn test_configure_jieba_inline_word_with_freq_and_tag() {
        with_clean_jieba(|| {
            configure_jieba(
                &[],
                &["树搜索引擎 9000 n".to_string()],
                &[],
            );
            let tokens = jieba_cut("树搜索引擎支持FTS5检索");
            assert!(tokens.iter().any(|t| t == "树搜索引擎"),
                "expected '树搜索引擎' in {:?}", tokens);
        });
    }

    #[test]
    fn test_configure_jieba_dict_file() {
        with_clean_jieba(|| {
            let mut tmp = tempfile::NamedTempFile::new().unwrap();
            writeln!(tmp, "图神经网络 9000 n").unwrap();
            writeln!(tmp, "知识蒸馏 9000 n").unwrap();
            tmp.flush().unwrap();

            configure_jieba(&[tmp.path().to_path_buf()], &[], &[]);
            let tokens = jieba_cut("图神经网络与知识蒸馏的结合");
            assert!(tokens.iter().any(|t| t == "图神经网络"));
            assert!(tokens.iter().any(|t| t == "知识蒸馏"));
        });
    }

    #[test]
    fn test_configure_jieba_del_words_suppresses_known_term() {
        with_clean_jieba(|| {
            let baseline = jieba_cut("计算机科学很有趣");
            assert!(baseline.iter().any(|t| t == "计算机科学"));

            configure_jieba(&[], &[], &["计算机科学".to_string()]);
            let tokens = jieba_cut("计算机科学很有趣");
            assert!(!tokens.iter().any(|t| t == "计算机科学"),
                "expected '计算机科学' to be suppressed in {:?}", tokens);
        });
    }

    #[test]
    fn test_configure_jieba_idempotent_no_reload() {
        with_clean_jieba(|| {
            configure_jieba(&[], &["量子纠缠引擎 9000 n".to_string()], &[]);
            let first = jieba_cut("量子纠缠引擎是新概念");
            assert!(first.iter().any(|t| t == "量子纠缠引擎"));

            // Same config → no rebuild, results unchanged.
            configure_jieba(&[], &["量子纠缠引擎 9000 n".to_string()], &[]);
            let second = jieba_cut("量子纠缠引擎是新概念");
            assert_eq!(first, second);
        });
    }

    #[test]
    fn test_configure_jieba_runtime_reload_on_change() {
        with_clean_jieba(|| {
            configure_jieba(&[], &[], &[]);
            let baseline = jieba_cut("超新星协议是新协议");
            assert!(!baseline.iter().any(|t| t == "超新星协议"));

            configure_jieba(&[], &["超新星协议 9000 n".to_string()], &[]);
            let after = jieba_cut("超新星协议是新协议");
            assert!(after.iter().any(|t| t == "超新星协议"));
        });
    }

    #[test]
    fn test_parse_user_word_variants() {
        assert_eq!(
            parse_user_word("石墨烯"),
            Some(("石墨烯".to_string(), None, None)),
        );
        assert_eq!(
            parse_user_word("石墨烯 9000"),
            Some(("石墨烯".to_string(), Some(9000), None)),
        );
        assert_eq!(
            parse_user_word("石墨烯 9000 n"),
            Some(("石墨烯".to_string(), Some(9000), Some("n".to_string()))),
        );
        assert_eq!(
            parse_user_word("石墨烯 n"),
            Some(("石墨烯".to_string(), None, Some("n".to_string()))),
        );
        assert_eq!(parse_user_word(""), None);
        assert_eq!(parse_user_word("   "), None);
    }
}
