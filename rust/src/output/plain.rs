//! Plain text output for search results (no ANSI escape codes).
//!
//! Same layout as TTY output but suitable for piping to other tools.

use unicode_width::UnicodeWidthStr;

use crate::document::SearchResult;

use super::OutputFormat;

/// Max display width for plain text output.
const MAX_WIDTH: usize = 100;

/// Truncate a string to fit within `max_width` display columns,
/// accounting for CJK double-width characters.
fn truncate_to_width(s: &str, max_width: usize) -> String {
    if s.width() <= max_width {
        return s.to_string();
    }
    let mut width = 0;
    let mut result = String::new();
    for ch in s.chars() {
        let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + cw + 3 > max_width {
            result.push_str("...");
            break;
        }
        result.push(ch);
        width += cw;
    }
    result
}

/// Plain text output formatter (no colors, no ANSI codes).
pub struct PlainOutput;

impl PlainOutput {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PlainOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputFormat for PlainOutput {
    fn render(&self, results: &[SearchResult], verbose: u8) -> String {
        if results.is_empty() {
            return "No results found.\n".to_string();
        }

        let mut out = String::new();

        for (i, result) in results.iter().enumerate() {
            // Result separator
            out.push_str(&format!("── Result {} ──\n", i + 1));

            // Header: source_path [source_type] (score: X.XX)
            out.push_str(&format!(
                "{} [{}] (score: {:.2})\n",
                result.source_path, result.source_type, result.score
            ));

            // Breadcrumb path with tree characters
            if !result.breadcrumb.is_empty() {
                let crumb_len = result.breadcrumb.len();
                for (j, crumb) in result.breadcrumb.iter().enumerate() {
                    let is_last = j == crumb_len - 1;
                    let prefix = if is_last { "  └── " } else { "  ├── " };
                    let truncated = truncate_to_width(crumb, MAX_WIDTH.saturating_sub(8));
                    out.push_str(&format!("{}{}\n", prefix, truncated));
                }
            }

            // Title
            if !result.title.is_empty() {
                out.push_str(&format!(
                    "  Title: {}\n",
                    truncate_to_width(&result.title, MAX_WIDTH.saturating_sub(10))
                ));
            }

            // Summary (verbose >= 1)
            if verbose >= 1 && !result.summary.is_empty() {
                out.push_str(&format!(
                    "  Summary: {}\n",
                    truncate_to_width(&result.summary, MAX_WIDTH.saturating_sub(12))
                ));
            }

            // Text (verbose >= 2)
            if verbose >= 2 && !result.text.is_empty() {
                out.push_str(&format!(
                    "  Text: {}\n",
                    truncate_to_width(&result.text, MAX_WIDTH.saturating_sub(10))
                ));
            }

            // Line numbers
            if let (Some(start), Some(end)) = (result.line_start, result.line_end) {
                out.push_str(&format!("  Lines: {}-{}\n", start, end));
            }

            out.push('\n'); // Blank line between results
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> SearchResult {
        SearchResult {
            node_id: "1".into(),
            doc_id: "doc1".into(),
            doc_name: "test.py".into(),
            title: "Data Processing".into(),
            summary: "Process input data for the pipeline".into(),
            text: "def process_data(input): ...".into(),
            source_type: "code".into(),
            source_path: "src/process.py".into(),
            line_start: Some(42),
            line_end: Some(58),
            score: 0.72,
            depth: 2,
            breadcrumb: vec!["Module".into(), "Pipeline".into(), "Data Processing".into()],
        }
    }

    #[test]
    fn test_plain_output_basic() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 1);
        assert!(output.contains("src/process.py"));
        assert!(output.contains("[code]"));
        assert!(output.contains("0.72"));
        assert!(output.contains("Data Processing"));
    }

    #[test]
    fn test_plain_output_empty() {
        let plain = PlainOutput::new();
        let output = plain.render(&[], 0);
        assert_eq!(output, "No results found.\n");
    }

    #[test]
    fn test_plain_output_no_ansi() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 2);
        // Should not contain ANSI escape sequences
        assert!(!output.contains("\x1b["));
    }

    #[test]
    fn test_plain_output_breadcrumb() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 0);
        assert!(output.contains("├──"));
        assert!(output.contains("└──"));
    }

    #[test]
    fn test_plain_output_verbose_0() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 0);
        assert!(output.contains("Title:"));
        // verbose=0 should NOT include summary
        assert!(!output.contains("Summary:"));
    }

    #[test]
    fn test_plain_output_verbose_1() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 1);
        assert!(output.contains("Summary:"));
        // verbose=1 should NOT include text
        assert!(!output.contains("Text:"));
    }

    #[test]
    fn test_plain_output_verbose_2() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 2);
        assert!(output.contains("Summary:"));
        assert!(output.contains("Text:"));
        assert!(output.contains("Lines: 42-58"));
    }

    #[test]
    fn test_plain_output_line_numbers() {
        let plain = PlainOutput::new();
        let output = plain.render(&[sample_result()], 0);
        assert!(output.contains("Lines: 42-58"));
    }

    #[test]
    fn test_plain_output_multiple_results() {
        let plain = PlainOutput::new();
        let results = vec![sample_result(), sample_result()];
        let output = plain.render(&results, 0);
        assert!(output.contains("Result 1"));
        assert!(output.contains("Result 2"));
    }

    #[test]
    fn test_truncate_to_width_short() {
        assert_eq!(truncate_to_width("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_to_width_exact() {
        let s = "abcde";
        assert_eq!(truncate_to_width(s, 5), "abcde");
    }

    #[test]
    fn test_truncate_to_width_long() {
        let long = "a".repeat(200);
        let truncated = truncate_to_width(&long, 50);
        assert!(truncated.ends_with("..."));
        assert!(truncated.width() <= 50);
    }
}
