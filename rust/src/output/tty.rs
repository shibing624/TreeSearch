//! Colored terminal output for search results.
//!
//! Uses `termcolor` for cross-platform ANSI color support and
//! `unicode-width` for correct CJK character alignment.

use std::io::Write;

use termcolor::{Buffer, BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};
use unicode_width::UnicodeWidthStr;

use crate::document::SearchResult;

use super::OutputFormat;

/// Terminal width used for truncation. Falls back to 100.
fn terminal_width() -> usize {
    // Simple fallback — a full implementation could query the terminal.
    100
}

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
            // Leave room for "..."
            result.push_str("...");
            break;
        }
        result.push(ch);
        width += cw;
    }
    result
}

/// Highlight matching query terms in text by wrapping them with ANSI yellow.
/// Returns segments for writing with color support.
struct HighlightSegment {
    text: String,
    is_highlight: bool,
}

fn highlight_terms(text: &str, terms: &[String]) -> Vec<HighlightSegment> {
    if terms.is_empty() {
        return vec![HighlightSegment {
            text: text.to_string(),
            is_highlight: false,
        }];
    }

    let text_lower = text.to_lowercase();
    let mut segments = Vec::new();
    let mut last_end = 0;

    // Collect all match ranges
    let mut matches: Vec<(usize, usize)> = Vec::new();
    for term in terms {
        let term_lower = term.to_lowercase();
        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(&term_lower) {
            let abs_pos = start + pos;
            matches.push((abs_pos, abs_pos + term.len()));
            start = abs_pos + 1;
        }
    }

    // Sort by start position
    matches.sort_by_key(|m| m.0);

    // Merge overlapping ranges and build segments
    for (mstart, mend) in matches {
        if mstart < last_end {
            continue; // Skip overlapping
        }
        if mstart > last_end {
            segments.push(HighlightSegment {
                text: text[last_end..mstart].to_string(),
                is_highlight: false,
            });
        }
        segments.push(HighlightSegment {
            text: text[mstart..mend].to_string(),
            is_highlight: true,
        });
        last_end = mend;
    }

    if last_end < text.len() {
        segments.push(HighlightSegment {
            text: text[last_end..].to_string(),
            is_highlight: false,
        });
    }

    if segments.is_empty() {
        segments.push(HighlightSegment {
            text: text.to_string(),
            is_highlight: false,
        });
    }

    segments
}

/// TTY output formatter with colors and tree structure.
pub struct TtyOutput {
    /// Query terms for highlighting matches.
    pub query_terms: Vec<String>,
}

impl TtyOutput {
    pub fn new(query_terms: Vec<String>) -> Self {
        Self { query_terms }
    }

    /// Write a single result entry to the buffer with colors.
    fn write_result(&self, buf: &mut Buffer, result: &SearchResult, _verbose: u8) -> std::io::Result<()> {
        let tw = terminal_width();

        // Header: source_path [source_type] (score: X.XX)
        buf.set_color(ColorSpec::new().set_fg(Some(Color::Green)).set_bold(true))?;
        write!(buf, "{}", result.source_path)?;
        buf.set_color(ColorSpec::new().set_fg(Some(Color::Cyan)))?;
        write!(buf, " [{}]", result.source_type)?;
        buf.set_color(ColorSpec::new().set_fg(Some(Color::White)).set_dimmed(true))?;
        writeln!(buf, " (score: {:.2})", result.score)?;
        buf.reset()?;

        // Breadcrumb path with tree chars
        if !result.breadcrumb.is_empty() {
            let crumb_len = result.breadcrumb.len();
            for (i, crumb) in result.breadcrumb.iter().enumerate() {
                let is_last = i == crumb_len - 1;
                let prefix = if is_last { "  └── " } else { "  ├── " };

                buf.set_color(ColorSpec::new().set_fg(Some(Color::White)).set_dimmed(true))?;
                write!(buf, "{}", prefix)?;
                buf.reset()?;

                let truncated = truncate_to_width(crumb, tw.saturating_sub(8));
                write!(buf, "{}", truncated)?;
                writeln!(buf)?;

                // Vertical connector for non-last items
                if !is_last {
                    buf.set_color(ColorSpec::new().set_fg(Some(Color::White)).set_dimmed(true))?;
                    buf.reset()?;
                }
            }
        }

        // Title (bold)
        if !result.title.is_empty() {
            buf.set_color(ColorSpec::new().set_bold(true))?;
            write!(buf, "  Title: ")?;
            let segments = highlight_terms(&result.title, &self.query_terms);
            for seg in &segments {
                if seg.is_highlight {
                    buf.set_color(
                        ColorSpec::new()
                            .set_fg(Some(Color::Yellow))
                            .set_bold(true),
                    )?;
                } else {
                    buf.set_color(ColorSpec::new().set_bold(true))?;
                }
                write!(buf, "{}", seg.text)?;
            }
            buf.reset()?;
            writeln!(buf)?;
        }

        // Summary (normal)
        if !result.summary.is_empty() {
            let summary = truncate_to_width(&result.summary, tw.saturating_sub(12));
            write!(buf, "  Summary: ")?;
            let segments = highlight_terms(&summary, &self.query_terms);
            for seg in &segments {
                if seg.is_highlight {
                    buf.set_color(
                        ColorSpec::new()
                            .set_fg(Some(Color::Yellow))
                            .set_bold(true),
                    )?;
                } else {
                    buf.reset()?;
                }
                write!(buf, "{}", seg.text)?;
            }
            buf.reset()?;
            writeln!(buf)?;
        }

        // Text snippet (verbose >= 2)
        if _verbose >= 2 && !result.text.is_empty() {
            let text = truncate_to_width(&result.text, tw.saturating_sub(12));
            write!(buf, "  Text: ")?;
            let segments = highlight_terms(&text, &self.query_terms);
            for seg in &segments {
                if seg.is_highlight {
                    buf.set_color(
                        ColorSpec::new()
                            .set_fg(Some(Color::Yellow))
                            .set_bold(true),
                    )?;
                } else {
                    buf.reset()?;
                }
                write!(buf, "{}", seg.text)?;
            }
            buf.reset()?;
            writeln!(buf)?;
        }

        // Line numbers (dim)
        if let (Some(start), Some(end)) = (result.line_start, result.line_end) {
            buf.set_color(ColorSpec::new().set_fg(Some(Color::White)).set_dimmed(true))?;
            writeln!(buf, "  Lines: {}-{}", start, end)?;
            buf.reset()?;
        }

        writeln!(buf)?; // Blank line between results
        Ok(())
    }
}

impl OutputFormat for TtyOutput {
    fn render(&self, results: &[SearchResult], verbose: u8) -> String {
        let writer = BufferWriter::stdout(ColorChoice::Auto);
        let mut buf = writer.buffer();

        if results.is_empty() {
            let _ = writeln!(buf, "No results found.");
            return String::from_utf8_lossy(buf.as_slice()).to_string();
        }

        for (i, result) in results.iter().enumerate() {
            // Result number header
            let _ = buf.set_color(ColorSpec::new().set_fg(Some(Color::White)).set_dimmed(true));
            let _ = writeln!(buf, "── Result {} ──", i + 1);
            let _ = buf.reset();
            let _ = self.write_result(&mut buf, result, verbose);
        }

        String::from_utf8_lossy(buf.as_slice()).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> SearchResult {
        SearchResult {
            node_id: "1".into(),
            doc_id: "doc1".into(),
            doc_name: "test.rs".into(),
            title: "Machine Learning Overview".into(),
            summary: "An overview of machine learning techniques".into(),
            text: "Machine learning is a subset of AI".into(),
            source_type: "code".into(),
            source_path: "src/test.rs".into(),
            line_start: Some(10),
            line_end: Some(25),
            score: 0.85,
            depth: 1,
            breadcrumb: vec!["Root".into(), "Chapter 1".into(), "ML Overview".into()],
        }
    }

    #[test]
    fn test_tty_output_nonempty() {
        let tty = TtyOutput::new(vec!["machine".into()]);
        let output = tty.render(&[sample_result()], 1);
        assert!(!output.is_empty());
        assert!(output.contains("src/test.rs"));
        assert!(output.contains("0.85"));
    }

    #[test]
    fn test_tty_output_empty_results() {
        let tty = TtyOutput::new(vec![]);
        let output = tty.render(&[], 0);
        assert!(output.contains("No results"));
    }

    #[test]
    fn test_truncate_to_width_short() {
        assert_eq!(truncate_to_width("hello", 100), "hello");
    }

    #[test]
    fn test_truncate_to_width_long() {
        let long = "a".repeat(200);
        let truncated = truncate_to_width(&long, 50);
        assert!(truncated.ends_with("..."));
        assert!(truncated.width() <= 50);
    }

    #[test]
    fn test_highlight_terms_no_terms() {
        let segs = highlight_terms("hello world", &[]);
        assert_eq!(segs.len(), 1);
        assert!(!segs[0].is_highlight);
    }

    #[test]
    fn test_highlight_terms_basic() {
        let terms = vec!["hello".into()];
        let segs = highlight_terms("say hello world", &terms);
        assert!(segs.iter().any(|s| s.is_highlight && s.text == "hello"));
    }

    #[test]
    fn test_highlight_terms_no_match() {
        let terms = vec!["xyz".into()];
        let segs = highlight_terms("hello world", &terms);
        assert!(segs.iter().all(|s| !s.is_highlight));
    }
}
