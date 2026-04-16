use crate::document::{Document, Node, SourceType};
use anyhow::Result;
use std::path::Path;

/// Parser for plain-text files (`.txt`, `.text`, `.log`, `.csv`).
///
/// Splits content by double newlines (paragraphs). Each paragraph becomes a flat
/// node with the first line (truncated to 80 chars) as the title and the full
/// paragraph text as the body.
pub struct PlainTextParser;

/// Maximum character length for a paragraph title.
const MAX_TITLE_LEN: usize = 80;

impl super::Parser for PlainTextParser {
    fn extensions(&self) -> &[&str] {
        &["txt", "text", "log", "csv"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Text
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let doc_id = path.to_string_lossy().to_string();
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Text);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let paragraphs = split_paragraphs(content);
        let mut line_offset: u32 = 1;

        for para in &paragraphs {
            let trimmed = para.trim();
            if trimmed.is_empty() {
                line_offset += para.lines().count().max(1) as u32;
                continue;
            }

            let title = make_title(trimmed);
            let line_count = trimmed.lines().count() as u32;

            let mut node = Node::new("", &title);
            node.text = trimmed.to_string();
            node.line_start = Some(line_offset);
            node.line_end = Some(line_offset + line_count.saturating_sub(1));

            doc.structure.push(node);

            // Advance past this paragraph text lines + separator gap.
            line_offset += para.lines().count().max(1) as u32 + 1;
        }

        doc.assign_node_ids();
        Ok(doc)
    }
}

/// Split content into paragraphs separated by one or more blank lines.
fn split_paragraphs(content: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'\n' {
            let mut j = i + 1;
            // Skip horizontal whitespace.
            while j < len && (bytes[j] == b' ' || bytes[j] == b'\t' || bytes[j] == b'\r') {
                j += 1;
            }
            if j < len && bytes[j] == b'\n' {
                // Paragraph break found.
                if start < i {
                    result.push(&content[start..i]);
                }
                // Skip past all consecutive blank lines.
                let mut k = j + 1;
                while k < len
                    && (bytes[k] == b'\n'
                        || bytes[k] == b'\r'
                        || bytes[k] == b' '
                        || bytes[k] == b'\t')
                {
                    k += 1;
                }
                start = k;
                i = k;
                continue;
            }
        }
        i += 1;
    }

    // Trailing paragraph.
    if start < len {
        result.push(&content[start..]);
    }

    result
}

/// Create a title from the first line of a paragraph, truncated to `MAX_TITLE_LEN`.
fn make_title(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("");
    let trimmed = first_line.trim();
    if trimmed.len() <= MAX_TITLE_LEN {
        trimmed.to_string()
    } else {
        // Truncate at a char boundary.
        let mut end = MAX_TITLE_LEN;
        while !trimmed.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        format!("{}...", &trimmed[..end])
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use super::*;

    fn parse(content: &str) -> Document {
        let parser = PlainTextParser;
        parser
            .parse(Path::new("notes.txt"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_empty() {
        let doc = parse("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let doc = parse("   \n\n  \n  ");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_single_paragraph() {
        let doc = parse("Hello world.\nThis is a test.");
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].title, "Hello world.");
        assert!(doc.structure[0].text.contains("This is a test"));
    }

    #[test]
    fn test_multiple_paragraphs() {
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 3);
        assert_eq!(doc.structure[0].title, "First paragraph.");
        assert_eq!(doc.structure[1].title, "Second paragraph.");
        assert_eq!(doc.structure[2].title, "Third paragraph.");
    }

    #[test]
    fn test_title_truncation() {
        let long_line = "A".repeat(120);
        let doc = parse(&long_line);
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].title.len() <= MAX_TITLE_LEN + 3);
        assert!(doc.structure[0].title.ends_with("..."));
    }

    #[test]
    fn test_line_numbers() {
        let content = "Para one line1\nPara one line2\n\nPara two.";
        let doc = parse(content);
        assert_eq!(doc.structure[0].line_start, Some(1));
        assert_eq!(doc.structure[0].line_end, Some(2));
    }

    #[test]
    fn test_node_ids_assigned() {
        let content = "A\n\nB\n\nC";
        let doc = parse(content);
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[1].node_id, "1");
        assert_eq!(doc.structure[2].node_id, "2");
    }

    #[test]
    fn test_source_type() {
        let doc = parse("hello");
        assert_eq!(doc.source_type, SourceType::Text);
        assert_eq!(doc.doc_id, "notes.txt");
    }

    #[test]
    fn test_multiple_blank_lines() {
        let content = "First\n\n\n\nSecond";
        let doc = parse(content);
        assert_eq!(doc.structure.len(), 2);
    }

    #[test]
    fn test_flat_structure() {
        let content = "A\n\nB\n\nC";
        let doc = parse(content);
        for node in &doc.structure {
            assert!(node.children.is_empty());
        }
    }

    #[test]
    fn test_unicode_title_truncation() {
        // Each CJK character is 3 bytes in UTF-8.
        let long_cjk = "你".repeat(100);
        let doc = parse(&long_cjk);
        // Should truncate without panicking (char boundary safe).
        assert!(doc.structure[0].title.ends_with("..."));
    }
}
