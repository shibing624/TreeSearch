use crate::document::{Document, Node, SourceType};
use anyhow::Result;
use scraper::{Html, Node as HtmlNode, Selector};
use std::path::Path;

/// Parser for HTML (`.html`, `.htm`) files.
///
/// Parses h1-h6 headings into a hierarchical tree structure, same nesting logic
/// as the markdown parser: h2 nests under h1, h3 under h2, etc.
/// Text content is extracted by stripping tags and collecting inner text.
pub struct HtmlParser;

impl super::Parser for HtmlParser {
    fn extensions(&self) -> &[&str] {
        &["html", "htm"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Html
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let doc_id = path.to_string_lossy().to_string();
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Html);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let html = Html::parse_document(content);

        // Extract <title> for doc_description.
        if let Ok(sel) = Selector::parse("title") {
            if let Some(title_el) = html.select(&sel).next() {
                let title_text: String = title_el.text().collect::<Vec<_>>().join(" ");
                doc.doc_description = title_text.trim().to_string();
            }
        }

        // Walk the DOM, building a tree from heading elements.
        let mut stack: Vec<(u8, Node)> = Vec::new();
        let mut roots: Vec<Node> = Vec::new();
        let mut current_text = String::new();

        for node_ref in html.tree.nodes() {
            match node_ref.value() {
                HtmlNode::Element(el) => {
                    if let Some(level) = heading_level(el.name()) {
                        // Flush accumulated text to current node.
                        flush_text(&mut stack, &mut roots, &mut current_text);
                        // Collapse deeper headings.
                        collapse_stack(&mut stack, &mut roots, level);
                        // Extract heading text from child text nodes.
                        let mut title = String::new();
                        for child in node_ref.children() {
                            collect_all_text(child, &mut title);
                        }
                        let node = Node::new("", title.trim());
                        stack.push((level, node));
                    }
                }
                HtmlNode::Text(text) => {
                    // Skip text that belongs to heading elements (already collected above).
                    if !is_inside_heading(&html, node_ref.id()) {
                        let t = text.trim();
                        if !t.is_empty() {
                            if !current_text.is_empty() {
                                current_text.push(' ');
                            }
                            current_text.push_str(t);
                        }
                    }
                }
                _ => {}
            }
        }

        // Flush remaining text.
        flush_text(&mut stack, &mut roots, &mut current_text);
        // Collapse entire stack.
        collapse_stack(&mut stack, &mut roots, 0);

        // Fallback: if no headings found, create a single node with all body text.
        if roots.is_empty() && !content.trim().is_empty() {
            let body_text = extract_body_text(&html);
            if !body_text.is_empty() {
                let mut node = Node::new("", &file_name);
                node.text = body_text;
                roots.push(node);
            }
        }

        doc.structure = roots;
        doc.assign_node_ids();
        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn heading_level(tag: &str) -> Option<u8> {
    match tag {
        "h1" => Some(1),
        "h2" => Some(2),
        "h3" => Some(3),
        "h4" => Some(4),
        "h5" => Some(5),
        "h6" => Some(6),
        _ => None,
    }
}

/// Check if a node is a descendant of any heading element.
fn is_inside_heading(html: &Html, node_id: ego_tree::NodeId) -> bool {
    let mut current = Some(node_id);
    while let Some(id) = current {
        if let Some(node) = html.tree.get(id) {
            if let HtmlNode::Element(el) = node.value() {
                if heading_level(el.name()).is_some() {
                    return true;
                }
            }
            current = node.parent().map(|p| p.id());
        } else {
            break;
        }
    }
    false
}


/// Recursively collect all text from a node and its descendants.
fn collect_all_text(node_ref: ego_tree::NodeRef<'_, scraper::Node>, out: &mut String) {
    match node_ref.value() {
        HtmlNode::Text(t) => {
            out.push_str(t);
        }
        _ => {
            for child in node_ref.children() {
                collect_all_text(child, out);
            }
        }
    }
}

/// Extract all visible text from <body>, or from the whole document as fallback.
fn extract_body_text(html: &Html) -> String {
    let mut parts = Vec::new();
    if let Ok(sel) = Selector::parse("body") {
        if let Some(body) = html.select(&sel).next() {
            for text in body.text() {
                let t = text.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
        }
    }
    if parts.is_empty() {
        for node in html.tree.nodes() {
            if let HtmlNode::Text(t) = node.value() {
                let t = t.trim();
                if !t.is_empty() {
                    parts.push(t.to_string());
                }
            }
        }
    }
    parts.join(" ")
}

fn collapse_stack(stack: &mut Vec<(u8, Node)>, roots: &mut Vec<Node>, target_level: u8) {
    while let Some(&(lvl, _)) = stack.last() {
        if lvl >= target_level {
            let (_, node) = stack.pop().unwrap();
            if let Some(parent) = stack.last_mut() {
                parent.1.children.push(node);
            } else {
                roots.push(node);
            }
        } else {
            break;
        }
    }
}

fn flush_text(stack: &mut Vec<(u8, Node)>, roots: &mut Vec<Node>, text: &mut String) {
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        text.clear();
        return;
    }
    if let Some(top) = stack.last_mut() {
        if top.1.text.is_empty() {
            top.1.text = trimmed;
        } else {
            top.1.text.push_str("\n\n");
            top.1.text.push_str(&trimmed);
        }
    } else {
        let mut node = Node::new("", "");
        node.text = trimmed;
        roots.push(node);
    }
    text.clear();
}

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use super::*;

    fn parse(content: &str) -> Document {
        let parser = HtmlParser;
        parser
            .parse(Path::new("test.html"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_empty() {
        let doc = parse("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_single_heading() {
        let doc = parse("<h1>Hello</h1><p>World</p>");
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].title, "Hello");
        assert!(doc.structure[0].text.contains("World"));
    }

    #[test]
    fn test_nested_headings() {
        let html = r#"
            <h1>Top</h1>
            <p>Intro</p>
            <h2>Section A</h2>
            <p>Text A</p>
            <h3>Sub A1</h3>
            <p>Deep</p>
            <h2>Section B</h2>
            <p>Text B</p>
        "#;
        let doc = parse(html);
        assert_eq!(doc.structure.len(), 1);
        let root = &doc.structure[0];
        assert_eq!(root.title, "Top");
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].title, "Section A");
        assert_eq!(root.children[1].title, "Section B");
        assert_eq!(root.children[0].children.len(), 1);
        assert_eq!(root.children[0].children[0].title, "Sub A1");
    }

    #[test]
    fn test_no_headings() {
        let doc = parse("<p>Just text</p><p>More text</p>");
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].text.contains("Just text"));
    }

    #[test]
    fn test_node_ids_assigned() {
        let doc = parse("<h1>A</h1><h2>B</h2><h2>C</h2>");
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[0].children[0].node_id, "1");
        assert_eq!(doc.structure[0].children[1].node_id, "2");
    }

    #[test]
    fn test_source_type() {
        let doc = parse("<h1>Hi</h1>");
        assert_eq!(doc.source_type, SourceType::Html);
        assert_eq!(doc.doc_id, "test.html");
    }

    #[test]
    fn test_full_html_document() {
        let html = r#"<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Title</h1>
<p>Content here.</p>
</body>
</html>"#;
        let doc = parse(html);
        assert_eq!(doc.doc_description, "Test Page");
        // The parser should produce at least one node from the document
        assert!(!doc.structure.is_empty());
    }

    #[test]
    fn test_multiple_h1() {
        let doc = parse("<h1>First</h1><p>A</p><h1>Second</h1><p>B</p>");
        assert_eq!(doc.structure.len(), 2);
        assert_eq!(doc.structure[0].title, "First");
        assert_eq!(doc.structure[1].title, "Second");
    }

    #[test]
    fn test_heading_with_nested_tags() {
        let doc = parse("<h1>Hello <strong>World</strong></h1><p>Body</p>");
        assert_eq!(doc.structure[0].title, "Hello World");
    }

    #[test]
    fn test_heading_level_fn() {
        assert_eq!(heading_level("h1"), Some(1));
        assert_eq!(heading_level("h6"), Some(6));
        assert_eq!(heading_level("p"), None);
        assert_eq!(heading_level("div"), None);
    }
}
