pub mod config_file;
pub mod html;
pub mod markdown;
pub mod plaintext;
pub mod treesitter;

use crate::document::{Document, SourceType};
use anyhow::Result;
use std::path::Path;

/// Trait for document parsers.
pub trait Parser: Send + Sync {
    /// File extensions this parser handles (without dot, lowercase).
    fn extensions(&self) -> &[&str];

    /// Source type for parsed documents.
    fn source_type(&self) -> SourceType;

    /// Parse file content into a Document.
    /// Implementations must call `doc.assign_node_ids()` before returning.
    fn parse(&self, path: &Path, content: &str) -> Result<Document>;
}

/// Registry of all parsers. Routes files to the appropriate parser by extension.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn Parser>>,
}

impl ParserRegistry {
    /// Create a new registry with all built-in parsers registered.
    pub fn new() -> Self {
        let parsers: Vec<Box<dyn Parser>> = vec![
            Box::new(markdown::MarkdownParser),
            Box::new(html::HtmlParser),
            Box::new(config_file::JsonParser),
            Box::new(config_file::YamlParser),
            Box::new(config_file::TomlParser),
            Box::new(treesitter::TreeSitterParser),
            // Plaintext last — most generic fallback.
            Box::new(plaintext::PlainTextParser),
        ];
        Self { parsers }
    }

    /// Find the parser for a given file extension (case-insensitive, without dot).
    fn find_parser_by_ext(&self, ext: &str) -> Option<&dyn Parser> {
        let ext_lower = ext.to_lowercase();
        self.parsers
            .iter()
            .find(|p| p.extensions().iter().any(|e| *e == ext_lower))
            .map(|p| p.as_ref())
    }

    /// Parse a file from disk. Returns `Ok(None)` if no parser handles the extension.
    /// Sets `source_path` to the canonical absolute path.
    pub fn parse_file(&self, path: &Path) -> Result<Option<Document>> {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => return Ok(None),
        };

        let parser = match self.find_parser_by_ext(ext) {
            Some(p) => p,
            None => return Ok(None),
        };

        let content = std::fs::read_to_string(path)?;
        let abs_path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut doc = parser.parse(path, &content)?;
        doc.source_path = abs_path.to_string_lossy().to_string();
        Ok(Some(doc))
    }

    /// Parse file content that has already been read. Returns `Ok(None)` if no
    /// parser handles the extension.
    pub fn parse_content(&self, path: &Path, content: &str) -> Result<Option<Document>> {
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e,
            None => return Ok(None),
        };

        let parser = match self.find_parser_by_ext(ext) {
            Some(p) => p,
            None => return Ok(None),
        };

        let mut doc = parser.parse(path, content)?;
        doc.source_path = path.to_string_lossy().to_string();
        Ok(Some(doc))
    }

    /// Check whether a file extension is supported by any registered parser.
    pub fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|ext| self.find_parser_by_ext(ext).is_some())
            .unwrap_or(false)
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_registry_supports() {
        let reg = ParserRegistry::new();
        assert!(reg.supports(Path::new("readme.md")));
        assert!(reg.supports(Path::new("index.html")));
        assert!(reg.supports(Path::new("main.rs")));
        assert!(reg.supports(Path::new("config.json")));
        assert!(reg.supports(Path::new("data.yaml")));
        assert!(reg.supports(Path::new("data.yml")));
        assert!(reg.supports(Path::new("Cargo.toml")));
        assert!(reg.supports(Path::new("notes.txt")));
        assert!(!reg.supports(Path::new("image.png")));
        assert!(!reg.supports(Path::new("noext")));
    }

    #[test]
    fn test_registry_parse_file_markdown() {
        let mut tmp = NamedTempFile::with_suffix(".md").unwrap();
        writeln!(tmp, "# Hello\n\nWorld").unwrap();
        let reg = ParserRegistry::new();
        let doc = reg.parse_file(tmp.path()).unwrap().unwrap();
        assert_eq!(doc.source_type, SourceType::Markdown);
        assert!(!doc.structure.is_empty());
    }

    #[test]
    fn test_registry_parse_file_json() {
        let mut tmp = NamedTempFile::with_suffix(".json").unwrap();
        write!(tmp, r#"{{"key": "value"}}"#).unwrap();
        let reg = ParserRegistry::new();
        let doc = reg.parse_file(tmp.path()).unwrap().unwrap();
        assert_eq!(doc.source_type, SourceType::Json);
    }

    #[test]
    fn test_registry_parse_content() {
        let reg = ParserRegistry::new();
        let doc = reg
            .parse_content(Path::new("test.md"), "# Hello\n\nWorld")
            .unwrap()
            .unwrap();
        assert_eq!(doc.source_type, SourceType::Markdown);
        assert!(!doc.structure.is_empty());
    }

    #[test]
    fn test_registry_unsupported_extension() {
        let tmp = NamedTempFile::with_suffix(".png").unwrap();
        let reg = ParserRegistry::new();
        assert!(reg.parse_file(tmp.path()).unwrap().is_none());
    }

    #[test]
    fn test_registry_no_extension() {
        let reg = ParserRegistry::new();
        assert!(!reg.supports(Path::new("Makefile_no_ext")));
    }

    #[test]
    fn test_registry_case_insensitive_ext() {
        let reg = ParserRegistry::new();
        assert!(reg.supports(Path::new("README.MD")));
        assert!(reg.supports(Path::new("page.HTML")));
        assert!(reg.supports(Path::new("data.JSON")));
    }
}
