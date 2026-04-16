use crate::document::{Document, Node, SourceType};
use anyhow::Result;
use regex::Regex;
use std::path::Path;

/// Phase 1 tree-sitter stub: uses regex heuristics to detect function/class
/// boundaries in source code files. Each detected definition becomes a node
/// with the function/class name as title, its body as text, and line numbers.
///
/// Phase 2 will replace the regex approach with actual tree-sitter grammars.
pub struct TreeSitterParser;

/// File extensions handled by this parser (all source code files).
const CODE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "cpp", "h", "hpp", "cs", "rb",
    "php", "swift", "kt", "scala", "sh", "bash", "zsh", "fish", "lua", "r", "m", "mm", "pl",
    "pm", "ex", "exs", "erl", "hs", "ml", "mli", "clj", "cljs", "el", "vim", "sql", "graphql",
    "proto", "tf", "hcl", "zig", "nim", "v", "d", "dart", "cmake", "makefile", "dockerfile",
    "css", "scss", "sass", "less", "vue", "svelte",
];

impl super::Parser for TreeSitterParser {
    fn extensions(&self) -> &[&str] {
        CODE_EXTENSIONS
    }

    fn source_type(&self) -> SourceType {
        SourceType::Code
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let doc_id = path.to_string_lossy().to_string();
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Code);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let boundaries = detect_boundaries(content, &ext);

        if boundaries.is_empty() {
            // No detectable boundaries — treat entire file as one node.
            let mut node = Node::new("", &file_name);
            node.text = content.to_string();
            node.line_start = Some(1);
            node.line_end = Some(content.lines().count().max(1) as u32);
            doc.structure.push(node);
        } else {
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();

            // Capture preamble before the first boundary (imports, comments, etc.)
            if boundaries[0].line > 0 {
                let preamble_text: String = lines[..boundaries[0].line].join("\n");
                let trimmed = preamble_text.trim();
                if !trimmed.is_empty() {
                    let mut node = Node::new("", "(preamble)");
                    node.text = trimmed.to_string();
                    node.line_start = Some(1);
                    node.line_end = Some(boundaries[0].line as u32);
                    doc.structure.push(node);
                }
            }

            for (i, boundary) in boundaries.iter().enumerate() {
                let start = boundary.line;
                let end = if i + 1 < boundaries.len() {
                    boundaries[i + 1].line
                } else {
                    total_lines
                };

                let body: String = lines[start..end].join("\n");
                let trimmed = body.trim_end();

                let mut node = Node::new("", &boundary.name);
                node.text = trimmed.to_string();
                node.line_start = Some(start as u32 + 1); // 1-based
                node.line_end = Some(end as u32);

                doc.structure.push(node);
            }
        }

        doc.assign_node_ids();
        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// Boundary detection
// ---------------------------------------------------------------------------

/// A detected function/class/struct boundary.
struct Boundary {
    /// 0-based line index in the source.
    line: usize,
    /// Extracted definition name (e.g. "main", "Foo", "hello").
    name: String,
}

/// Detect function/class boundaries by matching regex patterns against each line.
fn detect_boundaries(content: &str, ext: &str) -> Vec<Boundary> {
    let patterns = get_patterns(ext);
    if patterns.is_empty() {
        return Vec::new();
    }

    let regexes: Vec<Regex> = patterns
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect();

    let mut boundaries = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        for re in &regexes {
            if let Some(caps) = re.captures(trimmed) {
                let name = caps
                    .get(1)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_else(|| trimmed.to_string());
                boundaries.push(Boundary {
                    line: line_idx,
                    name,
                });
                break; // one match per line
            }
        }
    }

    boundaries
}

/// Return regex patterns for the given language extension.
/// Each pattern must contain a capture group (1) for the definition name.
fn get_patterns(ext: &str) -> Vec<&'static str> {
    match ext {
        // Rust
        "rs" => vec![
            r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
            r"^\s*(?:pub\s+)?struct\s+(\w+)",
            r"^\s*(?:pub\s+)?enum\s+(\w+)",
            r"^\s*(?:pub\s+)?trait\s+(\w+)",
            r"^\s*(?:pub\s+)?impl(?:<[^>]*>)?\s+(\w+)",
            r"^\s*(?:pub\s+)?mod\s+(\w+)",
        ],
        // Python
        "py" => vec![
            r"^(?:async\s+)?def\s+(\w+)",
            r"^class\s+(\w+)",
        ],
        // JavaScript / TypeScript
        "js" | "jsx" | "ts" | "tsx" | "vue" | "svelte" => vec![
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
            r"^\s*(?:export\s+)?class\s+(\w+)",
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>",
            r"^\s*(?:export\s+)?interface\s+(\w+)",
            r"^\s*(?:export\s+)?type\s+(\w+)",
        ],
        // Go
        "go" => vec![
            r"^func\s+(?:\([^)]+\)\s+)?(\w+)",
            r"^type\s+(\w+)\s+struct",
            r"^type\s+(\w+)\s+interface",
        ],
        // Java / Kotlin / Scala
        "java" | "kt" | "scala" => vec![
            r"(?:public|private|protected|static|final|abstract)?\s*(?:class|interface|enum)\s+(\w+)",
            r"(?:public|private|protected|static|final|abstract|override)?\s*(?:fun|void|int|String|boolean|long|double|float|Object|var|val)\s+(\w+)\s*\(",
        ],
        // C / C++
        "c" | "cpp" | "h" | "hpp" | "cc" | "cxx" => vec![
            r"^\s*(?:static\s+)?(?:inline\s+)?(?:virtual\s+)?(?:const\s+)?(?:\w+[\s*&]+)+(\w+)\s*\([^;]*$",
            r"^\s*(?:class|struct|enum)\s+(\w+)",
            r"^\s*namespace\s+(\w+)",
        ],
        // C#
        "cs" => vec![
            r"(?:public|private|protected|internal|static|virtual|abstract|override|async)?\s*(?:class|interface|struct|enum)\s+(\w+)",
            r"(?:public|private|protected|internal|static|virtual|abstract|override|async)?\s*(?:void|int|string|bool|Task|var|dynamic|\w+)\s+(\w+)\s*\(",
        ],
        // Ruby
        "rb" => vec![
            r"^\s*def\s+(\w+)",
            r"^\s*class\s+(\w+)",
            r"^\s*module\s+(\w+)",
        ],
        // PHP
        "php" => vec![
            r"^\s*(?:public|private|protected|static)?\s*function\s+(\w+)",
            r"^\s*class\s+(\w+)",
        ],
        // Swift
        "swift" => vec![
            r"^\s*(?:public|private|internal|open|fileprivate)?\s*(?:static|class)?\s*func\s+(\w+)",
            r"^\s*(?:public|private|internal|open|fileprivate)?\s*class\s+(\w+)",
            r"^\s*(?:public|private|internal|open|fileprivate)?\s*struct\s+(\w+)",
            r"^\s*(?:public|private|internal|open|fileprivate)?\s*enum\s+(\w+)",
            r"^\s*(?:public|private|internal|open|fileprivate)?\s*protocol\s+(\w+)",
        ],
        // Lua
        "lua" => vec![
            r"^\s*(?:local\s+)?function\s+(\w[\w.:]*)",
        ],
        // Shell
        "sh" | "bash" | "zsh" | "fish" => vec![
            r"^\s*(?:function\s+)?(\w+)\s*\(\)",
            r"^\s*function\s+(\w+)",
        ],
        // Elixir
        "ex" | "exs" => vec![
            r"^\s*def[p]?\s+(\w+)",
            r"^\s*defmodule\s+(\w[\w.]*)",
        ],
        // Haskell
        "hs" => vec![
            r"^(\w+)\s+::",
        ],
        // SQL
        "sql" => vec![
            r"(?i)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|FUNCTION|PROCEDURE|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        ],
        // Dart
        "dart" => vec![
            r"^\s*(?:abstract\s+)?class\s+(\w+)",
            r"^\s*(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:async\s*)?\{",
        ],
        // Zig
        "zig" => vec![
            r"^\s*(?:pub\s+)?fn\s+(\w+)",
            r"^\s*(?:pub\s+)?const\s+(\w+)\s*=\s*struct",
        ],
        // Nim
        "nim" => vec![
            r"^\s*(?:proc|func|method|template|macro)\s+(\w+)",
            r"^\s*type\s+(\w+)",
        ],
        // Default: no patterns → file treated as a single node
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use super::*;

    fn parse_code(content: &str, filename: &str) -> Document {
        let parser = TreeSitterParser;
        parser
            .parse(Path::new(filename), content)
            .expect("parse failed")
    }

    #[test]
    fn test_empty() {
        let doc = parse_code("", "main.rs");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_rust_functions() {
        let content = "use std::io;\n\nfn hello() {\n    println!(\"hello\");\n}\n\npub fn world() {\n    println!(\"world\");\n}\n";
        let doc = parse_code(content, "lib.rs");
        assert_eq!(doc.source_type, SourceType::Code);
        // preamble + two functions
        assert_eq!(doc.structure.len(), 3);
        assert_eq!(doc.structure[0].title, "(preamble)");
        assert_eq!(doc.structure[1].title, "hello");
        assert_eq!(doc.structure[2].title, "world");
    }

    #[test]
    fn test_rust_struct_and_impl() {
        let content = "pub struct Foo {\n    x: i32,\n}\n\nimpl Foo {\n    pub fn new() -> Self {\n        Self { x: 0 }\n    }\n}\n";
        let doc = parse_code(content, "foo.rs");
        assert!(doc.structure.len() >= 2);
        assert_eq!(doc.structure[0].title, "Foo"); // struct
    }

    #[test]
    fn test_python_functions() {
        let content = "import os\n\ndef hello():\n    print('hello')\n\nclass Foo:\n    def bar(self):\n        pass\n";
        let doc = parse_code(content, "app.py");
        assert!(doc.structure.len() >= 3);
        assert_eq!(doc.structure[1].title, "hello");
        assert_eq!(doc.structure[2].title, "Foo");
    }

    #[test]
    fn test_javascript_functions() {
        let content = "\nfunction greet(name) {\n    return `Hello ${name}`;\n}\n\nclass App {\n    constructor() {}\n}\n";
        let doc = parse_code(content, "app.js");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"greet"));
        assert!(titles.contains(&"App"));
    }

    #[test]
    fn test_go_functions() {
        let content = "package main\n\nfunc main() {\n    fmt.Println(\"hello\")\n}\n\nfunc (s *Server) Start() error {\n    return nil\n}\n\ntype Config struct {\n    Port int\n}\n";
        let doc = parse_code(content, "main.go");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"main"));
        assert!(titles.contains(&"Config"));
    }

    #[test]
    fn test_line_numbers() {
        let content = "fn foo() {\n    1\n}\n\nfn bar() {\n    2\n}\n";
        let doc = parse_code(content, "test.rs");
        let foo = &doc.structure[0];
        assert_eq!(foo.title, "foo");
        assert_eq!(foo.line_start, Some(1));
        let bar = &doc.structure[1];
        assert_eq!(bar.title, "bar");
        assert_eq!(bar.line_start, Some(5));
    }

    #[test]
    fn test_no_boundaries_fallback() {
        let content = "some random content\nwith lines\n";
        let doc = parse_code(content, "data.xyz");
        // No patterns for .xyz → single-node fallback
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].text.contains("some random content"));
    }

    #[test]
    fn test_node_ids_assigned() {
        let content = "fn a() {}\n\nfn b() {}\n";
        let doc = parse_code(content, "test.rs");
        for (i, node) in doc.structure.iter().enumerate() {
            assert_eq!(node.node_id, i.to_string());
        }
    }

    #[test]
    fn test_source_type() {
        let doc = parse_code("fn main() {}", "main.rs");
        assert_eq!(doc.source_type, SourceType::Code);
        assert_eq!(doc.doc_id, "main.rs");
    }

    #[test]
    fn test_ruby() {
        let content = "class Foo\n  def bar\n    puts 'hi'\n  end\nend\n";
        let doc = parse_code(content, "app.rb");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"Foo"));
    }

    #[test]
    fn test_shell() {
        let content = "#!/bin/bash\n\nhello() {\n    echo 'hi'\n}\n\nfunction world {\n    echo 'world'\n}\n";
        let doc = parse_code(content, "script.sh");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"hello"));
        assert!(titles.contains(&"world"));
    }

    #[test]
    fn test_sql() {
        let content = "CREATE TABLE users (\n  id INT PRIMARY KEY\n);\n\nCREATE VIEW active_users AS\nSELECT * FROM users;\n";
        let doc = parse_code(content, "schema.sql");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"users"));
        assert!(titles.contains(&"active_users"));
    }

    #[test]
    fn test_preamble_included() {
        let content = "// Copyright 2024\n// License: MIT\n\nfn main() {}\n";
        let doc = parse_code(content, "main.rs");
        assert_eq!(doc.structure[0].title, "(preamble)");
        assert!(doc.structure[0].text.contains("Copyright"));
    }

    #[test]
    fn test_rust_enum_and_trait() {
        let content = "pub enum Color {\n    Red,\n    Blue,\n}\n\npub trait Drawable {\n    fn draw(&self);\n}\n";
        let doc = parse_code(content, "lib.rs");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"Color"));
        assert!(titles.contains(&"Drawable"));
    }

    #[test]
    fn test_python_async_def() {
        let content = "async def fetch_data():\n    pass\n\ndef sync_func():\n    pass\n";
        let doc = parse_code(content, "app.py");
        let titles: Vec<&str> = doc.structure.iter().map(|n| n.title.as_str()).collect();
        assert!(titles.contains(&"fetch_data"));
        assert!(titles.contains(&"sync_func"));
    }
}
