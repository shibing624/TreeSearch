use crate::document::{Document, Node, SourceType};
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::Path;

// ===========================================================================
// JSON parser
// ===========================================================================

/// Parser for JSON (`.json`) files.
pub struct JsonParser;

impl super::Parser for JsonParser {
    fn extensions(&self) -> &[&str] {
        &["json"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Json
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let (doc_id, file_name) = path_parts(path);
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Json);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let value: Value =
            serde_json::from_str(content).context("failed to parse JSON")?;
        let root = value_to_node(&file_name, &value);
        doc.structure.push(root);
        doc.assign_node_ids();
        Ok(doc)
    }
}

// ===========================================================================
// YAML parser
// ===========================================================================

/// Parser for YAML (`.yaml`, `.yml`) files.
pub struct YamlParser;

impl super::Parser for YamlParser {
    fn extensions(&self) -> &[&str] {
        &["yaml", "yml"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Yaml
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let (doc_id, file_name) = path_parts(path);
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Yaml);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let yaml_value: serde_yaml::Value =
            serde_yaml::from_str(content).context("failed to parse YAML")?;
        let json_value = yaml_to_json(yaml_value);
        let root = value_to_node(&file_name, &json_value);
        doc.structure.push(root);
        doc.assign_node_ids();
        Ok(doc)
    }
}

// ===========================================================================
// TOML parser
// ===========================================================================

/// Parser for TOML (`.toml`) files.
pub struct TomlParser;

impl super::Parser for TomlParser {
    fn extensions(&self) -> &[&str] {
        &["toml"]
    }

    fn source_type(&self) -> SourceType {
        SourceType::Toml
    }

    fn parse(&self, path: &Path, content: &str) -> Result<Document> {
        let (doc_id, file_name) = path_parts(path);
        let mut doc = Document::new(&doc_id, &file_name, SourceType::Toml);

        if content.trim().is_empty() {
            doc.assign_node_ids();
            return Ok(doc);
        }

        let toml_value: toml::Value =
            toml::from_str(content).context("failed to parse TOML")?;
        let json_value = toml_to_json(&toml_value);
        let root = value_to_node(&file_name, &json_value);
        doc.structure.push(root);
        doc.assign_node_ids();
        Ok(doc)
    }
}

// ===========================================================================
// Shared: serde_json::Value → Node tree
// ===========================================================================

/// Convert a `serde_json::Value` tree into a `Node` tree.
///
/// - Objects become nodes; each key is either a child node (if the value is an
///   object/array) or a `key: value` line in the node's text (if scalar).
/// - Arrays become a node with indexed child elements.
/// - Scalars become leaf text.
fn value_to_node(title: &str, value: &Value) -> Node {
    let mut node = Node::new("", title);

    match value {
        Value::Object(map) => {
            for (key, val) in map {
                match val {
                    Value::Object(_) => {
                        let child = value_to_node(key, val);
                        node.children.push(child);
                    }
                    Value::Array(arr) => {
                        let child = array_to_node(key, arr);
                        node.children.push(child);
                    }
                    _ => {
                        let line = format!("{}: {}", key, format_scalar(val));
                        if node.text.is_empty() {
                            node.text = line;
                        } else {
                            node.text.push('\n');
                            node.text.push_str(&line);
                        }
                    }
                }
            }
        }
        Value::Array(arr) => {
            let inner = array_to_node(title, arr);
            node.text = inner.text;
            node.children = inner.children;
        }
        _ => {
            node.text = format_scalar(value);
        }
    }

    node
}

/// Convert a JSON array into a node. Object elements become child nodes; scalars
/// are listed as `- value` lines in the node's text.
fn array_to_node(title: &str, arr: &[Value]) -> Node {
    let mut node = Node::new("", title);
    let mut scalar_lines = Vec::new();

    for (i, item) in arr.iter().enumerate() {
        match item {
            Value::Object(_) => {
                let child_title = format!("{}[{}]", title, i);
                let child = value_to_node(&child_title, item);
                node.children.push(child);
            }
            Value::Array(inner) => {
                let child_title = format!("{}[{}]", title, i);
                let child = array_to_node(&child_title, inner);
                node.children.push(child);
            }
            _ => {
                scalar_lines.push(format!("- {}", format_scalar(item)));
            }
        }
    }

    if !scalar_lines.is_empty() {
        node.text = scalar_lines.join("\n");
    }

    node
}

fn format_scalar(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        _ => value.to_string(),
    }
}

// ===========================================================================
// Conversion: YAML → JSON, TOML → JSON
// ===========================================================================

fn yaml_to_json(yaml: serde_yaml::Value) -> Value {
    match yaml {
        serde_yaml::Value::Null => Value::Null,
        serde_yaml::Value::Bool(b) => Value::Bool(b),
        serde_yaml::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Number(i.into())
            } else if let Some(f) = n.as_f64() {
                serde_json::Number::from_f64(f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            } else {
                Value::Null
            }
        }
        serde_yaml::Value::String(s) => Value::String(s),
        serde_yaml::Value::Sequence(seq) => {
            Value::Array(seq.into_iter().map(yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(map) => {
            let obj = map
                .into_iter()
                .map(|(k, v)| {
                    let key = match k {
                        serde_yaml::Value::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    (key, yaml_to_json(v))
                })
                .collect();
            Value::Object(obj)
        }
        serde_yaml::Value::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

fn toml_to_json(toml_val: &toml::Value) -> Value {
    match toml_val {
        toml::Value::String(s) => Value::String(s.clone()),
        toml::Value::Integer(i) => Value::Number((*i).into()),
        toml::Value::Float(f) => serde_json::Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        toml::Value::Boolean(b) => Value::Bool(*b),
        toml::Value::Datetime(dt) => Value::String(dt.to_string()),
        toml::Value::Array(arr) => Value::Array(arr.iter().map(toml_to_json).collect()),
        toml::Value::Table(table) => {
            let obj = table
                .iter()
                .map(|(k, v)| (k.clone(), toml_to_json(v)))
                .collect();
            Value::Object(obj)
        }
    }
}

// ===========================================================================
// Utility
// ===========================================================================

fn path_parts(path: &Path) -> (String, String) {
    let doc_id = path.to_string_lossy().to_string();
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    (doc_id, file_name)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use crate::parser::Parser;
    use super::*;

    // --- JSON tests ---

    fn parse_json(content: &str) -> Document {
        let parser = JsonParser;
        parser
            .parse(Path::new("config.json"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_json_empty() {
        let doc = parse_json("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_json_simple_object() {
        let doc = parse_json(r#"{"name": "Alice", "age": 30}"#);
        assert_eq!(doc.structure.len(), 1);
        let root = &doc.structure[0];
        assert!(root.text.contains("name: Alice"));
        assert!(root.text.contains("age: 30"));
    }

    #[test]
    fn test_json_nested_object() {
        let doc = parse_json(r#"{"db": {"host": "localhost", "port": 5432}}"#);
        assert_eq!(doc.structure.len(), 1);
        let root = &doc.structure[0];
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].title, "db");
        assert!(root.children[0].text.contains("host: localhost"));
        assert!(root.children[0].text.contains("port: 5432"));
    }

    #[test]
    fn test_json_array_of_scalars() {
        let doc = parse_json(r#"{"tags": ["a", "b", "c"]}"#);
        let root = &doc.structure[0];
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].title, "tags");
        assert!(root.children[0].text.contains("- a"));
        assert!(root.children[0].text.contains("- b"));
        assert!(root.children[0].text.contains("- c"));
    }

    #[test]
    fn test_json_array_of_objects() {
        let doc = parse_json(r#"{"items": [{"id": 1}, {"id": 2}]}"#);
        let root = &doc.structure[0];
        let items = &root.children[0];
        assert_eq!(items.title, "items");
        assert_eq!(items.children.len(), 2);
        assert_eq!(items.children[0].title, "items[0]");
        assert_eq!(items.children[1].title, "items[1]");
    }

    #[test]
    fn test_json_source_type() {
        let doc = parse_json(r#"{"x": 1}"#);
        assert_eq!(doc.source_type, SourceType::Json);
        assert_eq!(doc.doc_id, "config.json");
    }

    #[test]
    fn test_json_node_ids() {
        let doc = parse_json(r#"{"a": {"b": 1}}"#);
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[0].children[0].node_id, "1");
    }

    #[test]
    fn test_json_top_level_array() {
        let doc = parse_json(r#"[1, 2, 3]"#);
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].text.contains("- 1"));
    }

    #[test]
    fn test_json_scalar_value() {
        let doc = parse_json(r#""hello world""#);
        assert_eq!(doc.structure.len(), 1);
        assert_eq!(doc.structure[0].text, "hello world");
    }

    // --- YAML tests ---

    fn parse_yaml(content: &str) -> Document {
        let parser = YamlParser;
        parser
            .parse(Path::new("config.yaml"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_yaml_empty() {
        let doc = parse_yaml("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_yaml_simple() {
        let doc = parse_yaml("name: Bob\nage: 25");
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].text.contains("name: Bob"));
        assert!(doc.structure[0].text.contains("age: 25"));
    }

    #[test]
    fn test_yaml_nested() {
        let content = "database:\n  host: localhost\n  port: 3306";
        let doc = parse_yaml(content);
        let root = &doc.structure[0];
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].title, "database");
        assert!(root.children[0].text.contains("host: localhost"));
    }

    #[test]
    fn test_yaml_list() {
        let content = "items:\n  - one\n  - two\n  - three";
        let doc = parse_yaml(content);
        let root = &doc.structure[0];
        let items = &root.children[0];
        assert_eq!(items.title, "items");
        assert!(items.text.contains("- one"));
    }

    #[test]
    fn test_yaml_source_type() {
        let doc = parse_yaml("x: 1");
        assert_eq!(doc.source_type, SourceType::Yaml);
        assert_eq!(doc.doc_id, "config.yaml");
    }

    // --- TOML tests ---

    fn parse_toml(content: &str) -> Document {
        let parser = TomlParser;
        parser
            .parse(Path::new("config.toml"), content)
            .expect("parse failed")
    }

    #[test]
    fn test_toml_empty() {
        let doc = parse_toml("");
        assert!(doc.structure.is_empty());
    }

    #[test]
    fn test_toml_simple() {
        let doc = parse_toml("name = \"test\"\nversion = \"1.0\"");
        assert_eq!(doc.structure.len(), 1);
        assert!(doc.structure[0].text.contains("name: test"));
        assert!(doc.structure[0].text.contains("version: 1.0"));
    }

    #[test]
    fn test_toml_nested_table() {
        let content = "[package]\nname = \"foo\"\nversion = \"0.1\"";
        let doc = parse_toml(content);
        let root = &doc.structure[0];
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].title, "package");
        assert!(root.children[0].text.contains("name: foo"));
    }

    #[test]
    fn test_toml_source_type() {
        let doc = parse_toml("x = 1");
        assert_eq!(doc.source_type, SourceType::Toml);
        assert_eq!(doc.doc_id, "config.toml");
    }

    #[test]
    fn test_toml_array_of_tables() {
        let content = "[[servers]]\nname = \"alpha\"\n\n[[servers]]\nname = \"beta\"";
        let doc = parse_toml(content);
        let root = &doc.structure[0];
        let servers = root
            .children
            .iter()
            .find(|c| c.title == "servers")
            .expect("servers node");
        assert_eq!(servers.children.len(), 2);
    }

    // --- Shared logic tests ---

    #[test]
    fn test_value_to_node_scalar() {
        let node = value_to_node("root", &Value::String("hello".into()));
        assert_eq!(node.title, "root");
        assert_eq!(node.text, "hello");
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_value_to_node_deeply_nested() {
        let json: Value =
            serde_json::from_str(r#"{"a": {"b": {"c": "deep"}}}"#).unwrap();
        let node = value_to_node("root", &json);
        assert_eq!(node.children[0].title, "a");
        assert_eq!(node.children[0].children[0].title, "b");
        assert!(node.children[0].children[0].text.contains("c: deep"));
    }

    #[test]
    fn test_format_scalar_types() {
        assert_eq!(format_scalar(&Value::String("hi".into())), "hi");
        assert_eq!(format_scalar(&Value::Bool(true)), "true");
        assert_eq!(format_scalar(&Value::Null), "null");
        assert_eq!(
            format_scalar(&Value::Number(serde_json::Number::from(42))),
            "42"
        );
    }

    #[test]
    fn test_yaml_to_json_roundtrip() {
        let yaml: serde_yaml::Value =
            serde_yaml::from_str("key: value\nnum: 42").unwrap();
        let json = yaml_to_json(yaml);
        assert_eq!(json["key"], Value::String("value".into()));
        assert_eq!(json["num"], Value::Number(42.into()));
    }

    #[test]
    fn test_toml_to_json_roundtrip() {
        let toml_val: toml::Value =
            toml::from_str("key = \"value\"\nnum = 42").unwrap();
        let json = toml_to_json(&toml_val);
        assert_eq!(json["key"], Value::String("value".into()));
        assert_eq!(json["num"], Value::Number(42.into()));
    }
}
