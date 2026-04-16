use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a node within a document.
pub type NodeId = String;

/// Source type classification for documents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceType {
    Code,
    Markdown,
    Html,
    Text,
    Json,
    Yaml,
    Toml,
    #[serde(other)]
    Unknown,
}

impl SourceType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Code => "code",
            Self::Markdown => "markdown",
            Self::Html => "html",
            Self::Text => "text",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Toml => "toml",
            Self::Unknown => "unknown",
        }
    }

    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "md" | "mdx" | "markdown" => Self::Markdown,
            "html" | "htm" => Self::Html,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "toml" => Self::Toml,
            "txt" | "text" | "log" | "csv" => Self::Text,
            "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "go" | "java" | "c" | "cpp" | "h"
            | "hpp" | "cs" | "rb" | "php" | "swift" | "kt" | "scala" | "sh" | "bash" | "zsh"
            | "fish" | "lua" | "r" | "m" | "mm" | "pl" | "pm" | "ex" | "exs" | "erl" | "hs"
            | "ml" | "mli" | "clj" | "cljs" | "el" | "vim" | "sql" | "graphql" | "proto"
            | "tf" | "hcl" | "zig" | "nim" | "v" | "d" | "dart" | "cmake" | "makefile"
            | "dockerfile" | "css" | "scss" | "sass" | "less" | "vue" | "svelte" => Self::Code,
            _ => Self::Unknown,
        }
    }
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single node in the document tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub node_id: NodeId,
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub line_start: Option<u32>,
    #[serde(default)]
    pub line_end: Option<u32>,
    #[serde(default)]
    pub children: Vec<Node>,
}

impl Node {
    pub fn new(node_id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            title: title.into(),
            summary: String::new(),
            text: String::new(),
            line_start: None,
            line_end: None,
            children: Vec::new(),
        }
    }

    /// Flatten this node and all descendants into a vec.
    pub fn flatten(&self) -> Vec<&Node> {
        let mut result = vec![self];
        for child in &self.children {
            result.extend(child.flatten());
        }
        result
    }
}

/// A parsed document with its tree structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub doc_id: String,
    pub doc_name: String,
    pub source_type: SourceType,
    #[serde(default)]
    pub doc_description: String,
    #[serde(default)]
    pub source_path: String,
    pub structure: Vec<Node>,
}

impl Document {
    pub fn new(
        doc_id: impl Into<String>,
        doc_name: impl Into<String>,
        source_type: SourceType,
    ) -> Self {
        Self {
            doc_id: doc_id.into(),
            doc_name: doc_name.into(),
            source_type,
            doc_description: String::new(),
            source_path: String::new(),
            structure: Vec::new(),
        }
    }

    /// Flatten all nodes in the document tree.
    pub fn flatten_nodes(&self) -> Vec<&Node> {
        let mut nodes = Vec::new();
        for root in &self.structure {
            nodes.extend(root.flatten());
        }
        nodes
    }

    /// Build parent map: node_id -> parent_node_id (None for roots).
    pub fn build_parent_map(&self) -> HashMap<String, Option<String>> {
        let mut map = HashMap::new();
        fn walk(node: &Node, parent_id: Option<&str>, map: &mut HashMap<String, Option<String>>) {
            map.insert(node.node_id.clone(), parent_id.map(String::from));
            for child in &node.children {
                walk(child, Some(&node.node_id), map);
            }
        }
        for root in &self.structure {
            walk(root, None, &mut map);
        }
        map
    }

    /// Build depth map: node_id -> depth (0 for roots).
    pub fn build_depth_map(&self) -> HashMap<String, u32> {
        let mut map = HashMap::new();
        fn walk(node: &Node, depth: u32, map: &mut HashMap<String, u32>) {
            map.insert(node.node_id.clone(), depth);
            for child in &node.children {
                walk(child, depth + 1, map);
            }
        }
        for root in &self.structure {
            walk(root, 0, &mut map);
        }
        map
    }

    /// Build children map: node_id -> list of child node_ids.
    pub fn build_children_map(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        fn walk(node: &Node, map: &mut HashMap<String, Vec<String>>) {
            let child_ids: Vec<String> = node.children.iter().map(|c| c.node_id.clone()).collect();
            if !child_ids.is_empty() {
                map.insert(node.node_id.clone(), child_ids);
            }
            for child in &node.children {
                walk(child, map);
            }
        }
        for root in &self.structure {
            walk(root, &mut map);
        }
        map
    }

    /// Find a node by id.
    pub fn find_node(&self, node_id: &str) -> Option<&Node> {
        fn find_in<'a>(node: &'a Node, target: &str) -> Option<&'a Node> {
            if node.node_id == target {
                return Some(node);
            }
            for child in &node.children {
                if let Some(found) = find_in(child, target) {
                    return Some(found);
                }
            }
            None
        }
        for root in &self.structure {
            if let Some(found) = find_in(root, node_id) {
                return Some(found);
            }
        }
        None
    }

    /// Get path from root to a node (list of node_ids, root first).
    pub fn path_to_node(&self, node_id: &str) -> Vec<String> {
        let parent_map = self.build_parent_map();
        let mut path = Vec::new();
        let mut current = Some(node_id.to_string());
        while let Some(nid) = current {
            path.push(nid.clone());
            current = parent_map.get(&nid).and_then(|p| p.clone());
        }
        path.reverse();
        path
    }

    /// Assign sequential node IDs to all nodes.
    pub fn assign_node_ids(&mut self) {
        let mut counter = 0u32;
        fn assign(node: &mut Node, counter: &mut u32) {
            node.node_id = counter.to_string();
            *counter += 1;
            for child in &mut node.children {
                assign(child, counter);
            }
        }
        for root in &mut self.structure {
            assign(root, &mut counter);
        }
    }
}

/// Search result from FTS5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub node_id: NodeId,
    pub doc_id: String,
    pub doc_name: String,
    pub title: String,
    pub summary: String,
    pub text: String,
    pub source_type: String,
    pub source_path: String,
    pub line_start: Option<u32>,
    pub line_end: Option<u32>,
    pub score: f64,
    pub depth: u32,
    /// Breadcrumb path from root to this node.
    #[serde(default)]
    pub breadcrumb: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> Document {
        let mut doc = Document::new("test", "test.rs", SourceType::Code);
        let mut root = Node::new("0", "Root");
        let mut child1 = Node::new("1", "Child 1");
        child1.text = "some text".into();
        let child2 = Node::new("2", "Child 2");
        child1.children.push(Node::new("3", "Grandchild"));
        root.children.push(child1);
        root.children.push(child2);
        doc.structure.push(root);
        doc
    }

    #[test]
    fn test_flatten_nodes() {
        let doc = sample_doc();
        let flat = doc.flatten_nodes();
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].title, "Root");
        assert_eq!(flat[1].title, "Child 1");
        assert_eq!(flat[2].title, "Grandchild");
        assert_eq!(flat[3].title, "Child 2");
    }

    #[test]
    fn test_parent_map() {
        let doc = sample_doc();
        let pm = doc.build_parent_map();
        assert_eq!(pm["0"], None);
        assert_eq!(pm["1"], Some("0".into()));
        assert_eq!(pm["3"], Some("1".into()));
    }

    #[test]
    fn test_depth_map() {
        let doc = sample_doc();
        let dm = doc.build_depth_map();
        assert_eq!(dm["0"], 0);
        assert_eq!(dm["1"], 1);
        assert_eq!(dm["3"], 2);
    }

    #[test]
    fn test_find_node() {
        let doc = sample_doc();
        assert!(doc.find_node("3").is_some());
        assert_eq!(doc.find_node("3").unwrap().title, "Grandchild");
        assert!(doc.find_node("999").is_none());
    }

    #[test]
    fn test_path_to_node() {
        let doc = sample_doc();
        let path = doc.path_to_node("3");
        assert_eq!(path, vec!["0", "1", "3"]);
    }

    #[test]
    fn test_assign_node_ids() {
        let mut doc = Document::new("test", "test.rs", SourceType::Code);
        let mut root = Node::new("", "Root");
        root.children.push(Node::new("", "A"));
        root.children.push(Node::new("", "B"));
        doc.structure.push(root);
        doc.assign_node_ids();
        assert_eq!(doc.structure[0].node_id, "0");
        assert_eq!(doc.structure[0].children[0].node_id, "1");
        assert_eq!(doc.structure[0].children[1].node_id, "2");
    }

    #[test]
    fn test_source_type_from_extension() {
        assert_eq!(SourceType::from_extension("rs"), SourceType::Code);
        assert_eq!(SourceType::from_extension("md"), SourceType::Markdown);
        assert_eq!(SourceType::from_extension("html"), SourceType::Html);
        assert_eq!(SourceType::from_extension("json"), SourceType::Json);
        assert_eq!(SourceType::from_extension("xyz"), SourceType::Unknown);
    }
}
