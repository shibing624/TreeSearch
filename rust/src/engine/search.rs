//! Search pipeline: FTS5 pre-scoring + mode routing (flat / tree).
//!
//! Two-stage pipeline:
//! 1. FTS5 pre-scoring: batch score all documents
//! 2. Mode routing: flat mode (FTS5 results directly) or tree mode (tree walk + reranking)

use std::collections::HashMap;

use anyhow::{bail, Result};
use regex::{Regex, RegexBuilder};

use crate::config::{SearchMode, TreeSearchConfig};
use crate::document::{Document, Node, SearchResult, SourceType};
use crate::engine::fts::FTS5Index;
use crate::engine::tree_walker::TreeSearcher;

// ---------------------------------------------------------------------------
// Auto mode constants (ported from Python search.py)
// ---------------------------------------------------------------------------

/// Which source types benefit from tree walk.
fn benefits_from_tree(source_type: &SourceType) -> bool {
    matches!(source_type, SourceType::Markdown | SourceType::Json | SourceType::Yaml | SourceType::Toml | SourceType::Html)
}

/// Minimum tree depth for a doc to truly benefit from tree walk.
/// Docs with depth ≤ 1 (flat list of nodes) won't gain anything from BFS walk.
const MIN_TREE_DEPTH: u32 = 2;

/// If ≥30% of docs benefit from tree, use tree for all.
const TREE_RATIO_THRESHOLD: f64 = 0.3;

#[derive(Debug, Clone)]
struct QueryMode {
    effective_query: String,
    fts_expression: Option<String>,
    regex_pattern: Option<String>,
}

fn classify_query_mode(query: &str, fts_expression: Option<&str>, regex: bool) -> QueryMode {
    if regex {
        return QueryMode {
            effective_query: query.to_string(),
            fts_expression: None,
            regex_pattern: Some(query.to_string()),
        };
    }
    if let Some(expr) = fts_expression {
        return QueryMode {
            effective_query: if query.is_empty() {
                expr.to_string()
            } else {
                query.to_string()
            },
            fts_expression: Some(expr.to_string()),
            regex_pattern: None,
        };
    }
    let trimmed = query.trim();
    let no_internal_star = !trimmed[..trimmed.len().saturating_sub(1)].contains('*');
    let middle = if trimmed.len() > 2 {
        &trimmed[1..trimmed.len() - 1]
    } else {
        ""
    };
    let prefix_body = if !trimmed.is_empty() {
        &trimmed[..trimmed.len().saturating_sub(1)]
    } else {
        ""
    };

    if trimmed.starts_with('*')
        && trimmed.ends_with('*')
        && trimmed.len() > 2
        && !middle.contains('*')
        && !middle.chars().any(|c| c.is_whitespace())
    {
        let term = middle.to_string();
        return QueryMode {
            effective_query: term.clone(),
            fts_expression: None,
            regex_pattern: Some(regex::escape(&term)),
        };
    }

    if trimmed.ends_with('*')
        && !trimmed.starts_with('*')
        && trimmed.len() > 1
        && no_internal_star
        && !prefix_body.chars().any(|c| c.is_whitespace())
    {
        return QueryMode {
            effective_query: prefix_body.to_string(),
            fts_expression: Some(trimmed.to_string()),
            regex_pattern: None,
        };
    }

    QueryMode {
        effective_query: query.to_string(),
        fts_expression: None,
        regex_pattern: None,
    }
}

fn compile_contains_regex(pattern: &str) -> Result<Regex> {
    Ok(RegexBuilder::new(pattern).case_insensitive(true).build()?)
}

fn regex_score_doc(doc: &Document, regex: &Regex) -> HashMap<String, f64> {
    fn count_matches(regex: &Regex, text: &str) -> usize {
        regex.find_iter(text).count()
    }

    fn walk(node: &Node, regex: &Regex, scores: &mut HashMap<String, f64>) {
        let hit_count = count_matches(regex, &node.title)
            + count_matches(regex, &node.summary)
            + count_matches(regex, &node.text);
        if hit_count > 0 {
            scores.insert(node.node_id.clone(), hit_count as f64);
        }
        for child in &node.children {
            walk(child, regex, scores);
        }
    }

    let mut scores = HashMap::new();
    for root in &doc.structure {
        walk(root, regex, &mut scores);
    }

    if let Some(max_score) = scores.values().copied().reduce(f64::max) {
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score /= max_score;
            }
        }
    }
    scores
}

/// Check if a document's tree has enough depth for tree walk to help.
fn has_meaningful_depth(doc: &Document) -> bool {
    fn max_depth(node: &Node, current: u32) -> u32 {
        if node.children.is_empty() {
            return current;
        }
        node.children
            .iter()
            .map(|child| max_depth(child, current + 1))
            .max()
            .unwrap_or(current)
    }

    if doc.structure.is_empty() {
        return false;
    }
    let depth = doc
        .structure
        .iter()
        .map(|root| max_depth(root, 1))
        .max()
        .unwrap_or(0);
    depth >= MIN_TREE_DEPTH
}

/// Unified search entry point.
pub fn search(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
) -> Result<Vec<SearchResult>> {
    search_with_options(query, documents, fts_index, config, None, false)
}

pub fn search_with_options(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    fts_expression: Option<&str>,
    regex: bool,
) -> Result<Vec<SearchResult>> {
    if regex && fts_expression.is_some() {
        bail!("regex and fts_expression cannot be used together");
    }
    if query.trim().is_empty() && fts_expression.is_none() {
        return Ok(Vec::new());
    }

    let query_mode = classify_query_mode(query, fts_expression, regex);
    let mode = resolve_mode(config.search_mode, documents);
    match mode {
        SearchMode::Flat => search_flat(documents, fts_index, config, &query_mode),
        SearchMode::Tree => search_tree(documents, fts_index, config, &query_mode),
        SearchMode::Auto => unreachable!("resolve_mode should never return Auto"),
    }
}

/// Resolve Auto mode to concrete Flat or Tree.
///
/// Strategy (ported from Python `_resolve_auto_mode`):
/// 1. Count docs whose source_type benefits from tree walk (markdown, json, yaml, toml, html).
/// 2. For those, verify they actually have meaningful depth (≥ MIN_TREE_DEPTH).
///    A markdown file with no headings is effectively flat.
/// 3. If the ratio of truly-hierarchical docs ≥ TREE_RATIO_THRESHOLD (30%) → tree mode.
///    Otherwise → flat mode.
///
/// This avoids "1 markdown among 50 code files → tree for everything" while
/// still activating tree mode when it helps.
fn resolve_mode(mode: SearchMode, documents: &[Document]) -> SearchMode {
    match mode {
        SearchMode::Flat => SearchMode::Flat,
        SearchMode::Tree => SearchMode::Tree,
        SearchMode::Auto => {
            if documents.is_empty() {
                return SearchMode::Flat;
            }
            let total = documents.len();
            let tree_count = documents
                .iter()
                .filter(|doc| benefits_from_tree(&doc.source_type) && has_meaningful_depth(doc))
                .count();
            let ratio = tree_count as f64 / total as f64;
            if ratio >= TREE_RATIO_THRESHOLD {
                tracing::debug!(
                    "Auto mode → tree: {}/{} docs ({:.0}%) have meaningful hierarchy",
                    tree_count, total, ratio * 100.0,
                );
                SearchMode::Tree
            } else {
                tracing::debug!(
                    "Auto mode → flat: {}/{} docs ({:.0}%) have hierarchy (threshold {:.0}%)",
                    tree_count, total, ratio * 100.0, TREE_RATIO_THRESHOLD * 100.0,
                );
                SearchMode::Flat
            }
        }
    }
}

/// Flat search: FTS5 results directly, ranked by BM25.
fn search_flat(
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    query_mode: &QueryMode,
) -> Result<Vec<SearchResult>> {
    let top_k = config.max_nodes_per_doc * config.top_k_docs;

    let doc_map: HashMap<&str, &Document> = documents
        .iter()
        .map(|d| (d.doc_id.as_str(), d))
        .collect();

    if let Some(pattern) = &query_mode.regex_pattern {
        let regex = compile_contains_regex(pattern)?;
        let mut results = Vec::new();
        for doc in documents {
            let scores = regex_score_doc(doc, &regex);
            for (node_id, score) in scores {
                if let Some(node) = doc.find_node(&node_id) {
                    let breadcrumb = doc.path_to_node(&node_id);
                    let breadcrumb_titles: Vec<String> = breadcrumb
                        .iter()
                        .filter_map(|nid| doc.find_node(nid).map(|n| n.title.clone()))
                        .collect();
                    results.push(SearchResult {
                        node_id,
                        doc_id: doc.doc_id.clone(),
                        doc_name: doc.doc_name.clone(),
                        title: node.title.clone(),
                        summary: node.summary.clone(),
                        text: node.text.clone(),
                        source_type: doc.source_type.to_string(),
                        source_path: doc.source_path.clone(),
                        line_start: node.line_start,
                        line_end: node.line_end,
                        score,
                        depth: 0,
                        breadcrumb: breadcrumb_titles,
                    });
                }
            }
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        return Ok(results);
    }

    if let Some(fts_expression) = query_mode.fts_expression.as_deref() {
        let doc_ids: Vec<String> = documents.iter().map(|d| d.doc_id.clone()).collect();
        let batch = fts_index.score_nodes_batch_with_expr(
            &query_mode.effective_query,
            Some(&doc_ids),
            0.0,
            Some(fts_expression),
        )?;
        let mut results = Vec::new();
        for doc in documents {
            let Some(scores) = batch.get(&doc.doc_id) else {
                continue;
            };
            let mut ranked_nodes: Vec<(&String, &f64)> = scores.iter().collect();
            ranked_nodes.sort_by(|a, b| {
                b.1.partial_cmp(a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (node_id, score) in ranked_nodes {
                let Some(node) = doc.find_node(node_id) else {
                    continue;
                };
                let breadcrumb = doc.path_to_node(node_id);
                let breadcrumb_titles: Vec<String> = breadcrumb
                    .iter()
                    .filter_map(|nid| doc.find_node(nid).map(|n| n.title.clone()))
                    .collect();
                results.push(SearchResult {
                    node_id: node_id.clone(),
                    doc_id: doc.doc_id.clone(),
                    doc_name: doc.doc_name.clone(),
                    title: node.title.clone(),
                    summary: node.summary.clone(),
                    text: node.text.clone(),
                    source_type: doc.source_type.to_string(),
                    source_path: doc.source_path.clone(),
                    line_start: node.line_start,
                    line_end: node.line_end,
                    score: *score,
                    depth: 0,
                    breadcrumb: breadcrumb_titles,
                });
            }
        }
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        return Ok(results);
    }

    let fts_results = fts_index.search_with_expr(
        &query_mode.effective_query,
        None,
        top_k,
        None,
    )?;

    let mut results: Vec<SearchResult> = fts_results
        .into_iter()
        .filter_map(|fts| {
            let doc = doc_map.get(fts.doc_id.as_str())?;
            let node = doc.find_node(&fts.node_id);
            let (text, line_start, line_end) = match node {
                Some(n) => (n.text.clone(), n.line_start, n.line_end),
                None => (String::new(), None, None),
            };
            let breadcrumb = doc.path_to_node(&fts.node_id);
            let breadcrumb_titles: Vec<String> = breadcrumb
                .iter()
                .filter_map(|nid| doc.find_node(nid).map(|n| n.title.clone()))
                .collect();

            Some(SearchResult {
                node_id: fts.node_id,
                doc_id: fts.doc_id,
                doc_name: doc.doc_name.clone(),
                title: fts.title,
                summary: fts.summary,
                text,
                source_type: doc.source_type.to_string(),
                source_path: doc.source_path.clone(),
                line_start,
                line_end,
                score: fts.fts_score,
                depth: fts.depth,
                breadcrumb: breadcrumb_titles,
            })
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    Ok(results)
}

/// Tree search: anchor retrieval + tree walk + path scoring + flat reranking.
fn search_tree(
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    query_mode: &QueryMode,
) -> Result<Vec<SearchResult>> {
    let searcher = TreeSearcher::new(config);

    // Get FTS5 scores for all documents (single batch query)
    let doc_ids: Vec<String> = documents.iter().map(|d| d.doc_id.clone()).collect();
    let fts_score_map = if let Some(pattern) = &query_mode.regex_pattern {
        let regex = compile_contains_regex(pattern)?;
        documents
            .iter()
            .filter_map(|doc| {
                let scores = regex_score_doc(doc, &regex);
                if scores.is_empty() {
                    None
                } else {
                    Some((doc.doc_id.clone(), scores))
                }
            })
            .collect()
    } else {
        fts_index.score_nodes_batch_with_expr(
            &query_mode.effective_query,
            Some(&doc_ids),
            0.6,
            query_mode.fts_expression.as_deref(),
        )?
    };

    // Run tree search
    let (paths, flat_nodes) = searcher.search(&query_mode.effective_query, documents, &fts_score_map);

    // Convert flat_nodes to SearchResults
    let doc_map: HashMap<&str, &Document> = documents
        .iter()
        .map(|d| (d.doc_id.as_str(), d))
        .collect();

    let top_k = config.max_nodes_per_doc * config.top_k_docs;
    let mut results: Vec<SearchResult> = flat_nodes
        .into_iter()
        .take(top_k)
        .filter_map(|flat| {
            let doc = doc_map.get(flat.doc_id.as_str())?;
            let node = doc.find_node(&flat.node_id);
            let (text, summary, line_start, line_end) = match node {
                Some(n) => (
                    n.text.clone(),
                    n.summary.clone(),
                    n.line_start,
                    n.line_end,
                ),
                None => (flat.text, String::new(), None, None),
            };
            let breadcrumb = doc.path_to_node(&flat.node_id);
            let breadcrumb_titles: Vec<String> = breadcrumb
                .iter()
                .filter_map(|nid| doc.find_node(nid).map(|n| n.title.clone()))
                .collect();

            Some(SearchResult {
                node_id: flat.node_id,
                doc_id: flat.doc_id.clone(),
                doc_name: flat.doc_name,
                title: flat.title,
                summary,
                text,
                source_type: doc.source_type.to_string(),
                source_path: doc.source_path.clone(),
                line_start,
                line_end,
                score: flat.score,
                depth: 0,
                breadcrumb: breadcrumb_titles,
            })
        })
        .collect();

    // Also inject path results if they scored higher
    for path in &paths {
        let doc = match doc_map.get(path.doc_id.as_str()) {
            Some(d) => d,
            None => continue,
        };
        let node = doc.find_node(&path.target_node_id);
        let already_present = results.iter().any(|r| {
            r.doc_id == path.doc_id && r.node_id == path.target_node_id
        });
        if !already_present {
            if let Some(n) = node {
                let breadcrumb_titles: Vec<String> = path
                    .path
                    .iter()
                    .map(|p| p.title.clone())
                    .collect();
                results.push(SearchResult {
                    node_id: path.target_node_id.clone(),
                    doc_id: path.doc_id.clone(),
                    doc_name: path.doc_name.clone(),
                    title: n.title.clone(),
                    summary: n.summary.clone(),
                    text: n.text.clone(),
                    source_type: doc.source_type.to_string(),
                    source_path: doc.source_path.clone(),
                    line_start: n.line_start,
                    line_end: n.line_end,
                    score: path.score,
                    depth: 0,
                    breadcrumb: breadcrumb_titles,
                });
            }
        }
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::{search, search_with_options};
    use crate::config::{SearchMode, TreeSearchConfig};
    use crate::document::{Document, Node, SourceType};
    use crate::engine::fts::FTS5Index;

    fn wildcard_documents() -> Vec<Document> {
        let mut exact = Document::new("exact", "Exact Auth", SourceType::Text);
        let mut exact_root = Node::new("0", "Exact Auth");
        exact_root.summary = "Contains the exact auth token.".to_string();
        exact_root.text = "Use auth tokens for API access.".to_string();
        exact.structure.push(exact_root);

        let mut prefix = Document::new("prefix", "Authentication Guide", SourceType::Text);
        let mut prefix_root = Node::new("0", "Authentication");
        prefix_root.summary = "Authentication and authorizer details.".to_string();
        prefix_root.text = "Authentication depends on an authorizer service.".to_string();
        prefix.structure.push(prefix_root);

        let mut contains = Document::new("contains", "OAuth Guide", SourceType::Text);
        let mut contains_root = Node::new("0", "OAuth");
        contains_root.summary = "OAuth callback handling.".to_string();
        contains_root.text = "OAuth callbacks must be validated.".to_string();
        contains.structure.push(contains_root);

        vec![exact, prefix, contains]
    }

    fn default_config() -> TreeSearchConfig {
        let mut config = TreeSearchConfig::default();
        config.search_mode = SearchMode::Flat;
        config.top_k_docs = 3;
        config.max_nodes_per_doc = 5;
        config
    }

    #[test]
    fn test_plain_query_preserves_exact_term_behavior() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("auth", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(!doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_suffix_star_query_uses_prefix_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("auth*", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_fts_expression_uses_prefix_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search_with_options(
            "ignored",
            &docs,
            &index,
            &default_config(),
            Some("auth*"),
            false,
        )
        .unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(!doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_surrounded_star_query_uses_contains_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("*auth*", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_regex_query_uses_regex_matching() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search_with_options(
            "o?auth",
            &docs,
            &index,
            &default_config(),
            None,
            true,
        )
        .unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();

        assert!(doc_names.contains(&"Exact Auth"));
        assert!(doc_names.contains(&"Authentication Guide"));
        assert!(doc_names.contains(&"OAuth Guide"));
    }

    #[test]
    fn test_explicit_regex_invalid_pattern_returns_error() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let error = search_with_options(
            "(",
            &docs,
            &index,
            &default_config(),
            None,
            true,
        )
        .unwrap_err();

        assert!(error.to_string().contains("regex parse error"));
    }

    #[test]
    fn test_unsupported_wildcard_shape_falls_back_to_plain_query() {
        let index = FTS5Index::new(None, None).unwrap();
        let docs = wildcard_documents();
        for doc in &docs {
            index.index_document(doc, false).unwrap();
        }

        let results = search("au*th", &docs, &index, &default_config()).unwrap();
        let doc_names: Vec<&str> = results.iter().map(|r| r.doc_name.as_str()).collect();
        assert_eq!(doc_names, vec!["Exact Auth"]);
    }
}
