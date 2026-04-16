//! Search pipeline: FTS5 pre-scoring + mode routing (flat / tree).
//!
//! Two-stage pipeline:
//! 1. FTS5 pre-scoring: batch score all documents
//! 2. Mode routing: flat mode (FTS5 results directly) or tree mode (tree walk + reranking)

use std::collections::HashMap;

use anyhow::Result;

use crate::config::{SearchMode, TreeSearchConfig};
use crate::document::{Document, SearchResult};
use crate::engine::fts::FTS5Index;
use crate::engine::tree_walker::TreeSearcher;

/// Unified search entry point.
pub fn search(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
) -> Result<Vec<SearchResult>> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mode = resolve_mode(config.search_mode, documents);
    match mode {
        SearchMode::Flat => search_flat(query, documents, fts_index, config),
        SearchMode::Tree => search_tree(query, documents, fts_index, config),
        SearchMode::Auto => unreachable!("resolve_mode should never return Auto"),
    }
}

/// Resolve Auto mode to concrete Flat or Tree.
fn resolve_mode(mode: SearchMode, documents: &[Document]) -> SearchMode {
    match mode {
        SearchMode::Flat => SearchMode::Flat,
        SearchMode::Tree => SearchMode::Tree,
        SearchMode::Auto => {
            // Auto: use tree mode for documents with heading hierarchy,
            // flat for code-only documents
            let has_hierarchy = documents.iter().any(|doc| {
                doc.structure.iter().any(|root| !root.children.is_empty())
            });
            if has_hierarchy {
                SearchMode::Tree
            } else {
                SearchMode::Flat
            }
        }
    }
}

/// Flat search: FTS5 results directly, ranked by BM25.
fn search_flat(
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
) -> Result<Vec<SearchResult>> {
    let top_k = config.max_nodes_per_doc * config.top_k_docs;
    let fts_results = fts_index.search(query, None, top_k)?;

    let doc_map: HashMap<&str, &Document> = documents
        .iter()
        .map(|d| (d.doc_id.as_str(), d))
        .collect();

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
    query: &str,
    documents: &[Document],
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
) -> Result<Vec<SearchResult>> {
    let searcher = TreeSearcher::new(config);

    // Get FTS5 scores for all documents (single batch query)
    let doc_ids: Vec<String> = documents.iter().map(|d| d.doc_id.clone()).collect();
    let fts_score_map = fts_index.score_nodes_batch(query, Some(&doc_ids), 0.6)?;

    // Run tree search
    let (paths, flat_nodes) = searcher.search(query, documents, &fts_score_map);

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
