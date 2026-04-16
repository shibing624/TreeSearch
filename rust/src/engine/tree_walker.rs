//! Tree Walker — Best-First Search over document trees.
//!
//! Core algorithm (ported from Python tree_searcher.py):
//! 1. Anchor Retrieval: use FTS5 scores to find high-value entry nodes
//! 2. Tree Walk: BFS expansion from anchors along parent/child/sibling edges
//! 3. Path Aggregation: select best root-to-leaf paths as results

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::config::TreeSearchConfig;
use crate::document::{Document, Node};
use crate::scorer::heuristics::{
    QueryPlan, build_query_plan, check_phrase_match, check_title_match,
    compute_term_overlap, estimate_idf, is_generic_section, score_anchor, score_path,
    score_walk_node,
};

/// State in the Best-First Search frontier.
#[derive(Debug, Clone)]
pub struct SearchState {
    pub doc_id: String,
    pub node_id: String,
    pub score: f64,
    pub hop: usize,
    pub source: String,
    pub path: Vec<String>,
    pub max_ancestor_score: f64,
}

impl PartialEq for SearchState {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for SearchState {}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: higher score = higher priority
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// A scored root-to-answer path.
#[derive(Debug, Clone)]
pub struct PathResult {
    pub doc_id: String,
    pub doc_name: String,
    pub score: f64,
    pub anchor_node_id: String,
    pub target_node_id: String,
    pub path: Vec<PathNode>,
    pub snippet: String,
}

#[derive(Debug, Clone)]
pub struct PathNode {
    pub node_id: String,
    pub title: String,
}

/// Flat node from tree walker reranking.
#[derive(Debug, Clone)]
pub struct FlatNode {
    pub node_id: String,
    pub doc_id: String,
    pub doc_name: String,
    pub title: String,
    pub score: f64,
    pub text: String,
}

/// Tree searcher engine.
pub struct TreeSearcher<'a> {
    config: &'a TreeSearchConfig,
}

impl<'a> TreeSearcher<'a> {
    pub fn new(config: &'a TreeSearchConfig) -> Self {
        Self { config }
    }

    /// Run tree search across documents.
    pub fn search(
        &self,
        query: &str,
        documents: &[Document],
        fts_score_map: &HashMap<String, HashMap<String, f64>>,
    ) -> (Vec<PathResult>, Vec<FlatNode>) {
        let plan = build_query_plan(query);
        let mut all_paths: Vec<PathResult> = Vec::new();
        let mut all_walked_nodes: Vec<(String, String, f64, f64, usize)> = Vec::new();

        // Sort documents by max FTS5 score descending
        let mut scored_docs: Vec<(f64, &Document, &HashMap<String, f64>)> = documents
            .iter()
            .filter_map(|doc| {
                let scores = fts_score_map.get(&doc.doc_id)?;
                if scores.is_empty() {
                    return None;
                }
                let max_score = scores.values().cloned().fold(0.0_f64, f64::max);
                Some((max_score, doc, scores))
            })
            .collect();
        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored_docs.truncate(20);

        for (_, doc, doc_scores) in &scored_docs {
            let all_nodes = doc.flatten_nodes();

            // IDF estimation for large documents
            let idf = if !plan.terms.is_empty() && all_nodes.len() > 20 && doc_scores.len() >= 5 {
                let corpus: Vec<&str> = all_nodes.iter().map(|n| n.text.as_str()).collect();
                Some(estimate_idf(&plan.terms, &corpus))
            } else {
                None
            };

            // Stage 1: Anchor retrieval
            let anchors = self.select_anchors(doc, doc_scores, &plan, idf.as_ref());
            if anchors.is_empty() {
                continue;
            }

            // Stage 2: Tree walk
            let (doc_paths, walked_states) =
                self.tree_walk(doc, &anchors, doc_scores, &plan, idf.as_ref());
            all_paths.extend(doc_paths);

            for state in &walked_states {
                let fts_s = doc_scores.get(&state.node_id).copied().unwrap_or(0.0);
                let combined = 0.3 * state.score + 0.7 * fts_s;
                all_walked_nodes.push((
                    doc.doc_id.clone(),
                    state.node_id.clone(),
                    combined,
                    fts_s,
                    state.hop,
                ));
            }
        }

        // Stage 3: Select top paths globally
        all_paths.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        all_paths.truncate(self.config.path_top_k);

        // Build flat nodes with reranking
        let doc_map: HashMap<&str, &Document> = documents
            .iter()
            .map(|d| (d.doc_id.as_str(), d))
            .collect();
        let flat_nodes = self.build_flat_nodes(
            &all_paths,
            &all_walked_nodes,
            &doc_map,
            fts_score_map,
            &plan,
        );

        (all_paths, flat_nodes)
    }

    // ---------------------------------------------------------------
    // Stage 1: Anchor Retrieval
    // ---------------------------------------------------------------

    fn select_anchors(
        &self,
        doc: &Document,
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
        idf: Option<&HashMap<String, f64>>,
    ) -> Vec<SearchState> {
        let max_candidates = self.config.anchor_top_k * 3;
        let threshold = if doc_scores.len() > max_candidates {
            let mut scores: Vec<f64> = doc_scores.values().copied().collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            scores.get(max_candidates.saturating_sub(1)).copied().unwrap_or(0.0)
        } else {
            0.0
        };

        let depth_map = doc.build_depth_map();
        let parent_map = doc.build_parent_map();

        let mut candidates: Vec<(f64, String, &Node)> = Vec::new();
        for (nid, &fts_score) in doc_scores {
            if fts_score < threshold {
                continue;
            }
            let node = match doc.find_node(nid) {
                Some(n) => n,
                None => continue,
            };
            let depth = depth_map.get(nid).copied().unwrap_or(0);
            let full_text = format!("{} {}", node.title, node.text);
            let a_score = score_anchor(
                fts_score,
                depth,
                check_title_match(&node.title, &plan.terms),
                check_phrase_match(&full_text, &plan.phrases),
                compute_term_overlap(&node.text, &plan.terms, idf),
                6,
            );
            candidates.push((a_score, nid.clone(), node));
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<SearchState> = Vec::new();
        let mut selected_paths: HashSet<String> = HashSet::new();

        for (a_score, nid, _node) in &candidates {
            if selected.len() >= self.config.anchor_top_k {
                break;
            }
            let path_to_root = path_to_root_via_map(nid, &parent_map);
            let path_key = path_to_root.iter().take(3).cloned().collect::<Vec<_>>().join(">");
            if selected_paths.contains(&path_key) {
                continue;
            }
            selected_paths.insert(path_key);

            selected.push(SearchState {
                doc_id: doc.doc_id.clone(),
                node_id: nid.clone(),
                score: *a_score,
                hop: 0,
                source: "anchor".into(),
                path: path_to_root,
                max_ancestor_score: doc_scores.get(nid.as_str()).copied().unwrap_or(0.0),
            });
        }

        selected
    }

    // ---------------------------------------------------------------
    // Stage 2: Tree Walk (Best-First Search)
    // ---------------------------------------------------------------

    fn tree_walk(
        &self,
        doc: &Document,
        anchors: &[SearchState],
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
        idf: Option<&HashMap<String, f64>>,
    ) -> (Vec<PathResult>, Vec<SearchState>) {
        let mut visited: HashSet<String> = HashSet::new();
        let mut frontier: BinaryHeap<SearchState> = BinaryHeap::new();
        let mut best_states: Vec<SearchState> = Vec::new();
        let mut expansion_count = 0;

        let parent_map = doc.build_parent_map();
        let children_map = doc.build_children_map();

        // Pre-cache term overlap for FTS5-scored nodes
        let mut overlap_cache: HashMap<String, f64> = HashMap::new();
        if !plan.terms.is_empty() {
            for nid in doc_scores.keys() {
                if let Some(node) = doc.find_node(nid) {
                    overlap_cache.insert(
                        nid.clone(),
                        compute_term_overlap(&node.text, &plan.terms, idf),
                    );
                }
            }
        }

        // Initialize frontier
        for anchor in anchors {
            frontier.push(anchor.clone());
        }

        while let Some(state) = frontier.pop() {
            if expansion_count >= self.config.max_expansions {
                break;
            }
            if visited.contains(&state.node_id) {
                continue;
            }
            visited.insert(state.node_id.clone());
            best_states.push(state.clone());
            expansion_count += 1;

            if state.score >= self.config.early_stop_score {
                break;
            }
            if state.score < self.config.min_frontier_score {
                continue;
            }
            if state.hop >= self.config.max_hops {
                continue;
            }

            // Expand neighbors
            let neighbors = get_neighbors(
                &state.node_id,
                &parent_map,
                &children_map,
                self.config.max_siblings,
            );
            for (nid, relation) in neighbors {
                if visited.contains(&nid) {
                    continue;
                }
                let node = match doc.find_node(&nid) {
                    Some(n) => n,
                    None => continue,
                };
                let lexical = doc_scores.get(&nid).copied().unwrap_or(0.0);
                let overlap = overlap_cache
                    .get(&nid)
                    .copied()
                    .unwrap_or_else(|| {
                        if !plan.terms.is_empty() {
                            let ov = compute_term_overlap(&node.text, &plan.terms, idf);
                            overlap_cache.insert(nid.clone(), ov);
                            ov
                        } else {
                            0.0
                        }
                    });
                let new_max_anc = state
                    .max_ancestor_score
                    .max(doc_scores.get(&state.node_id).copied().unwrap_or(0.0));
                let full_text = format!("{} {}", node.title, node.text);
                let w_score = score_walk_node(
                    lexical,
                    check_title_match(&node.title, &plan.terms),
                    check_phrase_match(&full_text, &plan.phrases),
                    overlap,
                    new_max_anc,
                    (state.hop + 1) as u32,
                    false,
                    self.config.max_hops as u32,
                );

                let new_path = if relation == "child" {
                    let mut p = state.path.clone();
                    p.push(nid.clone());
                    p
                } else {
                    path_to_root_via_map(&nid, &parent_map)
                };

                frontier.push(SearchState {
                    doc_id: doc.doc_id.clone(),
                    node_id: nid,
                    score: w_score,
                    hop: state.hop + 1,
                    source: relation,
                    path: new_path,
                    max_ancestor_score: new_max_anc,
                });
            }
        }

        let paths = self.states_to_paths(doc, &mut best_states, doc_scores, plan);
        (paths, best_states)
    }

    fn states_to_paths(
        &self,
        doc: &Document,
        states: &mut [SearchState],
        doc_scores: &HashMap<String, f64>,
        plan: &QueryPlan,
    ) -> Vec<PathResult> {
        states.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let mut results: Vec<PathResult> = Vec::new();
        let mut seen_targets: HashSet<String> = HashSet::new();
        let max_to_process = self.config.path_top_k * 2;

        let parent_map = doc.build_parent_map();

        for state in states.iter() {
            if results.len() >= max_to_process {
                break;
            }
            if seen_targets.contains(&state.node_id) {
                continue;
            }
            seen_targets.insert(state.node_id.clone());

            let full_path = path_to_root_via_map(&state.node_id, &parent_map);
            let mut path_titles = Vec::new();
            let mut path_texts = Vec::new();
            let mut path_nodes = Vec::new();
            for pid in &full_path {
                if let Some(pnode) = doc.find_node(pid) {
                    path_titles.push(pnode.title.clone());
                    path_texts.push(pnode.text.clone());
                    path_nodes.push(PathNode {
                        node_id: pid.clone(),
                        title: pnode.title.clone(),
                    });
                }
            }

            let p_score = score_path(
                state.score,
                &path_titles,
                &path_texts,
                &plan.terms,
                full_path.len(),
                doc_scores.get(&state.node_id).copied().unwrap_or(0.0),
                6,
            );

            let snippet = doc
                .find_node(&state.node_id)
                .map(|n| {
                    let t = &n.text;
                    if t.len() > 300 {
                        t[..300].to_string()
                    } else {
                        t.clone()
                    }
                })
                .unwrap_or_default();

            let anchor_id = state.path.first().cloned().unwrap_or(state.node_id.clone());

            results.push(PathResult {
                doc_id: doc.doc_id.clone(),
                doc_name: doc.doc_name.clone(),
                score: (p_score * 10000.0).round() / 10000.0,
                anchor_node_id: anchor_id,
                target_node_id: state.node_id.clone(),
                path: path_nodes,
                snippet,
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.path_top_k);
        results
    }

    // ---------------------------------------------------------------
    // Stage 3: Build flat nodes with reranking
    // ---------------------------------------------------------------

    fn build_flat_nodes(
        &self,
        _paths: &[PathResult],
        walked_nodes: &[(String, String, f64, f64, usize)],
        doc_map: &HashMap<&str, &Document>,
        fts_score_map: &HashMap<String, HashMap<String, f64>>,
        plan: &QueryPlan,
    ) -> Vec<FlatNode> {
        let mut node_scores: HashMap<(String, String), f64> = HashMap::new();

        // 1. Base: FTS5 scores
        for (doc_id, doc_scores) in fts_score_map {
            for (nid, &fts_s) in doc_scores {
                node_scores.insert((doc_id.clone(), nid.clone()), fts_s);
            }
        }

        // 2. Generic section demotion + leaf preference (merged pass)
        for ((doc_id, nid), score) in node_scores.iter_mut() {
            let doc = match doc_map.get(doc_id.as_str()) {
                Some(d) => d,
                None => continue,
            };
            let node = match doc.find_node(nid) {
                Some(n) => n,
                None => continue,
            };
            let depth_map = doc.build_depth_map();
            let depth = depth_map.get(nid.as_str()).copied().unwrap_or(0);

            // Generic section demotion
            if depth > 0 && is_generic_section(&node.title, depth) {
                let demote = if !plan.terms.is_empty() {
                    let base_title = node.title.to_lowercase();
                    !plan.terms.iter().any(|t| base_title.contains(t.as_str()))
                } else {
                    true
                };
                if demote {
                    *score *= 0.70;
                }
            }

            // Leaf preference
            if node.children.is_empty() && node.text.len() > 100 {
                *score *= 1.08;
            }
        }

        // 3. Walk boost
        for (doc_id, nid, combined_score, _fts_s, _hop) in walked_nodes {
            let key = (doc_id.clone(), nid.clone());
            if let Some(score) = node_scores.get_mut(&key) {
                *score += 0.15 * combined_score;
            }
        }

        // 4. Title match boost
        if !plan.terms.is_empty() {
            let keys: Vec<(String, String)> = node_scores.keys().cloned().collect();
            for key in &keys {
                let score = match node_scores.get(key) {
                    Some(&s) if s >= 0.05 => s,
                    _ => continue,
                };
                let doc = match doc_map.get(key.0.as_str()) {
                    Some(d) => d,
                    None => continue,
                };
                let node = match doc.find_node(&key.1) {
                    Some(n) => n,
                    None => continue,
                };
                let title_lower = node.title.to_lowercase();
                let title_hits = plan.terms.iter().filter(|t| title_lower.contains(t.as_str())).count();
                if title_hits > 0 {
                    let title_overlap = title_hits as f64 / plan.terms.len() as f64;
                    let title_bonus = 0.15 * title_overlap * score.max(0.10);
                    node_scores.insert(key.clone(), score + title_bonus);
                }
            }
        }

        // Build flat node list
        let mut flat_nodes: Vec<FlatNode> = node_scores
            .into_iter()
            .filter_map(|((doc_id, nid), score)| {
                let doc = doc_map.get(doc_id.as_str())?;
                let node = doc.find_node(&nid)?;
                Some(FlatNode {
                    node_id: nid,
                    doc_id: doc_id.clone(),
                    doc_name: doc.doc_name.clone(),
                    title: node.title.clone(),
                    score: (score * 10000.0).round() / 10000.0,
                    text: node.text.clone(),
                })
            })
            .collect();

        flat_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        flat_nodes
    }
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

fn path_to_root_via_map(node_id: &str, parent_map: &HashMap<String, Option<String>>) -> Vec<String> {
    let mut path = Vec::new();
    let mut current = Some(node_id.to_string());
    while let Some(nid) = current {
        path.push(nid.clone());
        current = parent_map.get(&nid).and_then(|p| p.clone());
    }
    path.reverse();
    path
}

fn get_neighbors(
    node_id: &str,
    parent_map: &HashMap<String, Option<String>>,
    children_map: &HashMap<String, Vec<String>>,
    max_siblings: usize,
) -> Vec<(String, String)> {
    let mut neighbors = Vec::new();

    // Children
    if let Some(children) = children_map.get(node_id) {
        for cid in children {
            neighbors.push((cid.clone(), "child".into()));
        }
    }

    // Parent
    if let Some(Some(pid)) = parent_map.get(node_id) {
        neighbors.push((pid.clone(), "parent".into()));

        // Siblings (via parent's children)
        if let Some(siblings) = children_map.get(pid.as_str()) {
            let mut count = 0;
            for sid in siblings {
                if sid != node_id && count < max_siblings {
                    neighbors.push((sid.clone(), "sibling".into()));
                    count += 1;
                }
            }
        }
    }

    neighbors
}
