//! Indexer: parallel file discovery, parsing, and FTS5 batch insertion.
//!
//! Pipeline:
//!   File Discovery (ignore crate) → Parallel Parse (rayon) → Batch Insert (FTS5)
//!                                    ↓
//!                            mtime/size fingerprinting for incremental index

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::config::TreeSearchConfig;
use crate::document::Document;
use crate::engine::fts::FTS5Index;
use crate::parser::ParserRegistry;

/// File fingerprint for incremental indexing.
fn file_fingerprint(path: &Path) -> Option<String> {
    let meta = fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let size = meta.len();
    Some(format!("{}:{}", mtime, size))
}

/// Discover files respecting .gitignore and .treesearchignore.
pub fn discover_files(root: &Path, config: &TreeSearchConfig, follow_symlinks: bool) -> Vec<PathBuf> {
    let parser_registry = ParserRegistry::new();

    let walker = WalkBuilder::new(root)
        .follow_links(follow_symlinks)
        .hidden(true) // skip hidden files
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .add_custom_ignore_filename(".treesearchignore")
        .max_depth(None)
        .build();

    let mut files: Vec<PathBuf> = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Walk error: {}", e);
                continue;
            }
        };
        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.into_path();
        if parser_registry.supports(&path) {
            files.push(path);
        }
        if files.len() >= config.max_dir_files {
            warn!("Reached max_dir_files limit ({}), stopping discovery", config.max_dir_files);
            break;
        }
    }
    files
}

/// Index a directory into an FTS5 database.
pub fn index_directory(
    root: &Path,
    fts_index: &FTS5Index,
    config: &TreeSearchConfig,
    follow_symlinks: bool,
    show_progress: bool,
) -> Result<IndexStats> {
    let start = Instant::now();

    // Discover files
    let files = discover_files(root, config, follow_symlinks);
    if files.is_empty() {
        info!("No supported files found in {:?}", root);
        return Ok(IndexStats {
            files_found: 0,
            files_indexed: 0,
            files_skipped: 0,
            files_failed: 0,
            nodes_indexed: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    // Check existing fingerprints for incremental indexing
    let existing_meta = fts_index.get_all_index_meta()?;

    // Filter to files that need (re-)indexing
    let mut to_index: Vec<PathBuf> = Vec::new();
    let mut skipped = 0usize;
    for file in &files {
        let path_str = file.to_string_lossy().to_string();
        let fp = file_fingerprint(file);
        match (&fp, existing_meta.get(&path_str)) {
            (Some(new_fp), Some(old_fp)) if new_fp == old_fp => {
                skipped += 1;
            }
            _ => {
                to_index.push(file.clone());
            }
        }
    }

    info!(
        "Discovered {} files, {} unchanged (skipping), {} to index",
        files.len(),
        skipped,
        to_index.len()
    );

    if to_index.is_empty() {
        return Ok(IndexStats {
            files_found: files.len(),
            files_indexed: 0,
            files_skipped: skipped,
            files_failed: 0,
            nodes_indexed: 0,
            duration_ms: start.elapsed().as_millis() as u64,
        });
    }

    // Progress bar
    let pb = if show_progress {
        let pb = ProgressBar::new(to_index.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("█▓░"),
        );
        Some(pb)
    } else {
        None
    };

    // Parallel parsing
    let parser_registry = ParserRegistry::new();
    let parse_results: Vec<(PathBuf, Result<Document>)> = to_index
        .par_iter()
        .map(|path| {
            let result = parse_file(&parser_registry, path);
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
            (path.clone(), result)
        })
        .collect();

    if let Some(ref pb) = pb {
        pb.finish_with_message("Parsing complete");
    }

    // Batch insert into FTS5 (single-threaded for SQLite)
    let mut indexed = 0usize;
    let mut failed = 0usize;
    let mut total_nodes = 0usize;
    let mut new_meta: HashMap<String, String> = HashMap::new();

    for (path, result) in parse_results {
        match result {
            Ok(doc) => {
                let path_str = path.to_string_lossy().to_string();
                match fts_index.index_document(&doc, false) {
                    Ok(count) => {
                        total_nodes += count;
                        indexed += 1;
                        if let Some(fp) = file_fingerprint(&path) {
                            new_meta.insert(path_str, fp);
                        }
                    }
                    Err(e) => {
                        warn!("Index error for {:?}: {}", path, e);
                        failed += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Parse error for {:?}: {}", path, e);
                failed += 1;
            }
        }
    }

    // Batch commit fingerprints
    fts_index.commit()?;
    if !new_meta.is_empty() {
        fts_index.set_index_meta_batch(&new_meta)?;
    }

    // Prune deleted files
    let all_paths: std::collections::HashSet<String> = files
        .iter()
        .map(|f| f.to_string_lossy().to_string())
        .collect();
    let mut pruned = 0;
    for old_path in existing_meta.keys() {
        if !all_paths.contains(old_path) {
            if let Some(doc_id) = fts_index.get_doc_id_by_source_path(old_path)? {
                fts_index.delete_document(&doc_id)?;
                pruned += 1;
            }
        }
    }
    if pruned > 0 {
        info!("Pruned {} deleted documents from index", pruned);
    }

    let duration = start.elapsed().as_millis() as u64;
    info!(
        "Indexed {} files ({} nodes) in {}ms ({} failed, {} skipped)",
        indexed, total_nodes, duration, failed, skipped
    );

    Ok(IndexStats {
        files_found: files.len(),
        files_indexed: indexed,
        files_skipped: skipped,
        files_failed: failed,
        nodes_indexed: total_nodes,
        duration_ms: duration,
    })
}

/// Parse a single file using the parser registry.
fn parse_file(registry: &ParserRegistry, path: &Path) -> Result<Document> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read {:?}", path))?;
    registry
        .parse_content(path, &content)?
        .ok_or_else(|| anyhow::anyhow!("No parser found for {:?}", path))
}

/// Index statistics.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub files_found: usize,
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub nodes_indexed: usize,
    pub duration_ms: u64,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Indexed: {} files ({} nodes) in {}ms | {} skipped, {} failed",
            self.files_indexed, self.nodes_indexed, self.duration_ms,
            self.files_skipped, self.files_failed,
        )
    }
}
