use std::env;

/// Unified configuration for TreeSearch Rust CLI.
///
/// Priority (high -> low):
///   1. CLI flags / set_config()
///   2. Environment variables (TREESEARCH_*)
///   3. Built-in defaults
#[derive(Debug, Clone)]
pub struct TreeSearchConfig {
    // Search
    pub max_nodes_per_doc: usize,
    pub top_k_docs: usize,

    // Index
    pub max_concurrency: usize,
    pub max_dir_files: usize,
    pub max_node_chars: usize,
    pub max_result_chars: usize,

    // FTS weights
    pub fts_title_weight: f64,
    pub fts_summary_weight: f64,
    pub fts_body_weight: f64,
    pub fts_code_weight: f64,
    pub fts_front_matter_weight: f64,

    // Tree search
    pub search_mode: SearchMode,
    pub anchor_top_k: usize,
    pub max_anchor_per_doc: usize,
    pub max_expansions: usize,
    pub max_hops: usize,
    pub max_siblings: usize,
    pub min_frontier_score: f64,
    pub early_stop_score: f64,
    pub path_top_k: usize,

    // Tokenizer
    pub cjk_tokenizer: CjkTokenizerMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    Auto,
    Flat,
    Tree,
}

impl SearchMode {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Auto => "auto",
            Self::Flat => "flat",
            Self::Tree => "tree",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "flat" => Self::Flat,
            "tree" => Self::Tree,
            _ => Self::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CjkTokenizerMode {
    Auto,
    Jieba,
    Bigram,
    Char,
}

impl CjkTokenizerMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "jieba" => Self::Jieba,
            "bigram" => Self::Bigram,
            "char" => Self::Char,
            _ => Self::Auto,
        }
    }
}

impl Default for TreeSearchConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(256);

        Self {
            max_nodes_per_doc: 5,
            top_k_docs: 3,
            max_concurrency: cpus,
            max_dir_files: 10_000,
            max_node_chars: 8_000,
            max_result_chars: 32_000,

            fts_title_weight: 5.0,
            fts_summary_weight: 2.0,
            fts_body_weight: 10.0,
            fts_code_weight: 1.0,
            fts_front_matter_weight: 2.0,

            search_mode: SearchMode::Auto,
            anchor_top_k: 5,
            max_anchor_per_doc: 3,
            max_expansions: 40,
            max_hops: 3,
            max_siblings: 2,
            min_frontier_score: 0.1,
            early_stop_score: 0.95,
            path_top_k: 3,

            cjk_tokenizer: CjkTokenizerMode::Auto,
        }
    }
}

impl TreeSearchConfig {
    /// Create config from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(v) = env::var("TREESEARCH_CJK_TOKENIZER") {
            config.cjk_tokenizer = CjkTokenizerMode::from_str(&v);
        }
        if let Ok(v) = env::var("TREESEARCH_SEARCH_MODE") {
            config.search_mode = SearchMode::from_str(&v);
        }
        if let Ok(v) = env::var("TREESEARCH_MAX_NODES_PER_DOC") {
            if let Ok(n) = v.parse() {
                config.max_nodes_per_doc = n;
            }
        }
        if let Ok(v) = env::var("TREESEARCH_TOP_K_DOCS") {
            if let Ok(n) = v.parse() {
                config.top_k_docs = n;
            }
        }

        config
    }

    /// Column weights as an array for FTS5 bm25() function.
    /// Order: title, summary, body, code_blocks, front_matter
    pub fn fts_weights(&self) -> [f64; 5] {
        [
            self.fts_title_weight,
            self.fts_summary_weight,
            self.fts_body_weight,
            self.fts_code_weight,
            self.fts_front_matter_weight,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TreeSearchConfig::default();
        assert_eq!(config.max_nodes_per_doc, 5);
        assert_eq!(config.fts_body_weight, 10.0);
        assert_eq!(config.search_mode, SearchMode::Auto);
    }

    #[test]
    fn test_search_mode_parse() {
        assert_eq!(SearchMode::from_str("flat"), SearchMode::Flat);
        assert_eq!(SearchMode::from_str("TREE"), SearchMode::Tree);
        assert_eq!(SearchMode::from_str("blah"), SearchMode::Auto);
    }

    #[test]
    fn test_fts_weights() {
        let config = TreeSearchConfig::default();
        let w = config.fts_weights();
        assert_eq!(w, [5.0, 2.0, 10.0, 1.0, 2.0]);
    }
}
