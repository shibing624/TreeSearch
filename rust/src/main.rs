use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::process;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

use treesearch::config::{SearchMode, TreeSearchConfig};
use treesearch::engine::fts::FTS5Index;
use treesearch::engine::indexer::{self};
use treesearch::engine::search;
use treesearch::output::{self, OutputFormat};
use treesearch::tokenizer::cjk;

/// treesearch — structure-aware document search
#[derive(Parser)]
#[command(name = "ts", version, about = "Structure-aware document search CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (supports auth* prefix and *auth* contains matching)
    #[arg(value_name = "QUERY")]
    query: Option<String>,

    /// Path to search/index (default: current directory)
    #[arg(value_name = "PATH")]
    path: Option<PathBuf>,

    /// Output format
    #[arg(long, value_enum, global = true)]
    format: Option<FormatChoice>,

    /// Force JSON output
    #[arg(long, global = true)]
    json: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    no_color: bool,

    /// Verbosity level (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Follow symbolic links
    #[arg(long, global = true)]
    follow: bool,

    /// Search mode
    #[arg(long, value_enum, global = true)]
    mode: Option<ModeChoice>,

    /// Treat the provided query as a raw regex pattern
    #[arg(long, global = true)]
    regex: bool,

    /// Pass a raw FTS5 expression directly
    #[arg(long, global = true, value_name = "EXPR")]
    fts_expression: Option<String>,

    /// Maximum results to return
    #[arg(short = 'n', long, default_value = "15", global = true)]
    max_results: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for a query in indexed documents
    Search {
        /// Search query (supports auth* prefix and *auth* contains matching)
        query: Option<String>,
        /// Path to search (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Build or update the search index
    Index {
        /// Path to index (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Show index statistics
    Stats {
        /// Path to the index directory (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

#[derive(Copy, Clone, ValueEnum)]
enum FormatChoice {
    Tty,
    Json,
    Plain,
}

#[derive(Copy, Clone, ValueEnum)]
enum ModeChoice {
    Auto,
    Flat,
    Tree,
}

fn main() {
    let cli = normalize_cli(Cli::parse());

    // Initialize logging
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        _ => "debug",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(false)
        .with_writer(io::stderr)
        .init();

    if let Err(e) = run(cli) {
        eprintln!("error: {:#}", e);
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    // Determine output format
    let stdout = io::stdout();
    let is_tty = stdout.is_terminal();
    let format: Box<dyn OutputFormat> = if cli.json || matches!(cli.format, Some(FormatChoice::Json)) || !is_tty {
        Box::new(output::json::JsonOutput)
    } else if cli.no_color || matches!(cli.format, Some(FormatChoice::Plain)) {
        Box::new(output::plain::PlainOutput)
    } else {
        Box::new(output::tty::TtyOutput::new(Vec::new()))
    };

    // Build config
    let mut config = TreeSearchConfig::from_env();
    if let Some(mode) = cli.mode {
        config.search_mode = match mode {
            ModeChoice::Auto => SearchMode::Auto,
            ModeChoice::Flat => SearchMode::Flat,
            ModeChoice::Tree => SearchMode::Tree,
        };
    }

    // Apply jieba custom dictionary (no-op if none configured).
    cjk::configure_from(&config);
    let regex = cli.regex;
    let fts_expression = cli.fts_expression.clone();
    let query = cli.query.clone();
    let path = cli.path.clone();
    let verbose = cli.verbose;
    let follow = cli.follow;
    let max_results = cli.max_results;

    match cli.command {
        Some(Commands::Search { query, path }) => {
            let request = resolve_search_request(query, fts_expression.clone(), regex)?;
            cmd_search(
                &request.query,
                request.fts_expression.as_deref(),
                request.regex,
                &path,
                &config,
                &*format,
                verbose,
                follow,
                max_results,
            )
        }
        Some(Commands::Index { path }) => cmd_index(&path, &config, follow),
        Some(Commands::Stats { path }) => cmd_stats(&path),
        None => {
            let request = resolve_search_request(query, fts_expression, regex);
            if let Ok(request) = request {
                let path = path.unwrap_or_else(|| PathBuf::from("."));
                cmd_search(
                    &request.query,
                    request.fts_expression.as_deref(),
                    request.regex,
                    &path,
                    &config,
                    &*format,
                    verbose,
                    follow,
                    max_results,
                )
            } else {
                eprintln!("Usage: ts <QUERY> [PATH]");
                eprintln!("       ts --fts-expression <EXPR> [PATH]");
                eprintln!("       ts search <QUERY> [PATH]");
                eprintln!("       ts search --fts-expression <EXPR> [PATH]");
                eprintln!("       ts index [PATH]");
                eprintln!("       ts stats [PATH]");
                process::exit(1);
            }
        }
    }
}

struct SearchRequest {
    query: String,
    fts_expression: Option<String>,
    regex: bool,
}

fn normalize_cli(mut cli: Cli) -> Cli {
    if cli.command.is_none() && cli.fts_expression.is_some() && cli.query.is_some() && cli.path.is_none() {
        cli.path = cli.query.take().map(PathBuf::from);
    }
    if let Some(Commands::Search { query, path }) = cli.command.as_mut() {
        if cli.fts_expression.is_some() && query.is_some() && *path == PathBuf::from(".") {
            *path = PathBuf::from(query.take().unwrap());
        }
    }
    cli
}

fn resolve_search_request(
    query: Option<String>,
    fts_expression: Option<String>,
    regex: bool,
) -> Result<SearchRequest> {
    if regex && fts_expression.is_some() {
        anyhow::bail!("--regex and --fts-expression cannot be used together");
    }
    if query.is_some() && fts_expression.is_some() {
        anyhow::bail!("pass either a query or --fts-expression, not both");
    }
    if let Some(expr) = fts_expression {
        return Ok(SearchRequest {
            query: String::new(),
            fts_expression: Some(expr),
            regex: false,
        });
    }
    if let Some(query) = query {
        return Ok(SearchRequest {
            query,
            fts_expression: None,
            regex,
        });
    }
    anyhow::bail!("a query is required unless --fts-expression is provided")
}

fn db_path_for(root: &PathBuf) -> PathBuf {
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
    canonical.join(".treesearch").join("index.db")
}

fn cmd_search(
    query: &str,
    fts_expression: Option<&str>,
    regex: bool,
    path: &PathBuf,
    config: &TreeSearchConfig,
    format: &dyn OutputFormat,
    verbose: u8,
    follow_symlinks: bool,
    max_results: usize,
) -> Result<()> {
    let db = db_path_for(path);

    // Auto-index if no index exists
    if !db.exists() {
        eprintln!("No index found. Building index for {:?}...", path);
        if let Some(parent) = db.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let fts = FTS5Index::new(Some(db.to_str().unwrap()), None)?;
        let stats = indexer::index_directory(path, &fts, config, follow_symlinks, true)?;
        eprintln!("{}", stats);
        fts.close();
    }

    let fts = FTS5Index::new(Some(db.to_str().unwrap()), None)?;

    // Load documents
    let documents = fts.load_all_documents()?;
    if documents.is_empty() {
        eprintln!("Index is empty. Run: ts index {:?}", path);
        return Ok(());
    }

    // Search
    let start = std::time::Instant::now();
    let mut results = search::search_with_options(query, &documents, &fts, config, fts_expression, regex)?;
    let search_time = start.elapsed();

    results.truncate(max_results);

    if verbose >= 1 {
        let stats = fts.get_stats()?;
        eprintln!(
            "Indexed: {} files ({} nodes) | Search: {} results in {:.1}ms",
            stats.document_count,
            stats.node_count,
            results.len(),
            search_time.as_secs_f64() * 1000.0,
        );
    }

    // Output
    let output = format.render(&results, verbose);
    // Handle broken pipe gracefully
    match io::stdout().write_all(output.as_bytes()) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => {}
        Err(e) => return Err(e.into()),
    }

    fts.close();
    Ok(())
}

fn cmd_index(path: &PathBuf, config: &TreeSearchConfig, follow_symlinks: bool) -> Result<()> {
    let db = db_path_for(path);
    if let Some(parent) = db.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let fts = FTS5Index::new(Some(db.to_str().unwrap()), None)?;
    let stats = indexer::index_directory(path, &fts, config, follow_symlinks, true)?;
    println!("{}", stats);
    fts.close();
    Ok(())
}

fn cmd_stats(path: &PathBuf) -> Result<()> {
    let db = db_path_for(path);
    if !db.exists() {
        eprintln!("No index found at {:?}", db);
        eprintln!("Run: ts index {:?}", path);
        return Ok(());
    }

    let fts = FTS5Index::new(Some(db.to_str().unwrap()), None)?;
    let stats = fts.get_stats()?;

    println!("TreeSearch Index Statistics");
    println!("==========================");
    println!("Database:   {:?}", db);
    println!("Documents:  {}", stats.document_count);
    println!("Nodes:      {}", stats.node_count);

    // File size
    if let Ok(meta) = std::fs::metadata(&db) {
        let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
        println!("Index size: {:.2} MB", size_mb);
    }

    // File type distribution
    let docs = fts.load_all_documents()?;
    let mut type_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for doc in &docs {
        *type_counts.entry(doc.source_type.to_string()).or_insert(0) += 1;
    }
    if !type_counts.is_empty() {
        println!("\nFile types:");
        let mut sorted: Vec<_> = type_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        for (stype, count) in sorted {
            println!("  {:<12} {}", stype, count);
        }
    }

    fts.close();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_default_parser_accepts_regex_flag() {
        let cli = normalize_cli(Cli::parse_from(["ts", "--regex", "auth.*", "."]));
        assert!(cli.regex);
        assert_eq!(cli.query.as_deref(), Some("auth.*"));
    }

    #[test]
    fn test_default_parser_accepts_fts_expression() {
        let cli = normalize_cli(Cli::parse_from(["ts", "--fts-expression", "auth*", "."]));
        assert_eq!(cli.fts_expression.as_deref(), Some("auth*"));
        assert_eq!(cli.query, None);
    }

    #[test]
    fn test_search_subcommand_accepts_regex_flag() {
        let cli = normalize_cli(Cli::parse_from(["ts", "search", "--regex", "auth.*"]));
        match cli.command {
            Some(Commands::Search { query, .. }) => {
                assert_eq!(query.as_deref(), Some("auth.*"));
                assert!(cli.regex);
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn test_search_subcommand_accepts_fts_expression() {
        let cli = normalize_cli(Cli::parse_from(["ts", "search", "--fts-expression", "auth*"]));
        match cli.command {
            Some(Commands::Search { query, .. }) => {
                assert_eq!(query, None);
                assert_eq!(cli.fts_expression.as_deref(), Some("auth*"));
            }
            _ => panic!("expected search command"),
        }
    }

    #[test]
    fn test_search_subcommand_accepts_fts_expression_with_path() {
        let cli = normalize_cli(Cli::parse_from(["ts", "search", "--fts-expression", "auth*", "src"]));
        match cli.command {
            Some(Commands::Search { query, path }) => {
                assert_eq!(query, None);
                assert_eq!(path, PathBuf::from("src"));
                assert_eq!(cli.fts_expression.as_deref(), Some("auth*"));
            }
            _ => panic!("expected search command"),
        }
    }
}
