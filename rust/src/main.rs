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

/// treesearch — structure-aware document search
#[derive(Parser)]
#[command(name = "ts", version, about = "Structure-aware document search CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Search query (shorthand for `ts search <QUERY> [PATH]`)
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

    /// Maximum results to return
    #[arg(short = 'n', long, default_value = "15", global = true)]
    max_results: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for a query in indexed documents
    Search {
        /// Search query
        query: String,
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
    let cli = Cli::parse();

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

    match cli.command {
        Some(Commands::Search { query, path }) => {
            cmd_search(&query, &path, &config, &*format, cli.verbose, cli.follow, cli.max_results)
        }
        Some(Commands::Index { path }) => cmd_index(&path, &config, cli.follow),
        Some(Commands::Stats { path }) => cmd_stats(&path),
        None => {
            // Shorthand: `ts "query" [path]`
            if let Some(query) = cli.query {
                let path = cli.path.unwrap_or_else(|| PathBuf::from("."));
                cmd_search(&query, &path, &config, &*format, cli.verbose, cli.follow, cli.max_results)
            } else {
                eprintln!("Usage: ts <QUERY> [PATH]");
                eprintln!("       ts search <QUERY> [PATH]");
                eprintln!("       ts index [PATH]");
                eprintln!("       ts stats [PATH]");
                process::exit(1);
            }
        }
    }
}

fn db_path_for(root: &PathBuf) -> PathBuf {
    let canonical = std::fs::canonicalize(root).unwrap_or_else(|_| root.clone());
    canonical.join(".treesearch").join("index.db")
}

fn cmd_search(
    query: &str,
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
    let mut results = search::search(query, &documents, &fts, config)?;
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
