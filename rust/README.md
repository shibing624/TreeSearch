# TreeSearch Rust CLI

`treesearch` is a fast, structure-aware document search CLI built with Rust.
It indexes files into a local SQLite FTS5 database and searches by document
structure instead of chunking text into arbitrary fragments.

This crate publishes the `ts` executable.

## Install

**Homebrew (macOS / Linux)**

```bash
brew tap shibing624/tap
brew install treesearch
ts --help
```

**Cargo**

```bash
cargo install treesearch
ts --help
```

**Prebuilt binaries**

Download from GitHub Releases:

- Linux: `x86_64-unknown-linux-gnu`
- macOS Intel: `x86_64-apple-darwin`
- macOS Apple Silicon: `aarch64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`

Release page: <https://github.com/shibing624/TreeSearch/releases>

## Why TreeSearch

- No embeddings
- No vector database
- No chunk splitting
- SQLite FTS5 with persistent local indexes
- Structure-aware retrieval for Markdown, text, code, HTML, and more

## Quick Start

Search the current directory with the default auto mode:

```bash
ts "How does auth work?" .
```

Build the index explicitly:

```bash
ts index .
```

Inspect index stats:

```bash
ts stats .
```

## Wildcard Queries

`ts` supports a narrow set of wildcard shortcuts:

- `auth*`: prefix match
- `*auth*`: contains-style regex match
- other wildcard shapes currently fall back to regular query parsing

For explicit control:

- `ts --regex "o?auth" .` treats the query as a raw regex
- `ts search --regex "o?auth"` runs indexed search in regex mode
- `ts --fts-expression "auth*" .` passes a raw FTS5 expression
- `ts search --fts-expression "auth*"` runs indexed search with raw FTS5 syntax
- Invalid regex patterns raise an explicit error instead of silently returning no results

Examples:

```bash
ts "auth*" .
ts "*auth*" .
ts --regex "o?auth" .
ts --fts-expression "auth*" .
```

## Search Modes

`ts` supports three search modes:

- `auto`: default mode, automatically selects `flat` or `tree`
- `flat`: force FTS-style flat retrieval
- `tree`: force tree traversal retrieval

Examples:

```bash
ts "query" .               # auto (default)
ts "query" . --mode flat   # force flat
ts "query" . --mode tree   # force tree
```

In `auto` mode, TreeSearch uses the same three-layer decision logic as the
Python version:

1. Source type mapping: file types that benefit from tree search are marked explicitly.
2. Depth verification: only documents with real structure depth are treated as hierarchical.
3. Ratio threshold: if enough indexed documents benefit from tree mode, use `tree`; otherwise use `flat`.

## Commands

```text
ts [OPTIONS] [QUERY] [PATH]
ts search <QUERY> [PATH]
ts index [PATH]
ts stats [PATH]
```

Useful options:

- `--mode auto|flat|tree`
- `--format tty|json|plain`
- `--json`
- `--follow`
- `-n, --max-results <N>`

## Output Formats

- `tty`: colored terminal output
- `json`: machine-readable JSON output
- `plain`: plain text output

## Documentation

- Project homepage: <https://github.com/shibing624/TreeSearch>
- API docs: <https://docs.rs/treesearch>

## License

Apache-2.0
