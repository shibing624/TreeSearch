pub mod json;
pub mod plain;
pub mod tty;

use crate::document::SearchResult;

/// Trait for rendering search results in different output formats.
pub trait OutputFormat {
    /// Render search results to a string.
    ///
    /// `verbose` controls detail level:
    ///   0 = minimal (title + score)
    ///   1 = normal (title + summary + score)
    ///   2 = full (title + summary + text + line numbers)
    fn render(&self, results: &[SearchResult], verbose: u8) -> String;
}
