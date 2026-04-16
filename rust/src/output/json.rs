//! JSON output format for search results.
//!
//! Serializes results as a JSON object with a `version` field and `results` array
//! using `serde_json::to_string_pretty` for human-readable output.

use serde::Serialize;

use crate::document::SearchResult;

use super::OutputFormat;

/// Schema version for the JSON output format.
const SCHEMA_VERSION: &str = "1.0";

/// Top-level JSON output envelope.
#[derive(Serialize)]
struct JsonEnvelope<'a> {
    version: &'static str,
    results: &'a [SearchResult],
}

/// JSON output formatter.
pub struct JsonOutput;

impl JsonOutput {
    pub fn new() -> Self {
        Self
    }
}

impl Default for JsonOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputFormat for JsonOutput {
    fn render(&self, results: &[SearchResult], _verbose: u8) -> String {
        let envelope = JsonEnvelope {
            version: SCHEMA_VERSION,
            results,
        };
        serde_json::to_string_pretty(&envelope).unwrap_or_else(|e| {
            format!("{{\"error\": \"serialization failed: {}\"}}", e)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_result() -> SearchResult {
        SearchResult {
            node_id: "1".into(),
            doc_id: "doc1".into(),
            doc_name: "test.md".into(),
            title: "Test Title".into(),
            summary: "A test summary".into(),
            text: "Some body text".into(),
            source_type: "markdown".into(),
            source_path: "docs/test.md".into(),
            line_start: Some(1),
            line_end: Some(10),
            score: 0.95,
            depth: 0,
            breadcrumb: vec!["Root".into(), "Test Title".into()],
        }
    }

    #[test]
    fn test_json_output_has_version() {
        let json_out = JsonOutput::new();
        let output = json_out.render(&[sample_result()], 0);
        assert!(output.contains("\"version\": \"1.0\""));
    }

    #[test]
    fn test_json_output_has_results() {
        let json_out = JsonOutput::new();
        let output = json_out.render(&[sample_result()], 0);
        assert!(output.contains("\"results\""));
        assert!(output.contains("\"Test Title\""));
        assert!(output.contains("0.95"));
    }

    #[test]
    fn test_json_output_empty() {
        let json_out = JsonOutput::new();
        let output = json_out.render(&[], 0);
        assert!(output.contains("\"version\": \"1.0\""));
        assert!(output.contains("\"results\": []"));
    }

    #[test]
    fn test_json_output_valid_json() {
        let json_out = JsonOutput::new();
        let output = json_out.render(&[sample_result()], 0);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["version"], "1.0");
        assert!(parsed["results"].is_array());
        assert_eq!(parsed["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_json_output_pretty_printed() {
        let json_out = JsonOutput::new();
        let output = json_out.render(&[sample_result()], 0);
        // Pretty-printed JSON should contain newlines and indentation
        assert!(output.contains('\n'));
        assert!(output.contains("  "));
    }

    #[test]
    fn test_json_output_multiple_results() {
        let json_out = JsonOutput::new();
        let results = vec![sample_result(), sample_result()];
        let output = json_out.render(&results, 0);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["results"].as_array().unwrap().len(), 2);
    }
}
