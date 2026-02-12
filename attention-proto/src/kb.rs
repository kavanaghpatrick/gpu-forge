//! KB finding output — emit structured findings as JSON-lines to findings.jsonl

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

/// A single empirical finding from a prototype benchmark or analysis.
#[derive(Serialize)]
pub struct KbFinding {
    pub domain: String,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub confidence: f32,
    pub source: String,
}

/// Path to findings.jsonl in the project root (attention-proto/).
fn findings_path() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("findings.jsonl")
}

/// Serialize a KbFinding as one JSON line and append to findings.jsonl.
pub fn emit_finding(finding: &KbFinding) {
    let json = serde_json::to_string(finding).expect("failed to serialize KbFinding");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(findings_path())
        .expect("failed to open findings.jsonl");
    writeln!(file, "{}", json).expect("failed to write finding");
}

/// Remove findings.jsonl if it exists (for test cleanup).
pub fn clear_findings() {
    let path = findings_path();
    if path.exists() {
        std::fs::remove_file(path).expect("failed to remove findings.jsonl");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_finding() {
        // Clean up before test
        clear_findings();

        let finding = KbFinding {
            domain: "msl-kernels".to_string(),
            title: "Flash attention tile 32x128 achieves 85% ALU on M4".to_string(),
            content: "Tile configuration Br=32, Bc=128 with D=64 reaches 2.1 TFLOPS on M4 GPU."
                .to_string(),
            tags: vec![
                "flash-attention".to_string(),
                "simdgroup-matrix".to_string(),
                "M4".to_string(),
            ],
            confidence: 0.92,
            source: "attention-proto/proto1_flash".to_string(),
        };

        emit_finding(&finding);

        // Read findings.jsonl and verify valid JSON
        let contents = std::fs::read_to_string(findings_path())
            .expect("failed to read findings.jsonl");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1, "expected exactly one finding line");

        // Parse back as JSON value to verify validity
        let parsed: serde_json::Value =
            serde_json::from_str(lines[0]).expect("finding line is not valid JSON");
        assert_eq!(parsed["domain"], "msl-kernels");
        assert_eq!(parsed["confidence"], 0.92f64);
        assert_eq!(parsed["tags"].as_array().unwrap().len(), 3);

        // Emit a second finding to verify append behavior
        let finding2 = KbFinding {
            domain: "metal-compute".to_string(),
            title: "Second test finding".to_string(),
            content: "Verifying append behavior".to_string(),
            tags: vec!["test".to_string()],
            confidence: 0.5,
            source: "attention-proto/test".to_string(),
        };
        emit_finding(&finding2);

        let contents = std::fs::read_to_string(findings_path())
            .expect("failed to read findings.jsonl after second emit");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "expected two finding lines after second emit");

        // Verify both are valid JSON
        for line in &lines {
            let _: serde_json::Value =
                serde_json::from_str(line).expect("each line must be valid JSON");
        }

        // Clean up after test
        clear_findings();
    }
}
