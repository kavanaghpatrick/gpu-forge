//! Adaptive dictionary encoding for string columns.
//!
//! Builds a sorted dictionary of unique string values, mapping each to a compact
//! integer code (u32). This enables GPU-side string comparison via integer comparison:
//! instead of comparing strings on the GPU, we compare dictionary codes.
//!
//! Encoding strategy:
//! - If distinct values < 10K: dictionary encode (dict code per row)
//! - Otherwise: offset+data (future, not yet implemented)
//!
//! For WHERE col = 'value', the executor looks up the value's dict code on CPU,
//! then dispatches the existing INT64 filter kernel to compare codes on GPU.

use std::collections::HashMap;

/// Maximum number of distinct values for dictionary encoding.
/// Above this threshold, offset+data encoding would be used instead.
const DICT_THRESHOLD: usize = 10_000;

/// A dictionary encoding for a string column.
///
/// Maps sorted unique string values to sequential u32 codes (0, 1, 2, ...).
/// The dictionary is built on CPU after CSV/JSON parsing; encoded indices
/// are uploaded to a Metal buffer for GPU-side integer comparison.
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Sorted unique values. Index = dict code.
    values: Vec<String>,
    /// Reverse lookup: string value -> dict code.
    index_map: HashMap<String, u32>,
}

impl Dictionary {
    /// Build a dictionary from column values.
    ///
    /// Collects unique values, sorts them, and assigns sequential codes.
    /// Returns `Some(Dictionary)` if distinct count <= DICT_THRESHOLD,
    /// or `None` if there are too many distinct values.
    pub fn build(column_values: &[String]) -> Option<Self> {
        // Collect unique values
        let mut unique: Vec<String> = column_values
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if unique.len() > DICT_THRESHOLD {
            return None;
        }

        // Sort for deterministic code assignment
        unique.sort();

        // Build reverse index
        let index_map: HashMap<String, u32> = unique
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i as u32))
            .collect();

        Some(Self {
            values: unique,
            index_map,
        })
    }

    /// Encode a string value to its dictionary code.
    ///
    /// Returns `None` if the value is not in the dictionary.
    pub fn encode(&self, value: &str) -> Option<u32> {
        self.index_map.get(value).copied()
    }

    /// Decode a dictionary code back to its string value.
    ///
    /// # Panics
    /// Panics if `code` is out of bounds.
    pub fn decode(&self, code: u32) -> &str {
        &self.values[code as usize]
    }

    /// Encode all values in a column, returning u32 codes.
    ///
    /// Values not found in the dictionary are encoded as `u32::MAX` (sentinel for NULL/missing).
    pub fn encode_column(&self, column_values: &[String]) -> Vec<u32> {
        column_values
            .iter()
            .map(|v| self.encode(v).unwrap_or(u32::MAX))
            .collect()
    }

    /// Number of distinct values in the dictionary.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get all dictionary values (sorted).
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// Check if dictionary encoding is appropriate for the given distinct count.
    pub fn should_dict_encode(distinct_count: usize) -> bool {
        distinct_count <= DICT_THRESHOLD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_dictionary() {
        let values = vec![
            "Europe".to_string(),
            "Asia".to_string(),
            "Europe".to_string(),
            "Africa".to_string(),
            "Asia".to_string(),
        ];
        let dict = Dictionary::build(&values).unwrap();

        assert_eq!(dict.len(), 3);
        // Sorted: Africa=0, Asia=1, Europe=2
        assert_eq!(dict.values(), &["Africa", "Asia", "Europe"]);
    }

    #[test]
    fn test_encode_decode() {
        let values = vec![
            "Europe".to_string(),
            "Asia".to_string(),
            "Africa".to_string(),
        ];
        let dict = Dictionary::build(&values).unwrap();

        assert_eq!(dict.encode("Africa"), Some(0));
        assert_eq!(dict.encode("Asia"), Some(1));
        assert_eq!(dict.encode("Europe"), Some(2));
        assert_eq!(dict.encode("Antarctica"), None);

        assert_eq!(dict.decode(0), "Africa");
        assert_eq!(dict.decode(1), "Asia");
        assert_eq!(dict.decode(2), "Europe");
    }

    #[test]
    fn test_encode_column() {
        let values = vec![
            "Europe".to_string(),
            "Asia".to_string(),
            "Europe".to_string(),
            "Africa".to_string(),
        ];
        let dict = Dictionary::build(&values).unwrap();

        let encoded = dict.encode_column(&values);
        // Africa=0, Asia=1, Europe=2
        assert_eq!(encoded, vec![2, 1, 2, 0]);
    }

    #[test]
    fn test_empty_dictionary() {
        let values: Vec<String> = vec![];
        let dict = Dictionary::build(&values).unwrap();
        assert!(dict.is_empty());
        assert_eq!(dict.len(), 0);
    }

    #[test]
    fn test_single_value() {
        let values = vec!["only".to_string(), "only".to_string()];
        let dict = Dictionary::build(&values).unwrap();
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.encode("only"), Some(0));
    }

    #[test]
    fn test_missing_value_sentinel() {
        let values = vec!["a".to_string(), "b".to_string()];
        let dict = Dictionary::build(&values).unwrap();
        let encoded = dict.encode_column(&["a".to_string(), "c".to_string()]);
        assert_eq!(encoded, vec![0, u32::MAX]);
    }
}
