//! HashMap-based CPU baselines for group-by aggregate operations.
//!
//! Uses std::collections::HashMap to compute per-group aggregates
//! (sum, count, min, max) in a single pass over the data.

use std::collections::HashMap;

/// Per-group aggregate results.
#[derive(Debug, Clone)]
pub struct GroupAgg {
    pub sum: f64,
    pub count: u32,
    pub min: f32,
    pub max: f32,
}

/// HashMap-based group-by aggregate.
///
/// For each (key, value) pair, accumulates sum/count/min/max per group key.
/// Returns a HashMap mapping key -> GroupAgg.
pub fn hashmap_groupby(keys: &[u32], values: &[f32]) -> HashMap<u32, GroupAgg> {
    let mut groups: HashMap<u32, GroupAgg> = HashMap::new();

    for (&k, &v) in keys.iter().zip(values.iter()) {
        let entry = groups.entry(k).or_insert(GroupAgg {
            sum: 0.0,
            count: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        });
        entry.sum += v as f64;
        entry.count += 1;
        entry.min = entry.min.min(v);
        entry.max = entry.max.max(v);
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hashmap_groupby_basic() {
        let keys = vec![1, 2, 1, 2, 1];
        let values = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
        let result = hashmap_groupby(&keys, &values);

        assert_eq!(result.len(), 2);

        let g1 = &result[&1];
        assert_eq!(g1.count, 3);
        assert!((g1.sum - 90.0).abs() < 1e-6);
        assert!((g1.min - 10.0).abs() < 1e-6);
        assert!((g1.max - 50.0).abs() < 1e-6);

        let g2 = &result[&2];
        assert_eq!(g2.count, 2);
        assert!((g2.sum - 60.0).abs() < 1e-6);
        assert!((g2.min - 20.0).abs() < 1e-6);
        assert!((g2.max - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_hashmap_groupby_single_group() {
        let keys = vec![5, 5, 5];
        let values = vec![1.0f32, 2.0, 3.0];
        let result = hashmap_groupby(&keys, &values);

        assert_eq!(result.len(), 1);
        let g = &result[&5];
        assert_eq!(g.count, 3);
        assert!((g.sum - 6.0).abs() < 1e-6);
        assert!((g.min - 1.0).abs() < 1e-6);
        assert!((g.max - 3.0).abs() < 1e-6);
    }
}
