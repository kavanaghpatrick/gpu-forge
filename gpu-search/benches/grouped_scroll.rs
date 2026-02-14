//! Grouped scroll performance benchmarks.
//!
//! Validates that the core grouped virtual scroll operations stay within
//! acceptable latency budgets:
//! - `FlatRowModel::rebuild()` < 1ms for 10K rows
//! - `first_visible_row()` binary search is O(log n)
//! - Incremental grouping is fast for streaming appends

use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gpu_search::search::types::{ContentMatch, FileMatch};
use gpu_search::ui::path_utils::abbreviate_path;
use gpu_search::ui::results_list::{ContentGroup, FlatRowModel};

/// Create a ContentMatch at a given path and line.
fn make_cm(path: &str, line: u32) -> ContentMatch {
    ContentMatch {
        path: PathBuf::from(path),
        line_number: line,
        line_content: format!("    let result = process_data(input_{});", line),
        context_before: vec!["// context line before".to_string()],
        context_after: vec!["// context line after".to_string()],
        match_range: 18..30,
    }
}

/// Build content matches and groups for a given number of groups and matches per group.
fn build_groups(
    n_groups: usize,
    matches_per_group: usize,
) -> (Vec<FileMatch>, Vec<ContentMatch>, Vec<ContentGroup>) {
    let file_matches: Vec<FileMatch> = Vec::new();
    let mut content_matches = Vec::new();
    let mut content_groups: Vec<ContentGroup> = Vec::new();
    let mut group_index_map: HashMap<PathBuf, usize> = HashMap::new();
    let search_root = Path::new("/project");

    for g in 0..n_groups {
        let path = PathBuf::from(format!("/project/src/module_{}/handler.rs", g));
        for line in 0..matches_per_group {
            let cm = make_cm(path.to_str().unwrap(), (line as u32) + 1);
            let idx = content_matches.len();
            content_matches.push(cm);

            if let Some(&group_idx) = group_index_map.get(&path) {
                content_groups[group_idx].match_indices.push(idx);
            } else {
                let (dir_display, filename) = abbreviate_path(&path, search_root);
                let extension = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                let group_idx = content_groups.len();
                group_index_map.insert(path.clone(), group_idx);
                content_groups.push(ContentGroup {
                    path: path.clone(),
                    dir_display,
                    filename,
                    extension,
                    match_indices: vec![idx],
                });
            }
        }
    }

    (file_matches, content_matches, content_groups)
}

/// Standalone incremental grouping (mirrors GpuSearchApp::recompute_groups).
fn recompute_groups_incremental(
    content_matches: &[ContentMatch],
    content_groups: &mut Vec<ContentGroup>,
    group_index_map: &mut HashMap<PathBuf, usize>,
    last_grouped_index: &mut usize,
    search_root: &Path,
) {
    for i in *last_grouped_index..content_matches.len() {
        let cm = &content_matches[i];
        if let Some(&group_idx) = group_index_map.get(&cm.path) {
            content_groups[group_idx].match_indices.push(i);
        } else {
            let (dir_display, filename) = abbreviate_path(&cm.path, search_root);
            let extension = cm
                .path
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let group_idx = content_groups.len();
            group_index_map.insert(cm.path.clone(), group_idx);
            content_groups.push(ContentGroup {
                path: cm.path.clone(),
                dir_display,
                filename,
                extension,
                match_indices: vec![i],
            });
        }
    }
    *last_grouped_index = content_matches.len();
}

fn bench_grouped_scroll(c: &mut Criterion) {
    // -----------------------------------------------------------------------
    // bench_rebuild_flat_row_model_100: 10 groups, 10 matches each = 100 matches
    // -----------------------------------------------------------------------
    {
        let (fm, _cm, groups) = build_groups(10, 10);
        c.bench_function("rebuild_flat_row_model_100", |b| {
            b.iter(|| FlatRowModel::rebuild(&fm, &groups, Some(5)))
        });
    }

    // -----------------------------------------------------------------------
    // bench_rebuild_flat_row_model_10k: 500 groups, 20 matches each = 10K matches
    // -----------------------------------------------------------------------
    {
        let (fm, _cm, groups) = build_groups(500, 20);
        c.bench_function("rebuild_flat_row_model_10k", |b| {
            b.iter(|| FlatRowModel::rebuild(&fm, &groups, Some(500)))
        });
    }

    // -----------------------------------------------------------------------
    // bench_first_visible_row_binary_search: 10K rows, binary search lookup
    // -----------------------------------------------------------------------
    {
        let (fm, _cm, groups) = build_groups(500, 20);
        let model = FlatRowModel::rebuild(&fm, &groups, None);
        // Pick a viewport position roughly in the middle
        let viewport_top = model.total_height / 2.0;

        c.bench_function("first_visible_row_binary_search", |b| {
            b.iter(|| model.first_visible_row(viewport_top))
        });
    }

    // -----------------------------------------------------------------------
    // bench_recompute_groups_incremental_50: add 50 matches to existing 1000
    // -----------------------------------------------------------------------
    {
        let search_root = Path::new("/project");

        // Pre-build 1000 existing matches across 100 groups
        let mut base_matches: Vec<ContentMatch> = Vec::new();
        for g in 0..100 {
            let path = format!("/project/src/module_{}/handler.rs", g);
            for line in 0..10 {
                base_matches.push(make_cm(&path, (line as u32) + 1));
            }
        }

        c.bench_function("recompute_groups_incremental_50", |b| {
            b.iter_batched(
                || {
                    // Setup: group existing 1000 matches
                    let mut groups = Vec::new();
                    let mut map = HashMap::new();
                    let mut last = 0;
                    recompute_groups_incremental(
                        &base_matches,
                        &mut groups,
                        &mut map,
                        &mut last,
                        search_root,
                    );

                    // Prepare 50 new matches (mix of existing and new groups)
                    let mut extended = base_matches.clone();
                    for i in 0..50 {
                        let path = if i < 30 {
                            // 30 into existing groups
                            format!("/project/src/module_{}/handler.rs", i % 100)
                        } else {
                            // 20 into new groups
                            format!("/project/src/new_module_{}/handler.rs", i)
                        };
                        extended.push(make_cm(&path, 999));
                    }

                    (extended, groups, map, last)
                },
                |(extended, mut groups, mut map, mut last)| {
                    recompute_groups_incremental(
                        &extended,
                        &mut groups,
                        &mut map,
                        &mut last,
                        search_root,
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(benches, bench_grouped_scroll);
criterion_main!(benches);
