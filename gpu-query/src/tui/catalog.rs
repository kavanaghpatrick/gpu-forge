//! Data catalog tree view widget for the TUI dashboard.
//!
//! Displays a hierarchical tree: directory > table (with format badge) > columns (with types).
//! Navigation via arrow keys or j/k, Enter to expand/collapse or trigger DESCRIBE.

use crate::io::catalog::TableEntry;
use crate::io::format_detect::FileFormat;
use crate::storage::schema::DataType;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// Column information for display in the catalog tree.
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    /// Column name.
    pub name: String,
    /// Data type for display.
    pub type_name: String,
}

/// A catalog entry combining table metadata with display-relevant info.
#[derive(Debug, Clone)]
pub struct CatalogEntry {
    /// Table name (file stem).
    pub name: String,
    /// File format.
    pub format: FileFormat,
    /// Number of rows (if known).
    pub row_count: Option<u64>,
    /// Column definitions for tree expansion.
    pub columns: Vec<ColumnInfo>,
}

impl CatalogEntry {
    /// Create a CatalogEntry from an io::catalog::TableEntry.
    pub fn from_table_entry(entry: &TableEntry) -> Self {
        let columns = if let Some(ref meta) = entry.csv_metadata {
            meta.column_names
                .iter()
                .map(|name| ColumnInfo {
                    name: name.clone(),
                    type_name: "VARCHAR".into(), // default before schema inference
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            name: entry.name.clone(),
            format: entry.format,
            row_count: None,
            columns,
        }
    }

    /// Create a CatalogEntry with typed columns from a TableEntry and column types.
    pub fn from_table_entry_with_types(
        entry: &TableEntry,
        column_types: &[(String, DataType)],
    ) -> Self {
        let columns = column_types
            .iter()
            .map(|(name, dtype)| ColumnInfo {
                name: name.clone(),
                type_name: format_data_type(*dtype),
            })
            .collect();

        Self {
            name: entry.name.clone(),
            format: entry.format,
            row_count: None,
            columns,
        }
    }
}

/// Format a DataType for display in the catalog tree.
fn format_data_type(dt: DataType) -> String {
    match dt {
        DataType::Int64 => "INT64".into(),
        DataType::Float64 => "FLOAT64".into(),
        DataType::Varchar => "VARCHAR".into(),
        DataType::Bool => "BOOL".into(),
        DataType::Date => "DATE".into(),
    }
}

/// State for the catalog tree view navigation.
#[derive(Debug, Clone)]
pub struct CatalogState {
    /// All catalog entries (tables).
    pub entries: Vec<CatalogEntry>,
    /// Currently selected flat index in the visible tree.
    pub selected: usize,
    /// Set of table indices that are expanded (showing columns).
    pub expanded: Vec<bool>,
    /// Vertical scroll offset for long catalogs.
    pub scroll_offset: usize,
    /// Directory name shown as header.
    pub directory_name: String,
    /// Whether a DESCRIBE was triggered (table name).
    pub describe_requested: Option<String>,
}

impl CatalogState {
    /// Create a new empty catalog state.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            selected: 0,
            expanded: Vec::new(),
            scroll_offset: 0,
            directory_name: String::new(),
            describe_requested: None,
        }
    }

    /// Load catalog entries and reset selection.
    pub fn load(&mut self, entries: Vec<CatalogEntry>, directory_name: String) {
        self.expanded = vec![false; entries.len()];
        self.entries = entries;
        self.directory_name = directory_name;
        self.selected = 0;
        self.scroll_offset = 0;
        self.describe_requested = None;
    }

    /// Total number of visible rows in the tree (tables + expanded columns).
    pub fn visible_row_count(&self) -> usize {
        let mut count = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            count += 1; // table row
            if self.is_expanded(i) {
                count += entry.columns.len(); // column rows
            }
        }
        count
    }

    /// Check if a table entry is expanded.
    pub fn is_expanded(&self, table_index: usize) -> bool {
        self.expanded.get(table_index).copied().unwrap_or(false)
    }

    /// Toggle expansion of a table entry.
    pub fn toggle_expand(&mut self, table_index: usize) {
        if table_index < self.expanded.len() {
            self.expanded[table_index] = !self.expanded[table_index];
        }
    }

    /// Move selection up by one visible row.
    pub fn move_up(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
        self.adjust_scroll();
    }

    /// Move selection down by one visible row.
    pub fn move_down(&mut self) {
        let max = self.visible_row_count();
        if max > 0 && self.selected < max - 1 {
            self.selected += 1;
        }
        self.adjust_scroll();
    }

    /// Handle Enter key: toggle expand on table rows, trigger DESCRIBE on table rows.
    pub fn handle_enter(&mut self) {
        self.describe_requested = None;

        if let Some((table_idx, is_column, _col_idx)) = self.resolve_selected() {
            if is_column {
                // Enter on a column row: trigger DESCRIBE for the parent table
                self.describe_requested = Some(self.entries[table_idx].name.clone());
            } else {
                // Enter on a table row: toggle expand + trigger DESCRIBE
                self.toggle_expand(table_idx);
                self.describe_requested = Some(self.entries[table_idx].name.clone());
            }
        }
    }

    /// Resolve the flat selected index to (table_index, is_column_row, column_index).
    /// Returns None if selected is out of range.
    pub fn resolve_selected(&self) -> Option<(usize, bool, usize)> {
        let mut pos = 0;
        for (i, entry) in self.entries.iter().enumerate() {
            if pos == self.selected {
                return Some((i, false, 0));
            }
            pos += 1;
            if self.is_expanded(i) {
                for col_idx in 0..entry.columns.len() {
                    if pos == self.selected {
                        return Some((i, true, col_idx));
                    }
                    pos += 1;
                }
            }
        }
        None
    }

    /// Adjust scroll offset to keep selected row visible.
    fn adjust_scroll(&mut self) {
        // Allow 2 lines of buffer from the edges
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
        // We don't know viewport height here; render will clamp
    }

    /// Adjust scroll offset for a given viewport height.
    pub fn adjust_scroll_for_viewport(&mut self, viewport_height: usize) {
        if viewport_height == 0 {
            return;
        }
        if self.selected >= self.scroll_offset + viewport_height {
            self.scroll_offset = self.selected - viewport_height + 1;
        }
        if self.selected < self.scroll_offset {
            self.scroll_offset = self.selected;
        }
    }
}

impl Default for CatalogState {
    fn default() -> Self {
        Self::new()
    }
}

/// Color for format badges.
fn format_badge_color(format: FileFormat) -> Color {
    match format {
        FileFormat::Csv => Color::Rgb(80, 200, 80),       // green
        FileFormat::Parquet => Color::Rgb(80, 120, 255),   // blue
        FileFormat::Json => Color::Rgb(255, 200, 60),      // yellow
        FileFormat::Unknown => Color::Rgb(120, 120, 120),  // gray
    }
}

/// Format badge text for a file format.
fn format_badge(format: FileFormat) -> &'static str {
    match format {
        FileFormat::Csv => "[CSV]",
        FileFormat::Parquet => "[PQ]",
        FileFormat::Json => "[JSON]",
        FileFormat::Unknown => "[?]",
    }
}

/// Build the visible tree lines for rendering.
pub fn build_tree_lines<'a>(
    state: &CatalogState,
    text_color: Color,
    muted_color: Color,
    accent_color: Color,
    selection_style: Style,
) -> Vec<Line<'a>> {
    let mut lines: Vec<Line> = Vec::new();
    let mut flat_idx: usize = 0;

    // Directory header
    if !state.directory_name.is_empty() {
        lines.push(Line::from(Span::styled(
            format!(" {}/", state.directory_name),
            Style::default()
                .fg(accent_color)
                .add_modifier(Modifier::BOLD),
        )));
    }

    for (i, entry) in state.entries.iter().enumerate() {
        let is_selected = flat_idx == state.selected;
        let is_exp = state.is_expanded(i);

        // Tree prefix
        let tree_char = if is_exp { " v " } else { " > " };

        // Format badge
        let badge = format_badge(entry.format);
        let badge_color = format_badge_color(entry.format);

        // Row count badge (if available)
        let row_badge = entry
            .row_count
            .map(|c| format!(" ({} rows)", c))
            .unwrap_or_default();

        let base_style = if is_selected {
            selection_style
        } else {
            Style::default().fg(text_color)
        };

        let mut spans = vec![
            Span::styled(tree_char.to_string(), base_style),
            Span::styled(
                entry.name.clone(),
                if is_selected {
                    selection_style
                } else {
                    Style::default()
                        .fg(text_color)
                        .add_modifier(Modifier::BOLD)
                },
            ),
            Span::styled(" ", Style::default()),
            Span::styled(
                badge.to_string(),
                Style::default()
                    .fg(badge_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ];

        if !row_badge.is_empty() {
            spans.push(Span::styled(
                row_badge,
                Style::default().fg(muted_color),
            ));
        }

        lines.push(Line::from(spans));
        flat_idx += 1;

        // Expanded columns
        if is_exp {
            for (col_idx, col) in entry.columns.iter().enumerate() {
                let col_selected = flat_idx == state.selected;
                let type_color = match col.type_name.as_str() {
                    "INT64" => Color::Rgb(100, 180, 255),   // light blue
                    "FLOAT64" => Color::Rgb(180, 100, 255), // purple
                    "VARCHAR" => Color::Rgb(100, 255, 150),  // light green
                    "BOOL" => Color::Rgb(255, 180, 100),     // orange
                    "DATE" => Color::Rgb(255, 255, 100),     // yellow
                    _ => muted_color,
                };

                let is_last = col_idx == entry.columns.len() - 1;
                let branch = if is_last { "     L " } else { "     | " };

                lines.push(Line::from(vec![
                    Span::styled(branch.to_string(), Style::default().fg(muted_color)),
                    Span::styled(
                        col.name.clone(),
                        if col_selected {
                            selection_style
                        } else {
                            Style::default().fg(text_color)
                        },
                    ),
                    Span::styled(" : ", Style::default().fg(muted_color)),
                    Span::styled(
                        col.type_name.clone(),
                        if col_selected {
                            selection_style
                        } else {
                            Style::default()
                                .fg(type_color)
                                .add_modifier(Modifier::DIM)
                        },
                    ),
                ]));
                flat_idx += 1;
            }
        }
    }

    lines
}

/// Render the catalog tree view in the given area.
pub fn render_catalog_tree(
    f: &mut Frame,
    area: Rect,
    state: &mut CatalogState,
    is_focused: bool,
    border_style: Style,
    focus_border_style: Style,
    text_color: Color,
    muted_color: Color,
    accent_color: Color,
    selection_style: Style,
) {
    let bstyle = if is_focused {
        focus_border_style
    } else {
        border_style
    };

    let block = Block::default()
        .title(Span::styled(
            " Catalog ",
            Style::default()
                .fg(accent_color)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(bstyle);

    // Inner area height (subtract 2 for borders)
    let inner_height = if area.height > 2 {
        (area.height - 2) as usize
    } else {
        0
    };

    // Adjust scroll for viewport
    state.adjust_scroll_for_viewport(inner_height);

    let all_lines = build_tree_lines(state, text_color, muted_color, accent_color, selection_style);

    // Apply scroll offset: skip directory header (always shown) + scroll
    let has_header = !state.directory_name.is_empty();
    let header_lines = if has_header { 1 } else { 0 };

    let visible_lines: Vec<Line> = if has_header && !all_lines.is_empty() {
        // Always show directory header, then scroll the rest
        let mut result = vec![all_lines[0].clone()];
        let data_lines = &all_lines[header_lines..];
        let skip = state.scroll_offset;
        let take = if inner_height > header_lines {
            inner_height - header_lines
        } else {
            0
        };
        result.extend(data_lines.iter().skip(skip).take(take).cloned());
        result
    } else {
        all_lines
            .into_iter()
            .skip(state.scroll_offset)
            .take(inner_height)
            .collect()
    };

    let paragraph = Paragraph::new(visible_lines).block(block);
    f.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::format_detect::FileFormat;

    fn make_entry(name: &str, format: FileFormat, columns: Vec<(&str, &str)>) -> CatalogEntry {
        CatalogEntry {
            name: name.into(),
            format,
            row_count: None,
            columns: columns
                .into_iter()
                .map(|(n, t)| ColumnInfo {
                    name: n.into(),
                    type_name: t.into(),
                })
                .collect(),
        }
    }

    #[test]
    fn test_catalog_state_new() {
        let state = CatalogState::new();
        assert_eq!(state.selected, 0);
        assert!(state.entries.is_empty());
        assert!(state.expanded.is_empty());
        assert_eq!(state.scroll_offset, 0);
        assert!(state.describe_requested.is_none());
    }

    #[test]
    fn test_catalog_state_load() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("sales", FileFormat::Csv, vec![("id", "INT64"), ("amount", "FLOAT64")]),
            make_entry("users", FileFormat::Json, vec![("uid", "INT64")]),
        ];
        state.load(entries, "test-data".into());
        assert_eq!(state.entries.len(), 2);
        assert_eq!(state.expanded.len(), 2);
        assert!(!state.expanded[0]);
        assert!(!state.expanded[1]);
        assert_eq!(state.directory_name, "test-data");
    }

    #[test]
    fn test_visible_row_count_collapsed() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("a", FileFormat::Csv, vec![("x", "INT64"), ("y", "INT64")]),
            make_entry("b", FileFormat::Json, vec![("z", "VARCHAR")]),
        ];
        state.load(entries, "dir".into());
        // All collapsed: 2 table rows
        assert_eq!(state.visible_row_count(), 2);
    }

    #[test]
    fn test_visible_row_count_expanded() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("a", FileFormat::Csv, vec![("x", "INT64"), ("y", "INT64")]),
            make_entry("b", FileFormat::Json, vec![("z", "VARCHAR")]),
        ];
        state.load(entries, "dir".into());
        state.toggle_expand(0);
        // table a (1) + 2 columns + table b (1) = 4
        assert_eq!(state.visible_row_count(), 4);
    }

    #[test]
    fn test_navigate_up_down() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("a", FileFormat::Csv, vec![("x", "INT64")]),
            make_entry("b", FileFormat::Parquet, vec![]),
            make_entry("c", FileFormat::Json, vec![]),
        ];
        state.load(entries, "dir".into());
        assert_eq!(state.selected, 0);

        state.move_down();
        assert_eq!(state.selected, 1);

        state.move_down();
        assert_eq!(state.selected, 2);

        // At bottom, should stay
        state.move_down();
        assert_eq!(state.selected, 2);

        state.move_up();
        assert_eq!(state.selected, 1);

        state.move_up();
        assert_eq!(state.selected, 0);

        // At top, should stay
        state.move_up();
        assert_eq!(state.selected, 0);
    }

    #[test]
    fn test_navigate_through_expanded() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("a", FileFormat::Csv, vec![("x", "INT64"), ("y", "FLOAT64")]),
            make_entry("b", FileFormat::Parquet, vec![]),
        ];
        state.load(entries, "dir".into());

        // Expand table a
        state.toggle_expand(0);
        // visible: a, x, y, b (4 rows)
        assert_eq!(state.visible_row_count(), 4);

        state.move_down(); // -> x column
        assert_eq!(state.selected, 1);
        let (table_idx, is_col, col_idx) = state.resolve_selected().unwrap();
        assert_eq!(table_idx, 0);
        assert!(is_col);
        assert_eq!(col_idx, 0);

        state.move_down(); // -> y column
        assert_eq!(state.selected, 2);
        let (table_idx, is_col, col_idx) = state.resolve_selected().unwrap();
        assert_eq!(table_idx, 0);
        assert!(is_col);
        assert_eq!(col_idx, 1);

        state.move_down(); // -> table b
        assert_eq!(state.selected, 3);
        let (table_idx, is_col, _) = state.resolve_selected().unwrap();
        assert_eq!(table_idx, 1);
        assert!(!is_col);
    }

    #[test]
    fn test_handle_enter_expands_table() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("sales", FileFormat::Csv, vec![("id", "INT64"), ("amount", "FLOAT64")]),
        ];
        state.load(entries, "dir".into());

        // Enter on table row: should expand + trigger DESCRIBE
        state.handle_enter();
        assert!(state.is_expanded(0));
        assert_eq!(state.describe_requested, Some("sales".into()));

        // Enter again: should collapse
        state.handle_enter();
        assert!(!state.is_expanded(0));
    }

    #[test]
    fn test_handle_enter_on_column_triggers_describe() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("sales", FileFormat::Csv, vec![("id", "INT64"), ("amount", "FLOAT64")]),
        ];
        state.load(entries, "dir".into());
        state.toggle_expand(0);

        // Navigate to first column
        state.move_down();
        assert_eq!(state.selected, 1);

        // Enter on column: should trigger DESCRIBE for parent table
        state.handle_enter();
        assert_eq!(state.describe_requested, Some("sales".into()));
    }

    #[test]
    fn test_resolve_selected_empty() {
        let state = CatalogState::new();
        assert!(state.resolve_selected().is_none());
    }

    #[test]
    fn test_format_badge_colors() {
        // Ensure badge functions don't panic
        assert_eq!(format_badge(FileFormat::Csv), "[CSV]");
        assert_eq!(format_badge(FileFormat::Parquet), "[PQ]");
        assert_eq!(format_badge(FileFormat::Json), "[JSON]");
        assert_eq!(format_badge(FileFormat::Unknown), "[?]");

        let _ = format_badge_color(FileFormat::Csv);
        let _ = format_badge_color(FileFormat::Parquet);
        let _ = format_badge_color(FileFormat::Json);
        let _ = format_badge_color(FileFormat::Unknown);
    }

    #[test]
    fn test_build_tree_lines_basic() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("sales", FileFormat::Csv, vec![("id", "INT64")]),
        ];
        state.load(entries, "data".into());

        let lines = build_tree_lines(
            &state,
            Color::White,
            Color::Gray,
            Color::Yellow,
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        );

        // Should have: directory header + 1 table row = 2 lines
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_build_tree_lines_expanded() {
        let mut state = CatalogState::new();
        let entries = vec![
            make_entry("sales", FileFormat::Csv, vec![("id", "INT64"), ("amount", "FLOAT64")]),
        ];
        state.load(entries, "data".into());
        state.toggle_expand(0);

        let lines = build_tree_lines(
            &state,
            Color::White,
            Color::Gray,
            Color::Yellow,
            Style::default().fg(Color::Yellow),
        );

        // directory header (1) + table (1) + 2 columns = 4
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_scroll_adjustment() {
        let mut state = CatalogState::new();
        let entries: Vec<CatalogEntry> = (0..20)
            .map(|i| make_entry(&format!("table_{}", i), FileFormat::Csv, vec![]))
            .collect();
        state.load(entries, "dir".into());

        // Simulate scrolling down past viewport
        for _ in 0..15 {
            state.move_down();
        }
        state.adjust_scroll_for_viewport(10);

        // scroll_offset should adjust so selected (15) is visible
        assert!(state.scroll_offset > 0);
        assert!(state.selected >= state.scroll_offset);
        assert!(state.selected < state.scroll_offset + 10);
    }

    #[test]
    fn test_from_table_entry() {
        use crate::io::catalog::TableEntry;
        use std::path::PathBuf;

        let te = TableEntry {
            name: "test".into(),
            format: FileFormat::Csv,
            path: PathBuf::from("/tmp/test.csv"),
            csv_metadata: Some(crate::io::csv::CsvMetadata {
                column_names: vec!["a".into(), "b".into()],
                delimiter: b',',
                column_count: 2,
                file_path: PathBuf::from("/tmp/test.csv"),
            }),
        };

        let ce = CatalogEntry::from_table_entry(&te);
        assert_eq!(ce.name, "test");
        assert_eq!(ce.format, FileFormat::Csv);
        assert_eq!(ce.columns.len(), 2);
        assert_eq!(ce.columns[0].name, "a");
        assert_eq!(ce.columns[0].type_name, "VARCHAR"); // default before inference
    }

    #[test]
    fn test_from_table_entry_with_types() {
        use crate::io::catalog::TableEntry;
        use std::path::PathBuf;

        let te = TableEntry {
            name: "typed".into(),
            format: FileFormat::Parquet,
            path: PathBuf::from("/tmp/typed.parquet"),
            csv_metadata: None,
        };

        let types = vec![
            ("id".into(), DataType::Int64),
            ("amount".into(), DataType::Float64),
            ("name".into(), DataType::Varchar),
        ];

        let ce = CatalogEntry::from_table_entry_with_types(&te, &types);
        assert_eq!(ce.columns.len(), 3);
        assert_eq!(ce.columns[0].type_name, "INT64");
        assert_eq!(ce.columns[1].type_name, "FLOAT64");
        assert_eq!(ce.columns[2].type_name, "VARCHAR");
    }

    #[test]
    fn test_row_count_badge() {
        let mut entry = make_entry("big", FileFormat::Csv, vec![]);
        entry.row_count = Some(1_000_000);

        let mut state = CatalogState::new();
        state.load(vec![entry], "dir".into());

        let lines = build_tree_lines(
            &state,
            Color::White,
            Color::Gray,
            Color::Yellow,
            Style::default().fg(Color::Yellow),
        );

        // directory header + table row = 2 lines
        assert_eq!(lines.len(), 2);
        // The table row should contain row count text
        let table_line = &lines[1];
        let full_text: String = table_line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(full_text.contains("1000000 rows"), "line: {}", full_text);
    }

    #[test]
    fn test_default_impl() {
        let state = CatalogState::default();
        assert_eq!(state.selected, 0);
        assert!(state.entries.is_empty());
    }
}
