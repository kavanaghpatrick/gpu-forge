//! Tab-complete popup for the SQL editor.
//!
//! Provides autocomplete suggestions for:
//! - SQL keywords (SELECT, FROM, WHERE, etc.)
//! - Table names from the data catalog
//! - Column names with type info and cardinality
//! - SQL functions (COUNT, SUM, AVG, etc.)
//!
//! Supports fuzzy matching (case-insensitive prefix + subsequence).

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

/// Describes the kind of autocomplete item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionKind {
    /// SQL keyword (SELECT, FROM, WHERE, etc.)
    Keyword,
    /// SQL function (COUNT, SUM, etc.)
    Function,
    /// Table name from catalog.
    Table,
    /// Column name with type and cardinality info.
    Column {
        /// Column data type (INT64, FLOAT64, VARCHAR, etc.)
        data_type: String,
        /// Number of distinct values (0 = unknown).
        distinct_count: usize,
    },
}

/// A single autocomplete suggestion.
#[derive(Debug, Clone)]
pub struct CompletionItem {
    /// The completion text to insert.
    pub text: String,
    /// Kind of completion (for display and sorting).
    pub kind: CompletionKind,
    /// Display label (may include type info).
    pub label: String,
}

impl CompletionItem {
    /// Create a keyword completion.
    pub fn keyword(kw: &str) -> Self {
        Self {
            text: kw.to_string(),
            kind: CompletionKind::Keyword,
            label: format!("{} [keyword]", kw),
        }
    }

    /// Create a function completion.
    pub fn function(name: &str) -> Self {
        Self {
            text: format!("{}(", name),
            kind: CompletionKind::Function,
            label: format!("{}() [function]", name),
        }
    }

    /// Create a table completion.
    pub fn table(name: &str) -> Self {
        Self {
            text: name.to_string(),
            kind: CompletionKind::Table,
            label: format!("{} [table]", name),
        }
    }

    /// Create a column completion with type and cardinality info.
    pub fn column(name: &str, data_type: &str, distinct_count: usize) -> Self {
        let label = if distinct_count > 0 {
            format!("{} [{}, {} distinct]", name, data_type, distinct_count)
        } else {
            format!("{} [{}]", name, data_type)
        };
        Self {
            text: name.to_string(),
            kind: CompletionKind::Column {
                data_type: data_type.to_string(),
                distinct_count,
            },
            label,
        }
    }
}

/// SQL keywords available for completion.
const COMPLETION_KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
    "AS", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON",
    "GROUP", "BY", "ORDER", "ASC", "DESC", "HAVING", "LIMIT", "OFFSET",
    "DISTINCT", "BETWEEN", "LIKE", "EXISTS", "UNION", "ALL",
    "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "TRUE", "FALSE",
    "WITH", "DESCRIBE",
];

/// SQL functions available for completion.
const COMPLETION_FUNCTIONS: &[&str] = &[
    "COUNT", "SUM", "AVG", "MIN", "MAX",
    "COALESCE", "NULLIF", "UPPER", "LOWER", "LENGTH", "TRIM",
];

/// Autocomplete state managing suggestions and selection.
#[derive(Debug, Clone)]
pub struct AutocompleteState {
    /// Whether the popup is currently visible.
    pub visible: bool,
    /// Current list of filtered suggestions.
    pub items: Vec<CompletionItem>,
    /// Selected item index within `items`.
    pub selected: usize,
    /// The fragment being completed (for display).
    pub fragment: String,
    /// Available table names from catalog.
    tables: Vec<String>,
    /// Available columns: (table_name, column_name, type_name, distinct_count).
    columns: Vec<(String, String, String, usize)>,
}

impl Default for AutocompleteState {
    fn default() -> Self {
        Self::new()
    }
}

impl AutocompleteState {
    /// Create empty autocomplete state.
    pub fn new() -> Self {
        Self {
            visible: false,
            items: Vec::new(),
            selected: 0,
            fragment: String::new(),
            tables: Vec::new(),
            columns: Vec::new(),
        }
    }

    /// Register table names from the data catalog.
    pub fn set_tables(&mut self, tables: Vec<String>) {
        self.tables = tables;
    }

    /// Register column metadata for autocomplete.
    /// Each entry: (table_name, column_name, type_display, distinct_count).
    pub fn set_columns(&mut self, columns: Vec<(String, String, String, usize)>) {
        self.columns = columns;
    }

    /// Update suggestions based on the given text fragment.
    /// Returns true if there are suggestions to show.
    pub fn update(&mut self, fragment: &str) -> bool {
        self.fragment = fragment.to_string();
        self.items.clear();
        self.selected = 0;

        if fragment.is_empty() {
            self.visible = false;
            return false;
        }

        let lower = fragment.to_lowercase();

        // Collect matching keywords
        for kw in COMPLETION_KEYWORDS {
            if fuzzy_match(&lower, &kw.to_lowercase()) {
                self.items.push(CompletionItem::keyword(kw));
            }
        }

        // Collect matching functions
        for func in COMPLETION_FUNCTIONS {
            if fuzzy_match(&lower, &func.to_lowercase()) {
                self.items.push(CompletionItem::function(func));
            }
        }

        // Collect matching table names
        for table in &self.tables {
            if fuzzy_match(&lower, &table.to_lowercase()) {
                self.items.push(CompletionItem::table(table));
            }
        }

        // Collect matching columns (from all registered tables)
        for (_, col_name, type_name, distinct) in &self.columns {
            if fuzzy_match(&lower, &col_name.to_lowercase()) {
                self.items
                    .push(CompletionItem::column(col_name, type_name, *distinct));
            }
        }

        // Sort: exact prefix first, then by kind (columns first for data exploration),
        // then alphabetical
        self.items.sort_by(|a, b| {
            let a_prefix = a.text.to_lowercase().starts_with(&lower);
            let b_prefix = b.text.to_lowercase().starts_with(&lower);
            b_prefix
                .cmp(&a_prefix)
                .then_with(|| kind_order(&a.kind).cmp(&kind_order(&b.kind)))
                .then_with(|| a.text.cmp(&b.text))
        });

        // Limit to top 15 results
        self.items.truncate(15);

        self.visible = !self.items.is_empty();
        self.visible
    }

    /// Move selection down.
    pub fn select_next(&mut self) {
        if !self.items.is_empty() {
            self.selected = (self.selected + 1) % self.items.len();
        }
    }

    /// Move selection up.
    pub fn select_prev(&mut self) {
        if !self.items.is_empty() {
            self.selected = if self.selected == 0 {
                self.items.len() - 1
            } else {
                self.selected - 1
            };
        }
    }

    /// Get the currently selected completion item.
    pub fn selected_item(&self) -> Option<&CompletionItem> {
        self.items.get(self.selected)
    }

    /// Accept the currently selected completion.
    /// Returns the text to insert and clears the popup.
    pub fn accept(&mut self) -> Option<String> {
        let text = self.selected_item().map(|item| item.text.clone());
        self.dismiss();
        text
    }

    /// Dismiss the autocomplete popup.
    pub fn dismiss(&mut self) {
        self.visible = false;
        self.items.clear();
        self.selected = 0;
        self.fragment.clear();
    }

    /// Render the autocomplete popup as a floating widget.
    /// `anchor` is the Rect of the editor area; popup appears near the cursor.
    pub fn render(&self, f: &mut Frame, anchor: Rect, cursor_row: u16, cursor_col: u16) {
        if !self.visible || self.items.is_empty() {
            return;
        }

        let popup_height = (self.items.len() as u16 + 2).min(12); // +2 for borders
        let popup_width = self
            .items
            .iter()
            .map(|item| item.label.len() as u16 + 4) // padding
            .max()
            .unwrap_or(20)
            .min(60)
            .max(20);

        // Position popup below the cursor line, within the editor area
        let popup_x = (anchor.x + cursor_col + 1).min(anchor.x + anchor.width.saturating_sub(popup_width));
        let popup_y = anchor.y + cursor_row + 2; // below cursor line (+1 for border)

        // Ensure popup doesn't go off screen
        let screen_h = f.area().height;
        let popup_y = if popup_y + popup_height > screen_h {
            // Place above cursor if not enough room below
            anchor.y + cursor_row.saturating_sub(popup_height)
        } else {
            popup_y
        };

        let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

        // Build lines with selection highlight
        let lines: Vec<Line> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let is_selected = i == self.selected;
                let (icon, icon_color) = kind_icon(&item.kind);

                let mut spans = vec![
                    Span::styled(
                        format!(" {} ", icon),
                        Style::default().fg(icon_color),
                    ),
                ];

                let text_style = if is_selected {
                    Style::default()
                        .fg(Color::Rgb(255, 255, 255))
                        .bg(Color::Rgb(60, 60, 100))
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::Rgb(200, 200, 220))
                };

                // Split label into name + info parts for coloring
                if let Some(bracket_pos) = item.label.find('[') {
                    let name_part = &item.label[..bracket_pos];
                    let info_part = &item.label[bracket_pos..];
                    spans.push(Span::styled(name_part.to_string(), text_style));
                    let info_style = if is_selected {
                        Style::default()
                            .fg(Color::Rgb(180, 180, 200))
                            .bg(Color::Rgb(60, 60, 100))
                    } else {
                        Style::default().fg(Color::Rgb(120, 120, 150))
                    };
                    spans.push(Span::styled(info_part.to_string(), info_style));
                } else {
                    spans.push(Span::styled(item.label.clone(), text_style));
                }

                Line::from(spans)
            })
            .collect();

        let block = Block::default()
            .title(Span::styled(
                " Completions ",
                Style::default()
                    .fg(Color::Rgb(180, 180, 255))
                    .add_modifier(Modifier::BOLD),
            ))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Rgb(80, 80, 140)));

        // Clear the area behind the popup, then render
        f.render_widget(Clear, popup_area);
        f.render_widget(Paragraph::new(lines).block(block), popup_area);
    }
}

/// Fuzzy match: first tries prefix match, then subsequence match.
/// Both query and candidate should be lowercase.
fn fuzzy_match(query: &str, candidate: &str) -> bool {
    if candidate.starts_with(query) {
        return true;
    }
    // Subsequence match: every char of query appears in order in candidate
    let mut candidate_chars = candidate.chars();
    for qch in query.chars() {
        loop {
            match candidate_chars.next() {
                Some(cch) if cch == qch => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

/// Sort order for completion kinds (lower = higher priority).
fn kind_order(kind: &CompletionKind) -> u8 {
    match kind {
        CompletionKind::Column { .. } => 0, // columns first (most useful for data)
        CompletionKind::Table => 1,
        CompletionKind::Function => 2,
        CompletionKind::Keyword => 3,
    }
}

/// Get a display icon and color for a completion kind.
fn kind_icon(kind: &CompletionKind) -> (&'static str, Color) {
    match kind {
        CompletionKind::Keyword => ("K", Color::Rgb(100, 140, 255)),    // blue
        CompletionKind::Function => ("f", Color::Rgb(220, 120, 255)),   // magenta
        CompletionKind::Table => ("T", Color::Rgb(80, 220, 120)),       // green
        CompletionKind::Column { .. } => ("C", Color::Rgb(240, 220, 80)), // yellow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_match_prefix() {
        assert!(fuzzy_match("sel", "select"));
        assert!(fuzzy_match("from", "from"));
        assert!(!fuzzy_match("xyz", "select"));
    }

    #[test]
    fn test_fuzzy_match_subsequence() {
        assert!(fuzzy_match("slct", "select")); // s-l-c-t in s-e-l-e-c-t
        assert!(fuzzy_match("cnt", "count"));   // c-n-t in c-o-u-n-t
        assert!(!fuzzy_match("zz", "select"));
    }

    #[test]
    fn test_fuzzy_match_empty() {
        assert!(fuzzy_match("", "anything")); // empty query matches everything
    }

    #[test]
    fn test_completion_item_keyword() {
        let item = CompletionItem::keyword("SELECT");
        assert_eq!(item.text, "SELECT");
        assert_eq!(item.kind, CompletionKind::Keyword);
        assert!(item.label.contains("[keyword]"));
    }

    #[test]
    fn test_completion_item_function() {
        let item = CompletionItem::function("COUNT");
        assert_eq!(item.text, "COUNT(");
        assert_eq!(item.kind, CompletionKind::Function);
        assert!(item.label.contains("[function]"));
    }

    #[test]
    fn test_completion_item_table() {
        let item = CompletionItem::table("sales");
        assert_eq!(item.text, "sales");
        assert_eq!(item.kind, CompletionKind::Table);
        assert!(item.label.contains("[table]"));
    }

    #[test]
    fn test_completion_item_column_with_cardinality() {
        let item = CompletionItem::column("region", "VARCHAR", 8);
        assert_eq!(item.text, "region");
        assert!(item.label.contains("VARCHAR"));
        assert!(item.label.contains("8 distinct"));
        match &item.kind {
            CompletionKind::Column { data_type, distinct_count } => {
                assert_eq!(data_type, "VARCHAR");
                assert_eq!(*distinct_count, 8);
            }
            _ => panic!("expected Column kind"),
        }
    }

    #[test]
    fn test_completion_item_column_no_cardinality() {
        let item = CompletionItem::column("amount", "FLOAT64", 0);
        assert!(item.label.contains("FLOAT64"));
        assert!(!item.label.contains("distinct"));
    }

    #[test]
    fn test_autocomplete_update_keywords() {
        let mut ac = AutocompleteState::new();
        let found = ac.update("sel");
        assert!(found);
        assert!(ac.visible);
        // Should find SELECT
        assert!(ac.items.iter().any(|i| i.text == "SELECT"));
    }

    #[test]
    fn test_autocomplete_update_functions() {
        let mut ac = AutocompleteState::new();
        let found = ac.update("cou");
        assert!(found);
        assert!(ac.items.iter().any(|i| i.text == "COUNT("));
    }

    #[test]
    fn test_autocomplete_update_tables() {
        let mut ac = AutocompleteState::new();
        ac.set_tables(vec!["sales".into(), "users".into(), "orders".into()]);
        let found = ac.update("sa");
        assert!(found);
        assert!(ac.items.iter().any(|i| i.text == "sales"));
    }

    #[test]
    fn test_autocomplete_update_columns() {
        let mut ac = AutocompleteState::new();
        ac.set_columns(vec![
            ("sales".into(), "region".into(), "VARCHAR".into(), 8),
            ("sales".into(), "amount".into(), "FLOAT64".into(), 0),
        ]);
        let found = ac.update("re");
        assert!(found);
        let region_item = ac.items.iter().find(|i| i.text == "region").unwrap();
        assert!(region_item.label.contains("VARCHAR"));
        assert!(region_item.label.contains("8 distinct"));
    }

    #[test]
    fn test_autocomplete_select_region_pattern() {
        // Exact scenario from spec: type "re" + Tab -> shows "region [VARCHAR, 8 distinct]"
        let mut ac = AutocompleteState::new();
        ac.set_columns(vec![
            ("sales".into(), "region".into(), "VARCHAR".into(), 8),
            ("sales".into(), "revenue".into(), "FLOAT64".into(), 0),
        ]);
        let found = ac.update("re");
        assert!(found);

        // region should be in results
        let region = ac.items.iter().find(|i| i.text == "region").unwrap();
        assert_eq!(region.label, "region [VARCHAR, 8 distinct]");

        // revenue should also match
        assert!(ac.items.iter().any(|i| i.text == "revenue"));
    }

    #[test]
    fn test_autocomplete_empty_fragment() {
        let mut ac = AutocompleteState::new();
        let found = ac.update("");
        assert!(!found);
        assert!(!ac.visible);
    }

    #[test]
    fn test_autocomplete_dismiss() {
        let mut ac = AutocompleteState::new();
        ac.update("sel");
        assert!(ac.visible);
        ac.dismiss();
        assert!(!ac.visible);
        assert!(ac.items.is_empty());
    }

    #[test]
    fn test_autocomplete_navigation() {
        let mut ac = AutocompleteState::new();
        ac.set_tables(vec!["sales".into(), "schema".into()]);
        ac.update("s");

        let initial_selected = ac.selected;
        assert_eq!(initial_selected, 0);

        ac.select_next();
        assert_eq!(ac.selected, 1);

        ac.select_prev();
        assert_eq!(ac.selected, 0);

        // Wrap around
        ac.select_prev();
        assert_eq!(ac.selected, ac.items.len() - 1);
    }

    #[test]
    fn test_autocomplete_accept() {
        let mut ac = AutocompleteState::new();
        ac.set_tables(vec!["sales".into()]);
        ac.update("sa");
        assert!(ac.visible);

        // Find the "sales" item and select it
        let idx = ac.items.iter().position(|i| i.text == "sales");
        if let Some(idx) = idx {
            ac.selected = idx;
        }

        let text = ac.accept();
        assert_eq!(text, Some("sales".into()));
        assert!(!ac.visible);
    }

    #[test]
    fn test_autocomplete_sort_order() {
        // Columns should sort before keywords for data-oriented UX
        let mut ac = AutocompleteState::new();
        ac.set_columns(vec![
            ("t".into(), "sum_val".into(), "INT64".into(), 0),
        ]);
        ac.update("sum");

        // Should have both SUM function/keyword and sum_val column
        let col_idx = ac.items.iter().position(|i| i.text == "sum_val");
        let kw_idx = ac.items.iter().position(|i| i.text == "SUM" || i.text == "SUM(");

        // Column should come before keyword (lower index)
        if let (Some(ci), Some(ki)) = (col_idx, kw_idx) {
            assert!(ci < ki, "column should sort before keyword");
        }
    }

    #[test]
    fn test_kind_icon() {
        let (icon, _) = kind_icon(&CompletionKind::Keyword);
        assert_eq!(icon, "K");
        let (icon, _) = kind_icon(&CompletionKind::Function);
        assert_eq!(icon, "f");
        let (icon, _) = kind_icon(&CompletionKind::Table);
        assert_eq!(icon, "T");
        let (icon, _) = kind_icon(&CompletionKind::Column { data_type: "INT64".into(), distinct_count: 0 });
        assert_eq!(icon, "C");
    }
}
