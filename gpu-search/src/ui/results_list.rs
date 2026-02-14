//! Scrollable results list with grouped virtual scroll for 10K+ items.
//!
//! Two sections:
//! - **FILENAME MATCHES (N)**: file path with highlighted query substring
//! - **CONTENT MATCHES (N)**: grouped by file with colored dots, match rows with
//!   syntax highlighting + match_range overlay
//!
//! Selected item gets a background highlight (ACCENT tint) and left accent border.
//! Virtual scroll via `egui::ScrollArea::vertical().show_viewport()` with prefix-sum
//! binary search for O(log n) first-visible-row lookup and O(1) rendering.

use std::path::{Path, PathBuf};

use eframe::egui::{self, Color32, FontId, Rect, Sense, UiBuilder, Vec2};

use crate::search::types::{ContentMatch, FileMatch};
use super::highlight::{apply_match_range_overlay, StyledSpan, SyntaxHighlighter};
use super::path_utils::abbreviate_path;
use super::theme;

/// A group of content matches sharing the same file path.
///
/// Used to display content matches grouped by file with a single header
/// showing the abbreviated path, file type dot, and match count.
/// `match_indices` stores indices into the parent `content_matches` vec.
#[derive(Debug, Clone)]
pub struct ContentGroup {
    /// Full path to the file.
    pub path: PathBuf,
    /// Abbreviated directory display string (from `abbreviate_path`).
    pub dir_display: String,
    /// Filename component (from `abbreviate_path`).
    pub filename: String,
    /// File extension (lowercase, without dot), e.g. "rs", "py".
    pub extension: String,
    /// Indices into the `content_matches` vec for matches in this file.
    pub match_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Row model types for grouped virtual scroll
// ---------------------------------------------------------------------------

/// Section type for top-level headers in the results list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    FileMatches,
    ContentMatches,
}

/// A single row in the flattened results list.
///
/// The flat row model converts the hierarchical (sections -> groups -> matches)
/// structure into a linear sequence of rows, each with a known height.
/// This enables efficient virtual scroll via prefix-sum binary search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// Top-level section header ("FILENAME MATCHES", "CONTENT MATCHES").
    SectionHeader(SectionType),
    /// A file match row, indexed into `file_matches`.
    FileMatchRow(usize),
    /// A content group header, indexed into `content_groups`.
    GroupHeader(usize),
    /// A content match row within a group.
    MatchRow {
        /// Index into `content_groups`.
        group_idx: usize,
        /// Index within the group's `match_indices` vec.
        local_idx: usize,
    },
}

/// Height constants for the grouped layout (in logical pixels).
pub const SECTION_HEADER_HEIGHT: f32 = 24.0;
pub const GROUP_HEADER_HEIGHT: f32 = 28.0;
pub const MATCH_ROW_COMPACT: f32 = 24.0;
pub const MATCH_ROW_EXPANDED: f32 = 52.0;

/// A flattened representation of the grouped results list.
///
/// Rows are laid out linearly with variable heights. `cum_heights[i]` stores
/// the cumulative height up to (and including) row `i`, enabling O(log n)
/// viewport-to-row lookup via `partition_point`.
#[derive(Debug, Clone)]
pub struct FlatRowModel {
    /// The ordered sequence of rows.
    pub rows: Vec<RowKind>,
    /// Cumulative heights: `cum_heights[i]` = sum of heights for rows 0..=i.
    pub cum_heights: Vec<f32>,
    /// Total height of all rows combined.
    pub total_height: f32,
}

impl Default for FlatRowModel {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            cum_heights: Vec::new(),
            total_height: 0.0,
        }
    }
}

impl FlatRowModel {
    /// Build the flat row model from file_matches and content_groups.
    ///
    /// `selected_row_idx` is the index in this flat model (not `selected_index`)
    /// of the currently selected row. Selected `MatchRow` entries get
    /// `MATCH_ROW_EXPANDED` height; all others get `MATCH_ROW_COMPACT`.
    pub fn rebuild(
        file_matches: &[FileMatch],
        content_groups: &[ContentGroup],
        selected_row_idx: Option<usize>,
    ) -> Self {
        let mut rows = Vec::new();
        let mut cum_heights = Vec::new();
        let mut running_height: f32 = 0.0;

        // --- File matches section ---
        if !file_matches.is_empty() {
            rows.push(RowKind::SectionHeader(SectionType::FileMatches));
            running_height += SECTION_HEADER_HEIGHT;
            cum_heights.push(running_height);

            for i in 0..file_matches.len() {
                rows.push(RowKind::FileMatchRow(i));
                // FileMatchRow always uses compact height (no expanded state)
                running_height += MATCH_ROW_COMPACT;
                cum_heights.push(running_height);
            }
        }

        // --- Content matches section ---
        if !content_groups.is_empty() {
            rows.push(RowKind::SectionHeader(SectionType::ContentMatches));
            running_height += SECTION_HEADER_HEIGHT;
            cum_heights.push(running_height);

            for (g_idx, group) in content_groups.iter().enumerate() {
                rows.push(RowKind::GroupHeader(g_idx));
                running_height += GROUP_HEADER_HEIGHT;
                cum_heights.push(running_height);

                for local_idx in 0..group.match_indices.len() {
                    rows.push(RowKind::MatchRow {
                        group_idx: g_idx,
                        local_idx,
                    });
                    let row_idx = rows.len() - 1;
                    let height = if selected_row_idx == Some(row_idx) {
                        MATCH_ROW_EXPANDED
                    } else {
                        MATCH_ROW_COMPACT
                    };
                    running_height += height;
                    cum_heights.push(running_height);
                }
            }
        }

        let total_height = running_height;
        Self {
            rows,
            cum_heights,
            total_height,
        }
    }

    /// Find the index of the first row whose bottom edge is below `viewport_top`.
    ///
    /// Uses binary search on the prefix-sum `cum_heights` array for O(log n).
    /// Returns 0 if the viewport is at or above the top.
    pub fn first_visible_row(&self, viewport_top: f32) -> usize {
        if self.cum_heights.is_empty() {
            return 0;
        }
        // partition_point returns the first index where cum_heights[i] > viewport_top
        self.cum_heights
            .partition_point(|&h| h <= viewport_top)
    }

    /// Returns true if the row at `idx` is selectable (FileMatchRow or MatchRow).
    /// Section headers and group headers are not selectable.
    pub fn is_selectable(&self, idx: usize) -> bool {
        matches!(
            self.rows.get(idx),
            Some(RowKind::FileMatchRow(_)) | Some(RowKind::MatchRow { .. })
        )
    }

    /// Find the next selectable row after `from`, wrapping around.
    /// Returns `None` if there are no selectable rows.
    pub fn next_selectable_row(&self, from: usize) -> Option<usize> {
        let len = self.rows.len();
        if len == 0 {
            return None;
        }
        for offset in 1..=len {
            let idx = (from + offset) % len;
            if self.is_selectable(idx) {
                return Some(idx);
            }
        }
        None
    }

    /// Find the previous selectable row before `from`, wrapping around.
    /// Returns `None` if there are no selectable rows.
    pub fn prev_selectable_row(&self, from: usize) -> Option<usize> {
        let len = self.rows.len();
        if len == 0 {
            return None;
        }
        for offset in 1..=len {
            let idx = (from + len - offset) % len;
            if self.is_selectable(idx) {
                return Some(idx);
            }
        }
        None
    }

    /// Find the first selectable row in the content matches section.
    ///
    /// Locates the `SectionHeader(ContentMatches)` row and returns the first
    /// selectable row after it (typically a `MatchRow`).
    /// Returns `None` if no content matches section exists.
    pub fn first_selectable_in_content_section(&self) -> Option<usize> {
        // Find the ContentMatches section header
        let header_idx = self.rows.iter().position(|r| {
            matches!(r, RowKind::SectionHeader(SectionType::ContentMatches))
        })?;
        // Find first selectable row after the header
        for idx in (header_idx + 1)..self.rows.len() {
            if self.is_selectable(idx) {
                return Some(idx);
            }
        }
        None
    }

    /// Find the first selectable row in the file matches section.
    ///
    /// Locates the `SectionHeader(FileMatches)` row and returns the first
    /// selectable row after it (typically a `FileMatchRow`).
    /// Returns `None` if no file matches section exists.
    pub fn first_selectable_in_file_section(&self) -> Option<usize> {
        // Find the FileMatches section header
        let header_idx = self.rows.iter().position(|r| {
            matches!(r, RowKind::SectionHeader(SectionType::FileMatches))
        })?;
        // Find first selectable row after the header
        for idx in (header_idx + 1)..self.rows.len() {
            if self.is_selectable(idx) {
                return Some(idx);
            }
        }
        None
    }
}

/// Width of the left accent border for selected items.
const ACCENT_BORDER_WIDTH: f32 = 3.0;
/// Background alpha for selected item highlight.
const SELECTED_BG_ALPHA: u8 = 40;

/// Scrollable results list widget.
///
/// Manages selection state and provides virtual-scrolled rendering of
/// file matches and content matches with query highlighting.
#[derive(Default)]
pub struct ResultsList {
    /// Index of the currently selected item in the combined list
    /// (file matches first, then content matches).
    pub selected_index: usize,
    /// When true, the scroll area will auto-scroll to bring the selected item into view.
    pub scroll_to_selected: bool,
}

impl ResultsList {
    /// Create a new ResultsList with default state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of items across both sections.
    fn total_items(file_matches: &[FileMatch], content_matches: &[ContentMatch]) -> usize {
        file_matches.len() + content_matches.len()
    }

    /// Move selection to the next item, wrapping around.
    pub fn select_next(&mut self, file_matches: &[FileMatch], content_matches: &[ContentMatch]) {
        let total = Self::total_items(file_matches, content_matches);
        if total == 0 {
            return;
        }
        self.selected_index = (self.selected_index + 1) % total;
        self.scroll_to_selected = true;
    }

    /// Move selection to the previous item, wrapping around.
    pub fn select_prev(&mut self, file_matches: &[FileMatch], content_matches: &[ContentMatch]) {
        let total = Self::total_items(file_matches, content_matches);
        if total == 0 {
            return;
        }
        if self.selected_index == 0 {
            self.selected_index = total - 1;
        } else {
            self.selected_index -= 1;
        }
        self.scroll_to_selected = true;
    }

    /// Get the path of the currently selected item, if any.
    pub fn get_selected<'a>(
        &self,
        file_matches: &'a [FileMatch],
        content_matches: &'a [ContentMatch],
    ) -> Option<&'a Path> {
        let total = Self::total_items(file_matches, content_matches);
        if total == 0 || self.selected_index >= total {
            return None;
        }
        if self.selected_index < file_matches.len() {
            Some(&file_matches[self.selected_index].path)
        } else {
            let ci = self.selected_index - file_matches.len();
            Some(&content_matches[ci].path)
        }
    }

    /// Get the line number of the selected item (for content matches).
    /// Returns None for file matches or if no selection.
    pub fn get_selected_line(
        &self,
        file_matches: &[FileMatch],
        content_matches: &[ContentMatch],
    ) -> Option<u32> {
        let total = Self::total_items(file_matches, content_matches);
        if total == 0 || self.selected_index >= total {
            return None;
        }
        if self.selected_index >= file_matches.len() {
            let ci = self.selected_index - file_matches.len();
            Some(content_matches[ci].line_number)
        } else {
            None
        }
    }

    /// Render the results list with grouped virtual scrolling.
    ///
    /// Displays two sections using `show_viewport()` with prefix-sum binary search:
    /// 1. FILENAME MATCHES -- paths with highlighted query substring
    /// 2. CONTENT MATCHES -- grouped by file with colored dots, syntax highlighting,
    ///    and match_range overlay
    ///
    /// Only visible rows are rendered for O(1) frame cost regardless of total count.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        file_matches: &[FileMatch],
        content_matches: &[ContentMatch],
        content_groups: &[ContentGroup],
        flat_row_model: &FlatRowModel,
        query: &str,
        search_root: &Path,
        highlighter: &mut SyntaxHighlighter,
    ) {
        if flat_row_model.rows.is_empty() {
            return;
        }

        let total_height = flat_row_model.total_height;

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_viewport(ui, |ui, viewport| {
                // Tell egui the total content height for scrollbar sizing
                ui.set_height(total_height);

                let start_row = flat_row_model.first_visible_row(viewport.min.y);
                let top_of_ui = ui.max_rect().top();

                // Compute y offset of the first visible row
                let y_start = if start_row == 0 {
                    0.0
                } else {
                    flat_row_model.cum_heights[start_row - 1]
                };

                let mut y_pos = y_start;

                for row_idx in start_row..flat_row_model.rows.len() {
                    if y_pos > viewport.max.y {
                        break;
                    }

                    let row_kind = flat_row_model.rows[row_idx];
                    let row_height = if row_idx == 0 {
                        flat_row_model.cum_heights[0]
                    } else {
                        flat_row_model.cum_heights[row_idx]
                            - flat_row_model.cum_heights[row_idx - 1]
                    };

                    let row_rect = Rect::from_min_size(
                        egui::pos2(ui.max_rect().left(), top_of_ui + y_pos),
                        Vec2::new(ui.available_width(), row_height),
                    );

                    let row_ui_rect = Rect::from_min_size(
                        row_rect.min,
                        Vec2::new(ui.max_rect().width(), row_height),
                    );

                    ui.allocate_new_ui(UiBuilder::new().max_rect(row_ui_rect), |row_ui| {
                        let is_selected = self.selected_index == row_idx;

                        if self.scroll_to_selected && is_selected {
                            row_ui.scroll_to_cursor(Some(egui::Align::Center));
                        }

                        match row_kind {
                            RowKind::SectionHeader(section_type) => {
                                render_section_header(row_ui, section_type, file_matches, content_matches);
                            }
                            RowKind::FileMatchRow(fm_idx) => {
                                if let Some(fm) = file_matches.get(fm_idx) {
                                    let response = render_file_row(row_ui, fm, query, is_selected, search_root);
                                    if response.clicked() {
                                        self.selected_index = row_idx;
                                    }
                                }
                            }
                            RowKind::GroupHeader(g_idx) => {
                                if let Some(group) = content_groups.get(g_idx) {
                                    render_group_header(row_ui, group);
                                }
                            }
                            RowKind::MatchRow { group_idx, local_idx } => {
                                if let Some(group) = content_groups.get(group_idx) {
                                    if let Some(&cm_idx) = group.match_indices.get(local_idx) {
                                        if let Some(cm) = content_matches.get(cm_idx) {
                                            let response = render_match_row(
                                                row_ui, cm, &group.extension,
                                                is_selected, highlighter,
                                            );
                                            if response.clicked() {
                                                self.selected_index = row_idx;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });

                    y_pos += row_height;
                }
            });

        // Reset scroll flag after rendering
        self.scroll_to_selected = false;
    }
}

// ---------------------------------------------------------------------------
// Row rendering functions
// ---------------------------------------------------------------------------

/// Render a section header row ("FILENAME MATCHES (N)" or "CONTENT MATCHES (N)").
fn render_section_header(
    ui: &mut egui::Ui,
    section_type: SectionType,
    file_matches: &[FileMatch],
    content_matches: &[ContentMatch],
) {
    let (label, count) = match section_type {
        SectionType::FileMatches => ("FILENAME MATCHES", file_matches.len()),
        SectionType::ContentMatches => ("CONTENT MATCHES", content_matches.len()),
    };
    let desired_size = Vec2::new(ui.available_width(), SECTION_HEADER_HEIGHT);
    let (rect, _response) = ui.allocate_exact_size(desired_size, Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);
        let text = format!("{} ({})", label, count);
        let text_pos = rect.left_top() + Vec2::new(8.0, (SECTION_HEADER_HEIGHT - 11.0) / 2.0);
        painter.text(
            text_pos,
            egui::Align2::LEFT_TOP,
            &text,
            FontId::proportional(11.0),
            theme::TEXT_MUTED,
        );
    }
}

/// Render a single file match row with abbreviated path and query highlighting.
fn render_file_row(
    ui: &mut egui::Ui,
    fm: &FileMatch,
    query: &str,
    is_selected: bool,
    search_root: &Path,
) -> egui::Response {
    let desired_size = Vec2::new(ui.available_width(), MATCH_ROW_COMPACT);
    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);

        if is_selected {
            paint_selected_bg(&painter, rect);
        }

        let (dir_part, name_part) = abbreviate_path(&fm.path, search_root);

        let text_pos = rect.left_top() + Vec2::new(
            if is_selected { ACCENT_BORDER_WIDTH + 8.0 } else { 8.0 },
            (MATCH_ROW_COMPACT - 14.0) / 2.0,
        );

        // Draw directory portion (dimmed)
        let dir_galley = painter.layout_no_wrap(
            dir_part,
            FontId::proportional(14.0),
            theme::TEXT_MUTED,
        );
        let dir_width = dir_galley.rect.width();
        painter.galley(text_pos, dir_galley, Color32::TRANSPARENT);

        // Draw filename portion with query highlighting
        let name_pos = text_pos + Vec2::new(dir_width, 0.0);
        render_highlighted_text(
            &painter,
            name_pos,
            &name_part,
            query,
            theme::TEXT_PRIMARY,
            theme::ACCENT,
            14.0,
        );
    }

    response.on_hover_text(fm.path.display().to_string())
}

/// Render a content group header: 6px colored dot + bold filename + " -- N matches" + dir path.
fn render_group_header(
    ui: &mut egui::Ui,
    group: &ContentGroup,
) -> egui::Response {
    let desired_size = Vec2::new(ui.available_width(), GROUP_HEADER_HEIGHT);
    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::hover());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);
        let left_pad = 8.0;
        let y_center = rect.center().y;

        // 6px colored dot for file type
        let dot_color = theme::extension_dot_color(&group.extension);
        let dot_center = egui::pos2(rect.left() + left_pad + 3.0, y_center);
        painter.circle_filled(dot_center, 3.0, dot_color);

        // Bold filename
        let name_x = rect.left() + left_pad + 12.0;
        let name_galley = painter.layout_no_wrap(
            group.filename.clone(),
            FontId::proportional(13.0),
            theme::TEXT_PRIMARY,
        );
        let name_width = name_galley.rect.width();
        let text_y = y_center - name_galley.rect.height() / 2.0;
        painter.galley(egui::pos2(name_x, text_y), name_galley, Color32::TRANSPARENT);

        // " -- N matches" muted
        let count_text = format!(" -- {} matches", group.match_indices.len());
        let count_galley = painter.layout_no_wrap(
            count_text,
            FontId::proportional(12.0),
            theme::TEXT_MUTED,
        );
        let count_x = name_x + name_width;
        let count_width = count_galley.rect.width();
        painter.galley(egui::pos2(count_x, text_y), count_galley, Color32::TRANSPARENT);

        // Abbreviated dir path (muted, smaller)
        if !group.dir_display.is_empty() {
            let dir_text = format!("  {}", group.dir_display);
            let dir_galley = painter.layout_no_wrap(
                dir_text,
                FontId::proportional(11.0),
                theme::TEXT_MUTED,
            );
            let dir_x = count_x + count_width;
            painter.galley(egui::pos2(dir_x, text_y + 1.0), dir_galley, Color32::TRANSPARENT);
        }
    }

    response.on_hover_text(group.path.display().to_string())
}

/// Render a content match row.
///
/// Compact (24px): `:line_number` in ACCENT + line content with syntax + match_range overlay.
/// Expanded (52px): context_before + match line + context_after.
fn render_match_row(
    ui: &mut egui::Ui,
    cm: &ContentMatch,
    extension: &str,
    is_selected: bool,
    highlighter: &mut SyntaxHighlighter,
) -> egui::Response {
    let row_height = if is_selected { MATCH_ROW_EXPANDED } else { MATCH_ROW_COMPACT };
    let desired_size = Vec2::new(ui.available_width(), row_height);
    let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

    if ui.is_rect_visible(rect) {
        let painter = ui.painter_at(rect);

        if is_selected {
            paint_selected_bg(&painter, rect);
        }

        let left_pad = if is_selected { ACCENT_BORDER_WIDTH + 8.0 } else { 8.0 };

        if is_selected {
            // Expanded: context_before + match + context_after
            let mut y_offset = 2.0;

            // Context before (up to 1 line)
            for ctx_line in cm.context_before.iter().rev().take(1).rev() {
                let line_pos = rect.left_top() + Vec2::new(left_pad + 12.0, y_offset);
                let trimmed = truncate_line(ctx_line, 100);
                painter.text(
                    line_pos,
                    egui::Align2::LEFT_TOP,
                    &trimmed,
                    egui::FontId::monospace(11.0),
                    theme::TEXT_MUTED,
                );
                y_offset += 14.0;
            }

            // Line number in ACCENT
            let line_num_text = format!(":{}", cm.line_number);
            let line_num_pos = rect.left_top() + Vec2::new(left_pad, y_offset);
            let line_num_galley = painter.layout_no_wrap(
                line_num_text,
                FontId::monospace(12.0),
                theme::ACCENT,
            );
            let line_num_width = line_num_galley.rect.width();
            painter.galley(line_num_pos, line_num_galley, Color32::TRANSPARENT);

            // Match line with syntax + match_range highlighting
            let match_pos = rect.left_top() + Vec2::new(left_pad + line_num_width + 4.0, y_offset);
            let spans = highlighter.highlight_line(&cm.line_content, extension, None);
            let spans = apply_match_range_overlay(spans, cm.match_range.clone());
            render_styled_spans(&painter, match_pos, &spans, 12.0);
            y_offset += 14.0;

            // Context after (up to 1 line)
            for ctx_line in cm.context_after.iter().take(1) {
                let line_pos = rect.left_top() + Vec2::new(left_pad + 12.0, y_offset);
                let trimmed = truncate_line(ctx_line, 100);
                painter.text(
                    line_pos,
                    egui::Align2::LEFT_TOP,
                    &trimmed,
                    egui::FontId::monospace(11.0),
                    theme::TEXT_MUTED,
                );
            }
        } else {
            // Compact: :line_number + line content
            let y_center = (MATCH_ROW_COMPACT - 12.0) / 2.0;

            // Line number in ACCENT
            let line_num_text = format!(":{}", cm.line_number);
            let line_num_pos = rect.left_top() + Vec2::new(left_pad, y_center);
            let line_num_galley = painter.layout_no_wrap(
                line_num_text,
                FontId::monospace(12.0),
                theme::ACCENT,
            );
            let line_num_width = line_num_galley.rect.width();
            painter.galley(line_num_pos, line_num_galley, Color32::TRANSPARENT);

            // Line content with syntax + match_range highlighting
            let content_pos = rect.left_top() + Vec2::new(left_pad + line_num_width + 4.0, y_center);
            let spans = highlighter.highlight_line(&cm.line_content, extension, None);
            let spans = apply_match_range_overlay(spans, cm.match_range.clone());
            render_styled_spans(&painter, content_pos, &spans, 12.0);
        }
    }

    response
}

/// Paint selected item background highlight + left accent border.
fn paint_selected_bg(painter: &egui::Painter, rect: Rect) {
    let bg_color = Color32::from_rgba_premultiplied(
        theme::ACCENT.r(),
        theme::ACCENT.g(),
        theme::ACCENT.b(),
        SELECTED_BG_ALPHA,
    );
    painter.rect_filled(rect, 2.0, bg_color);

    let border_rect = egui::Rect::from_min_size(
        rect.left_top(),
        Vec2::new(ACCENT_BORDER_WIDTH, rect.height()),
    );
    painter.rect_filled(border_rect, 1.0, theme::ACCENT);
}

/// Render a sequence of styled spans using the egui painter.
///
/// Each span carries its own foreground color, optional background, and bold flag.
/// Uses monospace font for consistent character width.
fn render_styled_spans(
    painter: &egui::Painter,
    pos: egui::Pos2,
    spans: &[StyledSpan],
    font_size: f32,
) {
    let font_id = FontId::monospace(font_size);
    let char_width = font_size * 0.6;
    let mut x_offset = 0.0;

    for span in spans {
        let fg = Color32::from_rgba_premultiplied(span.fg.0, span.fg.1, span.fg.2, span.fg.3);
        let span_width = span.text.len() as f32 * char_width;

        // Draw background if present
        if let Some(bg) = span.bg {
            let bg_color = Color32::from_rgba_premultiplied(bg.0, bg.1, bg.2, bg.3);
            let bg_rect = Rect::from_min_size(
                pos + Vec2::new(x_offset, -1.0),
                Vec2::new(span_width, font_size + 2.0),
            );
            painter.rect_filled(bg_rect, 2.0, bg_color);
        }

        painter.text(
            pos + Vec2::new(x_offset, 0.0),
            egui::Align2::LEFT_TOP,
            &span.text,
            font_id.clone(),
            fg,
        );

        x_offset += span_width;
    }
}

/// Render text with case-insensitive query substring highlighted in accent color.
fn render_highlighted_text(
    painter: &egui::Painter,
    pos: egui::Pos2,
    text: &str,
    query: &str,
    normal_color: Color32,
    highlight_color: Color32,
    font_size: f32,
) {
    if query.is_empty() {
        painter.text(
            pos,
            egui::Align2::LEFT_TOP,
            text,
            egui::FontId::monospace(font_size),
            normal_color,
        );
        return;
    }

    let font_id = egui::FontId::monospace(font_size);
    // Approximate character width for monospace font
    let char_width = font_size * 0.6;

    let text_lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    let mut x_offset = 0.0;
    let mut last_end = 0;

    // Find all case-insensitive matches
    let mut search_from = 0;
    while let Some(start) = text_lower[search_from..].find(&query_lower) {
        let abs_start = search_from + start;
        let abs_end = abs_start + query.len();

        // Render text before match in normal color
        if abs_start > last_end {
            let before = &text[last_end..abs_start];
            painter.text(
                pos + Vec2::new(x_offset, 0.0),
                egui::Align2::LEFT_TOP,
                before,
                font_id.clone(),
                normal_color,
            );
            x_offset += before.len() as f32 * char_width;
        }

        // Render match range with highlight background + accent text
        let matched = &text[abs_start..abs_end];
        let match_width = matched.len() as f32 * char_width;
        let match_rect = egui::Rect::from_min_size(
            pos + Vec2::new(x_offset, -1.0),
            Vec2::new(match_width, font_size + 2.0),
        );
        painter.rect_filled(
            match_rect,
            2.0,
            Color32::from_rgba_premultiplied(
                highlight_color.r(),
                highlight_color.g(),
                highlight_color.b(),
                50,
            ),
        );
        painter.text(
            pos + Vec2::new(x_offset, 0.0),
            egui::Align2::LEFT_TOP,
            matched,
            font_id.clone(),
            highlight_color,
        );
        x_offset += match_width;

        last_end = abs_end;
        search_from = abs_end;
    }

    // Render remaining text after last match
    if last_end < text.len() {
        let remaining = &text[last_end..];
        painter.text(
            pos + Vec2::new(x_offset, 0.0),
            egui::Align2::LEFT_TOP,
            remaining,
            font_id,
            normal_color,
        );
    }
}

/// Truncate a line to max characters, appending "..." if truncated.
fn truncate_line(line: &str, max_chars: usize) -> String {
    let trimmed = line.trim();
    if trimmed.len() > max_chars {
        format!("{}...", &trimmed[..max_chars])
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_file_matches(n: usize) -> Vec<FileMatch> {
        (0..n)
            .map(|i| FileMatch {
                path: PathBuf::from(format!("src/file_{}.rs", i)),
                score: 1.0 - (i as f32 * 0.001),
            })
            .collect()
    }

    fn make_content_matches(n: usize) -> Vec<ContentMatch> {
        (0..n)
            .map(|i| ContentMatch {
                path: PathBuf::from(format!("src/module_{}.rs", i)),
                line_number: (i as u32) + 1,
                line_content: format!("    let result = search_pattern(input);"),
                context_before: vec!["// context before".to_string()],
                context_after: vec!["// context after".to_string()],
                match_range: 18..32,
            })
            .collect()
    }

    #[test]
    fn test_select_next_wraps() {
        let mut rl = ResultsList::new();
        let fm = make_file_matches(3);
        let cm: Vec<ContentMatch> = vec![];
        rl.select_next(&fm, &cm);
        assert_eq!(rl.selected_index, 1);
        rl.select_next(&fm, &cm);
        assert_eq!(rl.selected_index, 2);
        rl.select_next(&fm, &cm);
        assert_eq!(rl.selected_index, 0); // wraps
    }

    #[test]
    fn test_select_prev_wraps() {
        let mut rl = ResultsList::new();
        let fm = make_file_matches(3);
        let cm: Vec<ContentMatch> = vec![];
        rl.select_prev(&fm, &cm);
        assert_eq!(rl.selected_index, 2); // wraps from 0 to last
        rl.select_prev(&fm, &cm);
        assert_eq!(rl.selected_index, 1);
    }

    #[test]
    fn test_select_on_empty() {
        let mut rl = ResultsList::new();
        let fm: Vec<FileMatch> = vec![];
        let cm: Vec<ContentMatch> = vec![];
        rl.select_next(&fm, &cm);
        assert_eq!(rl.selected_index, 0);
        rl.select_prev(&fm, &cm);
        assert_eq!(rl.selected_index, 0);
    }

    #[test]
    fn test_get_selected_file_match() {
        let rl = ResultsList { selected_index: 1, scroll_to_selected: false };
        let fm = make_file_matches(3);
        let cm: Vec<ContentMatch> = vec![];
        let path = rl.get_selected(&fm, &cm).unwrap();
        assert_eq!(path, Path::new("src/file_1.rs"));
    }

    #[test]
    fn test_get_selected_content_match() {
        let rl = ResultsList { selected_index: 4, scroll_to_selected: false };
        let fm = make_file_matches(3);
        let cm = make_content_matches(5);
        let path = rl.get_selected(&fm, &cm).unwrap();
        assert_eq!(path, Path::new("src/module_1.rs"));
    }

    #[test]
    fn test_get_selected_none_on_empty() {
        let rl = ResultsList::new();
        let fm: Vec<FileMatch> = vec![];
        let cm: Vec<ContentMatch> = vec![];
        assert!(rl.get_selected(&fm, &cm).is_none());
    }

    #[test]
    fn test_get_selected_line_for_content() {
        let rl = ResultsList { selected_index: 3, scroll_to_selected: false };
        let fm = make_file_matches(2);
        let cm = make_content_matches(3);
        assert_eq!(rl.get_selected_line(&fm, &cm), Some(2)); // cm[1].line_number
    }

    #[test]
    fn test_get_selected_line_for_file_match() {
        let rl = ResultsList { selected_index: 0, scroll_to_selected: false };
        let fm = make_file_matches(2);
        let cm = make_content_matches(3);
        assert_eq!(rl.get_selected_line(&fm, &cm), None);
    }

    #[test]
    fn test_navigation_across_sections() {
        let mut rl = ResultsList::new();
        let fm = make_file_matches(2);
        let cm = make_content_matches(3);
        // Navigate: 0(fm) -> 1(fm) -> 2(cm) -> 3(cm) -> 4(cm) -> 0(fm)
        for expected in [1, 2, 3, 4, 0] {
            rl.select_next(&fm, &cm);
            assert_eq!(rl.selected_index, expected);
        }
    }

    #[test]
    fn test_truncate_line_short() {
        assert_eq!(truncate_line("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_line_long() {
        let long = "a".repeat(200);
        let result = truncate_line(&long, 100);
        assert_eq!(result.len(), 103); // 100 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_virtual_scroll_10k_items() {
        // Verify data structures handle 10K items without issue
        let fm = make_file_matches(5_000);
        let cm = make_content_matches(5_000);
        let mut rl = ResultsList::new();

        assert_eq!(ResultsList::total_items(&fm, &cm), 10_000);

        // Navigate to last item
        rl.selected_index = 9_999;
        let path = rl.get_selected(&fm, &cm).unwrap();
        assert_eq!(path, Path::new("src/module_4999.rs"));

        // Navigate forward wraps to 0
        rl.select_next(&fm, &cm);
        assert_eq!(rl.selected_index, 0);
        assert!(rl.scroll_to_selected);
    }

    // -----------------------------------------------------------------------
    // ContentGroup incremental grouping tests (U-GRP-1 through U-GRP-10)
    // -----------------------------------------------------------------------
    //
    // These tests replicate the grouping logic from GpuSearchApp::recompute_groups()
    // and sort_groups_by_count() in a standalone manner, operating directly on
    // ContentGroup, HashMap<PathBuf, usize>, and last_grouped_index.

    use std::collections::HashMap;
    use crate::ui::path_utils::abbreviate_path;

    /// Standalone helper that mirrors GpuSearchApp::recompute_groups().
    /// Incrementally groups content_matches[last_grouped_index..] by file path.
    fn recompute_groups(
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

    /// Standalone helper that mirrors GpuSearchApp::sort_groups_by_count().
    fn sort_groups_by_count(
        content_groups: &mut Vec<ContentGroup>,
        group_index_map: &mut HashMap<PathBuf, usize>,
    ) {
        content_groups.sort_by(|a, b| b.match_indices.len().cmp(&a.match_indices.len()));
        group_index_map.clear();
        for (idx, group) in content_groups.iter().enumerate() {
            group_index_map.insert(group.path.clone(), idx);
        }
    }

    /// Helper to create a ContentMatch with a specific file path and line number.
    fn make_cm(path: &str, line: u32) -> ContentMatch {
        ContentMatch {
            path: PathBuf::from(path),
            line_number: line,
            line_content: format!("line content at {}", line),
            context_before: vec![],
            context_after: vec![],
            match_range: 0..4,
        }
    }

    #[test]
    fn test_group_1_single_file_single_match() {
        // U-GRP-1: A single match produces exactly one group with one match_index.
        let matches = vec![make_cm("src/foo.rs", 10)];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].match_indices, vec![0]);
        assert_eq!(groups[0].path, PathBuf::from("src/foo.rs"));
        assert_eq!(last, 1);
    }

    #[test]
    fn test_group_2_single_file_multiple_matches() {
        // U-GRP-2: Multiple matches in the same file collapse into one group.
        let matches = vec![
            make_cm("src/foo.rs", 10),
            make_cm("src/foo.rs", 20),
            make_cm("src/foo.rs", 30),
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 1, "should be one group for one file");
        assert_eq!(groups[0].match_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_group_3_multiple_files() {
        // U-GRP-3: Matches in different files produce separate groups.
        let matches = vec![
            make_cm("src/a.rs", 1),
            make_cm("src/b.rs", 2),
            make_cm("src/c.py", 3),
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].path, PathBuf::from("src/a.rs"));
        assert_eq!(groups[1].path, PathBuf::from("src/b.rs"));
        assert_eq!(groups[2].path, PathBuf::from("src/c.py"));
    }

    #[test]
    fn test_group_4_incremental_add() {
        // U-GRP-4: Calling recompute_groups twice with growing content_matches
        // processes only the new matches the second time.
        let mut matches = vec![
            make_cm("src/a.rs", 1),
            make_cm("src/b.rs", 2),
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 2);
        assert_eq!(last, 2);

        // Add more matches (simulating a new streaming batch)
        matches.push(make_cm("src/a.rs", 10)); // existing group
        matches.push(make_cm("src/c.rs", 5));  // new group
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 3, "should add one new group for src/c.rs");
        assert_eq!(groups[0].match_indices, vec![0, 2], "src/a.rs should have 2 matches");
        assert_eq!(groups[2].match_indices, vec![3], "src/c.rs should have 1 match");
        assert_eq!(last, 4);
    }

    #[test]
    fn test_group_5_sort_by_count_desc() {
        // U-GRP-5: sort_groups_by_count orders groups by match_indices.len() descending.
        let matches = vec![
            make_cm("src/few.rs", 1),
            make_cm("src/many.rs", 1),
            make_cm("src/many.rs", 2),
            make_cm("src/many.rs", 3),
            make_cm("src/mid.rs", 1),
            make_cm("src/mid.rs", 2),
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), 3);
        sort_groups_by_count(&mut groups, &mut map);

        assert_eq!(groups[0].path, PathBuf::from("src/many.rs"), "most matches first");
        assert_eq!(groups[0].match_indices.len(), 3);
        assert_eq!(groups[1].path, PathBuf::from("src/mid.rs"));
        assert_eq!(groups[1].match_indices.len(), 2);
        assert_eq!(groups[2].path, PathBuf::from("src/few.rs"));
        assert_eq!(groups[2].match_indices.len(), 1);
    }

    #[test]
    fn test_group_6_stable_sort() {
        // U-GRP-6: Groups with equal match counts preserve insertion order
        // (Rust's sort_by is stable).
        let matches = vec![
            make_cm("src/alpha.rs", 1),
            make_cm("src/beta.rs", 1),
            make_cm("src/gamma.rs", 1),
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        sort_groups_by_count(&mut groups, &mut map);

        // All have 1 match, so insertion order should be preserved
        assert_eq!(groups[0].path, PathBuf::from("src/alpha.rs"));
        assert_eq!(groups[1].path, PathBuf::from("src/beta.rs"));
        assert_eq!(groups[2].path, PathBuf::from("src/gamma.rs"));
    }

    #[test]
    fn test_group_7_empty_input() {
        // U-GRP-7: Empty content_matches produces no groups.
        let matches: Vec<ContentMatch> = vec![];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert!(groups.is_empty());
        assert!(map.is_empty());
        assert_eq!(last, 0);
    }

    #[test]
    fn test_group_8_dir_display() {
        // U-GRP-8: dir_display is populated from abbreviate_path.
        let matches = vec![make_cm("/root/src/search/foo.rs", 1)];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups[0].dir_display, "src/search/");
        assert_eq!(groups[0].filename, "foo.rs");
    }

    #[test]
    fn test_group_9_extension_extraction() {
        // U-GRP-9: Extension is extracted correctly, lowercased, without dot.
        let matches = vec![
            make_cm("file.RS", 1),
            make_cm("file.py", 2),
            make_cm("Makefile", 3), // no extension
        ];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups[0].extension, "rs", "should be lowercased");
        assert_eq!(groups[1].extension, "py");
        assert_eq!(groups[2].extension, "", "no extension -> empty string");
    }

    #[test]
    fn test_group_10_idempotent_recompute() {
        // U-GRP-10: Calling recompute_groups when last_grouped_index == len
        // is a no-op (idempotent).
        let matches = vec![make_cm("src/a.rs", 1)];
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        let groups_after_first = groups.clone();
        // Call again with no new matches
        recompute_groups(&matches, &mut groups, &mut map, &mut last, Path::new("/root"));

        assert_eq!(groups.len(), groups_after_first.len());
        assert_eq!(groups[0].match_indices, groups_after_first[0].match_indices);
    }
}
