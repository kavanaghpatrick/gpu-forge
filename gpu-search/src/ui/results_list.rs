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

    // -----------------------------------------------------------------------
    // RowKind flattening tests (U-ROW-1 through U-ROW-9)
    // -----------------------------------------------------------------------

    /// Helper: build ContentGroups from content matches grouped by path.
    fn make_groups(paths_and_counts: &[(&str, usize)]) -> (Vec<ContentMatch>, Vec<ContentGroup>) {
        let mut cms = Vec::new();
        let mut groups = Vec::new();
        let mut idx = 0;
        for &(path, count) in paths_and_counts {
            let mut match_indices = Vec::new();
            for line in 0..count {
                cms.push(ContentMatch {
                    path: PathBuf::from(path),
                    line_number: (line as u32) + 1,
                    line_content: format!("match line {}", line),
                    context_before: vec![],
                    context_after: vec![],
                    match_range: 0..5,
                });
                match_indices.push(idx);
                idx += 1;
            }
            let ext = Path::new(path)
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            groups.push(ContentGroup {
                path: PathBuf::from(path),
                dir_display: "src/".to_string(),
                filename: Path::new(path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                extension: ext,
                match_indices,
            });
        }
        (cms, groups)
    }

    #[test]
    fn test_row_1_empty_model() {
        // U-ROW-1: Empty inputs produce empty model with zero total height.
        let model = FlatRowModel::rebuild(&[], &[], None);
        assert!(model.rows.is_empty());
        assert!(model.cum_heights.is_empty());
        assert_eq!(model.total_height, 0.0);
    }

    #[test]
    fn test_row_2_file_matches_only() {
        // U-ROW-2: File matches only produce SectionHeader + FileMatchRow entries.
        let fm = make_file_matches(3);
        let model = FlatRowModel::rebuild(&fm, &[], None);

        assert_eq!(model.rows.len(), 4); // 1 header + 3 rows
        assert_eq!(model.rows[0], RowKind::SectionHeader(SectionType::FileMatches));
        assert_eq!(model.rows[1], RowKind::FileMatchRow(0));
        assert_eq!(model.rows[2], RowKind::FileMatchRow(1));
        assert_eq!(model.rows[3], RowKind::FileMatchRow(2));

        let expected = SECTION_HEADER_HEIGHT + 3.0 * MATCH_ROW_COMPACT;
        assert!((model.total_height - expected).abs() < 0.01);
    }

    #[test]
    fn test_row_3_content_groups_only() {
        // U-ROW-3: Content groups produce SectionHeader + GroupHeader + MatchRow entries.
        let (_cms, groups) = make_groups(&[("src/a.rs", 2)]);
        let model = FlatRowModel::rebuild(&[], &groups, None);

        // 1 section header + 1 group header + 2 match rows = 4
        assert_eq!(model.rows.len(), 4);
        assert_eq!(model.rows[0], RowKind::SectionHeader(SectionType::ContentMatches));
        assert_eq!(model.rows[1], RowKind::GroupHeader(0));
        assert_eq!(model.rows[2], RowKind::MatchRow { group_idx: 0, local_idx: 0 });
        assert_eq!(model.rows[3], RowKind::MatchRow { group_idx: 0, local_idx: 1 });

        let expected = SECTION_HEADER_HEIGHT + GROUP_HEADER_HEIGHT + 2.0 * MATCH_ROW_COMPACT;
        assert!((model.total_height - expected).abs() < 0.01);
    }

    #[test]
    fn test_row_4_both_sections() {
        // U-ROW-4: Both file matches and content groups produce two section headers.
        let fm = make_file_matches(2);
        let (_cms, groups) = make_groups(&[("src/x.rs", 1)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // File section: 1 header + 2 FileMatchRow = 3
        // Content section: 1 header + 1 GroupHeader + 1 MatchRow = 3
        assert_eq!(model.rows.len(), 6);
        assert_eq!(model.rows[0], RowKind::SectionHeader(SectionType::FileMatches));
        assert_eq!(model.rows[3], RowKind::SectionHeader(SectionType::ContentMatches));
    }

    #[test]
    fn test_row_5_selected_match_row_expanded() {
        // U-ROW-5: Selected MatchRow gets MATCH_ROW_EXPANDED height.
        let (_cms, groups) = make_groups(&[("src/a.rs", 3)]);
        // Row indices: 0=SectionHeader, 1=GroupHeader, 2=MatchRow0, 3=MatchRow1, 4=MatchRow2
        let model = FlatRowModel::rebuild(&[], &groups, Some(3)); // select MatchRow1

        let height_row3 = model.cum_heights[3] - model.cum_heights[2];
        assert!((height_row3 - MATCH_ROW_EXPANDED).abs() < 0.01,
            "selected MatchRow should be expanded, got {}", height_row3);

        // Non-selected match rows should be compact
        let height_row2 = model.cum_heights[2] - model.cum_heights[1];
        assert!((height_row2 - MATCH_ROW_COMPACT).abs() < 0.01,
            "non-selected MatchRow should be compact, got {}", height_row2);
    }

    #[test]
    fn test_row_6_selected_file_match_not_expanded() {
        // U-ROW-6: Selected FileMatchRow always uses MATCH_ROW_COMPACT (no expanded state).
        let fm = make_file_matches(2);
        // Row indices: 0=SectionHeader, 1=FileMatchRow(0), 2=FileMatchRow(1)
        let model = FlatRowModel::rebuild(&fm, &[], Some(1));

        let height_row1 = model.cum_heights[1] - model.cum_heights[0];
        assert!((height_row1 - MATCH_ROW_COMPACT).abs() < 0.01,
            "FileMatchRow should always be compact, got {}", height_row1);
    }

    #[test]
    fn test_row_7_multiple_groups_ordering() {
        // U-ROW-7: Multiple content groups produce GroupHeader + MatchRow sequences in order.
        let (_cms, groups) = make_groups(&[("src/a.rs", 2), ("src/b.rs", 1)]);
        let model = FlatRowModel::rebuild(&[], &groups, None);

        // SectionHeader, GroupHeader(0), MatchRow{0,0}, MatchRow{0,1},
        // GroupHeader(1), MatchRow{1,0}
        assert_eq!(model.rows.len(), 6);
        assert_eq!(model.rows[1], RowKind::GroupHeader(0));
        assert_eq!(model.rows[4], RowKind::GroupHeader(1));
        assert_eq!(model.rows[5], RowKind::MatchRow { group_idx: 1, local_idx: 0 });
    }

    #[test]
    fn test_row_8_is_selectable() {
        // U-ROW-8: Only FileMatchRow and MatchRow are selectable.
        let fm = make_file_matches(1);
        let (_cms, groups) = make_groups(&[("src/a.rs", 1)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // rows: SectionHeader(FM), FileMatchRow(0), SectionHeader(CM), GroupHeader(0), MatchRow{0,0}
        assert!(!model.is_selectable(0), "SectionHeader not selectable");
        assert!(model.is_selectable(1), "FileMatchRow selectable");
        assert!(!model.is_selectable(2), "SectionHeader not selectable");
        assert!(!model.is_selectable(3), "GroupHeader not selectable");
        assert!(model.is_selectable(4), "MatchRow selectable");
        assert!(!model.is_selectable(999), "out of bounds not selectable");
    }

    #[test]
    fn test_row_9_selectable_navigation_skips_headers() {
        // U-ROW-9: next/prev_selectable_row skip headers.
        let fm = make_file_matches(1);
        let (_cms, groups) = make_groups(&[("src/a.rs", 1)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // rows: 0=SH(FM), 1=FMR(0), 2=SH(CM), 3=GH(0), 4=MR{0,0}
        // From FileMatchRow(1), next selectable should skip SH(CM) and GH(0) to reach MR(4)
        assert_eq!(model.next_selectable_row(1), Some(4));
        // From MatchRow(4), next selectable wraps to FileMatchRow(1)
        assert_eq!(model.next_selectable_row(4), Some(1));
        // From MatchRow(4), prev selectable wraps backwards skipping headers to FileMatchRow(1)
        assert_eq!(model.prev_selectable_row(4), Some(1));
        // From FileMatchRow(1), prev selectable wraps to MatchRow(4)
        assert_eq!(model.prev_selectable_row(1), Some(4));
    }

    // -----------------------------------------------------------------------
    // Prefix-sum invariant tests (U-PFX-1 through U-PFX-8)
    // -----------------------------------------------------------------------

    #[test]
    fn test_pfx_1_cum_heights_len_equals_rows_len() {
        // U-PFX-1: cum_heights always has same length as rows.
        let fm = make_file_matches(5);
        let (_cms, groups) = make_groups(&[("src/a.rs", 3), ("src/b.rs", 2)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        assert_eq!(model.cum_heights.len(), model.rows.len());
    }

    #[test]
    fn test_pfx_2_cum_heights_monotonically_increasing() {
        // U-PFX-2: cum_heights is strictly monotonically increasing.
        let fm = make_file_matches(3);
        let (_cms, groups) = make_groups(&[("src/a.rs", 2), ("src/b.rs", 4)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        for i in 1..model.cum_heights.len() {
            assert!(
                model.cum_heights[i] > model.cum_heights[i - 1],
                "cum_heights not strictly increasing at index {}: {} <= {}",
                i, model.cum_heights[i], model.cum_heights[i - 1]
            );
        }
    }

    #[test]
    fn test_pfx_3_total_height_equals_last_cum_height() {
        // U-PFX-3: total_height equals the last element of cum_heights.
        let fm = make_file_matches(2);
        let (_cms, groups) = make_groups(&[("src/a.rs", 3)]);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        assert!(!model.cum_heights.is_empty());
        assert!((model.total_height - *model.cum_heights.last().unwrap()).abs() < 0.01);
    }

    #[test]
    fn test_pfx_4_first_visible_row_at_zero() {
        // U-PFX-4: first_visible_row(0.0) returns 0.
        let fm = make_file_matches(3);
        let model = FlatRowModel::rebuild(&fm, &[], None);

        assert_eq!(model.first_visible_row(0.0), 0);
    }

    #[test]
    fn test_pfx_5_first_visible_row_past_total() {
        // U-PFX-5: first_visible_row beyond total_height returns last row index + 1.
        let fm = make_file_matches(3);
        let model = FlatRowModel::rebuild(&fm, &[], None);

        let result = model.first_visible_row(model.total_height + 100.0);
        assert_eq!(result, model.rows.len());
    }

    #[test]
    fn test_pfx_6_first_visible_row_binary_search_accuracy() {
        // U-PFX-6: first_visible_row returns correct row for midpoint viewport positions.
        let fm = make_file_matches(10);
        let model = FlatRowModel::rebuild(&fm, &[], None);

        // After SectionHeader (24px), first FileMatchRow starts
        // first_visible_row(24.0) should return row 1 (since cum_heights[0]=24.0 <= 24.0)
        assert_eq!(model.first_visible_row(SECTION_HEADER_HEIGHT), 1);

        // After SectionHeader + 1 FileMatchRow (24 + 24 = 48px)
        assert_eq!(model.first_visible_row(SECTION_HEADER_HEIGHT + MATCH_ROW_COMPACT), 2);
    }

    #[test]
    fn test_pfx_7_first_visible_row_empty_model() {
        // U-PFX-7: first_visible_row on empty model returns 0.
        let model = FlatRowModel::rebuild(&[], &[], None);
        assert_eq!(model.first_visible_row(100.0), 0);
    }

    // U-PFX-8 is broken into 3 proptest property tests below.

    // -----------------------------------------------------------------------
    // Proptest property tests (counted as part of U-PFX-8a, U-PFX-8b, U-PFX-8c)
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    /// Strategy: generate (n_file_matches, vec of group sizes, optional selected row).
    fn model_strategy() -> impl Strategy<Value = (usize, Vec<usize>, Option<usize>)> {
        (0..20usize, proptest::collection::vec(1..10usize, 0..8))
            .prop_flat_map(|(n_fm, group_sizes)| {
                // Calculate total row count to bound selected_row_idx
                let mut total_rows = 0;
                if n_fm > 0 {
                    total_rows += 1 + n_fm; // section header + file match rows
                }
                if !group_sizes.is_empty() {
                    total_rows += 1; // section header
                    for &gs in &group_sizes {
                        total_rows += 1 + gs; // group header + match rows
                    }
                }
                let sel_strategy = if total_rows > 0 {
                    proptest::option::of(0..total_rows).boxed()
                } else {
                    Just(None).boxed()
                };
                (Just(n_fm), Just(group_sizes), sel_strategy)
            })
    }

    /// Build a FlatRowModel from the strategy parameters.
    fn build_model_from_params(n_fm: usize, group_sizes: &[usize], selected: Option<usize>) -> FlatRowModel {
        let fm = make_file_matches(n_fm);
        let group_specs: Vec<(&str, usize)> = group_sizes
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                // Use leaked strings for stable references (test only)
                let s: &'static str = Box::leak(format!("src/file_{}.rs", i).into_boxed_str());
                (s, count)
            })
            .collect();
        let (_cms, groups) = make_groups(&group_specs);
        FlatRowModel::rebuild(&fm, &groups, selected)
    }

    proptest! {
        #[test]
        fn test_pfx_8a_cum_heights_monotonic(
            (n_fm, group_sizes, selected) in model_strategy()
        ) {
            // Property: cum_heights is always strictly monotonically increasing.
            let model = build_model_from_params(n_fm, &group_sizes, selected);
            for i in 1..model.cum_heights.len() {
                prop_assert!(
                    model.cum_heights[i] > model.cum_heights[i - 1],
                    "Monotonicity violated at {}: {} <= {}",
                    i, model.cum_heights[i], model.cum_heights[i - 1]
                );
            }
        }

        #[test]
        fn test_pfx_8b_partition_point_correctness(
            (n_fm, group_sizes, selected) in model_strategy()
        ) {
            // Property: for every valid row index, first_visible_row at
            // that row's top y-position returns that row's index.
            let model = build_model_from_params(n_fm, &group_sizes, selected);
            if model.rows.is_empty() {
                return Ok(());
            }
            for idx in 0..model.rows.len() {
                let row_top = if idx == 0 { 0.0 } else { model.cum_heights[idx - 1] };
                let found = model.first_visible_row(row_top);
                prop_assert!(
                    found <= idx,
                    "first_visible_row({}) = {} but expected <= {} (row_top = {})",
                    row_top, found, idx, row_top
                );
                // The row we found should start at or before row_top
                let found_top = if found == 0 { 0.0 } else { model.cum_heights[found - 1] };
                prop_assert!(
                    found_top <= row_top + 0.01,
                    "found row {} starts at {} which is after viewport_top {}",
                    found, found_top, row_top
                );
            }
        }

        #[test]
        fn test_pfx_8c_round_trip_height_row_height(
            (n_fm, group_sizes, selected) in model_strategy()
        ) {
            // Property: sum of individual row heights equals total_height.
            let model = build_model_from_params(n_fm, &group_sizes, selected);
            let sum: f32 = (0..model.rows.len())
                .map(|i| {
                    if i == 0 {
                        model.cum_heights[0]
                    } else {
                        model.cum_heights[i] - model.cum_heights[i - 1]
                    }
                })
                .sum();
            let diff = (sum - model.total_height).abs();
            prop_assert!(
                diff < 0.01,
                "Sum of row heights ({}) != total_height ({}), diff={}",
                sum, model.total_height, diff
            );
        }
    }

    // -----------------------------------------------------------------------
    // Navigation integration tests (I-NAV-1 through I-NAV-7)
    // -----------------------------------------------------------------------

    /// Helper: build groups from content matches for navigation tests.
    fn build_groups_for_nav(
        content_matches: &[ContentMatch],
    ) -> Vec<ContentGroup> {
        let mut groups = Vec::new();
        let mut map = HashMap::new();
        let mut last = 0;
        recompute_groups(content_matches, &mut groups, &mut map, &mut last, Path::new("/root"));
        groups
    }

    #[test]
    fn test_i_nav_1_next_skips_section_header() {
        // I-NAV-1: next_selectable_row from a FileMatchRow skips SectionHeader(ContentMatches)
        // and GroupHeader, landing on the first MatchRow.
        let fm = make_file_matches(2);
        let cm = vec![make_cm("src/a.rs", 10)];
        let groups = build_groups_for_nav(&cm);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(FM), FileMatchRow(0), FileMatchRow(1),
        //          SectionHeader(CM), GroupHeader(0), MatchRow{0,0}]
        assert_eq!(model.rows.len(), 6);

        // From last FileMatchRow (idx=2), next should skip SectionHeader(CM) at idx=3
        // and GroupHeader at idx=4, landing on MatchRow at idx=5
        let next = model.next_selectable_row(2);
        assert_eq!(next, Some(5), "should skip section header and group header");
    }

    #[test]
    fn test_i_nav_2_next_skips_group_header() {
        // I-NAV-2: next_selectable_row skips GroupHeader between two groups.
        let cm = vec![
            make_cm("src/a.rs", 10),
            make_cm("src/b.rs", 20),
        ];
        let groups = build_groups_for_nav(&cm);
        let fm: Vec<FileMatch> = vec![];
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(CM), GroupHeader(0), MatchRow{0,0},
        //          GroupHeader(1), MatchRow{1,0}]
        assert_eq!(model.rows.len(), 5);

        // From MatchRow{0,0} at idx=2, next should skip GroupHeader(1) at idx=3
        let next = model.next_selectable_row(2);
        assert_eq!(next, Some(4), "should skip group header between groups");
    }

    #[test]
    fn test_i_nav_3_prev_skips_headers() {
        // I-NAV-3: prev_selectable_row skips GroupHeader and SectionHeader.
        let fm = make_file_matches(1);
        let cm = vec![make_cm("src/a.rs", 10)];
        let groups = build_groups_for_nav(&cm);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(FM), FileMatchRow(0),
        //          SectionHeader(CM), GroupHeader(0), MatchRow{0,0}]
        assert_eq!(model.rows.len(), 5);

        // From MatchRow at idx=4, prev should skip GroupHeader(idx=3) and
        // SectionHeader(CM)(idx=2), landing on FileMatchRow(idx=1)
        let prev = model.prev_selectable_row(4);
        assert_eq!(prev, Some(1), "should skip group header and section header backwards");
    }

    #[test]
    fn test_i_nav_4_wrap_around() {
        // I-NAV-4: Navigation wraps from last selectable to first selectable and vice versa.
        let fm = make_file_matches(1);
        let cm = vec![make_cm("src/a.rs", 10)];
        let groups = build_groups_for_nav(&cm);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(FM), FileMatchRow(0),
        //          SectionHeader(CM), GroupHeader(0), MatchRow{0,0}]

        // Forward wrap: from MatchRow(4), next wraps to FileMatchRow(1)
        let next = model.next_selectable_row(4);
        assert_eq!(next, Some(1), "should wrap from last to first selectable");

        // Backward wrap: from FileMatchRow(1), prev wraps to MatchRow(4)
        let prev = model.prev_selectable_row(1);
        assert_eq!(prev, Some(4), "should wrap from first to last selectable");
    }

    #[test]
    fn test_i_nav_5_tab_section_jump() {
        // I-NAV-5: first_selectable_in_content_section and first_selectable_in_file_section
        // implement Tab/Shift+Tab section jumping.
        let fm = make_file_matches(2);
        let cm = vec![
            make_cm("src/a.rs", 10),
            make_cm("src/a.rs", 20),
        ];
        let groups = build_groups_for_nav(&cm);
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(FM), FileMatchRow(0), FileMatchRow(1),
        //          SectionHeader(CM), GroupHeader(0), MatchRow{0,0}, MatchRow{0,1}]

        // Tab: jump to first content match (should be MatchRow at idx=5)
        let content_first = model.first_selectable_in_content_section();
        assert_eq!(content_first, Some(5), "Tab should jump to first content match row");

        // Shift+Tab: jump to first file match (should be FileMatchRow at idx=1)
        let file_first = model.first_selectable_in_file_section();
        assert_eq!(file_first, Some(1), "Shift+Tab should jump to first file match row");
    }

    #[test]
    fn test_i_nav_6_empty_results() {
        // I-NAV-6: Navigation on empty model returns None for all operations.
        let fm: Vec<FileMatch> = vec![];
        let groups: Vec<ContentGroup> = vec![];
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        assert!(model.rows.is_empty());
        assert_eq!(model.next_selectable_row(0), None);
        assert_eq!(model.prev_selectable_row(0), None);
        assert_eq!(model.first_selectable_in_content_section(), None);
        assert_eq!(model.first_selectable_in_file_section(), None);
    }

    #[test]
    fn test_i_nav_7_only_headers_edge_case() {
        // I-NAV-7: When only one section exists, the other section's jump returns None.
        // rebuild() always pairs headers with data, so we test cross-section absence.

        // File matches only -- no content section
        let fm = make_file_matches(1);
        let groups: Vec<ContentGroup> = vec![];
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        assert_eq!(model.first_selectable_in_content_section(), None,
            "no content section -> None");

        // Content matches only -- no file section
        let fm_empty: Vec<FileMatch> = vec![];
        let cm = vec![make_cm("src/a.rs", 1)];
        let groups2 = build_groups_for_nav(&cm);
        let model2 = FlatRowModel::rebuild(&fm_empty, &groups2, None);

        assert_eq!(model2.first_selectable_in_file_section(), None,
            "no file section -> None");
    }

    // -----------------------------------------------------------------------
    // Row expansion integration tests (I-EXP-1 through I-EXP-5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_i_exp_1_selected_match_row_gets_expanded_height() {
        // I-EXP-1: Selected MatchRow gets MATCH_ROW_EXPANDED (52.0) height.
        let cm = vec![make_cm("src/a.rs", 10), make_cm("src/a.rs", 20)];
        let groups = build_groups_for_nav(&cm);
        let fm: Vec<FileMatch> = vec![];

        // Layout: [SectionHeader(CM), GroupHeader(0), MatchRow{0,0}, MatchRow{0,1}]
        // Select MatchRow{0,0} at idx=2
        let model = FlatRowModel::rebuild(&fm, &groups, Some(2));

        // Height of row 2 (selected): cum_heights[2] - cum_heights[1]
        let h_selected = model.cum_heights[2] - model.cum_heights[1];
        assert!(
            (h_selected - MATCH_ROW_EXPANDED).abs() < f32::EPSILON,
            "selected MatchRow should have expanded height (52.0), got {}",
            h_selected
        );

        // Height of row 3 (not selected): cum_heights[3] - cum_heights[2]
        let h_compact = model.cum_heights[3] - model.cum_heights[2];
        assert!(
            (h_compact - MATCH_ROW_COMPACT).abs() < f32::EPSILON,
            "unselected MatchRow should have compact height (24.0), got {}",
            h_compact
        );
    }

    #[test]
    fn test_i_exp_2_no_selection_all_compact() {
        // I-EXP-2: Without selection, all MatchRows get compact height (24.0).
        let cm = vec![
            make_cm("src/a.rs", 10),
            make_cm("src/a.rs", 20),
            make_cm("src/a.rs", 30),
        ];
        let groups = build_groups_for_nav(&cm);
        let fm: Vec<FileMatch> = vec![];
        let model = FlatRowModel::rebuild(&fm, &groups, None);

        // Layout: [SectionHeader(CM), GroupHeader(0), MatchRow{0,0}, MatchRow{0,1}, MatchRow{0,2}]
        for idx in 2..5 {
            let h = model.cum_heights[idx] - model.cum_heights[idx - 1];
            assert!(
                (h - MATCH_ROW_COMPACT).abs() < f32::EPSILON,
                "row {} should have compact height, got {}",
                idx, h
            );
        }
    }

    #[test]
    fn test_i_exp_3_changing_selection_changes_heights() {
        // I-EXP-3: Rebuilding with a different selected_row_idx changes which row is expanded.
        let cm = vec![make_cm("src/a.rs", 10), make_cm("src/a.rs", 20)];
        let groups = build_groups_for_nav(&cm);
        let fm: Vec<FileMatch> = vec![];

        // Select first MatchRow (idx=2)
        let model_a = FlatRowModel::rebuild(&fm, &groups, Some(2));
        // Select second MatchRow (idx=3)
        let model_b = FlatRowModel::rebuild(&fm, &groups, Some(3));

        // model_a: row 2 expanded, row 3 compact
        let h_a2 = model_a.cum_heights[2] - model_a.cum_heights[1];
        let h_a3 = model_a.cum_heights[3] - model_a.cum_heights[2];
        assert!((h_a2 - MATCH_ROW_EXPANDED).abs() < f32::EPSILON);
        assert!((h_a3 - MATCH_ROW_COMPACT).abs() < f32::EPSILON);

        // model_b: row 2 compact, row 3 expanded
        let h_b2 = model_b.cum_heights[2] - model_b.cum_heights[1];
        let h_b3 = model_b.cum_heights[3] - model_b.cum_heights[2];
        assert!((h_b2 - MATCH_ROW_COMPACT).abs() < f32::EPSILON);
        assert!((h_b3 - MATCH_ROW_EXPANDED).abs() < f32::EPSILON);

        // Total heights should be equal (same count of expanded rows)
        assert!(
            (model_a.total_height - model_b.total_height).abs() < f32::EPSILON,
            "total height should be same when same count of expanded rows"
        );
    }

    #[test]
    fn test_i_exp_4_total_height_with_expansion() {
        // I-EXP-4: Total height correctly accounts for expanded row.
        let cm = vec![make_cm("src/a.rs", 10)];
        let groups = build_groups_for_nav(&cm);
        let fm: Vec<FileMatch> = vec![];

        // Layout: [SectionHeader(CM)=24, GroupHeader(0)=28, MatchRow{0,0}]
        let model_compact = FlatRowModel::rebuild(&fm, &groups, None);
        let model_expanded = FlatRowModel::rebuild(&fm, &groups, Some(2));

        let expected_compact = SECTION_HEADER_HEIGHT + GROUP_HEADER_HEIGHT + MATCH_ROW_COMPACT;
        let expected_expanded = SECTION_HEADER_HEIGHT + GROUP_HEADER_HEIGHT + MATCH_ROW_EXPANDED;

        assert!(
            (model_compact.total_height - expected_compact).abs() < f32::EPSILON,
            "compact total: expected {}, got {}",
            expected_compact, model_compact.total_height
        );
        assert!(
            (model_expanded.total_height - expected_expanded).abs() < f32::EPSILON,
            "expanded total: expected {}, got {}",
            expected_expanded, model_expanded.total_height
        );

        // Difference should be exactly MATCH_ROW_EXPANDED - MATCH_ROW_COMPACT = 28.0
        let diff = model_expanded.total_height - model_compact.total_height;
        assert!(
            (diff - (MATCH_ROW_EXPANDED - MATCH_ROW_COMPACT)).abs() < f32::EPSILON,
            "height difference should be 28.0, got {}",
            diff
        );
    }

    #[test]
    fn test_i_exp_5_file_match_row_always_compact() {
        // I-EXP-5: FileMatchRow always uses MATCH_ROW_COMPACT even when selected.
        let fm = make_file_matches(2);
        let groups: Vec<ContentGroup> = vec![];

        // Layout: [SectionHeader(FM), FileMatchRow(0), FileMatchRow(1)]
        // Select FileMatchRow(0) at idx=1
        let model = FlatRowModel::rebuild(&fm, &groups, Some(1));

        let h_selected = model.cum_heights[1] - model.cum_heights[0];
        assert!(
            (h_selected - MATCH_ROW_COMPACT).abs() < f32::EPSILON,
            "FileMatchRow should always be compact (24.0) even when selected, got {}",
            h_selected
        );
    }
}
