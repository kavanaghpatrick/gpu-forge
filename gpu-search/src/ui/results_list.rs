//! Scrollable results list with virtual scroll for 10K+ items.
//!
//! Two sections:
//! - **FILENAME MATCHES (N)**: file path with highlighted query substring
//! - **CONTENT MATCHES (N)**: `file:line` header + context lines with highlighted match
//!
//! Selected item gets a background highlight (ACCENT tint) and left accent border.
//! Virtual scroll via `egui::ScrollArea::vertical().show_rows()` for O(1) rendering
//! regardless of result count.

use std::path::Path;

use eframe::egui::{self, Color32, FontId, RichText, Sense, Vec2};

use crate::search::types::{ContentMatch, FileMatch};
use super::theme;

/// Height of a single filename match row in pixels.
const FILE_ROW_HEIGHT: f32 = 28.0;
/// Height of a single content match row in pixels (header + up to 3 context lines).
const CONTENT_ROW_HEIGHT: f32 = 80.0;
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

    /// Render the results list with virtual scrolling.
    ///
    /// Displays two sections:
    /// 1. FILENAME MATCHES (N) -- paths with highlighted query substring
    /// 2. CONTENT MATCHES (N) -- file:line headers + context with highlighted match
    ///
    /// Uses `show_rows()` for each section to only render visible rows.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        file_matches: &[FileMatch],
        content_matches: &[ContentMatch],
        query: &str,
    ) {
        let total = Self::total_items(file_matches, content_matches);
        if total == 0 {
            return;
        }

        // Clamp selected index to valid range
        if self.selected_index >= total {
            self.selected_index = total.saturating_sub(1);
        }

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // --- FILENAME MATCHES section ---
                if !file_matches.is_empty() {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(format!("FILENAME MATCHES ({})", file_matches.len()))
                            .color(theme::TEXT_MUTED)
                            .size(11.0)
                            .strong(),
                    );
                    ui.add_space(2.0);

                    self.show_file_matches(ui, file_matches, query);
                }

                // --- CONTENT MATCHES section ---
                if !content_matches.is_empty() {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("CONTENT MATCHES ({})", content_matches.len()))
                            .color(theme::TEXT_MUTED)
                            .size(11.0)
                            .strong(),
                    );
                    ui.add_space(2.0);

                    self.show_content_matches(ui, file_matches.len(), content_matches, query);
                }
            });

        // Reset scroll flag after rendering
        self.scroll_to_selected = false;
    }

    /// Render file match rows with virtual scrolling via show_rows().
    fn show_file_matches(
        &mut self,
        ui: &mut egui::Ui,
        file_matches: &[FileMatch],
        query: &str,
    ) {
        let total_rows = file_matches.len();
        // Use show_rows for virtual scrolling -- only renders visible rows
        egui::ScrollArea::vertical()
            .id_salt("file_matches_scroll")
            .max_height(FILE_ROW_HEIGHT * total_rows.min(200) as f32)
            .auto_shrink([false, true])
            .show_rows(ui, FILE_ROW_HEIGHT, total_rows, |ui, row_range| {
                for row in row_range {
                    let is_selected = row == self.selected_index;
                    let fm = &file_matches[row];

                    if self.scroll_to_selected && is_selected {
                        ui.scroll_to_cursor(Some(egui::Align::Center));
                    }

                    let response = self.render_file_row(ui, fm, query, is_selected);
                    if response.clicked() {
                        self.selected_index = row;
                    }
                }
            });
    }

    /// Render content match rows with virtual scrolling via show_rows().
    fn show_content_matches(
        &mut self,
        ui: &mut egui::Ui,
        file_offset: usize,
        content_matches: &[ContentMatch],
        query: &str,
    ) {
        let total_rows = content_matches.len();
        egui::ScrollArea::vertical()
            .id_salt("content_matches_scroll")
            .max_height(CONTENT_ROW_HEIGHT * total_rows.min(200) as f32)
            .auto_shrink([false, true])
            .show_rows(ui, CONTENT_ROW_HEIGHT, total_rows, |ui, row_range| {
                for row in row_range {
                    let global_index = file_offset + row;
                    let is_selected = global_index == self.selected_index;
                    let cm = &content_matches[row];

                    if self.scroll_to_selected && is_selected {
                        ui.scroll_to_cursor(Some(egui::Align::Center));
                    }

                    let response = self.render_content_row(ui, cm, query, is_selected);
                    if response.clicked() {
                        self.selected_index = global_index;
                    }
                }
            });
    }

    /// Render a single file match row with optional selection highlight.
    fn render_file_row(
        &self,
        ui: &mut egui::Ui,
        fm: &FileMatch,
        query: &str,
        is_selected: bool,
    ) -> egui::Response {
        let desired_size = Vec2::new(ui.available_width(), FILE_ROW_HEIGHT);
        let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter_at(rect);

            // Selected item: background highlight + left accent border
            if is_selected {
                let bg_color = Color32::from_rgba_premultiplied(
                    theme::ACCENT.r(),
                    theme::ACCENT.g(),
                    theme::ACCENT.b(),
                    SELECTED_BG_ALPHA,
                );
                painter.rect_filled(rect, 2.0, bg_color);

                // Left accent border
                let border_rect = egui::Rect::from_min_size(
                    rect.left_top(),
                    Vec2::new(ACCENT_BORDER_WIDTH, rect.height()),
                );
                painter.rect_filled(border_rect, 1.0, theme::ACCENT);
            }

            // Render: dimmed directory + highlighted filename
            let path_str = fm.path.display().to_string();
            let (dir_part, name_part) = match path_str.rfind('/') {
                Some(idx) => (&path_str[..=idx], &path_str[idx + 1..]),
                None => ("", path_str.as_str()),
            };

            let text_pos = rect.left_top() + Vec2::new(
                if is_selected { ACCENT_BORDER_WIDTH + 8.0 } else { 8.0 },
                (FILE_ROW_HEIGHT - 14.0) / 2.0,
            );

            // Draw directory portion (dimmed, no highlighting)
            let dir_galley = painter.layout_no_wrap(
                dir_part.to_string(),
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
                name_part,
                query,
                theme::TEXT_PRIMARY,
                theme::ACCENT,
                14.0,
            );
        }

        response
    }

    /// Render a single content match row: file:line header + context lines.
    fn render_content_row(
        &self,
        ui: &mut egui::Ui,
        cm: &ContentMatch,
        query: &str,
        is_selected: bool,
    ) -> egui::Response {
        let desired_size = Vec2::new(ui.available_width(), CONTENT_ROW_HEIGHT);
        let (rect, response) = ui.allocate_exact_size(desired_size, Sense::click());

        if ui.is_rect_visible(rect) {
            let painter = ui.painter_at(rect);

            // Selected item: background highlight + left accent border
            if is_selected {
                let bg_color = Color32::from_rgba_premultiplied(
                    theme::ACCENT.r(),
                    theme::ACCENT.g(),
                    theme::ACCENT.b(),
                    SELECTED_BG_ALPHA,
                );
                painter.rect_filled(rect, 2.0, bg_color);

                // Left accent border
                let border_rect = egui::Rect::from_min_size(
                    rect.left_top(),
                    Vec2::new(ACCENT_BORDER_WIDTH, rect.height()),
                );
                painter.rect_filled(border_rect, 1.0, theme::ACCENT);
            }

            let left_pad = if is_selected { ACCENT_BORDER_WIDTH + 8.0 } else { 8.0 };
            let mut y_offset = 4.0;

            // Header: file:line
            let header = format!("{}:{}", cm.path.display(), cm.line_number);
            let header_pos = rect.left_top() + Vec2::new(left_pad, y_offset);
            painter.text(
                header_pos,
                egui::Align2::LEFT_TOP,
                &header,
                egui::FontId::monospace(11.0),
                theme::TEXT_MUTED,
            );
            y_offset += 16.0;

            // Context before (up to 1 line to fit in row height)
            for ctx_line in cm.context_before.iter().rev().take(1).rev() {
                let line_pos = rect.left_top() + Vec2::new(left_pad + 12.0, y_offset);
                let trimmed = truncate_line(ctx_line, 100);
                painter.text(
                    line_pos,
                    egui::Align2::LEFT_TOP,
                    &trimmed,
                    egui::FontId::monospace(12.0),
                    theme::TEXT_MUTED,
                );
                y_offset += 14.0;
            }

            // Matched line with highlighted match range
            let match_pos = rect.left_top() + Vec2::new(left_pad + 12.0, y_offset);
            let line_content = truncate_line(&cm.line_content, 100);
            render_highlighted_text(
                &painter,
                match_pos,
                &line_content,
                query,
                theme::TEXT_PRIMARY,
                theme::ACCENT,
                12.0,
            );
            y_offset += 14.0;

            // Context after (up to 1 line)
            for ctx_line in cm.context_after.iter().take(1) {
                let line_pos = rect.left_top() + Vec2::new(left_pad + 12.0, y_offset);
                let trimmed = truncate_line(ctx_line, 100);
                painter.text(
                    line_pos,
                    egui::Align2::LEFT_TOP,
                    &trimmed,
                    egui::FontId::monospace(12.0),
                    theme::TEXT_MUTED,
                );
            }
        }

        response
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
}
