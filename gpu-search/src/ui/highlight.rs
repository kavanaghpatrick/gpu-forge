//! Syntax highlighting via syntect with custom Tokyo Night theme.
//!
//! Provides [`SyntaxHighlighter`] that maps source code lines to styled spans
//! using a curated Tokyo Night color palette (keywords, strings, comments,
//! types, functions). Query match text is overlaid with bold amber background
//! to stand out above syntax colors.
//!
//! Parse state is cached per file extension so repeated highlighting of the
//! same language re-uses the compiled grammar.

use std::collections::HashMap;
use std::ops::Range;
use std::str::FromStr;

use syntect::highlighting::{
    Color as SynColor, FontStyle, HighlightState, Highlighter, Style, StyleModifier, ThemeSettings,
};
use syntect::highlighting::{ScopeSelectors, Theme, ThemeItem};
use syntect::parsing::{ParseState, ScopeStack, SyntaxReference, SyntaxSet};

// Re-export for consumers (results_list, etc.)
pub use syntect::highlighting::Style as SynStyle;

// ── Tokyo Night syntax colors ──────────────────────────────────────────────

/// Keywords (purple): `fn`, `let`, `if`, `match`, `use`, `pub`, `struct`
const KW_COLOR: SynColor = SynColor { r: 0xBB, g: 0x9A, b: 0xF7, a: 0xFF };

/// Strings (green): `"hello"`, `'c'`
const STR_COLOR: SynColor = SynColor { r: 0x9E, g: 0xCE, b: 0x6A, a: 0xFF };

/// Comments (muted): `// ...`, `/* ... */`
const COMMENT_COLOR: SynColor = SynColor { r: 0x56, g: 0x5F, b: 0x89, a: 0xFF };

/// Types (cyan): `u32`, `String`, `Vec<T>`, trait names
const TYPE_COLOR: SynColor = SynColor { r: 0x2A, g: 0xC3, b: 0xDE, a: 0xFF };

/// Functions (blue): function calls and definitions
const FN_COLOR: SynColor = SynColor { r: 0x7A, g: 0xA2, b: 0xF7, a: 0xFF };

/// Default foreground (Tokyo Night primary text)
const FG_DEFAULT: SynColor = SynColor { r: 0xC0, g: 0xCA, b: 0xF5, a: 0xFF };

/// Default background (Tokyo Night base)
const BG_DEFAULT: SynColor = SynColor { r: 0x1A, g: 0x1B, b: 0x26, a: 0xFF };

/// Query match overlay: amber background
const MATCH_BG: SynColor = SynColor { r: 0xE0, g: 0xAF, b: 0x68, a: 0x60 };

/// Query match overlay: amber bold foreground
const MATCH_FG: SynColor = SynColor { r: 0xE0, g: 0xAF, b: 0x68, a: 0xFF };

// ── StyledSpan ─────────────────────────────────────────────────────────────

/// A contiguous run of text with uniform styling.
#[derive(Debug, Clone, PartialEq)]
pub struct StyledSpan {
    /// The text content of this span.
    pub text: String,
    /// Foreground color as (r, g, b, a).
    pub fg: (u8, u8, u8, u8),
    /// Background color as (r, g, b, a), or None for default.
    pub bg: Option<(u8, u8, u8, u8)>,
    /// Whether the text is bold.
    pub bold: bool,
}

impl StyledSpan {
    fn from_syntect(text: &str, style: &Style) -> Self {
        Self {
            text: text.to_string(),
            fg: (
                style.foreground.r,
                style.foreground.g,
                style.foreground.b,
                style.foreground.a,
            ),
            bg: None,
            bold: style.font_style.contains(FontStyle::BOLD),
        }
    }

    fn with_match_overlay(text: &str) -> Self {
        Self {
            text: text.to_string(),
            fg: (MATCH_FG.r, MATCH_FG.g, MATCH_FG.b, MATCH_FG.a),
            bg: Some((MATCH_BG.r, MATCH_BG.g, MATCH_BG.b, MATCH_BG.a)),
            bold: true,
        }
    }
}

// ── Tokyo Night theme builder ──────────────────────────────────────────────

fn parse_scope(s: &str) -> ScopeSelectors {
    ScopeSelectors::from_str(s).unwrap_or_else(|_| ScopeSelectors::from_str("source").unwrap())
}

/// Build a syntect Theme with Tokyo Night colors (4-5 syntax categories).
fn build_tokyo_night_theme() -> Theme {
    Theme {
        name: Some("Tokyo Night".to_string()),
        author: Some("gpu-search".to_string()),
        settings: ThemeSettings {
            foreground: Some(FG_DEFAULT),
            background: Some(BG_DEFAULT),
            caret: None,
            line_highlight: None,
            misspelling: None,
            accent: None,
            bracket_contents_foreground: None,
            bracket_contents_options: None,
            brackets_foreground: None,
            brackets_options: None,
            brackets_background: None,
            tags_foreground: None,
            tags_options: None,
            find_highlight: None,
            find_highlight_foreground: None,
            gutter: None,
            gutter_foreground: None,
            selection: None,
            selection_foreground: None,
            selection_border: None,
            inactive_selection: None,
            inactive_selection_foreground: None,
            guide: None,
            active_guide: None,
            stack_guide: None,
            highlight: None,
            shadow: None,
            minimap_border: None,
            popup_css: None,
            phantom_css: None,
        },
        scopes: vec![
            // 1. Keywords
            ThemeItem {
                scope: parse_scope("keyword"),
                style: StyleModifier {
                    foreground: Some(KW_COLOR),
                    background: None,
                    font_style: Some(FontStyle::BOLD),
                },
            },
            // Storage modifiers (pub, static, const)
            ThemeItem {
                scope: parse_scope("storage"),
                style: StyleModifier {
                    foreground: Some(KW_COLOR),
                    background: None,
                    font_style: Some(FontStyle::BOLD),
                },
            },
            // 2. Strings
            ThemeItem {
                scope: parse_scope("string"),
                style: StyleModifier {
                    foreground: Some(STR_COLOR),
                    background: None,
                    font_style: None,
                },
            },
            // 3. Comments
            ThemeItem {
                scope: parse_scope("comment"),
                style: StyleModifier {
                    foreground: Some(COMMENT_COLOR),
                    background: None,
                    font_style: Some(FontStyle::ITALIC),
                },
            },
            // 4. Types
            ThemeItem {
                scope: parse_scope("entity.name.type"),
                style: StyleModifier {
                    foreground: Some(TYPE_COLOR),
                    background: None,
                    font_style: None,
                },
            },
            ThemeItem {
                scope: parse_scope("support.type"),
                style: StyleModifier {
                    foreground: Some(TYPE_COLOR),
                    background: None,
                    font_style: None,
                },
            },
            ThemeItem {
                scope: parse_scope("storage.type"),
                style: StyleModifier {
                    foreground: Some(TYPE_COLOR),
                    background: None,
                    font_style: None,
                },
            },
            // 5. Functions
            ThemeItem {
                scope: parse_scope("entity.name.function"),
                style: StyleModifier {
                    foreground: Some(FN_COLOR),
                    background: None,
                    font_style: None,
                },
            },
            ThemeItem {
                scope: parse_scope("support.function"),
                style: StyleModifier {
                    foreground: Some(FN_COLOR),
                    background: None,
                    font_style: None,
                },
            },
        ],
    }
}

// ── SyntaxHighlighter ──────────────────────────────────────────────────────

/// Cached syntax highlighter that maps source lines to styled spans.
///
/// Maintains per-extension parse state cache for efficient re-highlighting.
/// Only processes lines that are passed in (caller decides visibility).
pub struct SyntaxHighlighter {
    syntax_set: SyntaxSet,
    theme: Theme,
    /// Cache: file extension -> (ParseState, HighlightState) after last highlighted line.
    /// This allows incremental parsing when highlighting sequential lines of the same file.
    parse_cache: HashMap<String, (ParseState, HighlightState)>,
}

impl SyntaxHighlighter {
    /// Create a new highlighter with the built-in Tokyo Night theme.
    pub fn new() -> Self {
        Self {
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme: build_tokyo_night_theme(),
            parse_cache: HashMap::new(),
        }
    }

    /// Get the syntect SyntaxReference for a file extension, falling back to plain text.
    fn syntax_for_ext(&self, ext: &str) -> &SyntaxReference {
        self.syntax_set
            .find_syntax_by_extension(ext)
            .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text())
    }

    /// Highlight a single line of code, returning styled spans.
    ///
    /// Uses the file extension to select the grammar and caches parse state
    /// for efficient sequential highlighting.
    ///
    /// # Arguments
    /// * `line` - The source line (may or may not include trailing newline)
    /// * `ext` - File extension (e.g., "rs", "py", "js")
    /// * `query` - Optional search query; matching substrings get amber overlay
    pub fn highlight_line(
        &mut self,
        line: &str,
        ext: &str,
        query: Option<&str>,
    ) -> Vec<StyledSpan> {
        let highlighter = Highlighter::new(&self.theme);

        // Ensure cache entry exists (avoids borrow conflict with syntax_for_ext)
        if !self.parse_cache.contains_key(ext) {
            let syntax = self.syntax_for_ext(ext);
            self.parse_cache.insert(
                ext.to_string(),
                (
                    ParseState::new(syntax),
                    HighlightState::new(&highlighter, ScopeStack::new()),
                ),
            );
        }

        // Get or create parse state for this extension
        let (parse_state, highlight_state) = self
            .parse_cache
            .get_mut(ext)
            .unwrap();

        // Ensure line ends with newline for syntect
        let line_with_nl = if line.ends_with('\n') {
            line.to_string()
        } else {
            format!("{}\n", line)
        };

        // Parse the line to get scope operations
        let ops = parse_state.parse_line(&line_with_nl, &self.syntax_set);

        // Convert scope ops to styled ranges using our theme
        let styles: Vec<(Style, &str)> =
            syntect::highlighting::RangedHighlightIterator::new(
                highlight_state,
                &ops.unwrap_or_default(),
                &line_with_nl,
                &highlighter,
            )
            .map(|(style, text, _range)| (style, text))
            .collect();

        // Build spans from syntect styles (trimming trailing newline)
        let mut syntax_spans: Vec<StyledSpan> = Vec::new();
        let mut total_len = 0usize;
        let line_len = line.len();

        for (style, text) in &styles {
            if total_len >= line_len {
                break;
            }
            let available = line_len - total_len;
            let segment = if text.len() > available {
                &text[..available]
            } else {
                text
            };
            if !segment.is_empty() {
                syntax_spans.push(StyledSpan::from_syntect(segment, style));
            }
            total_len += text.len();
        }

        // If no spans produced, return the raw line with default style
        if syntax_spans.is_empty() && !line.is_empty() {
            syntax_spans.push(StyledSpan {
                text: line.to_string(),
                fg: (FG_DEFAULT.r, FG_DEFAULT.g, FG_DEFAULT.b, FG_DEFAULT.a),
                bg: None,
                bold: false,
            });
        }

        // Apply query match overlay if provided
        if let Some(q) = query {
            if !q.is_empty() {
                syntax_spans = apply_match_overlay(syntax_spans, line, q);
            }
        }

        syntax_spans
    }

    /// Reset cached parse state for a given extension.
    ///
    /// Call when switching to a different file with the same extension,
    /// since parse state is sequential.
    pub fn reset_cache(&mut self, ext: &str) {
        self.parse_cache.remove(ext);
    }

    /// Reset all cached parse state.
    pub fn reset_all_caches(&mut self) {
        self.parse_cache.clear();
    }
}

impl Default for SyntaxHighlighter {
    fn default() -> Self {
        Self::new()
    }
}

// ── Query match overlay ────────────────────────────────────────────────────

/// Split styled spans at the given byte ranges, applying amber match overlay
/// to portions inside any range. Spans outside ranges keep their original style.
///
/// Ranges must be sorted and non-overlapping. Ranges are clamped to the total
/// text length of the input spans.
fn split_spans_at_ranges(spans: Vec<StyledSpan>, ranges: &[Range<usize>]) -> Vec<StyledSpan> {
    if ranges.is_empty() {
        return spans;
    }

    let mut result: Vec<StyledSpan> = Vec::new();
    let mut byte_offset = 0usize;

    for span in &spans {
        let span_start = byte_offset;
        let span_end = byte_offset + span.text.len();

        // Split this span at match boundaries
        let mut cursor = span_start;
        for range in ranges {
            // Skip ranges entirely before this span
            if range.end <= cursor {
                continue;
            }
            // Stop if range starts after this span
            if range.start >= span_end {
                break;
            }

            // Portion before the match range (syntax-colored)
            if cursor < range.start && range.start < span_end {
                let before_end = range.start.min(span_end);
                let text = &span.text[(cursor - span_start)..(before_end - span_start)];
                if !text.is_empty() {
                    result.push(StyledSpan {
                        text: text.to_string(),
                        fg: span.fg,
                        bg: span.bg,
                        bold: span.bold,
                    });
                }
                cursor = before_end;
            }

            // Match portion (amber overlay)
            let match_start = range.start.max(cursor);
            let match_end = range.end.min(span_end);
            if match_start < match_end {
                let text = &span.text[(match_start - span_start)..(match_end - span_start)];
                if !text.is_empty() {
                    result.push(StyledSpan::with_match_overlay(text));
                }
                cursor = match_end;
            }
        }

        // Remainder after all match ranges (syntax-colored)
        if cursor < span_end {
            let text = &span.text[(cursor - span_start)..(span_end - span_start)];
            if !text.is_empty() {
                result.push(StyledSpan {
                    text: text.to_string(),
                    fg: span.fg,
                    bg: span.bg,
                    bold: span.bold,
                });
            }
        }

        byte_offset = span_end;
    }

    result
}

/// Apply amber match overlay to spans where the query appears in the original line.
///
/// Case-insensitive matching. Match spans get bold + amber foreground + amber background,
/// overriding whatever syntax color was underneath.
fn apply_match_overlay(
    spans: Vec<StyledSpan>,
    original_line: &str,
    query: &str,
) -> Vec<StyledSpan> {
    // Find all match ranges in the original line (case-insensitive)
    let match_ranges = find_match_ranges(original_line, query);
    if match_ranges.is_empty() {
        return spans;
    }

    split_spans_at_ranges(spans, &match_ranges)
}

/// Apply amber match overlay at a specific byte range from the GPU match result.
///
/// This uses the GPU-reported `match_range` directly instead of re-searching
/// the line for query text, providing precise highlighting at the exact match
/// position reported by the GPU kernel.
///
/// The range is clamped to the total span text length. If `start >= total_len`,
/// spans are returned unchanged.
pub fn apply_match_range_overlay(
    spans: Vec<StyledSpan>,
    match_range: Range<usize>,
) -> Vec<StyledSpan> {
    // Compute total text length from spans
    let total_len: usize = spans.iter().map(|s| s.text.len()).sum();

    // If start is beyond the text, return unchanged
    if match_range.start >= total_len {
        return spans;
    }

    // Empty range: return unchanged
    if match_range.start >= match_range.end {
        return spans;
    }

    // Clamp end to total text length
    let clamped_end = match_range.end.min(total_len);
    let clamped_range = match_range.start..clamped_end;

    split_spans_at_ranges(spans, &[clamped_range])
}

/// Find all case-insensitive occurrences of `query` in `text`, returning byte ranges.
fn find_match_ranges(text: &str, query: &str) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    if query.is_empty() || text.is_empty() {
        return ranges;
    }

    let text_lower = text.to_lowercase();
    let query_lower = query.to_lowercase();
    let mut start = 0;

    while let Some(pos) = text_lower[start..].find(&query_lower) {
        let abs_start = start + pos;
        let abs_end = abs_start + query.len();
        ranges.push(abs_start..abs_end);
        start = abs_end;
    }

    ranges
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_highlight() {
        let mut hl = SyntaxHighlighter::new();

        // Highlight a Rust line with a keyword
        let spans = hl.highlight_line("fn main() {", "rs", None);
        assert!(!spans.is_empty(), "should produce at least one span");

        // Verify the spans cover the full text
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "fn main() {");

        // `fn` is a keyword -- should be purple (KW_COLOR or TYPE_COLOR for storage)
        let fn_span = &spans[0];
        assert_eq!(fn_span.text, "fn");
        // Should be keyword purple or type cyan (syntect may classify as storage.type)
        let is_keyword_or_type = (fn_span.fg.0 == KW_COLOR.r && fn_span.fg.2 == KW_COLOR.b)
            || (fn_span.fg.0 == TYPE_COLOR.r && fn_span.fg.2 == TYPE_COLOR.b);
        assert!(
            is_keyword_or_type,
            "fn should be keyword purple or type cyan, got {:?}",
            fn_span.fg
        );
    }

    #[test]
    fn test_highlight_rust_string() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line(r#"let s = "hello";"#, "rs", None);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, r#"let s = "hello";"#);

        // Find the span containing "hello" -- should be green (STR_COLOR)
        let string_span = spans
            .iter()
            .find(|s| s.text.contains("hello"))
            .expect("should have a span containing 'hello'");
        assert_eq!(
            string_span.fg.0, STR_COLOR.r,
            "string should be green, got {:?}",
            string_span.fg
        );
    }

    #[test]
    fn test_highlight_rust_comment() {
        let mut hl = SyntaxHighlighter::new();
        // Reset to ensure clean state
        hl.reset_cache("rs");
        let spans = hl.highlight_line("// a comment", "rs", None);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "// a comment");

        // Entire line should be comment color (muted)
        assert_eq!(
            spans[0].fg.0, COMMENT_COLOR.r,
            "comment should be muted color, got {:?}",
            spans[0].fg
        );
    }

    #[test]
    fn test_query_match_overlay() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line("fn search() {}", "rs", Some("search"));

        // Find the match overlay span
        let match_span = spans
            .iter()
            .find(|s| s.text == "search")
            .expect("should have a span for 'search'");

        // Match should be bold + amber
        assert!(match_span.bold, "match should be bold");
        assert_eq!(match_span.fg.0, MATCH_FG.r, "match fg should be amber");
        assert!(match_span.bg.is_some(), "match should have background");
    }

    #[test]
    fn test_query_match_case_insensitive() {
        let mut hl = SyntaxHighlighter::new();
        hl.reset_cache("rs");
        let spans = hl.highlight_line("fn Search() {}", "rs", Some("search"));

        let match_span = spans
            .iter()
            .find(|s| s.text == "Search")
            .expect("should match case-insensitively");
        assert!(match_span.bold);
        assert_eq!(match_span.fg.0, MATCH_FG.r);
    }

    #[test]
    fn test_plain_text_fallback() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line("just some text", "xyz_unknown", None);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "just some text");
    }

    #[test]
    fn test_empty_line() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line("", "rs", None);
        assert!(spans.is_empty() || spans.iter().all(|s| s.text.is_empty()));
    }

    #[test]
    fn test_cache_reset() {
        let mut hl = SyntaxHighlighter::new();
        hl.highlight_line("fn foo() {}", "rs", None);
        assert!(hl.parse_cache.contains_key("rs"));

        hl.reset_cache("rs");
        assert!(!hl.parse_cache.contains_key("rs"));

        hl.highlight_line("fn bar() {}", "rs", None);
        hl.highlight_line("def baz():", "py", None);
        assert!(hl.parse_cache.contains_key("rs"));
        assert!(hl.parse_cache.contains_key("py"));

        hl.reset_all_caches();
        assert!(hl.parse_cache.is_empty());
    }

    #[test]
    fn test_find_match_ranges() {
        let ranges = find_match_ranges("Hello HELLO hello", "hello");
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0], 0..5);
        assert_eq!(ranges[1], 6..11);
        assert_eq!(ranges[2], 12..17);
    }

    #[test]
    fn test_find_match_ranges_empty() {
        assert!(find_match_ranges("hello", "").is_empty());
        assert!(find_match_ranges("", "hello").is_empty());
        assert!(find_match_ranges("hello", "xyz").is_empty());
    }

    #[test]
    fn test_multiple_matches_on_line() {
        let mut hl = SyntaxHighlighter::new();
        hl.reset_cache("rs");
        let spans = hl.highlight_line("let x = fn_fn(fn);", "rs", Some("fn"));

        // Count how many spans are match-highlighted
        let match_count = spans.iter().filter(|s| s.bold && s.fg.0 == MATCH_FG.r).count();
        assert!(
            match_count >= 2,
            "should have at least 2 match spans, got {}",
            match_count
        );
    }

    #[test]
    fn test_tokyo_night_theme_colors() {
        // Verify the theme has the expected number of scope rules
        let theme = build_tokyo_night_theme();
        assert_eq!(theme.name.as_deref(), Some("Tokyo Night"));
        assert!(theme.scopes.len() >= 5, "should have at least 5 scope rules");
        assert_eq!(theme.settings.foreground, Some(FG_DEFAULT));
        assert_eq!(theme.settings.background, Some(BG_DEFAULT));
    }

    #[test]
    fn test_styled_span_from_syntect() {
        let style = Style {
            foreground: SynColor { r: 0xBB, g: 0x9A, b: 0xF7, a: 0xFF },
            background: SynColor { r: 0, g: 0, b: 0, a: 0 },
            font_style: FontStyle::BOLD,
        };
        let span = StyledSpan::from_syntect("test", &style);
        assert_eq!(span.text, "test");
        assert_eq!(span.fg, (0xBB, 0x9A, 0xF7, 0xFF));
        assert!(span.bold);
        assert!(span.bg.is_none());
    }

    #[test]
    fn test_python_highlighting() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line("def hello():", "py", None);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "def hello():");

        // `def` should be a keyword
        let def_span = &spans[0];
        assert_eq!(def_span.text, "def");
    }

    #[test]
    fn test_javascript_highlighting() {
        let mut hl = SyntaxHighlighter::new();
        let spans = hl.highlight_line("const x = 42;", "js", None);
        let combined: String = spans.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "const x = 42;");
    }

    // ── apply_match_range_overlay tests ────────────────────────────────────

    /// Helper: create a plain span with default fg, no bg, not bold.
    fn plain_span(text: &str) -> StyledSpan {
        StyledSpan {
            text: text.to_string(),
            fg: (FG_DEFAULT.r, FG_DEFAULT.g, FG_DEFAULT.b, FG_DEFAULT.a),
            bg: None,
            bold: false,
        }
    }

    /// Helper: check if a span has match overlay styling (amber bold).
    fn is_match_styled(span: &StyledSpan) -> bool {
        span.bold && span.fg.0 == MATCH_FG.r && span.bg.is_some()
    }

    #[test]
    fn test_match_range_overlay_basic() {
        // "hello world" with match_range covering "world" (6..11)
        let spans = vec![plain_span("hello world")];
        let result = apply_match_range_overlay(spans, 6..11);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello ");
        assert!(!is_match_styled(&result[0]));
        assert_eq!(result[1].text, "world");
        assert!(is_match_styled(&result[1]));
    }

    #[test]
    fn test_match_range_overlay_at_start() {
        // Match at start of line: "fn" in "fn main()"
        let spans = vec![plain_span("fn main()")];
        let result = apply_match_range_overlay(spans, 0..2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "fn");
        assert!(is_match_styled(&result[0]));
        assert_eq!(result[1].text, " main()");
        assert!(!is_match_styled(&result[1]));
    }

    #[test]
    fn test_match_range_overlay_at_end() {
        // Match at end of line: "()" in "fn main()"
        let spans = vec![plain_span("fn main()")];
        let result = apply_match_range_overlay(spans, 7..9);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "fn main");
        assert!(!is_match_styled(&result[0]));
        assert_eq!(result[1].text, "()");
        assert!(is_match_styled(&result[1]));
    }

    #[test]
    fn test_match_range_overlay_full_line() {
        // Match covers entire line
        let spans = vec![plain_span("hello")];
        let result = apply_match_range_overlay(spans, 0..5);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello");
        assert!(is_match_styled(&result[0]));
    }

    #[test]
    fn test_match_range_overlay_empty_range() {
        // Empty range (start == end) should return spans unchanged
        let spans = vec![plain_span("hello world")];
        let result = apply_match_range_overlay(spans.clone(), 3..3);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello world");
        assert!(!is_match_styled(&result[0]));
    }

    #[test]
    fn test_match_range_overlay_clamped() {
        // Range extends past end of text -- should clamp to text length
        let spans = vec![plain_span("hello")];
        let result = apply_match_range_overlay(spans, 3..100);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hel");
        assert!(!is_match_styled(&result[0]));
        assert_eq!(result[1].text, "lo");
        assert!(is_match_styled(&result[1]));
    }

    #[test]
    fn test_match_range_overlay_start_beyond_text() {
        // start >= total_len: should return spans unchanged
        let spans = vec![plain_span("hello")];
        let result = apply_match_range_overlay(spans.clone(), 10..15);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, "hello");
        assert!(!is_match_styled(&result[0]));
    }

    #[test]
    fn test_match_range_overlay_multi_span_split() {
        // Multiple input spans, match range crosses a span boundary
        // "hel" + "lo wor" + "ld" -> match "lo wo" (3..8)
        let spans = vec![
            plain_span("hel"),
            StyledSpan {
                text: "lo wor".to_string(),
                fg: (STR_COLOR.r, STR_COLOR.g, STR_COLOR.b, STR_COLOR.a),
                bg: None,
                bold: false,
            },
            plain_span("ld"),
        ];
        let result = apply_match_range_overlay(spans, 3..8);

        // Combined text should be preserved
        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "hello world");

        // "hel" unchanged, "lo wo" match-styled, "r" original, "ld" unchanged
        assert_eq!(result[0].text, "hel");
        assert!(!is_match_styled(&result[0]));

        // "lo wo" should be match-styled (spans 1..2 split)
        let matched: String = result.iter().filter(|s| is_match_styled(s)).map(|s| s.text.as_str()).collect();
        assert_eq!(matched, "lo wo");
    }

    #[test]
    fn test_split_spans_at_ranges_preserves_text() {
        // Verify that split_spans_at_ranges always preserves the full text
        let spans = vec![
            plain_span("abc"),
            plain_span("def"),
            plain_span("ghi"),
        ];
        let result = split_spans_at_ranges(spans, &[2..7]);

        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "abcdefghi");
    }

    #[test]
    fn test_apply_match_overlay_uses_split_spans() {
        // Verify the refactored apply_match_overlay still works correctly
        // by comparing with known output
        let spans = vec![plain_span("hello world")];
        let result = apply_match_overlay(spans, "hello world", "world");

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].text, "hello ");
        assert!(!is_match_styled(&result[0]));
        assert_eq!(result[1].text, "world");
        assert!(is_match_styled(&result[1]));
    }

    // ── U-MRO match_range overlay unit tests ──────────────────────────────

    /// Helper: create a span with specific color (simulating syntax highlighting).
    fn colored_span(text: &str, r: u8, g: u8, b: u8) -> StyledSpan {
        StyledSpan {
            text: text.to_string(),
            fg: (r, g, b, 0xFF),
            bg: None,
            bold: false,
        }
    }

    #[test]
    fn u_mro_1_basic_overlay_mid_line() {
        // Basic overlay in the middle of a multi-span line
        // Syntax: "let " (keyword) + "x" (default) + " = " (default) + "42" (number) + ";" (default)
        let spans = vec![
            colored_span("let ", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            plain_span("x = "),
            colored_span("42", TYPE_COLOR.r, TYPE_COLOR.g, TYPE_COLOR.b),
            plain_span(";"),
        ];
        // Match range covers "x = 4" (4..9)
        let result = apply_match_range_overlay(spans, 4..9);

        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "let x = 42;");

        let matched: String = result.iter().filter(|s| is_match_styled(s)).map(|s| s.text.as_str()).collect();
        assert_eq!(matched, "x = 4");

        // Non-match spans retain original styling
        assert_eq!(result[0].fg.0, KW_COLOR.r, "keyword span keeps its color");
    }

    #[test]
    fn u_mro_2_at_start_preserves_trailing_style() {
        // Match at position 0 with styled spans after
        let spans = vec![
            colored_span("fn", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            plain_span(" "),
            colored_span("main", FN_COLOR.r, FN_COLOR.g, FN_COLOR.b),
            plain_span("()"),
        ];
        let result = apply_match_range_overlay(spans, 0..2);

        assert!(is_match_styled(&result[0]), "first span should be match-styled");
        assert_eq!(result[0].text, "fn");

        // Remaining spans keep original styles
        let non_match: Vec<&StyledSpan> = result.iter().filter(|s| !is_match_styled(s)).collect();
        let trailing: String = non_match.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(trailing, " main()");
    }

    #[test]
    fn u_mro_3_at_end_preserves_leading_style() {
        // Match at the very last bytes of the line
        let spans = vec![
            colored_span("use ", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            plain_span("std::io"),
            plain_span(";"),
        ];
        // Match the semicolon at position 11..12
        let result = apply_match_range_overlay(spans, 11..12);

        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "use std::io;");

        // Only ";" is match-styled
        let matched: Vec<&StyledSpan> = result.iter().filter(|s| is_match_styled(s)).collect();
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].text, ";");
    }

    #[test]
    fn u_mro_4_full_line_multi_span() {
        // Full line match across multiple styled spans
        let spans = vec![
            colored_span("pub ", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            colored_span("fn", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            plain_span(" "),
            colored_span("run", FN_COLOR.r, FN_COLOR.g, FN_COLOR.b),
        ];
        let total_len: usize = spans.iter().map(|s| s.text.len()).sum();
        let result = apply_match_range_overlay(spans, 0..total_len);

        // Every span should be match-styled
        for span in &result {
            assert!(is_match_styled(span), "span '{}' should be match-styled", span.text);
        }
        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "pub fn run");
    }

    #[test]
    fn u_mro_5_span_boundary_split_preserves_all_styles() {
        // Match range that splits exactly at a span boundary (no partial spans)
        // "abc" (red) + "def" (green) + "ghi" (blue) -- match "def" exactly (3..6)
        let spans = vec![
            colored_span("abc", 0xFF, 0x00, 0x00),
            colored_span("def", 0x00, 0xFF, 0x00),
            colored_span("ghi", 0x00, 0x00, 0xFF),
        ];
        let result = apply_match_range_overlay(spans, 3..6);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].text, "abc");
        assert_eq!(result[0].fg.0, 0xFF, "first span keeps red");
        assert!(!is_match_styled(&result[0]));

        assert_eq!(result[1].text, "def");
        assert!(is_match_styled(&result[1]));

        assert_eq!(result[2].text, "ghi");
        assert_eq!(result[2].fg.2, 0xFF, "third span keeps blue");
        assert!(!is_match_styled(&result[2]));
    }

    #[test]
    fn u_mro_6_empty_range_various_positions() {
        // Empty ranges at various positions should all return unchanged
        let spans = vec![plain_span("hello"), colored_span(" world", STR_COLOR.r, STR_COLOR.g, STR_COLOR.b)];
        let original_len = spans.len();

        // Empty range at start
        let r1 = apply_match_range_overlay(spans.clone(), 0..0);
        assert_eq!(r1.len(), original_len);

        // Empty range in middle
        let r2 = apply_match_range_overlay(spans.clone(), 5..5);
        assert_eq!(r2.len(), original_len);

        // Inverted range (start > end) -- also treated as empty
        let r3 = apply_match_range_overlay(spans.clone(), 7..3);
        assert_eq!(r3.len(), original_len);
    }

    #[test]
    fn u_mro_7_clamped_to_line_preserves_all_text() {
        // Range starts inside text but extends far past -- verify no panic, all text preserved
        let spans = vec![
            colored_span("struct ", KW_COLOR.r, KW_COLOR.g, KW_COLOR.b),
            colored_span("Foo", TYPE_COLOR.r, TYPE_COLOR.g, TYPE_COLOR.b),
        ];
        let result = apply_match_range_overlay(spans, 5..1000);

        let combined: String = result.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(combined, "struct Foo");

        // "struc" unchanged, "t Foo" match-styled (clamped to len 9)
        let matched: String = result.iter().filter(|s| is_match_styled(s)).map(|s| s.text.as_str()).collect();
        assert_eq!(matched, "t Foo");
    }

    #[test]
    fn u_mro_8_match_range_vs_query_search_same_result() {
        // Compare: apply_match_range_overlay with explicit range vs
        // apply_match_overlay with query string -- both should produce
        // identical styled output for the same match position.
        let line = "fn search_query() {}";
        let query = "search";
        let query_start = line.find(query).unwrap();
        let query_end = query_start + query.len();

        let base_spans = vec![plain_span(line)];

        // Method 1: query-based overlay (finds "search" by string matching)
        let by_query = apply_match_overlay(base_spans.clone(), line, query);

        // Method 2: match_range overlay (uses explicit byte range)
        let by_range = apply_match_range_overlay(base_spans, query_start..query_end);

        // Both should produce the same number of spans with same text
        assert_eq!(by_query.len(), by_range.len(),
            "query overlay and range overlay should produce same span count");

        for (q_span, r_span) in by_query.iter().zip(by_range.iter()) {
            assert_eq!(q_span.text, r_span.text,
                "span text should match: query='{}' vs range='{}'", q_span.text, r_span.text);
            assert_eq!(is_match_styled(q_span), is_match_styled(r_span),
                "match styling should agree for span '{}'", q_span.text);
        }

        // Verify the match span specifically
        let q_match = by_query.iter().find(|s| s.text == "search").unwrap();
        let r_match = by_range.iter().find(|s| s.text == "search").unwrap();
        assert!(is_match_styled(q_match));
        assert!(is_match_styled(r_match));
    }
}
