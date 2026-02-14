//! Tokyo Night dark theme for gpu-search.
//!
//! Color palette based on Tokyo Night (https://github.com/enkia/tokyo-night-vscode-theme).
//! All colors avoid pure black (#000000) and pure white (#FFFFFF) for WCAG 2.1 AA compliance.

use eframe::egui::{self, Color32, Stroke, Style, Visuals};
use eframe::egui::style::{HandleShape, NumericColorSpace, Selection, WidgetVisuals, Widgets};
use eframe::epaint::{CornerRadius, Shadow};

// ── Tokyo Night color constants ──────────────────────────────────────────────

/// Base background (#1A1B26) -- main window/panel background
pub const BG_BASE: Color32 = Color32::from_rgb(0x1A, 0x1B, 0x26);

/// Surface background (#24263A) -- elevated surfaces, cards, popups
pub const BG_SURFACE: Color32 = Color32::from_rgb(0x24, 0x26, 0x3A);

/// Primary text (#C0CAF5) -- body text, labels
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xC0, 0xCA, 0xF5);

/// Muted text (#565F89) -- placeholders, secondary info
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x56, 0x5F, 0x89);

/// Accent / amber (#E0AF68) -- selection, highlights, active elements
pub const ACCENT: Color32 = Color32::from_rgb(0xE0, 0xAF, 0x68);

/// Error / red (#F7768E) -- error text, destructive actions
pub const ERROR: Color32 = Color32::from_rgb(0xF7, 0x76, 0x8E);

/// Success / green (#9ECE6A) -- success indicators
pub const SUCCESS: Color32 = Color32::from_rgb(0x9E, 0xCE, 0x6A);

/// Border (#3B3E52) -- separators, outlines, dividers
pub const BORDER: Color32 = Color32::from_rgb(0x3B, 0x3E, 0x52);

// ── Derived constants ────────────────────────────────────────────────────────

/// Slightly lighter surface for hover states
const BG_HOVER: Color32 = Color32::from_rgb(0x2A, 0x2C, 0x42);

/// Slightly lighter surface for active/pressed states
const BG_ACTIVE: Color32 = Color32::from_rgb(0x30, 0x32, 0x4A);

/// Faint background for striped rows
const BG_FAINT: Color32 = Color32::from_rgb(0x1E, 0x1F, 0x2C);

/// Extreme background (text edit interiors)
const BG_EXTREME: Color32 = Color32::from_rgb(0x16, 0x17, 0x21);

/// Strong text for active/focused widgets (brighter than primary)
const TEXT_STRONG: Color32 = Color32::from_rgb(0xD5, 0xDE, 0xF5);

/// Accent with reduced alpha for selection background
const ACCENT_SELECTION_BG: Color32 = Color32::from_rgba_premultiplied(0x50, 0x40, 0x28, 0x80);

/// Code background
const CODE_BG: Color32 = Color32::from_rgb(0x20, 0x22, 0x34);

/// Hyperlink blue (Tokyo Night blue)
const LINK_BLUE: Color32 = Color32::from_rgb(0x7A, 0xA2, 0xF7);

// ── Extension dot colors ───────────────────────────────────────────────────

/// Tokyo Night purple (#BB9AF7) -- Rust, etc.
const EXT_PURPLE: Color32 = Color32::from_rgb(0xBB, 0x9A, 0xF7);

/// Tokyo Night green (#9ECE6A) -- Python, shell scripts
const EXT_GREEN: Color32 = Color32::from_rgb(0x9E, 0xCE, 0x6A);

/// Tokyo Night amber (#E0AF68) -- JavaScript, Swift
const EXT_AMBER: Color32 = Color32::from_rgb(0xE0, 0xAF, 0x68);

/// Tokyo Night blue (#7AA2F7) -- TypeScript, C/C++, config
const EXT_BLUE: Color32 = Color32::from_rgb(0x7A, 0xA2, 0xF7);

/// Tokyo Night cyan (#2AC3DE) -- Markdown, Go
const EXT_CYAN: Color32 = Color32::from_rgb(0x2A, 0xC3, 0xDE);

/// Tokyo Night red (#F7768E) -- HTML
const EXT_RED: Color32 = Color32::from_rgb(0xF7, 0x76, 0x8E);

/// Muted (#565F89) -- unknown/default extensions
const EXT_MUTED: Color32 = Color32::from_rgb(0x56, 0x5F, 0x89);

// ── Public API ───────────────────────────────────────────────────────────────

/// Return a color for the file-type indicator dot based on file extension.
///
/// Uses the Tokyo Night palette to visually distinguish file types at a glance.
/// Extensions are matched case-sensitively (lowercase expected).
pub fn extension_dot_color(ext: &str) -> Color32 {
    match ext {
        "rs" => EXT_PURPLE,
        "py" => EXT_GREEN,
        "js" | "jsx" => EXT_AMBER,
        "ts" | "tsx" => EXT_BLUE,
        "md" | "txt" => EXT_CYAN,
        "toml" | "yaml" | "json" => EXT_BLUE,
        "sh" | "bash" | "zsh" => EXT_GREEN,
        "c" | "cpp" | "h" => EXT_BLUE,
        "go" => EXT_CYAN,
        "swift" => EXT_AMBER,
        "html" => EXT_RED,
        _ => EXT_MUTED,
    }
}

/// Apply the Tokyo Night dark theme to the given egui context.
///
/// Call this once during app setup (e.g., in `eframe::App::setup` or at the
/// start of the first `update` frame).
pub fn apply_theme(ctx: &egui::Context) {
    let style = Style {
        visuals: tokyo_night_visuals(),
        ..Default::default()
    };
    ctx.set_style(style);
}

/// Build a complete `Visuals` struct with Tokyo Night colors.
fn tokyo_night_visuals() -> Visuals {
    Visuals {
        dark_mode: true,
        override_text_color: None,

        widgets: Widgets {
            noninteractive: WidgetVisuals {
                weak_bg_fill: BG_BASE,
                bg_fill: BG_BASE,
                bg_stroke: Stroke::new(1.0, BORDER),
                fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
                corner_radius: CornerRadius::same(4),
                expansion: 0.0,
            },
            inactive: WidgetVisuals {
                weak_bg_fill: BG_SURFACE,
                bg_fill: BG_SURFACE,
                bg_stroke: Stroke::NONE,
                fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
                corner_radius: CornerRadius::same(4),
                expansion: 0.0,
            },
            hovered: WidgetVisuals {
                weak_bg_fill: BG_HOVER,
                bg_fill: BG_HOVER,
                bg_stroke: Stroke::new(1.0, ACCENT),
                fg_stroke: Stroke::new(1.5, TEXT_STRONG),
                corner_radius: CornerRadius::same(4),
                expansion: 1.0,
            },
            active: WidgetVisuals {
                weak_bg_fill: BG_ACTIVE,
                bg_fill: BG_ACTIVE,
                bg_stroke: Stroke::new(1.0, ACCENT),
                fg_stroke: Stroke::new(2.0, TEXT_STRONG),
                corner_radius: CornerRadius::same(4),
                expansion: 1.0,
            },
            open: WidgetVisuals {
                weak_bg_fill: BG_SURFACE,
                bg_fill: BG_BASE,
                bg_stroke: Stroke::new(1.0, BORDER),
                fg_stroke: Stroke::new(1.0, TEXT_PRIMARY),
                corner_radius: CornerRadius::same(4),
                expansion: 0.0,
            },
        },

        selection: Selection {
            bg_fill: ACCENT_SELECTION_BG,
            stroke: Stroke::new(1.0, ACCENT),
        },

        hyperlink_color: LINK_BLUE,
        faint_bg_color: BG_FAINT,
        extreme_bg_color: BG_EXTREME,
        code_bg_color: CODE_BG,

        warn_fg_color: ACCENT,
        error_fg_color: ERROR,

        // Window
        window_corner_radius: CornerRadius::same(8),
        window_shadow: Shadow {
            offset: [8, 16],
            blur: 12,
            spread: 0,
            color: Color32::from_black_alpha(80),
        },
        window_fill: BG_BASE,
        window_stroke: Stroke::new(1.0, BORDER),
        window_highlight_topmost: true,

        menu_corner_radius: CornerRadius::same(6),

        // Panel
        panel_fill: BG_BASE,

        popup_shadow: Shadow {
            offset: [4, 8],
            blur: 8,
            spread: 0,
            color: Color32::from_black_alpha(64),
        },

        resize_corner_size: 12.0,
        text_cursor: Default::default(),
        clip_rect_margin: 3.0,
        button_frame: true,
        collapsing_header_frame: false,
        indent_has_left_vline: true,
        striped: false,
        slider_trailing_fill: false,
        handle_shape: HandleShape::Circle,
        interact_cursor: None,
        image_loading_spinners: true,
        numeric_color_space: NumericColorSpace::GammaByte,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify no pure black or pure white in the theme.
    #[test]
    fn no_pure_black_or_white() {
        let colors = [
            ("BG_BASE", BG_BASE),
            ("BG_SURFACE", BG_SURFACE),
            ("TEXT_PRIMARY", TEXT_PRIMARY),
            ("TEXT_MUTED", TEXT_MUTED),
            ("ACCENT", ACCENT),
            ("ERROR", ERROR),
            ("SUCCESS", SUCCESS),
            ("BORDER", BORDER),
            ("BG_HOVER", BG_HOVER),
            ("BG_ACTIVE", BG_ACTIVE),
            ("BG_FAINT", BG_FAINT),
            ("BG_EXTREME", BG_EXTREME),
            ("TEXT_STRONG", TEXT_STRONG),
            ("CODE_BG", CODE_BG),
            ("LINK_BLUE", LINK_BLUE),
        ];

        for (name, color) in colors {
            let [r, g, b, _] = color.to_array();
            assert!(
                !(r == 0 && g == 0 && b == 0),
                "{name} is pure black (#000000)"
            );
            assert!(
                !(r == 255 && g == 255 && b == 255),
                "{name} is pure white (#FFFFFF)"
            );
        }
    }

    /// WCAG 2.1 AA requires contrast ratio >= 4.5:1 for normal text.
    /// Verify TEXT_PRIMARY on BG_BASE meets this threshold.
    #[test]
    fn wcag_aa_contrast_text_on_bg() {
        let ratio = contrast_ratio(TEXT_PRIMARY, BG_BASE);
        assert!(
            ratio >= 4.5,
            "TEXT_PRIMARY on BG_BASE contrast ratio {ratio:.2} < 4.5 (WCAG AA)"
        );
    }

    /// Verify TEXT_PRIMARY on BG_SURFACE meets WCAG AA.
    #[test]
    fn wcag_aa_contrast_text_on_surface() {
        let ratio = contrast_ratio(TEXT_PRIMARY, BG_SURFACE);
        assert!(
            ratio >= 4.5,
            "TEXT_PRIMARY on BG_SURFACE contrast ratio {ratio:.2} < 4.5 (WCAG AA)"
        );
    }

    /// Verify ACCENT on BG_BASE meets WCAG AA for large text (3:1).
    #[test]
    fn wcag_aa_contrast_accent_on_bg() {
        let ratio = contrast_ratio(ACCENT, BG_BASE);
        assert!(
            ratio >= 3.0,
            "ACCENT on BG_BASE contrast ratio {ratio:.2} < 3.0 (WCAG AA large text)"
        );
    }

    /// Verify ERROR on BG_BASE meets WCAG AA for large text (3:1).
    #[test]
    fn wcag_aa_contrast_error_on_bg() {
        let ratio = contrast_ratio(ERROR, BG_BASE);
        assert!(
            ratio >= 3.0,
            "ERROR on BG_BASE contrast ratio {ratio:.2} < 3.0 (WCAG AA large text)"
        );
    }

    /// Verify visuals build without panic.
    #[test]
    fn visuals_build() {
        let v = tokyo_night_visuals();
        assert!(v.dark_mode);
        assert_eq!(v.window_fill, BG_BASE);
        assert_eq!(v.panel_fill, BG_BASE);
        assert!(v.override_text_color.is_none());
    }

    // ── extension_dot_color tests ─────────────────────────────────────────

    #[test]
    fn ext_rs_is_purple() {
        assert_eq!(extension_dot_color("rs"), EXT_PURPLE);
    }

    #[test]
    fn ext_py_is_green() {
        assert_eq!(extension_dot_color("py"), EXT_GREEN);
    }

    #[test]
    fn ext_js_jsx_are_amber() {
        assert_eq!(extension_dot_color("js"), EXT_AMBER);
        assert_eq!(extension_dot_color("jsx"), EXT_AMBER);
    }

    #[test]
    fn ext_ts_tsx_are_blue() {
        assert_eq!(extension_dot_color("ts"), EXT_BLUE);
        assert_eq!(extension_dot_color("tsx"), EXT_BLUE);
    }

    #[test]
    fn ext_md_txt_are_cyan() {
        assert_eq!(extension_dot_color("md"), EXT_CYAN);
        assert_eq!(extension_dot_color("txt"), EXT_CYAN);
    }

    #[test]
    fn ext_config_formats_are_blue() {
        assert_eq!(extension_dot_color("toml"), EXT_BLUE);
        assert_eq!(extension_dot_color("yaml"), EXT_BLUE);
        assert_eq!(extension_dot_color("json"), EXT_BLUE);
    }

    #[test]
    fn ext_shell_scripts_are_green() {
        assert_eq!(extension_dot_color("sh"), EXT_GREEN);
        assert_eq!(extension_dot_color("bash"), EXT_GREEN);
        assert_eq!(extension_dot_color("zsh"), EXT_GREEN);
    }

    #[test]
    fn ext_c_cpp_h_are_blue() {
        assert_eq!(extension_dot_color("c"), EXT_BLUE);
        assert_eq!(extension_dot_color("cpp"), EXT_BLUE);
        assert_eq!(extension_dot_color("h"), EXT_BLUE);
    }

    #[test]
    fn ext_go_is_cyan() {
        assert_eq!(extension_dot_color("go"), EXT_CYAN);
    }

    #[test]
    fn ext_swift_is_amber() {
        assert_eq!(extension_dot_color("swift"), EXT_AMBER);
    }

    #[test]
    fn ext_html_is_red() {
        assert_eq!(extension_dot_color("html"), EXT_RED);
    }

    #[test]
    fn ext_unknown_is_muted() {
        assert_eq!(extension_dot_color("xyz"), EXT_MUTED);
        assert_eq!(extension_dot_color(""), EXT_MUTED);
        assert_eq!(extension_dot_color("pdf"), EXT_MUTED);
        assert_eq!(extension_dot_color("doc"), EXT_MUTED);
    }

    #[test]
    fn ext_all_known_are_not_muted() {
        let known = [
            "rs", "py", "js", "jsx", "ts", "tsx", "md", "txt",
            "toml", "yaml", "json", "sh", "bash", "zsh",
            "c", "cpp", "h", "go", "swift", "html",
        ];
        for ext in known {
            assert_ne!(
                extension_dot_color(ext), EXT_MUTED,
                "Extension '{ext}' should not be muted"
            );
        }
    }

    // U-DOT-5: Case sensitivity -- uppercase extensions should fall through to muted
    #[test]
    fn ext_case_sensitive_uppercase_is_muted() {
        // Extensions are matched case-sensitively (lowercase expected per doc comment)
        assert_eq!(extension_dot_color("RS"), EXT_MUTED);
        assert_eq!(extension_dot_color("Py"), EXT_MUTED);
        assert_eq!(extension_dot_color("Js"), EXT_MUTED);
        assert_eq!(extension_dot_color("TS"), EXT_MUTED);
        assert_eq!(extension_dot_color("Go"), EXT_MUTED);
        assert_eq!(extension_dot_color("HTML"), EXT_MUTED);
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Relative luminance per WCAG 2.1 (sRGB -> linear -> luminance).
    fn relative_luminance(c: Color32) -> f64 {
        let [r, g, b, _] = c.to_array();
        let to_linear = |v: u8| -> f64 {
            let s = v as f64 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)
    }

    /// WCAG 2.1 contrast ratio between two colors.
    fn contrast_ratio(fg: Color32, bg: Color32) -> f64 {
        let l1 = relative_luminance(fg);
        let l2 = relative_luminance(bg);
        let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
        (lighter + 0.05) / (darker + 0.05)
    }
}
