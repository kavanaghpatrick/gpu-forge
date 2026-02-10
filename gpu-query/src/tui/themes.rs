//! Gradient palette definitions for the TUI dashboard.
//!
//! Named palettes: thermal (GPU utilization bars), glow (throughput),
//! ice (cool metrics), neon (highlights), title (gradient title bar).

use super::gradient::{Gradient, GradientStop};
use ratatui::style::{Color, Modifier, Style};

/// A complete visual theme for the dashboard.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Name of the theme.
    pub name: String,
    /// Gradient for the title bar text.
    pub title_gradient: Gradient,
    /// Gradient for GPU utilization bars (green -> yellow -> red).
    pub thermal: Gradient,
    /// Gradient for throughput/speed indicators (cool -> hot).
    pub glow: Gradient,
    /// Primary border color.
    pub border: Color,
    /// Accent color for highlights.
    pub accent: Color,
    /// Dimmed/muted text color.
    pub muted: Color,
    /// Normal text color.
    pub text: Color,
    /// Background color (if overridden; None = terminal default).
    pub bg: Option<Color>,
    /// Style for selected/focused items.
    pub selection: Style,
    /// Style for borders.
    pub border_style: Style,
    /// Style for focused panel borders.
    pub focus_border_style: Style,
}

impl Theme {
    /// Get a theme by name. Defaults to "thermal" if unknown.
    pub fn by_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "glow" => Self::glow(),
            "mono" => Self::mono(),
            "ice" => Self::ice(),
            _ => Self::thermal(),
        }
    }

    /// Thermal theme: GPU heat gradient (green -> yellow -> red).
    /// Default theme. "Make it look sick."
    pub fn thermal() -> Self {
        Self {
            name: "thermal".into(),
            title_gradient: Gradient::new(vec![
                GradientStop::new(0.0, 255, 100, 0),   // deep orange
                GradientStop::new(0.3, 255, 200, 0),   // golden
                GradientStop::new(0.5, 255, 255, 100),  // hot yellow
                GradientStop::new(0.7, 255, 200, 0),   // golden
                GradientStop::new(1.0, 255, 100, 0),   // deep orange
            ]),
            thermal: Gradient::new(vec![
                GradientStop::new(0.0, 0, 200, 80),     // green
                GradientStop::new(0.4, 100, 220, 0),    // lime
                GradientStop::new(0.6, 255, 220, 0),    // yellow
                GradientStop::new(0.8, 255, 140, 0),    // orange
                GradientStop::new(1.0, 255, 40, 40),    // red
            ]),
            glow: Gradient::new(vec![
                GradientStop::new(0.0, 80, 80, 200),    // cool blue
                GradientStop::new(0.3, 100, 180, 255),  // light blue
                GradientStop::new(0.6, 180, 100, 255),  // purple
                GradientStop::new(1.0, 255, 80, 180),   // hot pink
            ]),
            border: Color::Rgb(100, 100, 120),
            accent: Color::Rgb(255, 180, 0),
            muted: Color::Rgb(100, 100, 110),
            text: Color::Rgb(220, 220, 230),
            bg: None,
            selection: Style::default()
                .fg(Color::Rgb(255, 220, 100))
                .add_modifier(Modifier::BOLD),
            border_style: Style::default().fg(Color::Rgb(80, 80, 100)),
            focus_border_style: Style::default()
                .fg(Color::Rgb(255, 180, 0))
                .add_modifier(Modifier::BOLD),
        }
    }

    /// Glow theme: neon blue/purple glow.
    pub fn glow() -> Self {
        Self {
            name: "glow".into(),
            title_gradient: Gradient::new(vec![
                GradientStop::new(0.0, 0, 150, 255),   // electric blue
                GradientStop::new(0.3, 100, 80, 255),   // blue-purple
                GradientStop::new(0.5, 200, 0, 255),    // purple
                GradientStop::new(0.7, 100, 80, 255),   // blue-purple
                GradientStop::new(1.0, 0, 150, 255),   // electric blue
            ]),
            thermal: Gradient::new(vec![
                GradientStop::new(0.0, 0, 100, 200),
                GradientStop::new(0.5, 100, 0, 255),
                GradientStop::new(1.0, 255, 0, 200),
            ]),
            glow: Gradient::new(vec![
                GradientStop::new(0.0, 0, 200, 255),
                GradientStop::new(0.5, 150, 0, 255),
                GradientStop::new(1.0, 255, 100, 255),
            ]),
            border: Color::Rgb(60, 60, 120),
            accent: Color::Rgb(100, 100, 255),
            muted: Color::Rgb(80, 80, 130),
            text: Color::Rgb(200, 200, 240),
            bg: None,
            selection: Style::default()
                .fg(Color::Rgb(150, 150, 255))
                .add_modifier(Modifier::BOLD),
            border_style: Style::default().fg(Color::Rgb(60, 60, 120)),
            focus_border_style: Style::default()
                .fg(Color::Rgb(100, 100, 255))
                .add_modifier(Modifier::BOLD),
        }
    }

    /// Ice theme: cold blue/cyan palette.
    pub fn ice() -> Self {
        Self {
            name: "ice".into(),
            title_gradient: Gradient::new(vec![
                GradientStop::new(0.0, 100, 200, 255),  // sky blue
                GradientStop::new(0.5, 200, 240, 255),  // ice white
                GradientStop::new(1.0, 100, 200, 255),  // sky blue
            ]),
            thermal: Gradient::new(vec![
                GradientStop::new(0.0, 0, 150, 200),
                GradientStop::new(0.5, 0, 200, 220),
                GradientStop::new(1.0, 200, 100, 100),
            ]),
            glow: Gradient::new(vec![
                GradientStop::new(0.0, 100, 180, 255),
                GradientStop::new(1.0, 200, 240, 255),
            ]),
            border: Color::Rgb(80, 120, 140),
            accent: Color::Rgb(100, 200, 255),
            muted: Color::Rgb(80, 110, 130),
            text: Color::Rgb(200, 220, 240),
            bg: None,
            selection: Style::default()
                .fg(Color::Rgb(150, 220, 255))
                .add_modifier(Modifier::BOLD),
            border_style: Style::default().fg(Color::Rgb(80, 120, 140)),
            focus_border_style: Style::default()
                .fg(Color::Rgb(100, 200, 255))
                .add_modifier(Modifier::BOLD),
        }
    }

    /// Mono theme: grayscale for simplicity.
    pub fn mono() -> Self {
        Self {
            name: "mono".into(),
            title_gradient: Gradient::new(vec![
                GradientStop::new(0.0, 180, 180, 180),
                GradientStop::new(0.5, 255, 255, 255),
                GradientStop::new(1.0, 180, 180, 180),
            ]),
            thermal: Gradient::new(vec![
                GradientStop::new(0.0, 80, 80, 80),
                GradientStop::new(1.0, 240, 240, 240),
            ]),
            glow: Gradient::new(vec![
                GradientStop::new(0.0, 120, 120, 120),
                GradientStop::new(1.0, 255, 255, 255),
            ]),
            border: Color::Rgb(100, 100, 100),
            accent: Color::White,
            muted: Color::Rgb(100, 100, 100),
            text: Color::Rgb(220, 220, 220),
            bg: None,
            selection: Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
            border_style: Style::default().fg(Color::Rgb(100, 100, 100)),
            focus_border_style: Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_by_name() {
        assert_eq!(Theme::by_name("thermal").name, "thermal");
        assert_eq!(Theme::by_name("glow").name, "glow");
        assert_eq!(Theme::by_name("ice").name, "ice");
        assert_eq!(Theme::by_name("mono").name, "mono");
        assert_eq!(Theme::by_name("unknown").name, "thermal"); // default
    }

    #[test]
    fn test_thermal_gradient_samples() {
        let theme = Theme::thermal();
        let colors = theme.title_gradient.sample(5);
        assert_eq!(colors.len(), 5);
    }

    #[test]
    fn test_glow_gradient_samples() {
        let theme = Theme::glow();
        let colors = theme.glow.sample(10);
        assert_eq!(colors.len(), 10);
    }

    #[test]
    fn test_all_themes_valid() {
        // Ensure no panics from any theme construction
        let _ = Theme::thermal();
        let _ = Theme::glow();
        let _ = Theme::ice();
        let _ = Theme::mono();
    }
}
