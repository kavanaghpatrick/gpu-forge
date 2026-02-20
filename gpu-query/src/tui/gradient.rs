//! Gradient color interpolation for true-color terminal rendering.
//!
//! Provides RGB lerp between gradient stops for smooth color transitions.
//! Used by the theme system to create thermal bars, glow effects, etc.

use ratatui::style::Color;

/// A single color stop in a gradient (position 0.0..=1.0, RGB color).
#[derive(Debug, Clone, Copy)]
pub struct GradientStop {
    pub position: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl GradientStop {
    pub const fn new(position: f32, r: u8, g: u8, b: u8) -> Self {
        Self { position, r, g, b }
    }
}

/// A multi-stop gradient for smooth color interpolation.
#[derive(Debug, Clone)]
pub struct Gradient {
    stops: Vec<GradientStop>,
}

impl Gradient {
    /// Create a gradient from a sorted list of stops.
    /// Stops must be sorted by position (0.0..=1.0).
    pub fn new(stops: Vec<GradientStop>) -> Self {
        assert!(stops.len() >= 2, "gradient needs at least 2 stops");
        Self { stops }
    }

    /// Interpolate the gradient at position t (0.0..=1.0).
    /// Returns a ratatui Color::Rgb.
    pub fn at(&self, t: f32) -> Color {
        let t = t.clamp(0.0, 1.0);

        // Find the two stops surrounding t
        if t <= self.stops[0].position {
            return Color::Rgb(self.stops[0].r, self.stops[0].g, self.stops[0].b);
        }

        let last = self.stops.len() - 1;
        if t >= self.stops[last].position {
            return Color::Rgb(self.stops[last].r, self.stops[last].g, self.stops[last].b);
        }

        for i in 0..last {
            let a = &self.stops[i];
            let b = &self.stops[i + 1];
            if t >= a.position && t <= b.position {
                let range = b.position - a.position;
                let local_t = if range > 0.0 {
                    (t - a.position) / range
                } else {
                    0.0
                };
                return lerp_rgb(a.r, a.g, a.b, b.r, b.g, b.b, local_t);
            }
        }

        // Fallback
        Color::Rgb(self.stops[last].r, self.stops[last].g, self.stops[last].b)
    }

    /// Sample the gradient into N discrete colors (for text coloring).
    pub fn sample(&self, n: usize) -> Vec<Color> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![self.at(0.5)];
        }
        (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                self.at(t)
            })
            .collect()
    }
}

/// Linearly interpolate between two RGB colors.
fn lerp_rgb(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8, t: f32) -> Color {
    let r = lerp_u8(r1, r2, t);
    let g = lerp_u8(g1, g2, t);
    let b = lerp_u8(b1, b2, t);
    Color::Rgb(r, g, b)
}

/// Linearly interpolate between two u8 values.
fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    let a = a as f32;
    let b = b as f32;
    (a + (b - a) * t).round() as u8
}

/// Apply a gradient to a string, coloring each character.
/// Returns a Vec of (char, Color) pairs.
pub fn gradient_text(text: &str, gradient: &Gradient) -> Vec<(char, Color)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 {
        return vec![];
    }
    let colors = gradient.sample(n);
    chars.into_iter().zip(colors).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_stop_gradient() -> Gradient {
        Gradient::new(vec![
            GradientStop::new(0.0, 0, 0, 0),
            GradientStop::new(1.0, 255, 255, 255),
        ])
    }

    #[test]
    fn test_gradient_at_endpoints() {
        let g = two_stop_gradient();
        assert_eq!(g.at(0.0), Color::Rgb(0, 0, 0));
        assert_eq!(g.at(1.0), Color::Rgb(255, 255, 255));
    }

    #[test]
    fn test_gradient_at_midpoint() {
        let g = two_stop_gradient();
        match g.at(0.5) {
            Color::Rgb(r, _g, b) => {
                assert!((r as i16 - 128).abs() <= 1);
                assert!((b as i16 - 128).abs() <= 1);
            }
            _ => panic!("expected Rgb"),
        }
    }

    #[test]
    fn test_gradient_clamp() {
        let g = two_stop_gradient();
        assert_eq!(g.at(-1.0), Color::Rgb(0, 0, 0));
        assert_eq!(g.at(2.0), Color::Rgb(255, 255, 255));
    }

    #[test]
    fn test_multi_stop_gradient() {
        let g = Gradient::new(vec![
            GradientStop::new(0.0, 255, 0, 0),
            GradientStop::new(0.5, 0, 255, 0),
            GradientStop::new(1.0, 0, 0, 255),
        ]);
        // At 0.0 = red
        assert_eq!(g.at(0.0), Color::Rgb(255, 0, 0));
        // At 0.5 = green
        assert_eq!(g.at(0.5), Color::Rgb(0, 255, 0));
        // At 1.0 = blue
        assert_eq!(g.at(1.0), Color::Rgb(0, 0, 255));
    }

    #[test]
    fn test_sample() {
        let g = two_stop_gradient();
        let colors = g.sample(3);
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], Color::Rgb(0, 0, 0));
        assert_eq!(colors[2], Color::Rgb(255, 255, 255));
    }

    #[test]
    fn test_gradient_text() {
        let g = two_stop_gradient();
        let result = gradient_text("AB", &g);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 'A');
        assert_eq!(result[1].0, 'B');
    }
}
