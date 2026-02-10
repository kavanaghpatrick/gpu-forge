/// Tracks mouse and keyboard input state for camera control and interaction.
pub struct InputState {
    /// Current cursor position in window coordinates
    pub cursor_x: f64,
    pub cursor_y: f64,
    /// Previous cursor position (for computing drag deltas)
    pub prev_cursor_x: f64,
    pub prev_cursor_y: f64,
    /// Whether the left mouse button is held
    pub left_held: bool,
    /// Whether the right mouse button is held
    pub right_held: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            cursor_x: 0.0,
            cursor_y: 0.0,
            prev_cursor_x: 0.0,
            prev_cursor_y: 0.0,
            left_held: false,
            right_held: false,
        }
    }
}

impl InputState {
    /// Returns the current cursor position in window coordinates.
    #[allow(dead_code)]
    pub fn cursor_position(&self) -> (f64, f64) {
        (self.cursor_x, self.cursor_y)
    }

    /// Update cursor position and return drag delta if right button is held.
    ///
    /// Returns `Some((dx, dy))` if right-dragging, `None` otherwise.
    pub fn update_cursor(&mut self, x: f64, y: f64) -> Option<(f32, f32)> {
        self.prev_cursor_x = self.cursor_x;
        self.prev_cursor_y = self.cursor_y;
        self.cursor_x = x;
        self.cursor_y = y;

        if self.right_held {
            let dx = (self.cursor_x - self.prev_cursor_x) as f32;
            let dy = (self.cursor_y - self.prev_cursor_y) as f32;
            Some((dx, dy))
        } else {
            None
        }
    }
}
