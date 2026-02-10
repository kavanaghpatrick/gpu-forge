//! Mouse and keyboard input handling for camera control, particle interaction,
//! physics parameter tuning, and pool scaling.
//!
//! Tracks cursor position, mouse buttons, shift modifier, drag deltas for orbit,
//! and provides cursor-to-world unprojection for mouse attraction and burst emission.

use glam::{Mat4, Vec4};
use winit::keyboard::KeyCode;

/// Unproject screen-space cursor coordinates to a world-space position on the z=0 plane.
///
/// Converts window coordinates to NDC, computes a ray via inverse(projection * view),
/// and intersects with the z=0 plane. Falls back to a fixed-depth plane if the ray
/// is nearly parallel to z=0.
pub fn unproject_cursor_to_world(
    cursor_x: f64,
    cursor_y: f64,
    window_width: u32,
    window_height: u32,
    view: &[[f32; 4]; 4],
    proj: &[[f32; 4]; 4],
) -> [f32; 3] {
    if window_width == 0 || window_height == 0 {
        return [0.0, 0.0, 0.0];
    }

    // Convert screen coords to NDC [-1, 1]
    let ndc_x = (cursor_x / window_width as f64) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor_y / window_height as f64) * 2.0;
    let ndc_x = ndc_x as f32;
    let ndc_y = ndc_y as f32;

    // Compute inverse of projection * view
    let view_mat = Mat4::from_cols_array_2d(view);
    let proj_mat = Mat4::from_cols_array_2d(proj);
    let inv_vp = (proj_mat * view_mat).inverse();

    // Near point (NDC z = -1 for RH)
    let near_clip = inv_vp * Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let near = near_clip.truncate() / near_clip.w;

    // Far point (NDC z = 1 for RH)
    let far_clip = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let far = far_clip.truncate() / far_clip.w;

    // Ray direction
    let dir = far - near;

    // Intersect with z=0 plane
    if dir.z.abs() > 1e-6 {
        let t = -near.z / dir.z;
        if t >= 0.0 {
            let hit = near + dir * t;
            return [hit.x, hit.y, hit.z];
        }
    }

    // Fallback: project onto a plane at a fixed depth along the ray
    // Use t=0.5 to get a point roughly in the middle of the frustum
    let fallback = near + dir * 0.5;
    [fallback.x, fallback.y, fallback.z]
}

/// Runtime-tunable physics parameters adjustable via keyboard.
///
/// - G/Shift+G: increase/decrease gravity magnitude
/// - D/Shift+D: increase/decrease drag coefficient
/// - A/Shift+A: increase/decrease mouse attraction strength
/// - R: reset all parameters to defaults
/// - E/Shift+E: increase/decrease emission rate
#[derive(Clone, Debug)]
pub struct PhysicsParams {
    /// Gravity magnitude (negative = downward). Default: -9.81
    pub gravity: f32,
    /// Drag coefficient (velocity damping per second). Default: 0.02
    pub drag_coefficient: f32,
    /// Mouse attraction strength. Default: 10.0
    pub mouse_attraction_strength: f32,
    /// Base emission rate per frame. Default: 10000
    pub emission_rate: u32,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            gravity: -9.81,
            drag_coefficient: 0.02,
            mouse_attraction_strength: 10.0,
            emission_rate: 10000,
        }
    }
}

impl PhysicsParams {
    /// Reset all parameters to their default values.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Handle a physics parameter key press.
    /// Returns true if a parameter was changed.
    pub fn handle_key(&mut self, key_code: KeyCode, shift_held: bool) -> bool {
        match key_code {
            KeyCode::KeyG => {
                if shift_held {
                    // Shift+G: decrease gravity magnitude (less negative = weaker)
                    self.gravity += 1.0;
                    self.gravity = self.gravity.min(0.0);
                } else {
                    // G: increase gravity magnitude (more negative = stronger)
                    self.gravity -= 1.0;
                    self.gravity = self.gravity.max(-50.0);
                }
                true
            }
            KeyCode::KeyD => {
                if shift_held {
                    // Shift+D: decrease drag
                    self.drag_coefficient -= 0.005;
                    self.drag_coefficient = self.drag_coefficient.max(0.0);
                } else {
                    // D: increase drag
                    self.drag_coefficient += 0.005;
                    self.drag_coefficient = self.drag_coefficient.min(1.0);
                }
                true
            }
            KeyCode::KeyA => {
                if shift_held {
                    // Shift+A: decrease attraction
                    self.mouse_attraction_strength -= 2.0;
                    self.mouse_attraction_strength = self.mouse_attraction_strength.max(0.0);
                } else {
                    // A: increase attraction
                    self.mouse_attraction_strength += 2.0;
                    self.mouse_attraction_strength = self.mouse_attraction_strength.min(200.0);
                }
                true
            }
            KeyCode::KeyR => {
                self.reset();
                true
            }
            KeyCode::KeyE => {
                if shift_held {
                    // Shift+E: decrease emission rate
                    self.emission_rate = self.emission_rate.saturating_sub(1000).max(1000);
                } else {
                    // E: increase emission rate
                    self.emission_rate = (self.emission_rate + 1000).min(100000);
                }
                true
            }
            _ => false,
        }
    }

    /// Format a short summary of current parameter values for display.
    pub fn summary(&self) -> String {
        format!(
            "G:{:.1} D:{:.3} A:{:.0} E:{}",
            self.gravity, self.drag_coefficient, self.mouse_attraction_strength, self.emission_rate
        )
    }
}

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
    /// Set to true on initial left mouse press (consumed each frame)
    pub burst_requested: bool,
    /// World-space position for the burst emission
    pub burst_world_pos: [f32; 3],
    /// Pending pool grow size (set by keyboard, consumed at frame boundary)
    pub pending_grow: Option<usize>,
    /// Whether shift key is currently held
    pub shift_held: bool,
    /// Runtime-tunable physics parameters
    pub physics: PhysicsParams,
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
            burst_requested: false,
            burst_world_pos: [0.0, 0.0, 0.0],
            pending_grow: None,
            shift_held: false,
            physics: PhysicsParams::default(),
        }
    }
}

impl InputState {
    /// Returns the current cursor position in window coordinates.
    #[allow(dead_code)]
    pub fn cursor_position(&self) -> (f64, f64) {
        (self.cursor_x, self.cursor_y)
    }

    /// Handle a key press for physics parameters or pool scaling.
    /// Physics keys: G, D, A, R, E (with Shift modifier).
    /// Pool keys: 1 -> 1M, 2 -> 2M, 5 -> 5M, 0 -> 10M.
    pub fn handle_key(&mut self, key_code: KeyCode) {
        // Try physics key first
        if self.physics.handle_key(key_code, self.shift_held) {
            return;
        }
        // Fall through to pool scaling keys
        self.handle_pool_key(key_code);
    }

    /// Handle a key press for pool scaling.
    /// Keys: 1 -> 1M, 2 -> 2M, 5 -> 5M, 0 -> 10M.
    /// Sets `pending_grow` which is consumed at the frame boundary.
    pub fn handle_pool_key(&mut self, key_code: KeyCode) {
        let target = match key_code {
            KeyCode::Digit1 => Some(1_000_000),
            KeyCode::Digit2 => Some(2_000_000),
            KeyCode::Digit5 => Some(5_000_000),
            KeyCode::Digit0 => Some(10_000_000),
            _ => None,
        };
        if let Some(size) = target {
            self.pending_grow = Some(size);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use winit::keyboard::KeyCode;

    // --- Default / construction ---

    #[test]
    fn default_state_is_zeroed() {
        let s = InputState::default();
        assert_eq!(s.cursor_x, 0.0);
        assert_eq!(s.cursor_y, 0.0);
        assert_eq!(s.prev_cursor_x, 0.0);
        assert_eq!(s.prev_cursor_y, 0.0);
        assert!(!s.left_held);
        assert!(!s.right_held);
        assert!(!s.burst_requested);
        assert_eq!(s.burst_world_pos, [0.0, 0.0, 0.0]);
        assert!(s.pending_grow.is_none());
        assert!(!s.shift_held);
        assert!((s.physics.gravity - (-9.81)).abs() < 1e-5);
    }

    // --- Cursor position tracking ---

    #[test]
    fn cursor_position_returns_current_coords() {
        let mut s = InputState::default();
        s.cursor_x = 150.0;
        s.cursor_y = 300.0;
        assert_eq!(s.cursor_position(), (150.0, 300.0));
    }

    #[test]
    fn update_cursor_tracks_position() {
        let mut s = InputState::default();
        s.update_cursor(100.0, 200.0);
        assert_eq!(s.cursor_x, 100.0);
        assert_eq!(s.cursor_y, 200.0);
        assert_eq!(s.cursor_position(), (100.0, 200.0));
    }

    #[test]
    fn update_cursor_stores_previous_position() {
        let mut s = InputState::default();
        s.update_cursor(10.0, 20.0);
        s.update_cursor(30.0, 40.0);
        // prev should be the first position
        assert_eq!(s.prev_cursor_x, 10.0);
        assert_eq!(s.prev_cursor_y, 20.0);
        // current should be the second position
        assert_eq!(s.cursor_x, 30.0);
        assert_eq!(s.cursor_y, 40.0);
    }

    // --- Click detection and burst flag ---

    #[test]
    fn burst_flag_initially_false() {
        let s = InputState::default();
        assert!(!s.burst_requested);
    }

    #[test]
    fn burst_flag_can_be_set_and_consumed() {
        let mut s = InputState::default();
        // Simulate a left click triggering burst
        s.left_held = true;
        s.burst_requested = true;
        s.burst_world_pos = [1.0, 2.0, 3.0];

        assert!(s.burst_requested);
        assert_eq!(s.burst_world_pos, [1.0, 2.0, 3.0]);

        // Simulate frame consumption
        s.burst_requested = false;
        assert!(!s.burst_requested);
        // World pos persists until next burst
        assert_eq!(s.burst_world_pos, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn left_held_tracks_mouse_button_state() {
        let mut s = InputState::default();
        assert!(!s.left_held);
        s.left_held = true;
        assert!(s.left_held);
        s.left_held = false;
        assert!(!s.left_held);
    }

    // --- Keyboard state: pool growth triggers ---

    #[test]
    fn handle_pool_key_digit1_sets_1m() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit1);
        assert_eq!(s.pending_grow, Some(1_000_000));
    }

    #[test]
    fn handle_pool_key_digit2_sets_2m() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit2);
        assert_eq!(s.pending_grow, Some(2_000_000));
    }

    #[test]
    fn handle_pool_key_digit5_sets_5m() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit5);
        assert_eq!(s.pending_grow, Some(5_000_000));
    }

    #[test]
    fn handle_pool_key_digit0_sets_10m() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit0);
        assert_eq!(s.pending_grow, Some(10_000_000));
    }

    #[test]
    fn handle_pool_key_unrecognized_key_does_nothing() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::KeyA);
        assert!(s.pending_grow.is_none());
        s.handle_pool_key(KeyCode::Digit3);
        assert!(s.pending_grow.is_none());
        s.handle_pool_key(KeyCode::Space);
        assert!(s.pending_grow.is_none());
    }

    #[test]
    fn handle_pool_key_overwrites_previous_pending_grow() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit1);
        assert_eq!(s.pending_grow, Some(1_000_000));
        s.handle_pool_key(KeyCode::Digit5);
        assert_eq!(s.pending_grow, Some(5_000_000));
    }

    #[test]
    fn pending_grow_consumed_by_take() {
        let mut s = InputState::default();
        s.handle_pool_key(KeyCode::Digit2);
        let val = s.pending_grow.take();
        assert_eq!(val, Some(2_000_000));
        assert!(s.pending_grow.is_none());
    }

    // --- Drag accumulation for camera orbit ---

    #[test]
    fn update_cursor_returns_none_when_right_not_held() {
        let mut s = InputState::default();
        let result = s.update_cursor(100.0, 200.0);
        assert!(result.is_none());
    }

    #[test]
    fn update_cursor_returns_delta_when_right_held() {
        let mut s = InputState::default();
        // First position - establishes baseline
        s.update_cursor(100.0, 200.0);
        // Now hold right button
        s.right_held = true;
        // Drag to a new position
        let result = s.update_cursor(130.0, 250.0);
        assert!(result.is_some());
        let (dx, dy) = result.unwrap();
        assert!((dx - 30.0).abs() < 1e-5);
        assert!((dy - 50.0).abs() < 1e-5);
    }

    #[test]
    fn drag_delta_accumulates_across_multiple_moves() {
        let mut s = InputState::default();
        s.right_held = true;

        // Move from (0,0) to (10,5)
        let d1 = s.update_cursor(10.0, 5.0).unwrap();
        assert!((d1.0 - 10.0).abs() < 1e-5);
        assert!((d1.1 - 5.0).abs() < 1e-5);

        // Move from (10,5) to (25,15)
        let d2 = s.update_cursor(25.0, 15.0).unwrap();
        assert!((d2.0 - 15.0).abs() < 1e-5);
        assert!((d2.1 - 10.0).abs() < 1e-5);

        // Move from (25,15) to (20,10) — negative delta
        let d3 = s.update_cursor(20.0, 10.0).unwrap();
        assert!((d3.0 - (-5.0)).abs() < 1e-5);
        assert!((d3.1 - (-5.0)).abs() < 1e-5);
    }

    #[test]
    fn drag_stops_when_right_released() {
        let mut s = InputState::default();
        s.right_held = true;
        let _ = s.update_cursor(10.0, 10.0);
        assert!(s.update_cursor(20.0, 20.0).is_some());

        // Release right button
        s.right_held = false;
        let result = s.update_cursor(30.0, 30.0);
        assert!(result.is_none());
        // Position still tracked
        assert_eq!(s.cursor_x, 30.0);
        assert_eq!(s.cursor_y, 30.0);
    }

    #[test]
    fn drag_delta_zero_when_cursor_stationary() {
        let mut s = InputState::default();
        s.update_cursor(50.0, 50.0);
        s.right_held = true;
        let result = s.update_cursor(50.0, 50.0).unwrap();
        assert!((result.0).abs() < 1e-5);
        assert!((result.1).abs() < 1e-5);
    }

    // --- Physics params ---

    #[test]
    fn physics_params_default_values() {
        let p = PhysicsParams::default();
        assert!((p.gravity - (-9.81)).abs() < 1e-5);
        assert!((p.drag_coefficient - 0.02).abs() < 1e-5);
        assert!((p.mouse_attraction_strength - 10.0).abs() < 1e-5);
        assert_eq!(p.emission_rate, 10000);
    }

    #[test]
    fn physics_gravity_increase_decrease() {
        let mut p = PhysicsParams::default();
        // G (no shift) = increase magnitude (more negative)
        p.handle_key(KeyCode::KeyG, false);
        assert!((p.gravity - (-10.81)).abs() < 1e-5);
        // Shift+G = decrease magnitude (less negative)
        p.handle_key(KeyCode::KeyG, true);
        assert!((p.gravity - (-9.81)).abs() < 1e-5);
    }

    #[test]
    fn physics_gravity_clamp_max() {
        let mut p = PhysicsParams::default();
        // Push gravity past 0
        for _ in 0..20 {
            p.handle_key(KeyCode::KeyG, true);
        }
        assert!(p.gravity >= 0.0 - 1e-5);
        assert!(p.gravity <= 0.0 + 1e-5);
    }

    #[test]
    fn physics_gravity_clamp_min() {
        let mut p = PhysicsParams::default();
        for _ in 0..100 {
            p.handle_key(KeyCode::KeyG, false);
        }
        assert!(p.gravity >= -50.0 - 1e-5);
    }

    #[test]
    fn physics_drag_increase_decrease() {
        let mut p = PhysicsParams::default();
        let orig = p.drag_coefficient;
        p.handle_key(KeyCode::KeyD, false);
        assert!(p.drag_coefficient > orig);
        p.handle_key(KeyCode::KeyD, true);
        assert!((p.drag_coefficient - orig).abs() < 1e-5);
    }

    #[test]
    fn physics_attraction_increase_decrease() {
        let mut p = PhysicsParams::default();
        let orig = p.mouse_attraction_strength;
        p.handle_key(KeyCode::KeyA, false);
        assert!(p.mouse_attraction_strength > orig);
        p.handle_key(KeyCode::KeyA, true);
        assert!((p.mouse_attraction_strength - orig).abs() < 1e-5);
    }

    #[test]
    fn physics_emission_increase_decrease() {
        let mut p = PhysicsParams::default();
        p.handle_key(KeyCode::KeyE, false);
        assert_eq!(p.emission_rate, 11000);
        p.handle_key(KeyCode::KeyE, true);
        assert_eq!(p.emission_rate, 10000);
    }

    #[test]
    fn physics_emission_clamp_min() {
        let mut p = PhysicsParams::default();
        for _ in 0..20 {
            p.handle_key(KeyCode::KeyE, true);
        }
        assert!(p.emission_rate >= 1000);
    }

    #[test]
    fn physics_reset() {
        let mut p = PhysicsParams::default();
        p.gravity = -20.0;
        p.drag_coefficient = 0.5;
        p.mouse_attraction_strength = 100.0;
        p.emission_rate = 50000;
        p.handle_key(KeyCode::KeyR, false);
        let d = PhysicsParams::default();
        assert!((p.gravity - d.gravity).abs() < 1e-5);
        assert!((p.drag_coefficient - d.drag_coefficient).abs() < 1e-5);
        assert!((p.mouse_attraction_strength - d.mouse_attraction_strength).abs() < 1e-5);
        assert_eq!(p.emission_rate, d.emission_rate);
    }

    #[test]
    fn physics_summary_format() {
        let p = PhysicsParams::default();
        let s = p.summary();
        assert!(s.contains("G:"));
        assert!(s.contains("D:"));
        assert!(s.contains("A:"));
        assert!(s.contains("E:"));
    }

    #[test]
    fn handle_key_dispatches_physics_before_pool() {
        let mut s = InputState::default();
        // KeyG should be handled by physics, not pool
        s.handle_key(KeyCode::KeyG);
        assert!(s.pending_grow.is_none());
        assert!((s.physics.gravity - (-10.81)).abs() < 1e-5);
    }

    #[test]
    fn handle_key_dispatches_pool_for_digits() {
        let mut s = InputState::default();
        s.handle_key(KeyCode::Digit2);
        assert_eq!(s.pending_grow, Some(2_000_000));
    }

    #[test]
    fn shift_held_modifies_physics_direction() {
        let mut s = InputState::default();
        s.shift_held = true;
        s.handle_key(KeyCode::KeyG);
        // Shift+G decreases magnitude (gravity goes toward 0)
        assert!(s.physics.gravity > -9.81);
    }

    // --- Unprojection ---

    #[test]
    fn unproject_zero_window_returns_origin() {
        let view = [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]];
        let proj = view;
        let result = unproject_cursor_to_world(100.0, 100.0, 0, 0, &view, &proj);
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn unproject_identity_center_hits_z_zero() {
        // Identity view and projection: NDC center (0,0) with ray along -Z
        // Using a simple orthographic-like projection
        let identity = [[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]];
        // Center of a 200x200 window => NDC (0, 0)
        let result = unproject_cursor_to_world(100.0, 100.0, 200, 200, &identity, &identity);
        // With identity matrices, near = (0,0,-1), far = (0,0,1), ray hits z=0 at t=0.5
        assert!((result[0]).abs() < 1e-4);
        assert!((result[1]).abs() < 1e-4);
        assert!((result[2]).abs() < 1e-4);
    }
}
