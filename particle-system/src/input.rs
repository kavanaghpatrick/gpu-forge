use glam::{Mat4, Vec4};

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
