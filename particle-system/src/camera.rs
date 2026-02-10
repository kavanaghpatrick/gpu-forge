//! 3D orbit camera with spherical coordinate control.
//!
//! Provides view and projection matrices for particle rendering. Right-drag orbits
//! the camera around a target point; scroll wheel zooms in/out.

use glam::{Mat4, Vec3};

/// Orbit camera for 3D perspective viewing of the particle field.
///
/// Orbits around a target point using azimuth/elevation spherical coordinates.
/// Right-drag orbits, scroll zooms.
pub struct OrbitCamera {
    /// Horizontal rotation angle in radians
    pub azimuth: f32,
    /// Vertical rotation angle in radians (clamped to avoid gimbal lock)
    pub elevation: f32,
    /// Distance from target point
    pub distance: f32,
    /// Center point the camera orbits around
    pub target: Vec3,
    /// Field of view in radians
    pub fov: f32,
    /// Viewport aspect ratio (width / height)
    pub aspect: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.3,
            distance: 15.0,
            target: Vec3::ZERO,
            fov: 60f32.to_radians(),
            aspect: 1280.0 / 720.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl OrbitCamera {
    /// Compute the camera eye position from spherical coordinates.
    fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();
        self.target + Vec3::new(x, y, z)
    }

    /// Compute the view matrix (right-handed, looking from eye toward target).
    pub fn view_matrix(&self) -> [[f32; 4]; 4] {
        let eye = self.eye_position();
        Mat4::look_at_rh(eye, self.target, Vec3::Y).to_cols_array_2d()
    }

    /// Compute the perspective projection matrix (right-handed).
    pub fn projection_matrix(&self) -> [[f32; 4]; 4] {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far).to_cols_array_2d()
    }

    /// Orbit the camera by mouse drag delta (in pixels).
    ///
    /// `delta_x` rotates azimuth, `delta_y` rotates elevation.
    /// Sensitivity is scaled so ~500px drag = full rotation.
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        let sensitivity = 0.005;
        self.azimuth += delta_x * sensitivity;
        self.elevation += delta_y * sensitivity;

        // Clamp elevation to avoid flipping (keep slightly away from poles)
        let max_elev = std::f32::consts::FRAC_PI_2 - 0.01;
        self.elevation = self.elevation.clamp(-max_elev, max_elev);
    }

    /// Zoom the camera by scroll delta.
    ///
    /// Positive delta zooms in (decreases distance), negative zooms out.
    pub fn zoom(&mut self, delta: f32) {
        self.distance -= delta;
        self.distance = self.distance.clamp(1.0, 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // ── view_matrix tests ──────────────────────────────────────────

    #[test]
    fn view_matrix_default_is_valid() {
        let cam = OrbitCamera::default();
        let v = cam.view_matrix();

        // Default: azimuth=0, elevation=0.3, distance=15, target=(0,0,0)
        // Eye position: x = 15 * cos(0.3) * sin(0) = 0
        //               y = 15 * sin(0.3)          ≈ 4.44
        //               z = 15 * cos(0.3) * cos(0) ≈ 14.33
        // View matrix should be a valid 4x4 matrix (not NaN/Inf)
        for row in &v {
            for &val in row {
                assert!(!val.is_nan(), "view_matrix contains NaN");
                assert!(!val.is_infinite(), "view_matrix contains Inf");
            }
        }
    }

    #[test]
    fn view_matrix_default_looks_at_origin() {
        let cam = OrbitCamera::default();
        let v = cam.view_matrix();
        let m = Mat4::from_cols_array_2d(&v);

        // Transform the target (origin) into view space: should be on negative Z axis
        let target_view = m * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        // In view space, the target should be in front of camera (negative z in RH)
        assert!(target_view.z < 0.0, "target should be in front of camera (negative z), got {}", target_view.z);
    }

    #[test]
    fn view_matrix_at_zero_elevation_eye_on_z_axis() {
        let cam = OrbitCamera {
            azimuth: 0.0,
            elevation: 0.0,
            distance: 10.0,
            target: Vec3::ZERO,
            ..OrbitCamera::default()
        };
        let eye = cam.eye_position();
        // With azimuth=0, elevation=0: eye should be at (0, 0, 10)
        assert!(approx_eq(eye.x, 0.0), "eye.x should be 0, got {}", eye.x);
        assert!(approx_eq(eye.y, 0.0), "eye.y should be 0, got {}", eye.y);
        assert!(approx_eq(eye.z, 10.0), "eye.z should be 10, got {}", eye.z);
    }

    // ── projection_matrix tests ────────────────────────────────────

    #[test]
    fn projection_matrix_is_valid() {
        let cam = OrbitCamera::default();
        let p = cam.projection_matrix();

        for row in &p {
            for &val in row {
                assert!(!val.is_nan(), "projection_matrix contains NaN");
                assert!(!val.is_infinite(), "projection_matrix contains Inf");
            }
        }
    }

    #[test]
    fn projection_matrix_encodes_fov() {
        let cam = OrbitCamera::default();
        let p = cam.projection_matrix();

        // For a perspective projection (column-major in glam):
        // p[1][1] = 1 / tan(fov/2)
        let expected_p11 = 1.0 / (cam.fov / 2.0).tan();
        assert!(
            approx_eq(p[1][1], expected_p11),
            "p[1][1] should encode FOV: expected {}, got {}",
            expected_p11,
            p[1][1]
        );
    }

    #[test]
    fn projection_matrix_encodes_aspect() {
        let cam = OrbitCamera::default();
        let p = cam.projection_matrix();

        // p[0][0] = 1 / (aspect * tan(fov/2))
        let expected_p00 = 1.0 / (cam.aspect * (cam.fov / 2.0).tan());
        assert!(
            approx_eq(p[0][0], expected_p00),
            "p[0][0] should encode aspect: expected {}, got {}",
            expected_p00,
            p[0][0]
        );
    }

    #[test]
    fn projection_different_aspect_changes_matrix() {
        let cam_wide = OrbitCamera {
            aspect: 2.0,
            ..OrbitCamera::default()
        };
        let cam_square = OrbitCamera {
            aspect: 1.0,
            ..OrbitCamera::default()
        };
        let p_wide = cam_wide.projection_matrix();
        let p_square = cam_square.projection_matrix();

        // p[0][0] depends on aspect, so they should differ
        assert!(
            !approx_eq(p_wide[0][0], p_square[0][0]),
            "different aspect ratios should produce different p[0][0]"
        );
        // p[1][1] depends only on FOV, should be the same
        assert!(
            approx_eq(p_wide[1][1], p_square[1][1]),
            "same FOV should produce same p[1][1]"
        );
    }

    // ── orbit tests ────────────────────────────────────────────────

    #[test]
    fn orbit_azimuth_changes_by_delta_x_times_sensitivity() {
        let mut cam = OrbitCamera::default();
        let initial_azimuth = cam.azimuth;
        let delta_x = 100.0;
        let sensitivity = 0.005;

        cam.orbit(delta_x, 0.0);

        let expected = initial_azimuth + delta_x * sensitivity;
        assert!(
            approx_eq(cam.azimuth, expected),
            "azimuth should change by delta_x * sensitivity: expected {}, got {}",
            expected,
            cam.azimuth
        );
    }

    #[test]
    fn orbit_elevation_changes_by_delta_y_times_sensitivity() {
        let mut cam = OrbitCamera::default();
        let initial_elevation = cam.elevation;
        let delta_y = 50.0;
        let sensitivity = 0.005;

        cam.orbit(0.0, delta_y);

        let expected = initial_elevation + delta_y * sensitivity;
        assert!(
            approx_eq(cam.elevation, expected),
            "elevation should change by delta_y * sensitivity: expected {}, got {}",
            expected,
            cam.elevation
        );
    }

    #[test]
    fn orbit_elevation_clamped_at_upper_bound() {
        let mut cam = OrbitCamera::default();
        let max_elev = FRAC_PI_2 - 0.01;

        // Push elevation way past the upper limit
        cam.orbit(0.0, 100_000.0);

        assert!(
            approx_eq(cam.elevation, max_elev),
            "elevation should clamp to max {}, got {}",
            max_elev,
            cam.elevation
        );
    }

    #[test]
    fn orbit_elevation_clamped_at_lower_bound() {
        let mut cam = OrbitCamera::default();
        let min_elev = -(FRAC_PI_2 - 0.01);

        // Push elevation way past the lower limit
        cam.orbit(0.0, -100_000.0);

        assert!(
            approx_eq(cam.elevation, min_elev),
            "elevation should clamp to min {}, got {}",
            min_elev,
            cam.elevation
        );
    }

    #[test]
    fn orbit_both_axes_simultaneously() {
        let mut cam = OrbitCamera::default();
        let initial_az = cam.azimuth;
        let initial_el = cam.elevation;
        let sensitivity = 0.005;

        cam.orbit(200.0, -100.0);

        assert!(
            approx_eq(cam.azimuth, initial_az + 200.0 * sensitivity),
            "azimuth should update when orbiting both axes"
        );
        assert!(
            approx_eq(cam.elevation, initial_el + (-100.0) * sensitivity),
            "elevation should update when orbiting both axes"
        );
    }

    // ── zoom tests ─────────────────────────────────────────────────

    #[test]
    fn zoom_in_decreases_distance() {
        let mut cam = OrbitCamera::default();
        let initial = cam.distance;

        cam.zoom(2.0);

        assert!(
            cam.distance < initial,
            "zoom in should decrease distance: was {}, now {}",
            initial,
            cam.distance
        );
        assert!(
            approx_eq(cam.distance, initial - 2.0),
            "distance should decrease by delta: expected {}, got {}",
            initial - 2.0,
            cam.distance
        );
    }

    #[test]
    fn zoom_out_increases_distance() {
        let mut cam = OrbitCamera::default();
        let initial = cam.distance;

        cam.zoom(-5.0);

        assert!(
            cam.distance > initial,
            "zoom out should increase distance: was {}, now {}",
            initial,
            cam.distance
        );
    }

    #[test]
    fn zoom_clamps_minimum_at_1() {
        let mut cam = OrbitCamera::default();

        // Zoom in way too much
        cam.zoom(1000.0);

        assert!(
            approx_eq(cam.distance, 1.0),
            "distance should clamp to 1.0, got {}",
            cam.distance
        );
    }

    #[test]
    fn zoom_clamps_maximum_at_100() {
        let mut cam = OrbitCamera::default();

        // Zoom out way too much
        cam.zoom(-1000.0);

        assert!(
            approx_eq(cam.distance, 100.0),
            "distance should clamp to 100.0, got {}",
            cam.distance
        );
    }

    #[test]
    fn zoom_at_boundary_stays_clamped() {
        let mut cam = OrbitCamera {
            distance: 1.0,
            ..OrbitCamera::default()
        };

        // Try zooming in further when already at minimum
        cam.zoom(1.0);

        assert!(
            approx_eq(cam.distance, 1.0),
            "should stay at 1.0 when already at minimum, got {}",
            cam.distance
        );

        // Reset to maximum
        cam.distance = 100.0;
        cam.zoom(-1.0);

        assert!(
            approx_eq(cam.distance, 100.0),
            "should stay at 100.0 when already at maximum, got {}",
            cam.distance
        );
    }
}
