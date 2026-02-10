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
