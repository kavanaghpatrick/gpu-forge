/// Rust-side Uniforms struct matching the MSL `Uniforms` in shaders/types.h.
///
/// Layout must be identical byte-for-byte. MSL float3 occupies 16 bytes
/// in structs (padded to 16-byte alignment), so we use `[f32; 3]` + explicit
/// padding float to match.
///
/// Total size: 208 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Uniforms {
    /// Camera view matrix (column-major float4x4, 64 bytes)
    pub view_matrix: [[f32; 4]; 4],
    /// Projection matrix (column-major float4x4, 64 bytes)
    pub projection_matrix: [[f32; 4]; 4],
    /// Mouse position in world space (float3 + pad = 16 bytes)
    pub mouse_world_pos: [f32; 3],
    pub _pad_mouse: f32,
    /// Frame delta time in seconds
    pub dt: f32,
    /// Gravity magnitude (negative = downward)
    pub gravity: f32,
    /// Drag coefficient (velocity damping per second)
    pub drag_coefficient: f32,
    /// Padding to align grid_bounds_min to 16 bytes
    pub _pad0: f32,
    /// Grid minimum bounds (float3 + pad = 16 bytes)
    pub grid_bounds_min: [f32; 3],
    pub _pad_grid_min: f32,
    /// Grid maximum bounds (float3 + pad = 16 bytes)
    pub grid_bounds_max: [f32; 3],
    pub _pad_grid_max: f32,
    /// Current frame number (for PRNG seeding)
    pub frame_number: u32,
    /// Global particle size multiplier
    pub particle_size_scale: f32,
    /// Number of particles to emit this frame
    pub emission_count: u32,
    /// Total pool capacity
    pub pool_size: u32,
}

impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            projection_matrix: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            mouse_world_pos: [0.0, 0.0, 0.0],
            _pad_mouse: 0.0,
            dt: 0.016,
            gravity: -9.8,
            drag_coefficient: 0.1,
            _pad0: 0.0,
            grid_bounds_min: [-10.0, -10.0, -10.0],
            _pad_grid_min: 0.0,
            grid_bounds_max: [10.0, 10.0, 10.0],
            _pad_grid_max: 0.0,
            frame_number: 0,
            particle_size_scale: 1.0,
            emission_count: 10000,
            pool_size: 1_000_000,
        }
    }
}

/// Indirect draw arguments matching MTLDrawPrimitivesIndirectArguments
/// and MSL `DrawArgs` in shaders/types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

impl Default for DrawArgs {
    fn default() -> Self {
        Self {
            vertex_count: 4,
            instance_count: 0,
            vertex_start: 0,
            base_instance: 0,
        }
    }
}
