use glam::{Mat4, Vec3};

/// Compute default view and projection matrices for particle rendering.
///
/// Camera at (0,0,5) looking at origin, with perspective projection
/// (60 deg FOV, 1280/720 aspect, 0.1-100 near/far).
#[allow(dead_code)]
pub fn default_camera_matrices() -> ([[f32; 4]; 4], [[f32; 4]; 4]) {
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(60f32.to_radians(), 1280.0 / 720.0, 0.1, 100.0);
    (view.to_cols_array_2d(), proj.to_cols_array_2d())
}

/// Rust-side Uniforms struct matching the MSL `Uniforms` in shaders/types.h.
///
/// Layout must be identical byte-for-byte. MSL float3 occupies 16 bytes
/// in structs (padded to 16-byte alignment), so we use `[f32; 3]` + explicit
/// padding float to match.
///
/// Total size: 208 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
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
        let (view_matrix, projection_matrix) = default_camera_matrices();
        Self {
            view_matrix,
            projection_matrix,
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
#[allow(dead_code)]
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

/// Counter header for dead/alive list buffers.
///
/// Layout: first 16 bytes of the buffer.
/// - offset 0:  `count` (u32, atomic counter)
/// - offset 4:  12 bytes padding (align to 16 bytes)
///
/// After the header, indices start at byte offset 16.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct CounterHeader {
    pub count: u32,
    pub _pad: [u32; 3],
}

/// Size of the counter header in bytes (16).
#[allow(dead_code)]
pub const COUNTER_HEADER_SIZE: usize = std::mem::size_of::<CounterHeader>();

/// Per-particle SoA buffer sizes in bytes for a given pool capacity.
#[allow(dead_code)]
#[derive(Debug)]
pub struct BufferSizes {
    /// float3 (12 bytes each)
    pub positions: usize,
    /// float3 (12 bytes each)
    pub velocities: usize,
    /// half2 (4 bytes each)
    pub lifetimes: usize,
    /// half4 (8 bytes each)
    pub colors: usize,
    /// half padded (4 bytes each)
    pub sizes: usize,
    /// 16B header + pool_size * 4B indices
    pub counter_list: usize,
    /// DrawArgs (32 bytes, but we use std::mem::size_of)
    pub indirect_args: usize,
    /// Uniforms buffer (256 bytes padded)
    pub uniforms: usize,
}

#[allow(dead_code)]
impl BufferSizes {
    /// Compute all buffer sizes for a given pool capacity.
    pub fn new(pool_size: usize) -> Self {
        Self {
            positions: pool_size * 12,
            velocities: pool_size * 12,
            lifetimes: pool_size * 4,
            colors: pool_size * 8,
            sizes: pool_size * 4,
            counter_list: COUNTER_HEADER_SIZE + pool_size * 4,
            indirect_args: 32, // MTLDrawPrimitivesIndirectArguments = 4 x u32 = 16, but padded to 32
            uniforms: 256,     // Uniforms padded to 256 bytes
        }
    }

    /// Total memory across all buffers (SoA + 1 dead + 2 alive + indirect + uniforms).
    pub fn total_bytes(&self) -> usize {
        self.positions
            + self.velocities
            + self.lifetimes
            + self.colors
            + self.sizes
            + self.counter_list     // dead list
            + self.counter_list * 2 // alive list A + B
            + self.indirect_args
            + self.uniforms
    }
}
