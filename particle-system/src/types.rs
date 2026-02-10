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
/// Total size: 256 bytes.
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
    /// Pressure gradient interaction strength (default: 0.001)
    pub interaction_strength: f32,
    /// Mouse attraction radius (particles within this distance are attracted)
    pub mouse_attraction_radius: f32,
    /// Mouse attraction strength (force magnitude scaling)
    pub mouse_attraction_strength: f32,
    pub _pad3: f32,
    /// Burst emission center in world space (float3 + pad = 16 bytes)
    pub burst_position: [f32; 3],
    pub _pad_burst: f32,
    /// Number of burst particles to emit this frame (0 = no burst)
    pub burst_count: u32,
    pub _pad4: f32,
    pub _pad5: f32,
    pub _pad6: f32,
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
            interaction_strength: 0.001,
            mouse_attraction_radius: 5.0,
            mouse_attraction_strength: 10.0,
            _pad3: 0.0,
            burst_position: [0.0, 0.0, 0.0],
            _pad_burst: 0.0,
            burst_count: 0,
            _pad4: 0.0,
            _pad5: 0.0,
            _pad6: 0.0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_uniforms_size_256() {
        // Uniforms must be exactly 256 bytes (padded to match Metal buffer allocation)
        assert_eq!(
            mem::size_of::<Uniforms>(),
            256,
            "Uniforms struct must be 256 bytes"
        );
    }

    #[test]
    fn test_draw_args_layout() {
        // DrawArgs matches MTLDrawPrimitivesIndirectArguments: 4 x u32 = 16 bytes
        assert_eq!(
            mem::size_of::<DrawArgs>(),
            16,
            "DrawArgs must be 16 bytes (4 x u32)"
        );

        // Verify field offsets match MTLDrawPrimitivesIndirectArguments layout:
        // offset 0: vertexCount (u32)
        // offset 4: instanceCount (u32)
        // offset 8: vertexStart (u32)
        // offset 12: baseInstance (u32)
        let args = DrawArgs::default();
        let base = &args as *const DrawArgs as *const u8;
        unsafe {
            let vertex_count_ptr = base as *const u32;
            assert_eq!(*vertex_count_ptr, 4, "vertexCount should be 4");

            let instance_count_ptr = base.add(4) as *const u32;
            assert_eq!(*instance_count_ptr, 0, "instanceCount should be 0");

            let vertex_start_ptr = base.add(8) as *const u32;
            assert_eq!(*vertex_start_ptr, 0, "vertexStart should be 0");

            let base_instance_ptr = base.add(12) as *const u32;
            assert_eq!(*base_instance_ptr, 0, "baseInstance should be 0");
        }
    }

    #[test]
    fn test_counter_header_size() {
        // CounterHeader must be 16 bytes (u32 count + 12 bytes padding)
        assert_eq!(
            mem::size_of::<CounterHeader>(),
            16,
            "CounterHeader must be 16 bytes"
        );
        assert_eq!(COUNTER_HEADER_SIZE, 16);
    }

    #[test]
    fn test_draw_args_default_values() {
        let args = DrawArgs::default();
        assert_eq!(args.vertex_count, 4);
        assert_eq!(args.instance_count, 0);
        assert_eq!(args.vertex_start, 0);
        assert_eq!(args.base_instance, 0);
    }
}

/// Per-particle SoA buffer sizes in bytes for a given pool capacity.
///
/// Mixed-precision strategy for bandwidth reduction:
/// - Positions/velocities: FP32 (packed_float3, 12B each) — physics needs full precision
/// - Lifetimes: FP16 (half2, 4B) — [0, ~5s] range fits FP16 (precision: ~0.001s)
/// - Colors: FP16 (half4, 8B) — [0,1] RGBA ideal for FP16 (precision: ~0.001)
/// - Sizes: FP16 (half, 2B) — [0.01, 0.05] range fits FP16
///
/// Total per-particle: 12+12+4+8+2 = 38B (vs 96B all-FP32) = 60% bandwidth reduction.
/// At 10M particles: 380 MB vs 960 MB read+write per frame.
#[allow(dead_code)]
#[derive(Debug)]
pub struct BufferSizes {
    /// packed_float3 (12 bytes each) — FP32 for physics precision
    pub positions: usize,
    /// packed_float3 (12 bytes each) — FP32 for physics precision
    pub velocities: usize,
    /// half2 (4 bytes each) — FP16: (age, max_age)
    pub lifetimes: usize,
    /// half4 (8 bytes each) — FP16: (r, g, b, a)
    pub colors: usize,
    /// half (2 bytes each) — FP16: particle radius
    pub sizes: usize,
    /// 16B header + pool_size * 4B indices
    pub counter_list: usize,
    /// Grid density: 64^3 cells x 4 bytes (uint32)
    pub grid_density: usize,
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
            lifetimes: pool_size * 4,  // half2 = 4 bytes
            colors: pool_size * 8,     // half4 = 8 bytes
            sizes: pool_size * 2,      // half = 2 bytes (FP16, not padded in arrays)
            counter_list: COUNTER_HEADER_SIZE + pool_size * 4,
            grid_density: 64 * 64 * 64 * 4, // 64^3 cells x 4 bytes (uint32) = 1,048,576 bytes
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
            + self.grid_density    // grid density (64^3 uint32)
            + self.indirect_args
            + self.uniforms
    }
}
