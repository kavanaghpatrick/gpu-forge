//! GPU integration tests for Metal compute kernels.
//!
//! These tests run actual Metal compute dispatches on the GPU and verify
//! buffer contents via CPU readback (SharedStorage mode).
//! Uses `waitUntilCompleted()` for synchronous test execution.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

/// Size of the counter header in bytes (4 x u32 = 16 bytes).
const COUNTER_HEADER_SIZE: usize = 16;
/// Number of u32 values in the counter header (used to skip to indices).
const COUNTER_HEADER_UINTS: usize = 4;

/// Uniforms struct matching the MSL `Uniforms` in shaders/types.h.
/// Must be exactly 256 bytes and match the shader layout byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct Uniforms {
    view_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    mouse_world_pos: [f32; 3],
    _pad_mouse: f32,
    dt: f32,
    gravity: f32,
    drag_coefficient: f32,
    _pad0: f32,
    grid_bounds_min: [f32; 3],
    _pad_grid_min: f32,
    grid_bounds_max: [f32; 3],
    _pad_grid_max: f32,
    frame_number: u32,
    particle_size_scale: f32,
    base_emission_rate: u32,
    pool_size: u32,
    interaction_strength: f32,
    mouse_attraction_radius: f32,
    mouse_attraction_strength: f32,
    _pad3: f32,
    burst_position: [f32; 3],
    _pad_burst: f32,
    burst_count: u32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
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
            base_emission_rate: 100,
            pool_size: 1000,
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

/// Allocate a Metal buffer with shared storage mode.
fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .unwrap_or_else(|| panic!("Failed to allocate Metal buffer ({} bytes)", size))
}

/// Get raw pointer to buffer contents.
unsafe fn buffer_ptr(buffer: &ProtocolObject<dyn MTLBuffer>) -> *mut c_void {
    buffer.contents().as_ptr()
}

/// Read the u32 counter at offset 0 of a counter+index list buffer.
unsafe fn read_counter(buffer: &ProtocolObject<dyn MTLBuffer>) -> u32 {
    let ptr = buffer.contents().as_ptr() as *const u32;
    std::ptr::read(ptr)
}

/// Find shaders.metallib in the build output directory.
///
/// Integration tests run from target/debug/deps/, so we look in
/// target/debug/build/*/out/shaders.metallib relative to the executable.
fn find_metallib() -> String {
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let target_dir = exe_path.parent().expect("Failed to get parent of exe");

    // Integration tests are in target/debug/deps/ -- go up one level to target/debug/
    let debug_dir = target_dir.parent().unwrap_or(target_dir);
    let build_dir = debug_dir.join("build");

    if build_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&build_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let metallib = path.join("out").join("shaders.metallib");
                if metallib.exists() {
                    return metallib.to_string_lossy().into_owned();
                }
            }
        }
    }

    // Also check target/debug/build/ from target_dir directly
    let build_dir2 = target_dir.join("build");
    if build_dir2.exists() {
        if let Ok(entries) = std::fs::read_dir(&build_dir2) {
            for entry in entries.flatten() {
                let path = entry.path();
                let metallib = path.join("out").join("shaders.metallib");
                if metallib.exists() {
                    return metallib.to_string_lossy().into_owned();
                }
            }
        }
    }

    panic!(
        "Could not find shaders.metallib. Searched: {} and {}",
        build_dir.display(),
        target_dir.join("build").display()
    );
}

/// Create a compute pipeline for a named kernel function.
fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
    let fn_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&fn_name)
        .unwrap_or_else(|| panic!("Failed to find function '{}' in metallib", name));
    #[allow(deprecated)]
    device
        .newComputePipelineStateWithFunction_error(&function)
        .unwrap_or_else(|e| panic!("Failed to create pipeline for '{}': {:?}", name, e))
}

/// Test helper: set up device, queue, library, and emission pipeline.
pub struct GpuTestContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub emission_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuTestContext {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let metallib_path = find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        let emission_pipeline = create_pipeline(&device, &library, "emission_kernel");

        Self {
            device,
            queue,
            emission_pipeline,
        }
    }
}

/// Buffers for emission test at a given pool size.
pub struct EmissionBuffers {
    pub uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub dead_list: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub alive_list: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub velocities: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub lifetimes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub colors: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub sizes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl EmissionBuffers {
    /// Create and initialize buffers for `pool_size` particles.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, pool_size: usize) -> Self {
        let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

        let uniforms = alloc_buffer(device, 256);
        let dead_list = alloc_buffer(device, counter_list_size);
        let alive_list = alloc_buffer(device, counter_list_size);
        let positions = alloc_buffer(device, pool_size * 12); // packed_float3 = 12 bytes
        let velocities = alloc_buffer(device, pool_size * 12);
        let lifetimes = alloc_buffer(device, pool_size * 4); // half2 = 4 bytes
        let colors = alloc_buffer(device, pool_size * 8); // half4 = 8 bytes
        let sizes = alloc_buffer(device, pool_size * 4); // half padded = 4 bytes

        // Initialize uniforms with base_emission_rate and pool_size
        unsafe {
            let ptr = buffer_ptr(&uniforms) as *mut Uniforms;
            let mut u = Uniforms::default();
            u.base_emission_rate = 100;
            u.pool_size = pool_size as u32;
            u.burst_count = 0;
            std::ptr::write(ptr, u);
        }

        // Initialize dead list: counter = pool_size, indices = [0, 1, ..., pool_size-1]
        unsafe {
            let ptr = buffer_ptr(&dead_list) as *mut u32;
            // Counter at offset 0
            std::ptr::write(ptr, pool_size as u32);
            // Padding at offsets 1-3
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            // Indices starting at offset COUNTER_HEADER_UINTS
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
            }
        }

        // Initialize alive list: counter = 0, indices zeroed
        unsafe {
            let ptr = buffer_ptr(&alive_list) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), 0u32);
            }
        }

        // Zero positions buffer so we can verify non-zero after emission
        unsafe {
            let ptr = buffer_ptr(&positions) as *mut u8;
            std::ptr::write_bytes(ptr, 0, pool_size * 12);
        }

        // GpuEmissionParams: 16 bytes (emission_count, actual_burst_count, _pad0, _pad1)
        // Manually set emission_count = base_emission_rate (100) for test isolation
        let gpu_emission_params = alloc_buffer(device, 16);
        unsafe {
            let ptr = buffer_ptr(&gpu_emission_params) as *mut u32;
            std::ptr::write(ptr, 100u32); // emission_count
            std::ptr::write(ptr.add(1), 0u32); // actual_burst_count
            std::ptr::write(ptr.add(2), 0u32); // _pad0
            std::ptr::write(ptr.add(3), 0u32); // _pad1
        }

        Self {
            uniforms,
            dead_list,
            alive_list,
            positions,
            velocities,
            lifetimes,
            colors,
            sizes,
            gpu_emission_params,
        }
    }
}

/// GPU integration test for the emission kernel.
///
/// Verifies that dispatching the emission_kernel on the GPU correctly:
/// - Decrements the dead list counter by emission_count
/// - Increments the alive list counter by emission_count
/// - Initializes particle positions to non-zero values
#[test]
fn test_emission_gpu_integration() {
    let pool_size: usize = 1000;
    let emission_count: u32 = 100;

    let ctx = GpuTestContext::new();
    let bufs = EmissionBuffers::new(&ctx.device, pool_size);

    // Verify initial state
    unsafe {
        assert_eq!(
            read_counter(&bufs.dead_list),
            pool_size as u32,
            "Dead list should start at pool_size"
        );
        assert_eq!(
            read_counter(&bufs.alive_list),
            0,
            "Alive list should start at 0"
        );
    }

    // Create command buffer and encode emission dispatch
    let cmd_buf = ctx
        .queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&ctx.emission_pipeline);

    // Bind buffers matching emission_kernel signature:
    // buffer(0) = uniforms, buffer(1) = dead_list, buffer(2) = alive_list,
    // buffer(3) = positions, buffer(4) = velocities, buffer(5) = lifetimes,
    // buffer(6) = colors, buffer(7) = sizes, buffer(8) = gpu_emission_params
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&bufs.gpu_emission_params), 0, 8);
    }

    // Dispatch ceil(emission_count / 256) threadgroups of 256 threads
    let threadgroup_size: usize = 256;
    let threadgroup_count = (emission_count as usize).div_ceil(threadgroup_size);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();

    // Commit and wait for GPU completion (synchronous for tests)
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();

    // --- Readback and verify ---

    // Alive list counter should be exactly emission_count
    let alive_count = unsafe { read_counter(&bufs.alive_list) };
    assert_eq!(
        alive_count, emission_count,
        "Alive list counter should be {} after emission, got {}",
        emission_count, alive_count
    );

    // Dead list counter should be pool_size - emission_count
    let dead_count = unsafe { read_counter(&bufs.dead_list) };
    assert_eq!(
        dead_count,
        (pool_size as u32) - emission_count,
        "Dead list counter should be {} after emission, got {}",
        pool_size as u32 - emission_count,
        dead_count
    );

    // Readback positions: all emitted particles should have non-zero positions.
    // The alive list contains particle indices; read each particle's position.
    unsafe {
        let alive_ptr = bufs.alive_list.contents().as_ptr() as *const u32;
        let pos_ptr = bufs.positions.contents().as_ptr() as *const u8;

        let mut nonzero_count = 0u32;
        for i in 0..emission_count as usize {
            let particle_idx =
                std::ptr::read(alive_ptr.add(COUNTER_HEADER_UINTS + i)) as usize;

            // packed_float3: 12 bytes per particle (3 x f32, no padding)
            let px = std::ptr::read((pos_ptr.add(particle_idx * 12)) as *const f32);
            let py = std::ptr::read((pos_ptr.add(particle_idx * 12 + 4)) as *const f32);
            let pz = std::ptr::read((pos_ptr.add(particle_idx * 12 + 8)) as *const f32);

            // At least one component should be non-zero (random position in sphere)
            if px != 0.0 || py != 0.0 || pz != 0.0 {
                nonzero_count += 1;
            }
        }

        // All emitted particles should have been initialized with non-zero positions.
        // Allow a tiny margin for the extremely unlikely case of exactly (0,0,0).
        assert!(
            nonzero_count >= emission_count - 1,
            "Expected at least {} non-zero positions, got {}",
            emission_count - 1,
            nonzero_count
        );
    }

    println!(
        "test_emission PASSED: alive={}, dead={}, nonzero positions verified",
        alive_count, dead_count
    );
}

// ---------------------------------------------------------------------------
// Helper: dispatch emission kernel and wait (reused by physics/compaction tests)
// ---------------------------------------------------------------------------

/// Dispatch the emission kernel on the given context and buffers, then wait.
#[allow(dead_code)]
fn dispatch_emission(ctx: &GpuTestContext, bufs: &EmissionBuffers, emission_count: u32) {
    // Write emission_count into gpu_emission_params for test isolation
    unsafe {
        let ptr = buffer_ptr(&bufs.gpu_emission_params) as *mut u32;
        std::ptr::write(ptr, emission_count); // emission_count
        // actual_burst_count stays as previously initialized (0)
    }

    let cmd_buf = ctx
        .queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&ctx.emission_pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&bufs.gpu_emission_params), 0, 8);
    }

    let threadgroup_size: usize = 256;
    let threadgroup_count = (emission_count as usize).div_ceil(threadgroup_size);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}

// ---------------------------------------------------------------------------
// Extended context with update and sync_indirect_args pipelines
// ---------------------------------------------------------------------------

/// Extended GPU test context with update_physics and sync_indirect_args pipelines.
pub struct GpuTestContextFull {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub emission_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub update_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub sync_indirect_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuTestContextFull {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let metallib_path = find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        let emission_pipeline = create_pipeline(&device, &library, "emission_kernel");
        let update_pipeline = create_pipeline(&device, &library, "update_physics_kernel");
        let sync_indirect_pipeline = create_pipeline(&device, &library, "sync_indirect_args");

        Self {
            device,
            queue,
            emission_pipeline,
            update_pipeline,
            sync_indirect_pipeline,
        }
    }
}

/// Extended buffers including alive_list_b, grid_density, and indirect_args.
pub struct PhysicsBuffers {
    pub uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub dead_list: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub alive_list_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub alive_list_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub velocities: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub lifetimes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub colors: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub sizes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub grid_density: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub indirect_args: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub update_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Grid dimension constant (must match shader).
const GRID_DIM: usize = 64;
/// Total grid cells (64^3).
const GRID_CELLS: usize = GRID_DIM * GRID_DIM * GRID_DIM;

impl PhysicsBuffers {
    /// Create and initialize all buffers needed for physics/compaction tests.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, pool_size: usize) -> Self {
        let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

        let uniforms = alloc_buffer(device, 256);
        let dead_list = alloc_buffer(device, counter_list_size);
        let alive_list_a = alloc_buffer(device, counter_list_size);
        let alive_list_b = alloc_buffer(device, counter_list_size);
        let positions = alloc_buffer(device, pool_size * 12);
        let velocities = alloc_buffer(device, pool_size * 12);
        let lifetimes = alloc_buffer(device, pool_size * 4);
        let colors = alloc_buffer(device, pool_size * 8);
        let sizes = alloc_buffer(device, pool_size * 4);
        let grid_density = alloc_buffer(device, GRID_CELLS * 4);
        let indirect_args = alloc_buffer(device, 16); // DrawArgs = 4 x u32 = 16 bytes

        // Initialize uniforms: disable mouse attraction and interaction for isolated tests
        unsafe {
            let ptr = buffer_ptr(&uniforms) as *mut Uniforms;
            let mut u = Uniforms::default();
            u.base_emission_rate = 100;
            u.pool_size = pool_size as u32;
            u.dt = 0.016;
            u.gravity = -9.81;
            u.drag_coefficient = 0.02;
            u.interaction_strength = 0.0;
            u.mouse_attraction_strength = 0.0;
            u.mouse_attraction_radius = 0.0;
            u.burst_count = 0;
            std::ptr::write(ptr, u);
        }

        // Initialize dead list: counter = pool_size, indices [0..pool_size-1]
        unsafe {
            let ptr = buffer_ptr(&dead_list) as *mut u32;
            std::ptr::write(ptr, pool_size as u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
            }
        }

        // Initialize alive_list_a: counter = 0
        unsafe {
            let ptr = buffer_ptr(&alive_list_a) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), 0u32);
            }
        }

        // Initialize alive_list_b: counter = 0
        unsafe {
            let ptr = buffer_ptr(&alive_list_b) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), 0u32);
            }
        }

        // Zero positions
        unsafe {
            let ptr = buffer_ptr(&positions) as *mut u8;
            std::ptr::write_bytes(ptr, 0, pool_size * 12);
        }

        // Zero grid density
        unsafe {
            let ptr = buffer_ptr(&grid_density) as *mut u8;
            std::ptr::write_bytes(ptr, 0, GRID_CELLS * 4);
        }

        // Zero indirect args
        unsafe {
            let ptr = buffer_ptr(&indirect_args) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
        }

        // GpuEmissionParams: 16 bytes (emission_count, actual_burst_count, _pad0, _pad1)
        // Manually set emission_count = base_emission_rate (100) for test isolation
        let gpu_emission_params = alloc_buffer(device, 16);
        unsafe {
            let ptr = buffer_ptr(&gpu_emission_params) as *mut u32;
            std::ptr::write(ptr, 100u32); // emission_count
            std::ptr::write(ptr.add(1), 0u32); // actual_burst_count
            std::ptr::write(ptr.add(2), 0u32); // _pad0
            std::ptr::write(ptr.add(3), 0u32); // _pad1
        }

        // DispatchArgs for update/grid_populate indirect dispatch: 12 bytes (3 x u32)
        // Zero-init; sync_indirect_args will write correct values
        let update_dispatch_args = alloc_buffer(device, 12);
        unsafe {
            let ptr = buffer_ptr(&update_dispatch_args) as *mut u32;
            std::ptr::write(ptr, 0u32); // threadgroupsPerGridX
            std::ptr::write(ptr.add(1), 1u32); // threadgroupsPerGridY
            std::ptr::write(ptr.add(2), 1u32); // threadgroupsPerGridZ
        }

        Self {
            uniforms,
            dead_list,
            alive_list_a,
            alive_list_b,
            positions,
            velocities,
            lifetimes,
            colors,
            sizes,
            grid_density,
            indirect_args,
            gpu_emission_params,
            update_dispatch_args,
        }
    }
}

/// Dispatch emission using PhysicsBuffers (emission writes to alive_list_a).
fn dispatch_emission_physics(
    ctx: &GpuTestContextFull,
    bufs: &PhysicsBuffers,
    emission_count: u32,
) {
    // Write emission_count into gpu_emission_params for test isolation
    unsafe {
        let ptr = buffer_ptr(&bufs.gpu_emission_params) as *mut u32;
        std::ptr::write(ptr, emission_count); // emission_count
        // actual_burst_count stays as previously initialized (0)
    }

    let cmd_buf = ctx
        .queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&ctx.emission_pipeline);

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_a), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&bufs.gpu_emission_params), 0, 8);
    }

    let threadgroup_size: usize = 256;
    let threadgroup_count = (emission_count as usize).div_ceil(threadgroup_size);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}

/// Dispatch the update_physics_kernel. Reads alive_list_a, writes alive_list_b.
fn dispatch_update(ctx: &GpuTestContextFull, bufs: &PhysicsBuffers, pool_size: usize) {
    let cmd_buf = ctx
        .queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&ctx.update_pipeline);

    // update_physics_kernel buffer bindings (0-9):
    // 0: uniforms, 1: dead_list, 2: alive_list_a (read), 3: alive_list_b (write),
    // 4: positions, 5: velocities, 6: lifetimes, 7: colors, 8: sizes, 9: grid_density
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_a), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_b), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 7);
        encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 8);
        encoder.setBuffer_offset_atIndex(Some(&bufs.grid_density), 0, 9);
    }

    let threadgroup_size: usize = 256;
    let threadgroup_count = pool_size.div_ceil(threadgroup_size);
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: threadgroup_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threadgroup_size,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}

/// Dispatch sync_indirect_args kernel. Reads alive_list counter, writes DrawArgs and DispatchArgs.
fn dispatch_sync_indirect(
    ctx: &GpuTestContextFull,
    alive_list: &ProtocolObject<dyn MTLBuffer>,
    indirect_args: &ProtocolObject<dyn MTLBuffer>,
    update_dispatch_args: &ProtocolObject<dyn MTLBuffer>,
) {
    let cmd_buf = ctx
        .queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(&ctx.sync_indirect_pipeline);

    // sync_indirect_args buffer bindings:
    // 0: alive_list, 1: indirect_args, 2: update_dispatch_args
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(alive_list), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(indirect_args), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(update_dispatch_args), 0, 2);
    }

    // Single thread dispatch
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}

// ---------------------------------------------------------------------------
// test_physics: emit 100 particles, run 1 physics step, verify gravity applied
// ---------------------------------------------------------------------------

/// GPU integration test for the physics update kernel.
///
/// Emits 100 particles, runs one physics step with dt=0.016, and verifies:
/// - Positions y-component decreased (gravity applied)
/// - All 100 particles survive (none dead at age=0.016s with max_age 1-5s)
#[test]
fn test_physics_gpu_integration() {
    let pool_size: usize = 1000;
    let emission_count: u32 = 100;

    let ctx = GpuTestContextFull::new();
    let bufs = PhysicsBuffers::new(&ctx.device, pool_size);

    // Step 1: Emit 100 particles into alive_list_a
    dispatch_emission_physics(&ctx, &bufs, emission_count);

    let alive_after_emission = unsafe { read_counter(&bufs.alive_list_a) };
    assert_eq!(
        alive_after_emission, emission_count,
        "Alive list A should have {} after emission, got {}",
        emission_count, alive_after_emission
    );

    // Record initial y-positions of emitted particles for comparison after physics
    let initial_y_positions: Vec<f32> = unsafe {
        let alive_ptr = bufs.alive_list_a.contents().as_ptr() as *const u32;
        let pos_ptr = bufs.positions.contents().as_ptr() as *const u8;

        (0..emission_count as usize)
            .map(|i| {
                let particle_idx =
                    std::ptr::read(alive_ptr.add(COUNTER_HEADER_UINTS + i)) as usize;
                std::ptr::read((pos_ptr.add(particle_idx * 12 + 4)) as *const f32)
            })
            .collect()
    };

    // Step 2: Run update_physics_kernel (reads alive_list_a, writes alive_list_b)
    dispatch_update(&ctx, &bufs, pool_size);

    // Step 3: Verify results

    // alive_list_b should have all survivors (~100, none dead at age 0.016s)
    let alive_after_update = unsafe { read_counter(&bufs.alive_list_b) };
    assert_eq!(
        alive_after_update, emission_count,
        "All 100 particles should survive after 1 step (age=0.016s << max_age). Got {}",
        alive_after_update
    );

    // Verify y-positions decreased due to gravity (gravity = -9.81)
    // After 1 step: vel.y += -9.81 * 0.016 = -0.157; pos.y += vel.y * 0.016
    // Even with initial upward velocity, gravity should decrease y relative to pre-update
    unsafe {
        let alive_ptr = bufs.alive_list_b.contents().as_ptr() as *const u32;
        let pos_ptr = bufs.positions.contents().as_ptr() as *const u8;

        let mut y_decreased_count = 0u32;
        for i in 0..alive_after_update as usize {
            let particle_idx =
                std::ptr::read(alive_ptr.add(COUNTER_HEADER_UINTS + i)) as usize;
            let new_y = std::ptr::read((pos_ptr.add(particle_idx * 12 + 4)) as *const f32);

            // Find this particle's initial y (particle indices may differ between lists,
            // but since emission assigns sequential indices, we can use particle_idx directly)
            // Initial positions were set by emission kernel; we need to compare by particle index
            // Since alive_list_a had these same indices in possibly different order,
            // we use the initial_y_positions array which was indexed by alive_list_a order.
            // However, alive_list_b may have different ordering due to atomics.
            // Instead, just verify that new_y < initial_y for the same particle_idx.
            // We stored initial_y by alive_list_a order. Find initial_y for this particle_idx:
            let initial_y = {
                let alive_a_ptr = bufs.alive_list_a.contents().as_ptr() as *const u32;
                let mut found_y = None;
                for j in 0..emission_count as usize {
                    let idx =
                        std::ptr::read(alive_a_ptr.add(COUNTER_HEADER_UINTS + j)) as usize;
                    if idx == particle_idx {
                        found_y = Some(initial_y_positions[j]);
                        break;
                    }
                }
                found_y.expect("Particle index from alive_list_b not found in alive_list_a")
            };

            if new_y < initial_y {
                y_decreased_count += 1;
            }
        }

        // With gravity=-9.81 and dt=0.016, most particles should have y decreased.
        // Some particles with strong upward initial velocity may still have y increased,
        // but gravity should pull the majority downward.
        // Expect at least 50% to have decreased y (conservative threshold).
        let threshold = alive_after_update / 2;
        assert!(
            y_decreased_count >= threshold,
            "Expected at least {} particles with decreased y (gravity), got {}/{}",
            threshold,
            y_decreased_count,
            alive_after_update
        );
    }

    println!(
        "test_physics PASSED: alive_after_update={}, gravity verified",
        alive_after_update
    );
}

// ---------------------------------------------------------------------------
// test_compaction: emit 100 particles, force death, verify all dead
// ---------------------------------------------------------------------------

/// GPU integration test for compaction (death via lifetime expiry).
///
/// Emits 100 particles, sets all lifetimes to age==max_age (forcing death),
/// runs update kernel, and verifies:
/// - alive_list_b counter == 0 (all particles dead)
/// - dead_list counter restored (all 100 returned + remaining 900 = 1000)
#[test]
fn test_compaction_gpu_integration() {
    let pool_size: usize = 1000;
    let emission_count: u32 = 100;

    let ctx = GpuTestContextFull::new();
    let bufs = PhysicsBuffers::new(&ctx.device, pool_size);

    // Step 1: Emit 100 particles into alive_list_a
    dispatch_emission_physics(&ctx, &bufs, emission_count);

    let alive_after_emission = unsafe { read_counter(&bufs.alive_list_a) };
    assert_eq!(alive_after_emission, emission_count);

    let dead_after_emission = unsafe { read_counter(&bufs.dead_list) };
    assert_eq!(dead_after_emission, (pool_size as u32) - emission_count);

    // Step 2: Force all emitted particles to die by setting age = max_age.
    // Lifetimes are half2(age, max_age). We set age = max_age for each alive particle.
    // The update kernel checks `if (age >= max_age)` and pushes to dead list.
    unsafe {
        let alive_ptr = bufs.alive_list_a.contents().as_ptr() as *const u32;
        let lt_ptr = bufs.lifetimes.contents().as_ptr() as *mut u16;

        for i in 0..emission_count as usize {
            let particle_idx =
                std::ptr::read(alive_ptr.add(COUNTER_HEADER_UINTS + i)) as usize;

            // Read current max_age (second half of half2)
            let max_age_bits = std::ptr::read(lt_ptr.add(particle_idx * 2 + 1));
            // Set age = max_age (first half of half2)
            std::ptr::write(lt_ptr.add(particle_idx * 2), max_age_bits);
        }
    }

    // Step 3: Run update kernel. All particles should die (age >= max_age after age += dt).
    dispatch_update(&ctx, &bufs, pool_size);

    // Step 4: Verify all particles are dead
    let alive_after_update = unsafe { read_counter(&bufs.alive_list_b) };
    assert_eq!(
        alive_after_update, 0,
        "All particles should be dead after compaction. Got alive={}",
        alive_after_update
    );

    // Dead list should have all pool_size indices restored
    let dead_after_update = unsafe { read_counter(&bufs.dead_list) };
    assert_eq!(
        dead_after_update, pool_size as u32,
        "Dead list should be fully restored to {}. Got {}",
        pool_size, dead_after_update
    );

    println!(
        "test_compaction PASSED: alive={}, dead={} (all particles recycled)",
        alive_after_update, dead_after_update
    );
}

// ---------------------------------------------------------------------------
// test_indirect_draw_args: verify sync_indirect_args writes correct DrawArgs
// ---------------------------------------------------------------------------

/// GPU integration test for indirect draw args synchronization.
///
/// After emission + physics (with survivors), runs sync_indirect_args and verifies:
/// - instanceCount == alive_count from alive_list_b
/// - vertexCount == 4 (billboard quad)
#[test]
fn test_indirect_draw_args_gpu_integration() {
    let pool_size: usize = 1000;
    let emission_count: u32 = 100;

    let ctx = GpuTestContextFull::new();
    let bufs = PhysicsBuffers::new(&ctx.device, pool_size);

    // Step 1: Emit 100 particles
    dispatch_emission_physics(&ctx, &bufs, emission_count);

    // Step 2: Run physics (all should survive at age=0.016s)
    dispatch_update(&ctx, &bufs, pool_size);

    let alive_count = unsafe { read_counter(&bufs.alive_list_b) };
    assert_eq!(
        alive_count, emission_count,
        "Expected {} alive after physics, got {}",
        emission_count, alive_count
    );

    // Step 3: Run sync_indirect_args (reads alive_list_b counter, writes indirect_args + update_dispatch_args)
    dispatch_sync_indirect(&ctx, &bufs.alive_list_b, &bufs.indirect_args, &bufs.update_dispatch_args);

    // Step 4: Read back indirect args and verify
    unsafe {
        let args_ptr = bufs.indirect_args.contents().as_ptr() as *const u32;
        let vertex_count = std::ptr::read(args_ptr);
        let instance_count = std::ptr::read(args_ptr.add(1));
        let vertex_start = std::ptr::read(args_ptr.add(2));
        let base_instance = std::ptr::read(args_ptr.add(3));

        assert_eq!(
            vertex_count, 4,
            "vertexCount should be 4 (billboard quad), got {}",
            vertex_count
        );
        assert_eq!(
            instance_count, alive_count,
            "instanceCount should be {} (alive particles), got {}",
            alive_count, instance_count
        );
        assert_eq!(
            vertex_start, 0,
            "vertexStart should be 0, got {}",
            vertex_start
        );
        assert_eq!(
            base_instance, 0,
            "baseInstance should be 0, got {}",
            base_instance
        );

        println!(
            "test_indirect_draw_args PASSED: vertexCount={}, instanceCount={}, vertexStart={}, baseInstance={}",
            vertex_count, instance_count, vertex_start, base_instance
        );
    }
}

// ---------------------------------------------------------------------------
// prepare_dispatch correctness tests
// ---------------------------------------------------------------------------

/// Helper: dispatch prepare_dispatch kernel and wait.
fn dispatch_prepare_dispatch(
    _device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    dead_list: &ProtocolObject<dyn MTLBuffer>,
    uniforms: &ProtocolObject<dyn MTLBuffer>,
    write_list: &ProtocolObject<dyn MTLBuffer>,
    emission_dispatch_args: &ProtocolObject<dyn MTLBuffer>,
    gpu_emission_params: &ProtocolObject<dyn MTLBuffer>,
) {
    let cmd_buf = queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    let encoder = cmd_buf
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    encoder.setComputePipelineState(pipeline);

    // buffer(0)=dead_list, buffer(1)=uniforms, buffer(2)=write_list,
    // buffer(3)=emission_dispatch_args, buffer(4)=gpu_emission_params
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(dead_list), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(uniforms), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(write_list), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(emission_dispatch_args), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(gpu_emission_params), 0, 4);
    }

    // Single-thread dispatch: 1 threadgroup of 32 threads (SIMD-aligned)
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    encoder.endEncoding();
    cmd_buf.commit();
    cmd_buf.waitUntilCompleted();
}

/// GPU integration test: prepare_dispatch correctness.
///
/// Verifies that the prepare_dispatch kernel correctly:
/// - Clamps emission_count to dead_count when base_emission_rate > dead_count
/// - Computes correct threadgroupsPerGridX = ceil(emission_count / 256)
/// - Resets write_list counter to 0
#[test]
fn test_prepare_dispatch_correctness() {
    let pool_size: usize = 1000;

    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    let metallib_path = find_metallib();
    let path_ns = NSString::from_str(&metallib_path);
    #[allow(deprecated)]
    let library = device
        .newLibraryWithFile_error(&path_ns)
        .expect("Failed to load shaders.metallib");

    let prepare_pipeline = create_pipeline(&device, &library, "prepare_dispatch");

    let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

    // Allocate buffers
    let dead_list = alloc_buffer(&device, counter_list_size);
    let uniforms_buf = alloc_buffer(&device, 256);
    let write_list = alloc_buffer(&device, counter_list_size);
    let emission_dispatch_args = alloc_buffer(&device, 12); // DispatchArgs: 3 x u32
    let gpu_emission_params = alloc_buffer(&device, 16); // GpuEmissionParams: 4 x u32

    // Initialize dead_list: counter = 500, fill 500 indices
    unsafe {
        let ptr = buffer_ptr(&dead_list) as *mut u32;
        std::ptr::write(ptr, 500u32); // dead_count = 500
        std::ptr::write(ptr.add(1), 0u32);
        std::ptr::write(ptr.add(2), 0u32);
        std::ptr::write(ptr.add(3), 0u32);
        for i in 0..500usize {
            std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
        }
    }

    // Initialize uniforms: base_emission_rate=1000, burst_count=0, pool_size=1000
    unsafe {
        let ptr = buffer_ptr(&uniforms_buf) as *mut Uniforms;
        let mut u = Uniforms::default();
        u.base_emission_rate = 1000;
        u.burst_count = 0;
        u.pool_size = pool_size as u32;
        std::ptr::write(ptr, u);
    }

    // Set write_list counter to a non-zero value to verify reset
    unsafe {
        let ptr = buffer_ptr(&write_list) as *mut u32;
        std::ptr::write(ptr, 42u32); // will be reset to 0 by kernel
    }

    // Dispatch prepare_dispatch
    dispatch_prepare_dispatch(
        &device,
        &queue,
        &prepare_pipeline,
        &dead_list,
        &uniforms_buf,
        &write_list,
        &emission_dispatch_args,
        &gpu_emission_params,
    );

    // Read back and verify
    unsafe {
        // gpu_emission_params: emission_count should be clamped to dead_count = 500
        let params_ptr = gpu_emission_params.contents().as_ptr() as *const u32;
        let emission_count = std::ptr::read(params_ptr);
        assert_eq!(
            emission_count, 500,
            "emission_count should be min(1000+0, 500) = 500, got {}",
            emission_count
        );

        // emission_dispatch_args: threadgroupsPerGridX = ceil(500/256) = 2
        let args_ptr = emission_dispatch_args.contents().as_ptr() as *const u32;
        let threadgroups_x = std::ptr::read(args_ptr);
        assert_eq!(
            threadgroups_x, 2,
            "threadgroupsPerGridX should be ceil(500/256) = 2, got {}",
            threadgroups_x
        );

        // write_list counter should be reset to 0
        let write_ptr = write_list.contents().as_ptr() as *const u32;
        let write_counter = std::ptr::read(write_ptr);
        assert_eq!(
            write_counter, 0,
            "write_list counter should be reset to 0, got {}",
            write_counter
        );
    }

    println!(
        "test_prepare_dispatch_correctness PASSED: emission_count=500, threadgroupsX=2, write_list reset"
    );
}

/// GPU integration test: prepare_dispatch with zero dead particles.
///
/// Verifies that when dead_count=0, emission_count=0 and threadgroups=0.
#[test]
fn test_prepare_dispatch_zero_dead() {
    let pool_size: usize = 1000;

    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    let metallib_path = find_metallib();
    let path_ns = NSString::from_str(&metallib_path);
    #[allow(deprecated)]
    let library = device
        .newLibraryWithFile_error(&path_ns)
        .expect("Failed to load shaders.metallib");

    let prepare_pipeline = create_pipeline(&device, &library, "prepare_dispatch");

    let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

    // Allocate buffers
    let dead_list = alloc_buffer(&device, counter_list_size);
    let uniforms_buf = alloc_buffer(&device, 256);
    let write_list = alloc_buffer(&device, counter_list_size);
    let emission_dispatch_args = alloc_buffer(&device, 12);
    let gpu_emission_params = alloc_buffer(&device, 16);

    // Initialize dead_list: counter = 0 (no dead particles)
    unsafe {
        let ptr = buffer_ptr(&dead_list) as *mut u32;
        std::ptr::write(ptr, 0u32); // dead_count = 0
        std::ptr::write(ptr.add(1), 0u32);
        std::ptr::write(ptr.add(2), 0u32);
        std::ptr::write(ptr.add(3), 0u32);
    }

    // Initialize uniforms: base_emission_rate=10000, burst_count=0, pool_size=1000
    unsafe {
        let ptr = buffer_ptr(&uniforms_buf) as *mut Uniforms;
        let mut u = Uniforms::default();
        u.base_emission_rate = 10000;
        u.burst_count = 0;
        u.pool_size = pool_size as u32;
        std::ptr::write(ptr, u);
    }

    // Set write_list counter to non-zero
    unsafe {
        let ptr = buffer_ptr(&write_list) as *mut u32;
        std::ptr::write(ptr, 99u32);
    }

    // Dispatch prepare_dispatch
    dispatch_prepare_dispatch(
        &device,
        &queue,
        &prepare_pipeline,
        &dead_list,
        &uniforms_buf,
        &write_list,
        &emission_dispatch_args,
        &gpu_emission_params,
    );

    // Read back and verify
    unsafe {
        // emission_count = min(10000+0, 0) = 0
        let params_ptr = gpu_emission_params.contents().as_ptr() as *const u32;
        let emission_count = std::ptr::read(params_ptr);
        assert_eq!(
            emission_count, 0,
            "emission_count should be 0 with zero dead, got {}",
            emission_count
        );

        // threadgroupsPerGridX = ceil(0/256) = 0
        let args_ptr = emission_dispatch_args.contents().as_ptr() as *const u32;
        let threadgroups_x = std::ptr::read(args_ptr);
        assert_eq!(
            threadgroups_x, 0,
            "threadgroupsPerGridX should be 0 with zero emission, got {}",
            threadgroups_x
        );
    }

    println!("test_prepare_dispatch_zero_dead PASSED: emission_count=0, threadgroupsX=0");
}

/// GPU integration test: prepare_dispatch burst clamping.
///
/// Verifies that burst_count is correctly clamped when dead_count < base+burst.
/// dead_count=100, base=50, burst=200:
///   emission_count = min(50+200, 100) = 100
///   actual_burst = min(200, 100) = 100
#[test]
fn test_prepare_dispatch_burst_clamping() {
    let pool_size: usize = 1000;

    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    let metallib_path = find_metallib();
    let path_ns = NSString::from_str(&metallib_path);
    #[allow(deprecated)]
    let library = device
        .newLibraryWithFile_error(&path_ns)
        .expect("Failed to load shaders.metallib");

    let prepare_pipeline = create_pipeline(&device, &library, "prepare_dispatch");

    let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

    // Allocate buffers
    let dead_list = alloc_buffer(&device, counter_list_size);
    let uniforms_buf = alloc_buffer(&device, 256);
    let write_list = alloc_buffer(&device, counter_list_size);
    let emission_dispatch_args = alloc_buffer(&device, 12);
    let gpu_emission_params = alloc_buffer(&device, 16);

    // Initialize dead_list: counter = 100
    unsafe {
        let ptr = buffer_ptr(&dead_list) as *mut u32;
        std::ptr::write(ptr, 100u32); // dead_count = 100
        std::ptr::write(ptr.add(1), 0u32);
        std::ptr::write(ptr.add(2), 0u32);
        std::ptr::write(ptr.add(3), 0u32);
        for i in 0..100usize {
            std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
        }
    }

    // Initialize uniforms: base_emission_rate=50, burst_count=200, pool_size=1000
    unsafe {
        let ptr = buffer_ptr(&uniforms_buf) as *mut Uniforms;
        let mut u = Uniforms::default();
        u.base_emission_rate = 50;
        u.burst_count = 200;
        u.pool_size = pool_size as u32;
        std::ptr::write(ptr, u);
    }

    // Dispatch prepare_dispatch
    dispatch_prepare_dispatch(
        &device,
        &queue,
        &prepare_pipeline,
        &dead_list,
        &uniforms_buf,
        &write_list,
        &emission_dispatch_args,
        &gpu_emission_params,
    );

    // Read back and verify
    unsafe {
        let params_ptr = gpu_emission_params.contents().as_ptr() as *const u32;

        // emission_count = min(50+200, 100) = 100
        let emission_count = std::ptr::read(params_ptr);
        assert_eq!(
            emission_count, 100,
            "emission_count should be min(50+200, 100) = 100, got {}",
            emission_count
        );

        // actual_burst = min(200, 100) = 100
        let actual_burst = std::ptr::read(params_ptr.add(1));
        assert_eq!(
            actual_burst, 100,
            "actual_burst should be min(200, 100) = 100, got {}",
            actual_burst
        );

        // threadgroupsPerGridX = ceil(100/256) = 1
        let args_ptr = emission_dispatch_args.contents().as_ptr() as *const u32;
        let threadgroups_x = std::ptr::read(args_ptr);
        assert_eq!(
            threadgroups_x, 1,
            "threadgroupsPerGridX should be ceil(100/256) = 1, got {}",
            threadgroups_x
        );
    }

    println!(
        "test_prepare_dispatch_burst_clamping PASSED: emission_count=100, actual_burst=100, threadgroupsX=1"
    );
}

// ---------------------------------------------------------------------------
// test_sync_indirect_writes_update_dispatch_args: verify DispatchArgs output
// ---------------------------------------------------------------------------

/// GPU integration test for sync_indirect_args writing update_dispatch_args.
///
/// Emits 100 particles, runs physics (all survive), runs sync_indirect_args,
/// then reads back update_dispatch_args and verifies:
/// - threadgroupsPerGridX == ceil(100 / 256) == 1
/// - threadgroupsPerGridY == 1
/// - threadgroupsPerGridZ == 1
#[test]
fn test_sync_indirect_writes_update_dispatch_args() {
    let pool_size: usize = 1000;
    let emission_count: u32 = 100;

    let ctx = GpuTestContextFull::new();
    let bufs = PhysicsBuffers::new(&ctx.device, pool_size);

    // Step 1: Emit 100 particles into alive_list_a
    dispatch_emission_physics(&ctx, &bufs, emission_count);

    let alive_after_emission = unsafe { read_counter(&bufs.alive_list_a) };
    assert_eq!(
        alive_after_emission, emission_count,
        "Alive list A should have {} after emission, got {}",
        emission_count, alive_after_emission
    );

    // Step 2: Run physics (reads alive_list_a, writes alive_list_b; all survive at age=0.016s)
    dispatch_update(&ctx, &bufs, pool_size);

    let alive_after_update = unsafe { read_counter(&bufs.alive_list_b) };
    assert_eq!(
        alive_after_update, emission_count,
        "All {} particles should survive after 1 step, got {}",
        emission_count, alive_after_update
    );

    // Step 3: Run sync_indirect_args (reads alive_list_b, writes indirect_args + update_dispatch_args)
    dispatch_sync_indirect(&ctx, &bufs.alive_list_b, &bufs.indirect_args, &bufs.update_dispatch_args);

    // Step 4: Read back update_dispatch_args and verify
    unsafe {
        let args_ptr = bufs.update_dispatch_args.contents().as_ptr() as *const u32;
        let threadgroups_x = std::ptr::read(args_ptr);
        let threadgroups_y = std::ptr::read(args_ptr.add(1));
        let threadgroups_z = std::ptr::read(args_ptr.add(2));

        // ceil(100 / 256) == 1, but shader uses max((alive_count + 255) / 256, 1) == 1
        let expected_x = (emission_count as u32 + 255) / 256;
        assert_eq!(
            threadgroups_x, expected_x,
            "threadgroupsPerGridX should be ceil({}/256) = {}, got {}",
            emission_count, expected_x, threadgroups_x
        );
        assert_eq!(
            threadgroups_y, 1,
            "threadgroupsPerGridY should be 1, got {}",
            threadgroups_y
        );
        assert_eq!(
            threadgroups_z, 1,
            "threadgroupsPerGridZ should be 1, got {}",
            threadgroups_z
        );

        println!(
            "test_sync_indirect_writes_update_dispatch_args PASSED: threadgroups=[{}, {}, {}]",
            threadgroups_x, threadgroups_y, threadgroups_z
        );
    }
}

// ---------------------------------------------------------------------------
// Full pipeline context: all 6 compute pipelines for round-trip testing
// ---------------------------------------------------------------------------

/// Complete GPU test context with all compute pipelines for full pipeline tests.
pub struct GpuTestContextFullPipeline {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub prepare_dispatch_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub emission_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grid_clear_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub grid_populate_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub update_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub sync_indirect_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuTestContextFullPipeline {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
        let queue = device
            .newCommandQueue()
            .expect("Failed to create command queue");

        let metallib_path = find_metallib();
        let path_ns = NSString::from_str(&metallib_path);
        #[allow(deprecated)]
        let library = device
            .newLibraryWithFile_error(&path_ns)
            .expect("Failed to load shaders.metallib");

        let prepare_dispatch_pipeline = create_pipeline(&device, &library, "prepare_dispatch");
        let emission_pipeline = create_pipeline(&device, &library, "emission_kernel");
        let grid_clear_pipeline = create_pipeline(&device, &library, "grid_clear_kernel");
        let grid_populate_pipeline = create_pipeline(&device, &library, "grid_populate_kernel");
        let update_pipeline = create_pipeline(&device, &library, "update_physics_kernel");
        let sync_indirect_pipeline = create_pipeline(&device, &library, "sync_indirect_args");

        Self {
            device,
            queue,
            prepare_dispatch_pipeline,
            emission_pipeline,
            grid_clear_pipeline,
            grid_populate_pipeline,
            update_pipeline,
            sync_indirect_pipeline,
        }
    }
}

/// Extended buffers with emission_dispatch_args for full pipeline tests.
pub struct FullPipelineBuffers {
    pub uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub dead_list: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub alive_list_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub alive_list_b: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub velocities: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub lifetimes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub colors: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub sizes: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub grid_density: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub indirect_args: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub update_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub emission_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl FullPipelineBuffers {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, pool_size: usize) -> Self {
        let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

        let uniforms = alloc_buffer(device, 256);
        let dead_list = alloc_buffer(device, counter_list_size);
        let alive_list_a = alloc_buffer(device, counter_list_size);
        let alive_list_b = alloc_buffer(device, counter_list_size);
        let positions = alloc_buffer(device, pool_size * 12);
        let velocities = alloc_buffer(device, pool_size * 12);
        let lifetimes = alloc_buffer(device, pool_size * 4);
        let colors = alloc_buffer(device, pool_size * 8);
        let sizes = alloc_buffer(device, pool_size * 4);
        let grid_density = alloc_buffer(device, GRID_CELLS * 4);
        let indirect_args = alloc_buffer(device, 16);
        let gpu_emission_params = alloc_buffer(device, 16);
        let update_dispatch_args = alloc_buffer(device, 12);
        let emission_dispatch_args = alloc_buffer(device, 12);

        // Initialize uniforms
        unsafe {
            let ptr = buffer_ptr(&uniforms) as *mut Uniforms;
            let mut u = Uniforms::default();
            u.base_emission_rate = 100;
            u.pool_size = pool_size as u32;
            u.dt = 0.016;
            u.gravity = -9.81;
            u.drag_coefficient = 0.02;
            u.interaction_strength = 0.0;
            u.mouse_attraction_strength = 0.0;
            u.mouse_attraction_radius = 0.0;
            u.burst_count = 0;
            std::ptr::write(ptr, u);
        }

        // Initialize dead list: counter = pool_size, indices [0..pool_size-1]
        unsafe {
            let ptr = buffer_ptr(&dead_list) as *mut u32;
            std::ptr::write(ptr, pool_size as u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
            }
        }

        // Initialize alive_list_a: counter = 0
        unsafe {
            let ptr = buffer_ptr(&alive_list_a) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), 0u32);
            }
        }

        // Initialize alive_list_b: counter = 0
        unsafe {
            let ptr = buffer_ptr(&alive_list_b) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
            for i in 0..pool_size {
                std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), 0u32);
            }
        }

        // Zero positions
        unsafe {
            let ptr = buffer_ptr(&positions) as *mut u8;
            std::ptr::write_bytes(ptr, 0, pool_size * 12);
        }

        // Zero grid density
        unsafe {
            let ptr = buffer_ptr(&grid_density) as *mut u8;
            std::ptr::write_bytes(ptr, 0, GRID_CELLS * 4);
        }

        // Zero indirect args
        unsafe {
            let ptr = buffer_ptr(&indirect_args) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
        }

        // Zero gpu_emission_params
        unsafe {
            let ptr = buffer_ptr(&gpu_emission_params) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 0u32);
            std::ptr::write(ptr.add(2), 0u32);
            std::ptr::write(ptr.add(3), 0u32);
        }

        // Bootstrap update_dispatch_args: pool_size/256 threadgroups for first frame
        unsafe {
            let ptr = buffer_ptr(&update_dispatch_args) as *mut u32;
            std::ptr::write(ptr, ((pool_size + 255) / 256) as u32);
            std::ptr::write(ptr.add(1), 1u32);
            std::ptr::write(ptr.add(2), 1u32);
        }

        // Zero emission_dispatch_args (prepare_dispatch will write correct values)
        unsafe {
            let ptr = buffer_ptr(&emission_dispatch_args) as *mut u32;
            std::ptr::write(ptr, 0u32);
            std::ptr::write(ptr.add(1), 1u32);
            std::ptr::write(ptr.add(2), 1u32);
        }

        Self {
            uniforms,
            dead_list,
            alive_list_a,
            alive_list_b,
            positions,
            velocities,
            lifetimes,
            colors,
            sizes,
            grid_density,
            indirect_args,
            gpu_emission_params,
            update_dispatch_args,
            emission_dispatch_args,
        }
    }
}

// ---------------------------------------------------------------------------
// test_indirect_dispatch_round_trip: full 6-kernel pipeline
// ---------------------------------------------------------------------------

/// GPU integration test: indirect dispatch round-trip.
///
/// Runs the full 6-kernel pipeline:
///   prepare_dispatch -> emission (indirect) -> grid_clear -> grid_populate -> update -> sync_indirect_args
///
/// Pool size 1000, base_emission_rate=100, burst=0.
/// After full pipeline:
///   - Read indirect_args.instanceCount: verify alive count > 0 and <= 100
///   - Verify conservation: alive_count + dead_count == pool_size
#[test]
fn test_indirect_dispatch_round_trip() {
    let pool_size: usize = 1000;

    let ctx = GpuTestContextFullPipeline::new();
    let bufs = FullPipelineBuffers::new(&ctx.device, pool_size);

    // --- Step 1: prepare_dispatch ---
    // Reads dead_list counter, computes emission params, writes emission_dispatch_args,
    // resets alive_list_b counter (write_list) to 0.
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.prepare_dispatch_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_b), 0, 2); // write_list
            encoder.setBuffer_offset_atIndex(Some(&bufs.emission_dispatch_args), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&bufs.gpu_emission_params), 0, 4);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 32, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // Verify prepare_dispatch set emission_count correctly
    let emission_count = unsafe {
        let ptr = bufs.gpu_emission_params.contents().as_ptr() as *const u32;
        std::ptr::read(ptr)
    };
    assert_eq!(
        emission_count, 100,
        "prepare_dispatch should set emission_count = min(100+0, 1000) = 100, got {}",
        emission_count
    );

    // --- Step 2: emission (indirect dispatch) ---
    // Uses emission_dispatch_args for threadgroup count (computed by prepare_dispatch).
    // Emission writes to alive_list_a (the read_list for this first frame).
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.emission_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_a), 0, 2); // emission target
            encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&bufs.gpu_emission_params), 0, 8);
            // Indirect dispatch using emission_dispatch_args
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &bufs.emission_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // Verify emission happened
    let alive_after_emission = unsafe { read_counter(&bufs.alive_list_a) };
    assert_eq!(
        alive_after_emission, 100,
        "Emission should produce 100 alive particles, got {}",
        alive_after_emission
    );

    // --- Step 3: grid_clear ---
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.grid_clear_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.grid_density), 0, 0);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1024, height: 1, depth: 1 },
            MTLSize { width: 256, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // --- Step 4: grid_populate (indirect dispatch via update_dispatch_args) ---
    // grid_populate reads alive_list_a (read_list) and writes to grid_density.
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.grid_populate_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_a), 0, 1); // read_list
            encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&bufs.grid_density), 0, 3);
            // Indirect dispatch using update_dispatch_args (bootstrap value)
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &bufs.update_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // --- Step 5: update (indirect dispatch via update_dispatch_args) ---
    // Reads alive_list_a, writes survivors to alive_list_b.
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.update_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_a), 0, 2); // read
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_b), 0, 3); // write
            encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 6);
            encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 7);
            encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 8);
            encoder.setBuffer_offset_atIndex(Some(&bufs.grid_density), 0, 9);
            // Indirect dispatch using update_dispatch_args
            encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                &bufs.update_dispatch_args,
                0,
                MTLSize { width: 256, height: 1, depth: 1 },
            );
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // --- Step 6: sync_indirect_args ---
    // Reads alive_list_b counter, writes DrawArgs and update_dispatch_args.
    {
        let cmd_buf = ctx.queue.commandBuffer().expect("cmd buf");
        let encoder = cmd_buf.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&ctx.sync_indirect_pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list_b), 0, 0); // write_list
            encoder.setBuffer_offset_atIndex(Some(&bufs.indirect_args), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&bufs.update_dispatch_args), 0, 2);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: 1, height: 1, depth: 1 },
            MTLSize { width: 1, height: 1, depth: 1 },
        );
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
    }

    // --- Verify results ---

    // Read alive count from indirect_args (DrawArgs.instanceCount at offset 1)
    let alive_count = unsafe {
        let ptr = bufs.indirect_args.contents().as_ptr() as *const u32;
        std::ptr::read(ptr.add(1)) // instanceCount
    };

    assert!(
        alive_count > 0,
        "alive_count should be > 0 after full pipeline, got 0",
    );
    assert!(
        alive_count <= 100,
        "alive_count should be <= 100 (emission_count), got {}",
        alive_count
    );

    // Read dead_count from dead_list counter
    let dead_count = unsafe { read_counter(&bufs.dead_list) };

    // Conservation invariant: alive_count + dead_count == pool_size
    assert_eq!(
        alive_count + dead_count,
        pool_size as u32,
        "Conservation invariant failed: alive({}) + dead({}) = {} != pool_size({})",
        alive_count,
        dead_count,
        alive_count + dead_count,
        pool_size
    );

    println!(
        "test_indirect_dispatch_round_trip PASSED: alive={}, dead={}, conservation OK (sum={})",
        alive_count, dead_count, alive_count + dead_count
    );
}

// ---------------------------------------------------------------------------
// test_write_list_reset_by_gpu: verify prepare_dispatch resets write_list counter
// ---------------------------------------------------------------------------

/// GPU integration test: write_list counter reset by prepare_dispatch.
///
/// Manually sets write_list counter to 999, runs prepare_dispatch,
/// verifies counter is reset to 0.
#[test]
fn test_write_list_reset_by_gpu() {
    let pool_size: usize = 1000;

    let device = MTLCreateSystemDefaultDevice().expect("No Metal device available");
    let queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    let metallib_path = find_metallib();
    let path_ns = NSString::from_str(&metallib_path);
    #[allow(deprecated)]
    let library = device
        .newLibraryWithFile_error(&path_ns)
        .expect("Failed to load shaders.metallib");

    let prepare_pipeline = create_pipeline(&device, &library, "prepare_dispatch");

    let counter_list_size = COUNTER_HEADER_SIZE + pool_size * 4;

    // Allocate buffers
    let dead_list = alloc_buffer(&device, counter_list_size);
    let uniforms_buf = alloc_buffer(&device, 256);
    let write_list = alloc_buffer(&device, counter_list_size);
    let emission_dispatch_args = alloc_buffer(&device, 12);
    let gpu_emission_params = alloc_buffer(&device, 16);

    // Initialize dead_list with some particles
    unsafe {
        let ptr = buffer_ptr(&dead_list) as *mut u32;
        std::ptr::write(ptr, 500u32);
        std::ptr::write(ptr.add(1), 0u32);
        std::ptr::write(ptr.add(2), 0u32);
        std::ptr::write(ptr.add(3), 0u32);
        for i in 0..500usize {
            std::ptr::write(ptr.add(COUNTER_HEADER_UINTS + i), i as u32);
        }
    }

    // Initialize uniforms
    unsafe {
        let ptr = buffer_ptr(&uniforms_buf) as *mut Uniforms;
        let mut u = Uniforms::default();
        u.base_emission_rate = 100;
        u.pool_size = pool_size as u32;
        u.burst_count = 0;
        std::ptr::write(ptr, u);
    }

    // Set write_list counter to 999 manually (the value we want the GPU to reset)
    unsafe {
        let ptr = buffer_ptr(&write_list) as *mut u32;
        std::ptr::write(ptr, 999u32);
    }

    // Verify it's set
    let counter_before = unsafe { read_counter(&write_list) };
    assert_eq!(counter_before, 999, "write_list counter should be 999 before prepare_dispatch");

    // Dispatch prepare_dispatch
    dispatch_prepare_dispatch(
        &device,
        &queue,
        &prepare_pipeline,
        &dead_list,
        &uniforms_buf,
        &write_list,
        &emission_dispatch_args,
        &gpu_emission_params,
    );

    // Read back write_list counter: should be 0
    let counter_after = unsafe { read_counter(&write_list) };
    assert_eq!(
        counter_after, 0,
        "write_list counter should be reset to 0 by prepare_dispatch, got {}",
        counter_after
    );

    println!(
        "test_write_list_reset_by_gpu PASSED: counter before={}, after={}",
        counter_before, counter_after
    );
}
