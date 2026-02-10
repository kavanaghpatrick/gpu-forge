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
    emission_count: u32,
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
            emission_count: 100,
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

        // Initialize uniforms with emission_count and pool_size
        unsafe {
            let ptr = buffer_ptr(&uniforms) as *mut Uniforms;
            let mut u = Uniforms::default();
            u.emission_count = 100;
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

        Self {
            uniforms,
            dead_list,
            alive_list,
            positions,
            velocities,
            lifetimes,
            colors,
            sizes,
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
    // buffer(6) = colors, buffer(7) = sizes
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&bufs.uniforms), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&bufs.dead_list), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&bufs.alive_list), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&bufs.positions), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&bufs.velocities), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&bufs.lifetimes), 0, 5);
        encoder.setBuffer_offset_atIndex(Some(&bufs.colors), 0, 6);
        encoder.setBuffer_offset_atIndex(Some(&bufs.sizes), 0, 7);
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
