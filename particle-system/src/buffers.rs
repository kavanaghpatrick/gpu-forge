/// GPU buffer pool for 1M+ particle SoA data, dead/alive lists, and indirect args.
///
/// All buffers use `MTLResourceStorageModeShared` for CPU+GPU access on Apple Silicon
/// unified memory. Dead list is initialized with all indices; alive lists start empty.

use std::ffi::c_void;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::types::{BufferSizes, CounterHeader, DrawArgs, Uniforms, COUNTER_HEADER_SIZE};

/// Particle pool holding all SoA buffers, free lists, and indirect draw args.
#[allow(dead_code)]
pub struct ParticlePool {
    /// Pool capacity (number of particles).
    pub pool_size: usize,

    // --- SoA particle attribute buffers ---
    /// float3 per particle (12 bytes each).
    pub positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// float3 per particle (12 bytes each).
    pub velocities: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half2 per particle (4 bytes each): (age, max_age).
    pub lifetimes: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half4 per particle (8 bytes each): (r, g, b, a).
    pub colors: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half per particle, padded to 4 bytes each.
    pub sizes: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Free / alive lists ---
    /// Dead list: 16B counter header + pool_size * 4B indices.
    pub dead_list: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Alive list A (ping): same layout as dead list.
    pub alive_list_a: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Alive list B (pong): same layout as dead list.
    pub alive_list_b: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Indirect draw / dispatch ---
    /// Indirect draw arguments (MTLDrawPrimitivesIndirectArguments, 32 bytes).
    pub indirect_args: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Uniforms ---
    /// Uniforms buffer (256 bytes padded).
    pub uniforms: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Allocate a Metal buffer of `size` bytes with shared storage mode.
#[allow(dead_code)]
fn alloc_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    size: usize,
    label: &str,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
        .unwrap_or_else(|| panic!("Failed to allocate Metal buffer '{}' ({} bytes)", label, size))
}

/// Get a mutable raw pointer to buffer contents.
///
/// # Safety
/// Caller must ensure exclusive access and correct type alignment.
#[allow(dead_code)]
unsafe fn buffer_ptr(buffer: &ProtocolObject<dyn MTLBuffer>) -> *mut c_void {
    buffer.contents().as_ptr()
}

#[allow(dead_code)]
impl ParticlePool {
    /// Create a new particle pool with `pool_size` capacity.
    ///
    /// Allocates all SoA buffers, dead/alive lists, indirect args, and uniforms.
    /// Initializes dead list with all indices [0..pool_size-1], alive lists empty,
    /// and indirect args to default draw state.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, pool_size: usize) -> Self {
        let sizes = BufferSizes::new(pool_size);

        // Allocate SoA attribute buffers
        let positions = alloc_buffer(device, sizes.positions, "positions");
        let velocities = alloc_buffer(device, sizes.velocities, "velocities");
        let lifetimes = alloc_buffer(device, sizes.lifetimes, "lifetimes");
        let colors = alloc_buffer(device, sizes.colors, "colors");
        let particle_sizes = alloc_buffer(device, sizes.sizes, "sizes");

        // Allocate counter + index list buffers
        let dead_list = alloc_buffer(device, sizes.counter_list, "dead_list");
        let alive_list_a = alloc_buffer(device, sizes.counter_list, "alive_list_a");
        let alive_list_b = alloc_buffer(device, sizes.counter_list, "alive_list_b");

        // Allocate indirect args and uniforms
        let indirect_args = alloc_buffer(device, sizes.indirect_args, "indirect_args");
        let uniforms = alloc_buffer(device, sizes.uniforms, "uniforms");

        let pool = Self {
            pool_size,
            positions,
            velocities,
            lifetimes,
            colors,
            sizes: particle_sizes,
            dead_list,
            alive_list_a,
            alive_list_b,
            indirect_args,
            uniforms,
        };

        // Initialize buffer contents on CPU
        pool.init_dead_list();
        pool.init_alive_list(&pool.alive_list_a);
        pool.init_alive_list(&pool.alive_list_b);
        pool.init_indirect_args();
        pool.init_uniforms();

        // Log total allocated memory
        let total = sizes.total_bytes();
        println!(
            "ParticlePool allocated: {:.1} MB ({} bytes) for {} particles",
            total as f64 / (1024.0 * 1024.0),
            total,
            pool_size
        );
        assert!(
            total < 200 * 1024 * 1024,
            "Total allocation {} MB exceeds 200 MB limit",
            total / (1024 * 1024)
        );

        pool
    }

    /// Initialize dead list: counter = pool_size, indices = [0, 1, 2, ..., pool_size-1].
    fn init_dead_list(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.dead_list);

            // Write counter header
            let header = ptr as *mut CounterHeader;
            std::ptr::write(
                header,
                CounterHeader {
                    count: self.pool_size as u32,
                    _pad: [0; 3],
                },
            );

            // Write indices starting at offset 16
            let indices_ptr =
                (ptr as *mut u8).add(COUNTER_HEADER_SIZE) as *mut u32;
            let indices =
                std::slice::from_raw_parts_mut(indices_ptr, self.pool_size);
            for (i, slot) in indices.iter_mut().enumerate() {
                *slot = i as u32;
            }
        }
    }

    /// Initialize an alive list: counter = 0, indices zeroed.
    fn init_alive_list(&self, buffer: &ProtocolObject<dyn MTLBuffer>) {
        unsafe {
            let ptr = buffer_ptr(buffer);

            // Write counter header with count = 0
            let header = ptr as *mut CounterHeader;
            std::ptr::write(
                header,
                CounterHeader {
                    count: 0,
                    _pad: [0; 3],
                },
            );

            // Zero out indices
            let indices_ptr =
                (ptr as *mut u8).add(COUNTER_HEADER_SIZE) as *mut u32;
            let indices =
                std::slice::from_raw_parts_mut(indices_ptr, self.pool_size);
            for slot in indices.iter_mut() {
                *slot = 0;
            }
        }
    }

    /// Initialize indirect args: vertexCount=4, instanceCount=0, vertexStart=0, baseInstance=0.
    fn init_indirect_args(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.indirect_args) as *mut DrawArgs;
            std::ptr::write(ptr, DrawArgs::default());
        }
    }

    /// Initialize uniforms buffer with default values.
    fn init_uniforms(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.uniforms) as *mut Uniforms;
            std::ptr::write(ptr, Uniforms::default());
        }
    }
}
