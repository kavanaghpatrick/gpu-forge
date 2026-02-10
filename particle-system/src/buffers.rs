//! GPU buffer pool for 1M+ particle SoA data, dead/alive lists, and indirect args.
//!
//! All buffers use `MTLResourceStorageModeShared` for CPU+GPU access on Apple Silicon
//! unified memory. Dead list is initialized with all indices; alive lists start empty.

use std::ffi::c_void;
use std::ptr;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::types::{BufferSizes, CounterHeader, DispatchArgs, DrawArgs, GpuEmissionParams, Uniforms, COUNTER_HEADER_SIZE};

/// Particle pool holding all SoA buffers, free lists, and indirect draw args.
#[allow(dead_code)]
pub struct ParticlePool {
    /// Pool capacity (number of particles).
    pub pool_size: usize,

    /// Reference to the Metal device for buffer allocation during grow().
    device: Retained<ProtocolObject<dyn MTLDevice>>,

    // --- SoA particle attribute buffers ---
    /// float3 per particle (12 bytes each).
    pub positions: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// float3 per particle (12 bytes each).
    pub velocities: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half2 per particle (4 bytes each): (age, max_age).
    pub lifetimes: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half4 per particle (8 bytes each): (r, g, b, a).
    pub colors: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// half per particle (2 bytes each, FP16).
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
    /// Indirect dispatch arguments for emission kernel (12 bytes = 3 x u32).
    /// Written by `prepare_dispatch` kernel each frame.
    pub emission_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// GPU-computed emission parameters (16 bytes = 4 x u32).
    /// Written by `prepare_dispatch`, read by `emission_kernel`.
    pub gpu_emission_params: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Indirect dispatch arguments for update/grid_populate kernels (12 bytes = 3 x u32).
    /// Written by `sync_indirect_args` kernel each frame based on alive count.
    /// Bootstrap value uses pool_size for first frame (design R6).
    pub update_dispatch_args: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Debug telemetry ---
    /// Debug telemetry buffer (32 bytes = 8 x u32). Only allocated when debug-telemetry feature is enabled.
    #[cfg(feature = "debug-telemetry")]
    pub debug_telemetry: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Grid density ---
    /// Grid density buffer: 64^3 = 262144 uint32 cells (1.05 MB).
    pub grid_density: Retained<ProtocolObject<dyn MTLBuffer>>,

    // --- Uniforms ---
    /// Uniform ring buffer for triple buffering (768 bytes = 3 x 256).
    /// Each frame writes to slot [frame_index * 256 .. (frame_index+1) * 256].
    pub uniform_ring: Retained<ProtocolObject<dyn MTLBuffer>>,
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
    /// Allocates all SoA buffers, dead/alive lists, indirect args, and uniform ring.
    /// Initializes dead list with all indices [0..pool_size-1], alive lists empty,
    /// and indirect args to default draw state.
    pub fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>, pool_size: usize) -> Self {
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

        // Allocate grid density buffer: 64^3 cells x 4 bytes per uint32
        let grid_density = alloc_buffer(device, sizes.grid_density, "grid_density");

        // Allocate indirect args
        let indirect_args = alloc_buffer(device, sizes.indirect_args, "indirect_args");

        // Allocate uniform ring buffer for triple buffering (3 x 256 = 768 bytes)
        let uniform_ring_size = 3 * std::mem::size_of::<Uniforms>();
        let uniform_ring = alloc_buffer(device, uniform_ring_size, "uniform_ring");

        // Allocate GPU-centric dispatch buffers
        let emission_dispatch_args = alloc_buffer(
            device,
            std::mem::size_of::<DispatchArgs>(),
            "emission_dispatch_args",
        );
        let gpu_emission_params = alloc_buffer(
            device,
            std::mem::size_of::<GpuEmissionParams>(),
            "gpu_emission_params",
        );
        let update_dispatch_args = alloc_buffer(
            device,
            std::mem::size_of::<DispatchArgs>(),
            "update_dispatch_args",
        );

        // Allocate debug telemetry buffer (32 bytes) when feature is enabled
        #[cfg(feature = "debug-telemetry")]
        let debug_telemetry_buf = alloc_buffer(device, 32, "debug_telemetry");

        let pool = Self {
            pool_size,
            device: device.clone(),
            positions,
            velocities,
            lifetimes,
            colors,
            sizes: particle_sizes,
            dead_list,
            alive_list_a,
            alive_list_b,
            #[cfg(feature = "debug-telemetry")]
            debug_telemetry: debug_telemetry_buf,
            grid_density,
            indirect_args,
            emission_dispatch_args,
            gpu_emission_params,
            update_dispatch_args,
            uniform_ring,
        };

        // Initialize buffer contents on CPU
        pool.init_dead_list();
        pool.init_alive_list(&pool.alive_list_a);
        pool.init_alive_list(&pool.alive_list_b);
        pool.init_indirect_args();
        pool.init_emission_dispatch_args();
        pool.init_gpu_emission_params();
        pool.init_update_dispatch_args();
        pool.init_uniform_ring();
        #[cfg(feature = "debug-telemetry")]
        pool.init_debug_telemetry();

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

    /// Initialize emission dispatch args to default DispatchArgs (0 threadgroups X, 1 Y, 1 Z).
    fn init_emission_dispatch_args(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.emission_dispatch_args) as *mut DispatchArgs;
            std::ptr::write(ptr, DispatchArgs::default());
        }
    }

    /// Initialize GPU emission params to default (all zeros).
    fn init_gpu_emission_params(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.gpu_emission_params) as *mut GpuEmissionParams;
            std::ptr::write(ptr, GpuEmissionParams::default());
        }
    }

    /// Initialize update dispatch args with bootstrap value: threadgroups = ceil(pool_size / 256).
    /// This ensures the first frame dispatches enough threads to cover the full pool (design R6).
    fn init_update_dispatch_args(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.update_dispatch_args) as *mut DispatchArgs;
            std::ptr::write(
                ptr,
                DispatchArgs {
                    threadgroups_per_grid: [self.pool_size.div_ceil(256) as u32, 1, 1],
                },
            );
        }
    }

    /// Initialize uniform ring buffer with default Uniforms in all 3 slots.
    fn init_uniform_ring(&self) {
        unsafe {
            let base = buffer_ptr(&self.uniform_ring) as *mut u8;
            for i in 0..3 {
                let slot = base.add(Self::uniforms_offset(i)) as *mut Uniforms;
                std::ptr::write(slot, Uniforms::default());
            }
        }
    }

    /// Initialize debug telemetry buffer to all zeros (32 bytes).
    #[cfg(feature = "debug-telemetry")]
    fn init_debug_telemetry(&self) {
        unsafe {
            let ptr = buffer_ptr(&self.debug_telemetry) as *mut u8;
            std::ptr::write_bytes(ptr, 0, 32);
        }
    }

    /// Compute the byte offset into the uniform ring buffer for a given frame index.
    pub fn uniforms_offset(frame_index: usize) -> usize {
        frame_index * std::mem::size_of::<Uniforms>()
    }

    /// Grow the pool to `new_size` particles, preserving existing data.
    ///
    /// Only grows; if `new_size <= pool_size`, returns early.
    /// Allocates new SoA buffers, copies existing data via CPU-side memcpy,
    /// extends the dead list with new indices [old_size..new_size-1],
    /// and swaps buffer references. Grid density buffer is fixed at 64^3 and not resized.
    pub fn grow(&mut self, new_size: usize) {
        if new_size <= self.pool_size {
            return;
        }

        let old_size = self.pool_size;
        let new_sizes = BufferSizes::new(new_size);
        let old_sizes = BufferSizes::new(old_size);

        // Allocate new SoA buffers
        let new_positions = alloc_buffer(&self.device, new_sizes.positions, "positions");
        let new_velocities = alloc_buffer(&self.device, new_sizes.velocities, "velocities");
        let new_lifetimes = alloc_buffer(&self.device, new_sizes.lifetimes, "lifetimes");
        let new_colors = alloc_buffer(&self.device, new_sizes.colors, "colors");
        let new_particle_sizes = alloc_buffer(&self.device, new_sizes.sizes, "sizes");

        // Allocate new counter+index list buffers
        let new_dead_list = alloc_buffer(&self.device, new_sizes.counter_list, "dead_list");
        let new_alive_list_a = alloc_buffer(&self.device, new_sizes.counter_list, "alive_list_a");
        let new_alive_list_b = alloc_buffer(&self.device, new_sizes.counter_list, "alive_list_b");

        // Copy existing particle data (CPU-side memcpy via SharedStorage pointer access)
        unsafe {
            // SoA buffers: copy old_size * bytes_per_particle
            ptr::copy_nonoverlapping(
                self.positions.contents().as_ptr() as *const u8,
                new_positions.contents().as_ptr() as *mut u8,
                old_sizes.positions,
            );
            ptr::copy_nonoverlapping(
                self.velocities.contents().as_ptr() as *const u8,
                new_velocities.contents().as_ptr() as *mut u8,
                old_sizes.velocities,
            );
            ptr::copy_nonoverlapping(
                self.lifetimes.contents().as_ptr() as *const u8,
                new_lifetimes.contents().as_ptr() as *mut u8,
                old_sizes.lifetimes,
            );
            ptr::copy_nonoverlapping(
                self.colors.contents().as_ptr() as *const u8,
                new_colors.contents().as_ptr() as *mut u8,
                old_sizes.colors,
            );
            ptr::copy_nonoverlapping(
                self.sizes.contents().as_ptr() as *const u8,
                new_particle_sizes.contents().as_ptr() as *mut u8,
                old_sizes.sizes,
            );

            // Dead list: copy existing header + indices, then extend with new indices
            // Read current dead_count from the old dead list counter
            let old_dead_ptr = self.dead_list.contents().as_ptr() as *const CounterHeader;
            let dead_count = (*old_dead_ptr).count;

            // Copy existing dead list data (header + old indices)
            ptr::copy_nonoverlapping(
                self.dead_list.contents().as_ptr() as *const u8,
                new_dead_list.contents().as_ptr() as *mut u8,
                COUNTER_HEADER_SIZE + old_size * 4,
            );

            // Add new indices [old_size..new_size-1] to dead list starting at dead_count offset
            let new_dead_indices_ptr =
                (new_dead_list.contents().as_ptr() as *mut u8).add(COUNTER_HEADER_SIZE) as *mut u32;
            let new_particles = new_size - old_size;
            for i in 0..new_particles {
                *new_dead_indices_ptr.add(dead_count as usize + i) = (old_size + i) as u32;
            }

            // Update dead list counter: dead_count + new_particles
            let new_dead_header = new_dead_list.contents().as_ptr() as *mut CounterHeader;
            (*new_dead_header).count = dead_count + new_particles as u32;

            // Alive lists: copy existing data (counter + indices stay valid since we only added new)
            ptr::copy_nonoverlapping(
                self.alive_list_a.contents().as_ptr() as *const u8,
                new_alive_list_a.contents().as_ptr() as *mut u8,
                COUNTER_HEADER_SIZE + old_size * 4,
            );
            ptr::copy_nonoverlapping(
                self.alive_list_b.contents().as_ptr() as *const u8,
                new_alive_list_b.contents().as_ptr() as *mut u8,
                COUNTER_HEADER_SIZE + old_size * 4,
            );
        }

        // Reallocate GPU-centric dispatch buffers (no copy needed, GPU recomputes each frame)
        let new_emission_dispatch_args = alloc_buffer(
            &self.device,
            std::mem::size_of::<DispatchArgs>(),
            "emission_dispatch_args",
        );
        let new_gpu_emission_params = alloc_buffer(
            &self.device,
            std::mem::size_of::<GpuEmissionParams>(),
            "gpu_emission_params",
        );
        // Allocate new update dispatch args buffer (GPU recomputes each frame via sync_indirect_args)
        let new_update_dispatch_args = alloc_buffer(
            &self.device,
            std::mem::size_of::<DispatchArgs>(),
            "update_dispatch_args",
        );

        // Initialize dispatch buffers: emission to zero, update to bootstrap value for new pool_size
        unsafe {
            let ptr = buffer_ptr(&new_emission_dispatch_args) as *mut DispatchArgs;
            std::ptr::write(ptr, DispatchArgs::default());
            let ptr = buffer_ptr(&new_gpu_emission_params) as *mut GpuEmissionParams;
            std::ptr::write(ptr, GpuEmissionParams::default());
            let ptr = buffer_ptr(&new_update_dispatch_args) as *mut DispatchArgs;
            std::ptr::write(
                ptr,
                DispatchArgs {
                    threadgroups_per_grid: [new_size.div_ceil(256) as u32, 1, 1],
                },
            );
        }

        // Reallocate debug telemetry buffer when feature is enabled (no copy, GPU rewrites each frame)
        #[cfg(feature = "debug-telemetry")]
        {
            let new_debug_telemetry = alloc_buffer(&self.device, 32, "debug_telemetry");
            unsafe {
                std::ptr::write_bytes(new_debug_telemetry.contents().as_ptr() as *mut u8, 0, 32);
            }
            self.debug_telemetry = new_debug_telemetry;
        }

        // Swap buffer references (old buffers dropped when replaced)
        self.positions = new_positions;
        self.velocities = new_velocities;
        self.lifetimes = new_lifetimes;
        self.colors = new_colors;
        self.sizes = new_particle_sizes;
        self.dead_list = new_dead_list;
        self.alive_list_a = new_alive_list_a;
        self.alive_list_b = new_alive_list_b;
        self.emission_dispatch_args = new_emission_dispatch_args;
        self.gpu_emission_params = new_gpu_emission_params;
        self.update_dispatch_args = new_update_dispatch_args;
        // grid_density stays the same (64^3 fixed)
        // indirect_args stays the same
        // uniform_ring stays the same (768 bytes, not pool-size-dependent)

        self.pool_size = new_size;
    }

    /// Read the alive count from the given alive list buffer (CPU readback via SharedStorage).
    pub fn read_alive_count(&self, buffer: &ProtocolObject<dyn MTLBuffer>) -> u32 {
        unsafe {
            let header = buffer.contents().as_ptr() as *const CounterHeader;
            (*header).count
        }
    }

    /// Read the alive count from indirect_args buffer (instance_count field of DrawArgs).
    ///
    /// This avoids reading from the alive list counter directly, supporting
    /// GPU-centric architecture where indirect_args is the source of truth.
    pub fn read_alive_count_from_indirect(&self) -> u32 {
        unsafe {
            let draw_args = self.indirect_args.contents().as_ptr() as *const DrawArgs;
            (*draw_args).instance_count
        }
    }

    /// Get (read_list, write_list) based on ping-pong flag.
    ///
    /// - `ping_pong = false`: read from A (contains last frame's survivors), write to B
    /// - `ping_pong = true`:  read from B (contains last frame's survivors), write to A
    ///
    /// The read list holds previous survivors + receives new emissions.
    /// The write list receives this frame's survivors from the update kernel.
    pub fn get_ping_pong_lists(
        &self,
        ping_pong: bool,
    ) -> (
        &ProtocolObject<dyn MTLBuffer>,
        &ProtocolObject<dyn MTLBuffer>,
    ) {
        if ping_pong {
            (&self.alive_list_b, &self.alive_list_a)
        } else {
            (&self.alive_list_a, &self.alive_list_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CounterHeader, COUNTER_HEADER_SIZE};
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device available")
    }

    #[test]
    fn test_pool_new_buffer_lengths() {
        let device = get_device();
        let pool = ParticlePool::new(&device, 1_000_000);

        // Position buffer: 1M * 12 bytes (float3)
        assert_eq!(pool.positions.length(), 1_000_000 * 12);
        // Velocity buffer: 1M * 12 bytes (float3)
        assert_eq!(pool.velocities.length(), 1_000_000 * 12);
        // Lifetime buffer: 1M * 4 bytes (half2)
        assert_eq!(pool.lifetimes.length(), 1_000_000 * 4);
        // Color buffer: 1M * 8 bytes (half4)
        assert_eq!(pool.colors.length(), 1_000_000 * 8);
        // Size buffer: 1M * 2 bytes (half, FP16)
        assert_eq!(pool.sizes.length(), 1_000_000 * 2);
    }

    #[test]
    fn test_pool_new_dead_list_counter() {
        let device = get_device();
        let pool = ParticlePool::new(&device, 1_000_000);

        // Dead list counter should be initialized to pool_size
        unsafe {
            let header = pool.dead_list.contents().as_ptr() as *const CounterHeader;
            assert_eq!((*header).count, 1_000_000);
        }
    }

    #[test]
    fn test_pool_new_alive_list_counters() {
        let device = get_device();
        let pool = ParticlePool::new(&device, 1_000_000);

        // Both alive list counters should be 0
        unsafe {
            let header_a = pool.alive_list_a.contents().as_ptr() as *const CounterHeader;
            assert_eq!((*header_a).count, 0);

            let header_b = pool.alive_list_b.contents().as_ptr() as *const CounterHeader;
            assert_eq!((*header_b).count, 0);
        }
    }

    #[test]
    fn test_pool_grow_buffer_lengths() {
        let device = get_device();
        let mut pool = ParticlePool::new(&device, 1000);

        // Verify initial sizes
        assert_eq!(pool.positions.length(), 1000 * 12);
        assert_eq!(pool.pool_size, 1000);

        // Grow to 2000
        pool.grow(2000);

        assert_eq!(pool.pool_size, 2000);
        assert_eq!(pool.positions.length(), 2000 * 12);
        assert_eq!(pool.velocities.length(), 2000 * 12);
        assert_eq!(pool.lifetimes.length(), 2000 * 4);
        assert_eq!(pool.colors.length(), 2000 * 8);
        assert_eq!(pool.sizes.length(), 2000 * 2);
    }

    #[test]
    fn test_pool_grow_dead_list_extended() {
        let device = get_device();
        let mut pool = ParticlePool::new(&device, 1000);

        // Initially dead list has 1000 entries with indices [0..999]
        unsafe {
            let header = pool.dead_list.contents().as_ptr() as *const CounterHeader;
            assert_eq!((*header).count, 1000);
        }

        // Grow to 2000
        pool.grow(2000);

        unsafe {
            let header = pool.dead_list.contents().as_ptr() as *const CounterHeader;
            // Dead list counter should be 1000 (original) + 1000 (new) = 2000
            assert_eq!((*header).count, 2000);

            // Verify new indices [1000..1999] are present after the original 1000
            let indices_ptr =
                (pool.dead_list.contents().as_ptr() as *const u8).add(COUNTER_HEADER_SIZE)
                    as *const u32;
            let indices = std::slice::from_raw_parts(indices_ptr, 2000);

            // Original indices [0..999] should still be there
            for i in 0..1000 {
                assert_eq!(indices[i], i as u32, "Original index {} mismatch", i);
            }

            // New indices [1000..1999] should be at positions [1000..1999]
            for i in 0..1000 {
                assert_eq!(
                    indices[1000 + i],
                    (1000 + i) as u32,
                    "New index {} mismatch",
                    1000 + i
                );
            }
        }
    }

    #[test]
    fn test_pool_grow_noop_for_same_size() {
        let device = get_device();
        let mut pool = ParticlePool::new(&device, 1000);

        // Growing to same or smaller size should be a no-op
        pool.grow(1000);
        assert_eq!(pool.pool_size, 1000);

        pool.grow(500);
        assert_eq!(pool.pool_size, 1000);
    }
}
