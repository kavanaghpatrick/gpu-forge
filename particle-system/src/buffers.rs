//! GPU buffer pool for 1M+ particle SoA data, dead/alive lists, and indirect args.
//!
//! All buffers use `MTLResourceStorageModeShared` for CPU+GPU access on Apple Silicon
//! unified memory. Dead list is initialized with all indices; alive lists start empty.

use std::ffi::c_void;
use std::ptr;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::types::{BufferSizes, CounterHeader, DrawArgs, Uniforms, COUNTER_HEADER_SIZE};

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

    // --- Grid density ---
    /// Grid density buffer: 64^3 = 262144 uint32 cells (1.05 MB).
    pub grid_density: Retained<ProtocolObject<dyn MTLBuffer>>,

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

        // Allocate indirect args and uniforms
        let indirect_args = alloc_buffer(device, sizes.indirect_args, "indirect_args");
        let uniforms = alloc_buffer(device, sizes.uniforms, "uniforms");

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
            grid_density,
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

    /// Grow the pool to `new_size` particles, preserving existing data.
    ///
    /// Only grows; if `new_size <= pool_size`, returns early.
    /// Allocates new SoA buffers, copies existing data via CPU-side memcpy,
    /// extends the dead list with new indices [old_size..new_size-1],
    /// and swaps buffer references. Grid density buffer is fixed at 64^3 and not resized.
    pub fn grow(&mut self, new_size: usize) {
        if new_size <= self.pool_size {
            println!("grow: new_size {} <= pool_size {}, skipping", new_size, self.pool_size);
            return;
        }

        let old_size = self.pool_size;
        let new_sizes = BufferSizes::new(new_size);
        let old_sizes = BufferSizes::new(old_size);

        println!(
            "Growing pool: {}M -> {}M ({:.1} MB -> {:.1} MB)",
            old_size / 1_000_000,
            new_size / 1_000_000,
            old_sizes.total_bytes() as f64 / (1024.0 * 1024.0),
            new_sizes.total_bytes() as f64 / (1024.0 * 1024.0),
        );

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

        // Swap buffer references (old buffers dropped when replaced)
        self.positions = new_positions;
        self.velocities = new_velocities;
        self.lifetimes = new_lifetimes;
        self.colors = new_colors;
        self.sizes = new_particle_sizes;
        self.dead_list = new_dead_list;
        self.alive_list_a = new_alive_list_a;
        self.alive_list_b = new_alive_list_b;
        // grid_density stays the same (64^3 fixed)
        // indirect_args stays the same
        // uniforms stays the same

        self.pool_size = new_size;

        println!(
            "Pool grown to {} particles ({:.1} MB)",
            new_size,
            new_sizes.total_bytes() as f64 / (1024.0 * 1024.0),
        );
    }

    /// Read the alive count from the given alive list buffer (CPU readback via SharedStorage).
    pub fn read_alive_count(&self, buffer: &ProtocolObject<dyn MTLBuffer>) -> u32 {
        unsafe {
            let header = buffer.contents().as_ptr() as *const CounterHeader;
            (*header).count
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
