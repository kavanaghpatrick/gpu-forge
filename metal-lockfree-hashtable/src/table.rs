//! GPU hash table: insert, lookup, clear across V1/V2/V3 kernel variants.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputePipelineState};

use crate::metal_ctx::{
    alloc_buffer, alloc_buffer_with_data, dispatch_1d, gpu_elapsed_ms, read_buffer_slice,
    MetalContext,
};

/// Hash table version — selects which kernel set to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// V1: SoA layout, simple hash, atomic lookup
    V1,
    /// V2: SoA layout, MurmurHash3, non-atomic lookup
    V2,
    /// V3: AoS interleaved (key+value in same cache line), MurmurHash3
    V3,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Version::V1 => write!(f, "V1"),
            Version::V2 => write!(f, "V2"),
            Version::V3 => write!(f, "V3"),
        }
    }
}

/// Parameters passed to Metal kernels. Must match `HashTableParams` in types.h.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct HashTableParams {
    capacity: u32,
    num_ops: u32,
}

/// A lock-free GPU hash table backed by Metal compute kernels.
pub struct GpuHashTable {
    version: Version,
    capacity: u32,
    ctx: MetalContext,

    // V1/V2: separate key and value buffers
    // V3: single interleaved buffer
    buf_table: Retained<ProtocolObject<dyn MTLBuffer>>,
    // V1/V2 only: separate value buffer
    buf_values: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    pso_insert: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_lookup: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl GpuHashTable {
    /// Create a new GPU hash table.
    ///
    /// `capacity` is rounded up to the next power of 2.
    /// The table supports up to `capacity / 2` unique keys (50% load factor).
    pub fn new(version: Version, capacity: u32) -> Self {
        let capacity = capacity.next_power_of_two();

        let ctx = MetalContext::new();

        let (insert_name, lookup_name) = match version {
            Version::V1 => ("ht_insert_v1", "ht_lookup_v1"),
            Version::V2 => ("ht_insert_v2", "ht_lookup_v2"),
            Version::V3 => ("ht_insert_v3", "ht_lookup_v3"),
        };

        let pso_insert = ctx.make_pipeline(insert_name);
        let pso_lookup = ctx.make_pipeline(lookup_name);

        let (buf_table, buf_values) = match version {
            Version::V1 | Version::V2 => {
                // SoA: separate key buffer (atomic_uint) and value buffer (uint)
                let keys = alloc_buffer(
                    &ctx.device,
                    capacity as usize * std::mem::size_of::<u32>(),
                );
                let values = alloc_buffer(
                    &ctx.device,
                    capacity as usize * std::mem::size_of::<u32>(),
                );
                (keys, Some(values))
            }
            Version::V3 => {
                // AoS: interleaved [key, val, key, val, ...] = capacity * 2 * 4 bytes
                let table = alloc_buffer(
                    &ctx.device,
                    capacity as usize * 2 * std::mem::size_of::<u32>(),
                );
                (table, None)
            }
        };

        let mut ht = Self {
            version,
            capacity,
            ctx,
            buf_table,
            buf_values,
            pso_insert,
            pso_lookup,
        };
        ht.clear();
        ht
    }

    /// Clear the table (fill with EMPTY_KEY sentinel).
    pub fn clear(&mut self) {
        match self.version {
            Version::V1 | Version::V2 => {
                // Fill keys with 0xFF (EMPTY_KEY)
                unsafe {
                    let ptr = self.buf_table.contents().as_ptr() as *mut u8;
                    std::ptr::write_bytes(ptr, 0xFF, self.capacity as usize * 4);
                }
                // Fill values with 0xFF
                if let Some(ref values) = self.buf_values {
                    unsafe {
                        let ptr = values.contents().as_ptr() as *mut u8;
                        std::ptr::write_bytes(ptr, 0xFF, self.capacity as usize * 4);
                    }
                }
            }
            Version::V3 => {
                // Fill entire AoS table with 0xFF
                unsafe {
                    let ptr = self.buf_table.contents().as_ptr() as *mut u8;
                    std::ptr::write_bytes(ptr, 0xFF, self.capacity as usize * 8);
                }
            }
        }
    }

    /// Insert key-value pairs into the table.
    ///
    /// Returns GPU execution time in milliseconds.
    pub fn insert(&self, keys: &[u32], values: &[u32]) -> f64 {
        assert_eq!(keys.len(), values.len());
        let n = keys.len();

        let buf_input_keys = alloc_buffer_with_data(&self.ctx.device, keys);
        let buf_input_values = alloc_buffer_with_data(&self.ctx.device, values);
        let params = HashTableParams {
            capacity: self.capacity,
            num_ops: n as u32,
        };
        let buf_params = alloc_buffer_with_data(&self.ctx.device, &[params]);

        let cmd = self.ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        match self.version {
            Version::V1 | Version::V2 => {
                dispatch_1d(
                    &enc,
                    &self.pso_insert,
                    &[
                        (self.buf_table.as_ref(), 0),
                        (self.buf_values.as_ref().unwrap().as_ref(), 1),
                        (buf_input_keys.as_ref(), 2),
                        (buf_input_values.as_ref(), 3),
                        (buf_params.as_ref(), 4),
                    ],
                    n,
                );
            }
            Version::V3 => {
                dispatch_1d(
                    &enc,
                    &self.pso_insert,
                    &[
                        (self.buf_table.as_ref(), 0),
                        (buf_input_keys.as_ref(), 1),
                        (buf_input_values.as_ref(), 2),
                        (buf_params.as_ref(), 3),
                    ],
                    n,
                );
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        gpu_elapsed_ms(&cmd).unwrap_or(0.0)
    }

    /// Lookup keys in the table.
    ///
    /// Returns `(values, gpu_ms)`. Missing keys return `0xFFFFFFFF`.
    pub fn lookup(&self, keys: &[u32]) -> (Vec<u32>, f64) {
        let n = keys.len();

        let buf_query_keys = alloc_buffer_with_data(&self.ctx.device, keys);
        let buf_output = alloc_buffer(
            &self.ctx.device,
            n * std::mem::size_of::<u32>(),
        );
        let params = HashTableParams {
            capacity: self.capacity,
            num_ops: n as u32,
        };
        let buf_params = alloc_buffer_with_data(&self.ctx.device, &[params]);

        let cmd = self.ctx.command_buffer();
        let enc = cmd.computeCommandEncoder().unwrap();

        match self.version {
            Version::V1 | Version::V2 => {
                dispatch_1d(
                    &enc,
                    &self.pso_lookup,
                    &[
                        (self.buf_table.as_ref(), 0),
                        (self.buf_values.as_ref().unwrap().as_ref(), 1),
                        (buf_query_keys.as_ref(), 2),
                        (buf_output.as_ref(), 3),
                        (buf_params.as_ref(), 4),
                    ],
                    n,
                );
            }
            Version::V3 => {
                dispatch_1d(
                    &enc,
                    &self.pso_lookup,
                    &[
                        (self.buf_table.as_ref(), 0),
                        (buf_query_keys.as_ref(), 1),
                        (buf_output.as_ref(), 2),
                        (buf_params.as_ref(), 3),
                    ],
                    n,
                );
            }
        }

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        let gpu_ms = gpu_elapsed_ms(&cmd).unwrap_or(0.0);
        let results = unsafe { read_buffer_slice(&buf_output, n) };

        (results, gpu_ms)
    }

    /// Table capacity (power of 2).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Table version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Get a reference to the Metal context.
    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }
}
