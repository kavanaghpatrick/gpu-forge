//! Lock-free GPU hash table on Apple Silicon Metal.
//!
//! Three versions showing progressive optimization:
//! - **V1**: SoA layout, simple hash, atomic lookup (baseline)
//! - **V2**: MurmurHash3, non-atomic lookup
//! - **V3**: AoS interleaved — key+value in same cache line (fastest)
//!
//! # Quick Start
//!
//! ```no_run
//! use metal_lockfree_hashtable::{GpuHashTable, Version};
//!
//! let table = GpuHashTable::new(Version::V3, 2_000_000);
//! let keys = vec![1u32, 2, 3, 4, 5];
//! let values = vec![10u32, 20, 30, 40, 50];
//!
//! table.insert(&keys, &values);
//! let (results, gpu_ms) = table.lookup(&keys);
//! assert_eq!(results[0], 10);
//! ```

pub mod bench;
pub mod metal_ctx;
pub mod table;

pub use bench::{run_benchmarks, BenchConfig, BenchResult};
pub use metal_ctx::MetalContext;
pub use table::{GpuHashTable, Version};
