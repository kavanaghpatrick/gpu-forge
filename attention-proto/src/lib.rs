//! attention-proto: 8 GPU kernel prototypes for trait Attention<Q,K,V> on Apple Silicon Metal
//!
//! Prototypes:
//! 1. Metal Flash Attention — simdgroup_matrix tiled attention, tile size sweep
//! 2. Function Stitching — [[stitchable]] inner loop overhead measurement
//! 3. PagedAttention V2 — partitioned mode with block table, 32KB constraint
//! 4. Function Constants — compilation overhead for 72 variants
//! 5. CubeCL MSL Quality — compare generated vs hand-written MSL
//! 6. FLA Linear Attention — chunk_h/chunk_o kernels ported to Metal
//! 7. RoPE/ALiBi/GQA — empirical per-variant timing on M4
//! 8. Burn Extension Trait — AttentionBackend supertrait viability

// Shared infrastructure (uncomment as modules are implemented)
pub mod device;
pub mod pipeline;
pub mod encode;
pub mod timing;
pub mod types;
pub mod kb;

// Prototype modules (uncomment as implemented)
pub mod proto1_flash;
pub mod proto2_stitch;
pub mod proto3_paged;
pub mod proto4_constants;
pub mod proto6_fla;
pub mod proto7_variants;

// Feature-gated prototype modules
// #[cfg(feature = "cubecl")]
// pub mod proto5_cubecl;

// #[cfg(feature = "burn-ext")]
// pub mod proto8_burn;
