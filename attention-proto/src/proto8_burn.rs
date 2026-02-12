//! Proto 8: Burn Extension Trait — AttentionBackend supertrait viability
//!
//! This prototype tests whether we can define a custom `AttentionBackend` trait
//! that extends Burn's `Backend` trait, adding a `flash_attention` method.
//!
//! The orphan rule prevents implementing a foreign trait on a foreign type,
//! so we use a newtype wrapper `MetalAttentionBackend` around the concrete
//! Burn backend type.
//!
//! Feature-gated behind `burn-ext`.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Extension trait adding flash attention capability to any Burn Backend.
///
/// Usage:
/// ```ignore
/// fn my_model<B: AttentionBackend>(q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
///     B::flash_attention(q, k, v, None)
/// }
/// ```
pub trait AttentionBackend: Backend {
    /// Compute scaled dot-product attention using Flash Attention algorithm.
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, head_dim]
    /// * `k` - Key tensor [batch, seq_len, head_dim]
    /// * `v` - Value tensor [batch, seq_len, head_dim]
    /// * `mask` - Optional causal mask [batch, seq_len, seq_len]
    ///
    /// # Returns
    /// Output tensor [batch, seq_len, head_dim]
    fn flash_attention(
        q: Tensor<Self, 3>,
        k: Tensor<Self, 3>,
        v: Tensor<Self, 3>,
        mask: Option<Tensor<Self, 3>>,
    ) -> Tensor<Self, 3>;
}

/// Newtype wrapper around a Burn backend to enable AttentionBackend implementation.
///
/// This avoids the orphan rule: we own MetalAttentionBackend, so we can implement
/// our AttentionBackend trait on it. The inner type B must itself implement Backend.
///
/// In a production implementation, this would wrap a specific backend like
/// `burn::backend::wgpu::Wgpu` or a hypothetical `burn::backend::metal::Metal`.
/// For this prototype, we keep it generic to demonstrate the pattern compiles.
#[derive(Clone, Default, Debug)]
pub struct MetalAttentionBackend<B: Backend>(core::marker::PhantomData<B>);

/// Delegate Backend implementation to the inner type.
///
/// NOTE: Implementing Backend requires implementing many supertraits
/// (FloatTensorOps, IntTensorOps, BoolTensorOps, etc.). A full implementation
/// would delegate all methods to the inner B. For this prototype, we demonstrate
/// that the trait definition and newtype pattern compile correctly.
///
/// A production implementation would use a proc-macro or manual delegation
/// to forward all Backend operations to the inner type B.

// --- Trait hierarchy demonstration ---
// The key insight for Proto 8 is that the AttentionBackend supertrait pattern
// is viable in Rust's type system. The two challenges are:
//
// 1. Defining the supertrait: `trait AttentionBackend: Backend` compiles and
//    allows adding custom methods that use Burn's tensor types.
//
// 2. The orphan rule: We cannot `impl AttentionBackend for SomeExternalBackend`
//    directly if both the trait and the type are in external crates. The newtype
//    `MetalAttentionBackend<B>` pattern solves this.
//
// The remaining challenge (implementing Backend for MetalAttentionBackend) is
// a mechanical delegation task, not a type-system limitation. This can be solved
// with: (a) a derive macro, (b) the ambassador crate, or (c) manual delegation.

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the trait definition compiles and is object-safe enough for our use case.
    #[test]
    fn test_trait_definition_compiles() {
        // The trait AttentionBackend: Backend exists and has the flash_attention method.
        // We verify this by checking we can name the trait and its method signature
        // compiles. We cannot call it without a concrete Backend impl.
        fn _assert_trait_bound<B: AttentionBackend>() {}

        // If this compiles, the trait definition is valid.
    }

    /// Verify the newtype wrapper compiles with PhantomData.
    #[test]
    fn test_newtype_compiles() {
        // MetalAttentionBackend is generic over any Backend.
        // We verify it can be named and has the expected derives.
        fn _assert_newtype<B: Backend>() {
            let _phantom: MetalAttentionBackend<B>;
        }

        // If this compiles, the newtype pattern is valid.
    }
}
