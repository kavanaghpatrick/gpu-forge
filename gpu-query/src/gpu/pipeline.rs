//! Pipeline State Object (PSO) cache with function constant specialization.
//!
//! Metal function constants allow compile-time specialization of kernel code,
//! eliminating runtime branches. Each unique combination of function constants
//! produces a different PSO that must be compiled and cached.
//!
//! The `PsoCache` stores compiled PSOs keyed by function name + constant values,
//! so repeated queries with the same operator/type combination reuse the PSO.

use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};

/// Key for PSO cache lookup: function name + function constant values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PsoKey {
    /// Kernel function name (e.g., "column_filter").
    pub function_name: String,
    /// Serialized function constant values for identity.
    /// Format: vec of (index, type_tag, value_bytes).
    pub constants: Vec<(usize, u8, Vec<u8>)>,
}

/// Cache of compiled Metal compute pipeline states.
///
/// Avoids recompiling the same function+constants combination on every query.
/// PSO compilation with function constants triggers the Metal compiler at runtime,
/// so caching is essential for interactive performance.
pub struct PsoCache {
    cache: HashMap<PsoKey, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Default for PsoCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PsoCache {
    /// Create a new empty PSO cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Get or create a PSO for the given function name and constant values.
    ///
    /// If a matching PSO exists in the cache, returns it immediately.
    /// Otherwise, compiles a new PSO using the Metal compiler and caches it.
    pub fn get_or_create(
        &mut self,
        library: &ProtocolObject<dyn MTLLibrary>,
        key: &PsoKey,
    ) -> &ProtocolObject<dyn MTLComputePipelineState> {
        if !self.cache.contains_key(key) {
            let pso = Self::compile_pso(library, key);
            self.cache.insert(key.clone(), pso);
        }
        &self.cache[key]
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Compile a PSO with the given function constants.
    fn compile_pso(
        library: &ProtocolObject<dyn MTLLibrary>,
        key: &PsoKey,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let constant_values = MTLFunctionConstantValues::new();

        for (index, type_tag, value_bytes) in &key.constants {
            let mtl_type = match type_tag {
                0 => MTLDataType::UInt, // uint
                1 => MTLDataType::Bool, // bool
                _ => panic!("Unknown function constant type tag: {}", type_tag),
            };

            unsafe {
                let ptr = NonNull::new(value_bytes.as_ptr() as *mut std::ffi::c_void)
                    .expect("constant value pointer is null");
                constant_values.setConstantValue_type_atIndex(ptr, mtl_type, *index);
            }
        }

        let fn_name = NSString::from_str(&key.function_name);
        let function = library
            .newFunctionWithName_constantValues_error(&fn_name, &constant_values)
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to create function '{}' with constants: {:?}",
                    key.function_name, e
                )
            });

        let device = library.device();
        device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap_or_else(|e| panic!("Failed to create PSO for '{}': {:?}", key.function_name, e))
    }
}

/// Compare operation codes matching the Metal shader's COMPARE_OP constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CompareOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// Column type codes matching the Metal shader's COLUMN_TYPE constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ColumnTypeCode {
    Int64 = 0,
    Float64 = 1,
}

/// Build a PsoKey for the column_filter kernel with the given parameters.
pub fn filter_pso_key(
    compare_op: CompareOp,
    column_type: ColumnTypeCode,
    has_null_check: bool,
) -> PsoKey {
    let op_val = compare_op as u32;
    let type_val = column_type as u32;
    let null_val: u8 = if has_null_check { 1 } else { 0 };

    PsoKey {
        function_name: "column_filter".to_string(),
        constants: vec![
            // index 0: COMPARE_OP (uint)
            (0, 0, op_val.to_ne_bytes().to_vec()),
            // index 1: COLUMN_TYPE (uint)
            (1, 0, type_val.to_ne_bytes().to_vec()),
            // index 2: HAS_NULL_CHECK (bool)
            (2, 1, vec![null_val]),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pso_key_equality() {
        let k1 = filter_pso_key(CompareOp::Gt, ColumnTypeCode::Int64, false);
        let k2 = filter_pso_key(CompareOp::Gt, ColumnTypeCode::Int64, false);
        assert_eq!(k1, k2, "Same parameters should produce equal keys");
    }

    #[test]
    fn test_pso_key_inequality() {
        let k1 = filter_pso_key(CompareOp::Gt, ColumnTypeCode::Int64, false);
        let k2 = filter_pso_key(CompareOp::Lt, ColumnTypeCode::Int64, false);
        assert_ne!(k1, k2, "Different ops should produce different keys");

        let k3 = filter_pso_key(CompareOp::Gt, ColumnTypeCode::Float64, false);
        assert_ne!(k1, k3, "Different types should produce different keys");

        let k4 = filter_pso_key(CompareOp::Gt, ColumnTypeCode::Int64, true);
        assert_ne!(k1, k4, "Different null check should produce different keys");
    }

    #[test]
    fn test_compare_op_values() {
        assert_eq!(CompareOp::Eq as u32, 0);
        assert_eq!(CompareOp::Ne as u32, 1);
        assert_eq!(CompareOp::Lt as u32, 2);
        assert_eq!(CompareOp::Le as u32, 3);
        assert_eq!(CompareOp::Gt as u32, 4);
        assert_eq!(CompareOp::Ge as u32, 5);
    }

    #[test]
    fn test_column_type_code_values() {
        assert_eq!(ColumnTypeCode::Int64 as u32, 0);
        assert_eq!(ColumnTypeCode::Float64 as u32, 1);
    }
}
