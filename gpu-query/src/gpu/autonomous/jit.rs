//! JIT Metal shader compiler with plan-structure caching.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLLibrary};

use crate::gpu::autonomous::loader::ColumnInfo;
use crate::sql::physical_plan::PhysicalPlan;
use crate::sql::types::{AggFunc, CompareOp};

/// Compute a structure-only hash of a physical plan.
///
/// The hash captures plan node types, operators, column references, aggregate
/// functions, and GROUP BY columns. It deliberately does NOT hash literal values
/// (filter thresholds), so two plans that differ only in constants produce the
/// same hash. This enables JIT PSO cache reuse across parameter changes.
pub fn plan_structure_hash(plan: &PhysicalPlan) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_plan_node(plan, &mut hasher);
    hasher.finish()
}

/// Recursively hash a plan node's structure into the hasher.
fn hash_plan_node(plan: &PhysicalPlan, hasher: &mut DefaultHasher) {
    match plan {
        PhysicalPlan::GpuScan { table, columns } => {
            "GpuScan".hash(hasher);
            table.hash(hasher);
            for col in columns {
                col.hash(hasher);
            }
        }

        PhysicalPlan::GpuFilter {
            compare_op,
            column,
            value: _, // deliberately skip literal value
            input,
        } => {
            "GpuFilter".hash(hasher);
            compare_op.hash(hasher);
            column.hash(hasher);
            // Do NOT hash value -- same structure with different literals shares PSO
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuCompoundFilter { op, left, right } => {
            "GpuCompoundFilter".hash(hasher);
            // LogicalOp doesn't derive Hash, so discriminate manually
            match op {
                crate::sql::types::LogicalOp::And => 0u8.hash(hasher),
                crate::sql::types::LogicalOp::Or => 1u8.hash(hasher),
            }
            hash_plan_node(left, hasher);
            hash_plan_node(right, hasher);
        }

        PhysicalPlan::GpuAggregate {
            functions,
            group_by,
            input,
        } => {
            "GpuAggregate".hash(hasher);
            for (func, col) in functions {
                func.hash(hasher);
                col.hash(hasher);
            }
            for col in group_by {
                col.hash(hasher);
            }
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuSort { order_by, input } => {
            "GpuSort".hash(hasher);
            for (col, asc) in order_by {
                col.hash(hasher);
                asc.hash(hasher);
            }
            hash_plan_node(input, hasher);
        }

        PhysicalPlan::GpuLimit { count, input } => {
            "GpuLimit".hash(hasher);
            count.hash(hasher);
            hash_plan_node(input, hasher);
        }
    }
}

// ---------------------------------------------------------------------------
// Extracted plan components for code generation
// ---------------------------------------------------------------------------

/// A filter predicate extracted from the plan tree.
#[derive(Debug, Clone)]
struct ExtractedFilter {
    column_name: String,
    compare_op: CompareOp,
}

/// All components extracted from a PhysicalPlan for Metal source generation.
#[derive(Debug)]
struct ExtractedPlan {
    filters: Vec<ExtractedFilter>,
    aggregates: Vec<(AggFunc, String)>,
    group_by: Vec<String>,
}

/// Extract filters, aggs, and group_by from a PhysicalPlan tree.
fn extract_plan(plan: &PhysicalPlan) -> ExtractedPlan {
    let mut filters = Vec::new();
    let mut aggregates = Vec::new();
    let mut group_by = Vec::new();

    fn collect_filters(node: &PhysicalPlan, out: &mut Vec<ExtractedFilter>) {
        match node {
            PhysicalPlan::GpuFilter {
                compare_op,
                column,
                input,
                ..
            } => {
                out.push(ExtractedFilter {
                    column_name: column.clone(),
                    compare_op: *compare_op,
                });
                collect_filters(input, out);
            }
            PhysicalPlan::GpuCompoundFilter { left, right, .. } => {
                collect_filters(left, out);
                collect_filters(right, out);
            }
            _ => {}
        }
    }

    // Walk plan tree to find the aggregate and its input
    match plan {
        PhysicalPlan::GpuAggregate {
            functions,
            group_by: gb,
            input,
        } => {
            aggregates = functions.clone();
            group_by = gb.clone();
            collect_filters(input, &mut filters);
        }
        _ => {
            // Non-aggregate plan: just collect filters
            collect_filters(plan, &mut filters);
        }
    }

    ExtractedPlan {
        filters,
        aggregates,
        group_by,
    }
}

// ---------------------------------------------------------------------------
// JIT Metal source generator
// ---------------------------------------------------------------------------

/// A compiled JIT plan: PSO + metadata.
pub struct CompiledPlan {
    pub pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub plan_hash: u64,
    pub source_len: usize,
}

/// JIT compiler that generates specialized Metal source from a PhysicalPlan,
/// compiles it at runtime via `newLibraryWithSource`, and caches PSOs by
/// plan structure hash.
pub struct JitCompiler {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    cache: HashMap<u64, CompiledPlan>,
}

impl JitCompiler {
    /// Create a new JIT compiler bound to the given Metal device.
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            cache: HashMap::new(),
        }
    }

    /// Compile a plan into a Metal PSO, returning a cached result on hit.
    ///
    /// On cache miss: generates specialized Metal source, compiles it via
    /// `newLibraryWithSource`, creates a compute pipeline state, and inserts
    /// into the cache. On cache hit: returns the existing `CompiledPlan`.
    pub fn compile(
        &mut self,
        plan: &PhysicalPlan,
        schema: &[ColumnInfo],
    ) -> Result<&CompiledPlan, String> {
        let hash = plan_structure_hash(plan);

        if self.cache.contains_key(&hash) {
            return Ok(&self.cache[&hash]);
        }

        let source = Self::generate_metal_source_for_jit(plan, schema);

        // Compile Metal source at runtime (no include paths needed — header inlined)
        let ns_source = NSString::from_str(&source);
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| format!("JIT compile failed: {}", e))?;

        let fn_name = NSString::from_str("fused_query_jit");
        let func = library
            .newFunctionWithName(&fn_name)
            .ok_or("Function 'fused_query_jit' not found in JIT library")?;

        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| format!("PSO creation failed: {}", e))?;

        let compiled = CompiledPlan {
            pso,
            plan_hash: hash,
            source_len: source.len(),
        };

        self.cache.insert(hash, compiled);
        Ok(&self.cache[&hash])
    }

    /// Number of cached compiled plans.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Generate specialized Metal shader source code from a physical plan.
    ///
    /// The generated kernel has the same buffer interface as the AOT fused_query
    /// kernel (buffers 0-3: params, data, columns, output) but with specialized
    /// inner logic: exact filter predicates inlined, exact aggregate functions,
    /// and GROUP BY handling baked in.
    ///
    /// Uses `#include "autonomous_types.h"` — suitable for AOT compilation via
    /// build.rs where `-I shaders/` resolves the include path.
    pub fn generate_metal_source(plan: &PhysicalPlan, schema: &[ColumnInfo]) -> String {
        let extracted = extract_plan(plan);
        let mut src = String::with_capacity(4096);

        // --- Header (with #include for AOT path) ---
        emit_header(&mut src);

        // --- SIMD helpers (always included for reductions) ---
        emit_simd_helpers(&mut src);

        // --- Column read helpers ---
        emit_column_readers(&mut src);

        // --- Device atomic helpers ---
        emit_atomic_helpers(&mut src);

        // --- Main kernel ---
        emit_kernel(
            &mut src,
            &extracted.filters,
            &extracted.aggregates,
            &extracted.group_by,
            schema,
        );

        src
    }

    /// Generate specialized Metal source for JIT runtime compilation.
    ///
    /// Unlike `generate_metal_source`, this version inlines the
    /// `autonomous_types.h` header content directly into the source string,
    /// because `newLibraryWithSource` does not have access to the build.rs
    /// `-I shaders/` include path.
    fn generate_metal_source_for_jit(plan: &PhysicalPlan, schema: &[ColumnInfo]) -> String {
        let extracted = extract_plan(plan);
        let mut src = String::with_capacity(8192);

        // --- Header with inlined types (no #include) ---
        emit_header_for_jit(&mut src);

        // --- SIMD helpers (always included for reductions) ---
        emit_simd_helpers(&mut src);

        // --- Column read helpers ---
        emit_column_readers(&mut src);

        // --- Device atomic helpers ---
        emit_atomic_helpers(&mut src);

        // --- Main kernel ---
        emit_kernel(
            &mut src,
            &extracted.filters,
            &extracted.aggregates,
            &extracted.group_by,
            schema,
        );

        src
    }
}

/// Resolve a column name to its index in the schema. Returns 0 if not found.
fn column_index(schema: &[ColumnInfo], name: &str) -> usize {
    schema.iter().position(|c| c.name == name).unwrap_or(0)
}

/// Map a ColumnInfo data type to a MSL column type constant.
fn column_type_constant(schema: &[ColumnInfo], name: &str) -> &'static str {
    use crate::storage::schema::DataType;
    if let Some(col) = schema.iter().find(|c| c.name == name) {
        match col.data_type {
            DataType::Int64 => "COLUMN_TYPE_INT64",
            DataType::Float64 => "COLUMN_TYPE_FLOAT32", // loader downcasts to f32
            DataType::Varchar => "COLUMN_TYPE_DICT_U32",
            DataType::Bool | DataType::Date => "COLUMN_TYPE_INT64",
        }
    } else {
        "COLUMN_TYPE_INT64"
    }
}

/// Map a CompareOp to its MSL operator string.
fn compare_op_str(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "==",
        CompareOp::Ne => "!=",
        CompareOp::Lt => "<",
        CompareOp::Le => "<=",
        CompareOp::Gt => ">",
        CompareOp::Ge => ">=",
    }
}

/// Map a CompareOp to its MSL column read function.
fn read_func_for_type(type_const: &str) -> &'static str {
    match type_const {
        "COLUMN_TYPE_INT64" => "read_int64",
        "COLUMN_TYPE_FLOAT32" => "read_float32",
        "COLUMN_TYPE_DICT_U32" => "read_dict_u32",
        _ => "read_int64",
    }
}

/// MSL type name for a column type constant.
fn msl_type_for_column(type_const: &str) -> &'static str {
    match type_const {
        "COLUMN_TYPE_INT64" => "long",
        "COLUMN_TYPE_FLOAT32" => "float",
        "COLUMN_TYPE_DICT_U32" => "uint",
        _ => "long",
    }
}

// ---------------------------------------------------------------------------
// Code emission helpers
// ---------------------------------------------------------------------------

fn emit_header(src: &mut String) {
    src.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
    src.push_str("#include \"autonomous_types.h\"\n\n");
    src.push_str("#define THREADGROUP_SIZE 256\n");
    src.push_str("#define INT64_MAX_VAL  0x7FFFFFFFFFFFFFFF\n");
    src.push_str("#define INT64_MIN_VAL  0x8000000000000000\n");
    src.push_str("#define FLOAT_MAX_VAL  3.402823466e+38f\n");
    src.push_str("#define FLOAT_MIN_VAL  (-3.402823466e+38f)\n\n");
}

/// Emit header for JIT runtime compilation: inlines the autonomous_types.h
/// content directly since `newLibraryWithSource` has no include path.
fn emit_header_for_jit(src: &mut String) {
    src.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Inline autonomous_types.h content (struct definitions + constants)
    src.push_str("// --- Inlined autonomous_types.h for JIT compilation ---\n\n");
    src.push_str("#define MAX_GROUPS  64\n");
    src.push_str("#define MAX_AGGS    5\n");
    src.push_str("#define MAX_FILTERS 4\n");
    src.push_str("#define MAX_GROUP_SLOTS 256\n\n");
    src.push_str("#define COLUMN_TYPE_INT64   0\n");
    src.push_str("#define COLUMN_TYPE_FLOAT32 1\n");
    src.push_str("#define COLUMN_TYPE_DICT_U32 2\n\n");
    src.push_str("#define COMPARE_OP_EQ 0\n");
    src.push_str("#define COMPARE_OP_NE 1\n");
    src.push_str("#define COMPARE_OP_LT 2\n");
    src.push_str("#define COMPARE_OP_LE 3\n");
    src.push_str("#define COMPARE_OP_GT 4\n");
    src.push_str("#define COMPARE_OP_GE 5\n\n");
    src.push_str("#define AGG_FUNC_COUNT 0\n");
    src.push_str("#define AGG_FUNC_SUM   1\n");
    src.push_str("#define AGG_FUNC_AVG   2\n");
    src.push_str("#define AGG_FUNC_MIN   3\n");
    src.push_str("#define AGG_FUNC_MAX   4\n\n");

    // FilterSpec (48 bytes)
    src.push_str(
        "struct FilterSpec {\n\
         \x20   uint     column_idx;\n\
         \x20   uint     compare_op;\n\
         \x20   uint     column_type;\n\
         \x20   uint     _pad0;\n\
         \x20   long     value_int;\n\
         \x20   uint     value_float_bits;\n\
         \x20   uint     _pad1;\n\
         \x20   uint     has_null_check;\n\
         \x20   uint     _pad2[3];\n\
         };\n\n",
    );

    // AggSpec (16 bytes)
    src.push_str(
        "struct AggSpec {\n\
         \x20   uint     agg_func;\n\
         \x20   uint     column_idx;\n\
         \x20   uint     column_type;\n\
         \x20   uint     _pad0;\n\
         };\n\n",
    );

    // QueryParamsSlot (512 bytes)
    src.push_str(
        "struct QueryParamsSlot {\n\
         \x20   uint       sequence_id;\n\
         \x20   uint       _pad_seq;\n\
         \x20   ulong      query_hash;\n\
         \x20   uint       filter_count;\n\
         \x20   uint       _pad_fc;\n\
         \x20   FilterSpec filters[4];\n\
         \x20   uint       agg_count;\n\
         \x20   uint       _pad_ac;\n\
         \x20   AggSpec    aggs[5];\n\
         \x20   uint       group_by_col;\n\
         \x20   uint       has_group_by;\n\
         \x20   uint       row_count;\n\
         \x20   char       _padding[196];\n\
         };\n\n",
    );

    // ColumnMeta (32 bytes)
    src.push_str(
        "struct ColumnMeta {\n\
         \x20   ulong    offset;\n\
         \x20   uint     column_type;\n\
         \x20   uint     stride;\n\
         \x20   ulong    null_offset;\n\
         \x20   uint     row_count;\n\
         \x20   uint     _pad;\n\
         };\n\n",
    );

    // AggResult (16 bytes)
    src.push_str(
        "struct AggResult {\n\
         \x20   long     value_int;\n\
         \x20   float    value_float;\n\
         \x20   uint     count;\n\
         };\n\n",
    );

    // OutputBuffer (22560 bytes)
    src.push_str(
        "struct OutputBuffer {\n\
         \x20   uint       ready_flag;\n\
         \x20   uint       sequence_id;\n\
         \x20   ulong      latency_ns;\n\
         \x20   uint       result_row_count;\n\
         \x20   uint       result_col_count;\n\
         \x20   uint       error_code;\n\
         \x20   uint       _pad;\n\
         \x20   long       group_keys[256];\n\
         \x20   AggResult  agg_results[256][5];\n\
         };\n\n",
    );

    src.push_str("// --- End inlined autonomous_types.h ---\n\n");

    // JIT-specific defines
    src.push_str("#define THREADGROUP_SIZE 256\n");
    src.push_str("#define INT64_MAX_VAL  0x7FFFFFFFFFFFFFFF\n");
    src.push_str("#define INT64_MIN_VAL  0x8000000000000000\n");
    src.push_str("#define FLOAT_MAX_VAL  3.402823466e+38f\n");
    src.push_str("#define FLOAT_MIN_VAL  (-3.402823466e+38f)\n\n");
}

fn emit_simd_helpers(src: &mut String) {
    src.push_str(
        "static inline long jit_simd_sum_int64(long value, uint simd_lane) {\n\
         \x20   int2 val;\n\
         \x20   val.x = static_cast<int>(static_cast<ulong>(value));\n\
         \x20   val.y = static_cast<int>(static_cast<ulong>(value) >> 32);\n\
         \x20   for (ushort offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       int2 other;\n\
         \x20       other.x = simd_shuffle_down(val.x, offset);\n\
         \x20       other.y = simd_shuffle_down(val.y, offset);\n\
         \x20       uint a_lo = static_cast<uint>(val.x);\n\
         \x20       uint b_lo = static_cast<uint>(other.x);\n\
         \x20       uint new_lo = a_lo + b_lo;\n\
         \x20       uint carry = (new_lo < a_lo) ? 1u : 0u;\n\
         \x20       val.x = static_cast<int>(new_lo);\n\
         \x20       val.y = val.y + other.y + static_cast<int>(carry);\n\
         \x20   }\n\
         \x20   ulong result = static_cast<ulong>(static_cast<uint>(val.x))\n\
         \x20                | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);\n\
         \x20   return static_cast<long>(result);\n\
         }\n\n",
    );
    src.push_str(
        "static inline long jit_simd_min_int64(long value, uint simd_lane) {\n\
         \x20   int2 val;\n\
         \x20   val.x = static_cast<int>(static_cast<ulong>(value));\n\
         \x20   val.y = static_cast<int>(static_cast<ulong>(value) >> 32);\n\
         \x20   for (ushort offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       int2 other;\n\
         \x20       other.x = simd_shuffle_down(val.x, offset);\n\
         \x20       other.y = simd_shuffle_down(val.y, offset);\n\
         \x20       bool other_less = (other.y < val.y) || (other.y == val.y && static_cast<uint>(other.x) < static_cast<uint>(val.x));\n\
         \x20       if (other_less) { val.x = other.x; val.y = other.y; }\n\
         \x20   }\n\
         \x20   ulong result = static_cast<ulong>(static_cast<uint>(val.x))\n\
         \x20                | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);\n\
         \x20   return static_cast<long>(result);\n\
         }\n\n",
    );
    src.push_str(
        "static inline long jit_simd_max_int64(long value, uint simd_lane) {\n\
         \x20   int2 val;\n\
         \x20   val.x = static_cast<int>(static_cast<ulong>(value));\n\
         \x20   val.y = static_cast<int>(static_cast<ulong>(value) >> 32);\n\
         \x20   for (ushort offset = 16; offset > 0; offset >>= 1) {\n\
         \x20       int2 other;\n\
         \x20       other.x = simd_shuffle_down(val.x, offset);\n\
         \x20       other.y = simd_shuffle_down(val.y, offset);\n\
         \x20       bool other_greater = (other.y > val.y) || (other.y == val.y && static_cast<uint>(other.x) > static_cast<uint>(val.x));\n\
         \x20       if (other_greater) { val.x = other.x; val.y = other.y; }\n\
         \x20   }\n\
         \x20   ulong result = static_cast<ulong>(static_cast<uint>(val.x))\n\
         \x20                | (static_cast<ulong>(static_cast<uint>(val.y)) << 32);\n\
         \x20   return static_cast<long>(result);\n\
         }\n\n",
    );
}

fn emit_column_readers(src: &mut String) {
    src.push_str(
        "static inline long read_int64(device const char* data, device const ColumnMeta& meta, uint row) {\n\
         \x20   return ((device const long*)(data + meta.offset))[row];\n\
         }\n\n",
    );
    src.push_str(
        "static inline float read_float32(device const char* data, device const ColumnMeta& meta, uint row) {\n\
         \x20   return ((device const float*)(data + meta.offset))[row];\n\
         }\n\n",
    );
    src.push_str(
        "static inline uint read_dict_u32(device const char* data, device const ColumnMeta& meta, uint row) {\n\
         \x20   return ((device const uint*)(data + meta.offset))[row];\n\
         }\n\n",
    );
}

fn emit_atomic_helpers(src: &mut String) {
    // atomic_add_int64
    src.push_str(
        "static inline void atomic_add_int64(device atomic_uint* lo_ptr, device atomic_uint* hi_ptr, long value) {\n\
         \x20   ulong val_u = static_cast<ulong>(value);\n\
         \x20   uint lo = static_cast<uint>(val_u);\n\
         \x20   uint hi = static_cast<uint>(val_u >> 32);\n\
         \x20   uint old_lo = atomic_fetch_add_explicit(lo_ptr, lo, memory_order_relaxed);\n\
         \x20   ulong sum_lo = static_cast<ulong>(old_lo) + static_cast<ulong>(lo);\n\
         \x20   uint carry = (sum_lo > 0xFFFFFFFFUL) ? 1u : 0u;\n\
         \x20   atomic_fetch_add_explicit(hi_ptr, hi + carry, memory_order_relaxed);\n\
         }\n\n",
    );
    // atomic_add_float
    src.push_str(
        "static inline void atomic_add_float(device atomic_uint* ptr, float value) {\n\
         \x20   if (value == 0.0f) return;\n\
         \x20   uint expected = atomic_load_explicit(ptr, memory_order_relaxed);\n\
         \x20   for (int i = 0; i < 64; i++) {\n\
         \x20       float cur = as_type<float>(expected);\n\
         \x20       float desired = cur + value;\n\
         \x20       uint desired_bits = as_type<uint>(desired);\n\
         \x20       if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits, memory_order_relaxed, memory_order_relaxed)) return;\n\
         \x20   }\n\
         }\n\n",
    );
    // atomic_min_int64
    src.push_str(
        "static inline void atomic_min_int64(device atomic_uint* lo_ptr, device atomic_uint* hi_ptr, long new_val) {\n\
         \x20   ulong new_u = static_cast<ulong>(new_val);\n\
         \x20   uint new_lo = static_cast<uint>(new_u); uint new_hi = static_cast<uint>(new_u >> 32);\n\
         \x20   for (int i = 0; i < 128; i++) {\n\
         \x20       uint cur_hi = atomic_load_explicit(hi_ptr, memory_order_relaxed);\n\
         \x20       uint cur_lo = atomic_load_explicit(lo_ptr, memory_order_relaxed);\n\
         \x20       uint cur_hi2 = atomic_load_explicit(hi_ptr, memory_order_relaxed);\n\
         \x20       if (cur_hi != cur_hi2) continue;\n\
         \x20       long cur_val = static_cast<long>(static_cast<ulong>(cur_lo) | (static_cast<ulong>(cur_hi) << 32));\n\
         \x20       if (new_val >= cur_val) return;\n\
         \x20       if (atomic_compare_exchange_weak_explicit(lo_ptr, &cur_lo, new_lo, memory_order_relaxed, memory_order_relaxed)) {\n\
         \x20           if (atomic_compare_exchange_weak_explicit(hi_ptr, &cur_hi, new_hi, memory_order_relaxed, memory_order_relaxed)) return;\n\
         \x20           atomic_store_explicit(lo_ptr, cur_lo, memory_order_relaxed);\n\
         \x20       }\n\
         \x20   }\n\
         }\n\n",
    );
    // atomic_max_int64
    src.push_str(
        "static inline void atomic_max_int64(device atomic_uint* lo_ptr, device atomic_uint* hi_ptr, long new_val) {\n\
         \x20   ulong new_u = static_cast<ulong>(new_val);\n\
         \x20   uint new_lo = static_cast<uint>(new_u); uint new_hi = static_cast<uint>(new_u >> 32);\n\
         \x20   for (int i = 0; i < 128; i++) {\n\
         \x20       uint cur_hi = atomic_load_explicit(hi_ptr, memory_order_relaxed);\n\
         \x20       uint cur_lo = atomic_load_explicit(lo_ptr, memory_order_relaxed);\n\
         \x20       uint cur_hi2 = atomic_load_explicit(hi_ptr, memory_order_relaxed);\n\
         \x20       if (cur_hi != cur_hi2) continue;\n\
         \x20       long cur_val = static_cast<long>(static_cast<ulong>(cur_lo) | (static_cast<ulong>(cur_hi) << 32));\n\
         \x20       if (new_val <= cur_val) return;\n\
         \x20       if (atomic_compare_exchange_weak_explicit(lo_ptr, &cur_lo, new_lo, memory_order_relaxed, memory_order_relaxed)) {\n\
         \x20           if (atomic_compare_exchange_weak_explicit(hi_ptr, &cur_hi, new_hi, memory_order_relaxed, memory_order_relaxed)) return;\n\
         \x20           atomic_store_explicit(lo_ptr, cur_lo, memory_order_relaxed);\n\
         \x20       }\n\
         \x20   }\n\
         }\n\n",
    );
    // atomic_min_float / atomic_max_float
    src.push_str(
        "static inline void atomic_min_float(device atomic_uint* ptr, float value) {\n\
         \x20   uint expected = atomic_load_explicit(ptr, memory_order_relaxed);\n\
         \x20   for (int i = 0; i < 64; i++) {\n\
         \x20       float cur = as_type<float>(expected);\n\
         \x20       if (value >= cur) return;\n\
         \x20       uint desired_bits = as_type<uint>(value);\n\
         \x20       if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits, memory_order_relaxed, memory_order_relaxed)) return;\n\
         \x20   }\n\
         }\n\n",
    );
    src.push_str(
        "static inline void atomic_max_float(device atomic_uint* ptr, float value) {\n\
         \x20   uint expected = atomic_load_explicit(ptr, memory_order_relaxed);\n\
         \x20   for (int i = 0; i < 64; i++) {\n\
         \x20       float cur = as_type<float>(expected);\n\
         \x20       if (value <= cur) return;\n\
         \x20       uint desired_bits = as_type<uint>(value);\n\
         \x20       if (atomic_compare_exchange_weak_explicit(ptr, &expected, desired_bits, memory_order_relaxed, memory_order_relaxed)) return;\n\
         \x20   }\n\
         }\n\n",
    );
}

fn emit_kernel(
    src: &mut String,
    filters: &[ExtractedFilter],
    aggregates: &[(AggFunc, String)],
    group_by: &[String],
    schema: &[ColumnInfo],
) {
    let has_group_by = !group_by.is_empty();
    let filter_count = filters.len();
    let agg_count = aggregates.len();

    // Kernel signature
    src.push_str(
        "kernel void fused_query_jit(\n\
         \x20   device const QueryParamsSlot* params [[buffer(0)]],\n\
         \x20   device const char* data [[buffer(1)]],\n\
         \x20   device const ColumnMeta* columns [[buffer(2)]],\n\
         \x20   device OutputBuffer* output [[buffer(3)]],\n\
         \x20   uint tid [[thread_position_in_grid]],\n\
         \x20   uint tgid [[threadgroup_position_in_grid]],\n\
         \x20   uint lid [[thread_position_in_threadgroup]],\n\
         \x20   uint simd_lane [[thread_index_in_simdgroup]],\n\
         \x20   uint simd_id [[simdgroup_index_in_threadgroup]]\n\
         ) {\n",
    );

    // Threadgroup accumulators
    if has_group_by || agg_count > 0 {
        emit_threadgroup_accumulators(src, agg_count);
    }

    // Row bounds check
    src.push_str("    uint row_count = params->row_count;\n");
    src.push_str("    bool row_passes = (tid < row_count);\n\n");

    // --- Filter phase (inlined) ---
    if filter_count > 0 {
        src.push_str(&format!(
            "    // === FILTER PHASE ({} predicates, inlined) ===\n",
            filter_count
        ));
        for (i, f) in filters.iter().enumerate() {
            let col_idx = column_index(schema, &f.column_name);
            let col_type = column_type_constant(schema, &f.column_name);
            let read_fn = read_func_for_type(col_type);
            let msl_type = msl_type_for_column(col_type);
            let op_str = compare_op_str(f.compare_op);

            // Read column value (reuse if same column as previous filter)
            src.push_str(&format!(
                "    if (row_passes) {{\n\
                 \x20       {msl_type} filter_{i}_val = {read_fn}(data, columns[{col_idx}], tid);\n",
            ));

            // Comparison value comes from params (literal values stored there)
            match col_type {
                "COLUMN_TYPE_INT64" => {
                    src.push_str(&format!(
                        "        if (!(filter_{i}_val {op_str} params->filters[{i}].value_int)) row_passes = false;\n",
                    ));
                }
                "COLUMN_TYPE_FLOAT32" => {
                    src.push_str(&format!(
                        "        float filter_{i}_threshold = as_type<float>(params->filters[{i}].value_float_bits);\n\
                         \x20       if (!(filter_{i}_val {op_str} filter_{i}_threshold)) row_passes = false;\n",
                    ));
                }
                "COLUMN_TYPE_DICT_U32" => {
                    src.push_str(&format!(
                        "        if (!(filter_{i}_val {op_str} static_cast<uint>(params->filters[{i}].value_int))) row_passes = false;\n",
                    ));
                }
                _ => {}
            }
            src.push_str("    }\n");
        }
        src.push('\n');
    }

    // --- GROUP BY phase ---
    src.push_str("    // === GROUP BY PHASE ===\n");
    src.push_str("    uint bucket = 0;\n");
    if has_group_by {
        let gb_col = &group_by[0]; // single GROUP BY column
        let gb_idx = column_index(schema, gb_col);
        let gb_type = column_type_constant(schema, gb_col);
        let gb_read = read_func_for_type(gb_type);
        let gb_msl = msl_type_for_column(gb_type);

        src.push_str(&format!(
            "    if (row_passes) {{\n\
             \x20       {gb_msl} group_val = {gb_read}(data, columns[{gb_idx}], tid);\n\
             \x20       bucket = static_cast<uint>((group_val >= 0 ? group_val : -group_val) % MAX_GROUPS);\n\
             \x20       accum[bucket].group_key = static_cast<long>(group_val);\n\
             \x20       accum[bucket].valid = 1;\n\
             \x20   }}\n\n",
        ));
    } else {
        src.push_str(
            "    if (row_passes) {\n\
             \x20       accum[0].valid = 1;\n\
             \x20   }\n\n",
        );
    }

    // --- Aggregate phase ---
    if agg_count > 0 {
        src.push_str(&format!(
            "    // === AGGREGATE PHASE ({} aggs) ===\n",
            agg_count
        ));
        emit_aggregate_phase(src, aggregates, schema, has_group_by, agg_count);
    }

    // --- Global reduction ---
    emit_global_reduction(src, aggregates, schema, agg_count);

    // --- Output metadata ---
    emit_output_metadata(src, has_group_by, agg_count);

    src.push_str("}\n");
}

fn emit_threadgroup_accumulators(src: &mut String, agg_count: usize) {
    src.push_str("    // Threadgroup-local group accumulators\n");
    src.push_str("    struct GroupAccumulator {\n");
    src.push_str("        long count;\n");
    src.push_str(&format!(
        "        long sum_int[{agg_count}];\n\
         \x20       float sum_float[{agg_count}];\n\
         \x20       long min_int[{agg_count}];\n\
         \x20       long max_int[{agg_count}];\n\
         \x20       float min_float[{agg_count}];\n\
         \x20       float max_float[{agg_count}];\n",
    ));
    src.push_str("        long group_key;\n        uint valid;\n    };\n");
    src.push_str("    threadgroup GroupAccumulator accum[MAX_GROUPS];\n\n");

    // Initialize accumulators
    src.push_str("    // Init accumulators\n");
    src.push_str(
        "    uint groups_per_thread = (MAX_GROUPS + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;\n",
    );
    src.push_str("    for (uint i = 0; i < groups_per_thread; i++) {\n");
    src.push_str("        uint g = lid * groups_per_thread + i;\n");
    src.push_str("        if (g < MAX_GROUPS) {\n");
    src.push_str("            accum[g].count = 0;\n");
    for a in 0..agg_count {
        src.push_str(&format!(
            "            accum[g].sum_int[{a}] = 0; accum[g].sum_float[{a}] = 0.0f;\n\
             \x20           accum[g].min_int[{a}] = static_cast<long>(INT64_MAX_VAL);\n\
             \x20           accum[g].max_int[{a}] = static_cast<long>(INT64_MIN_VAL);\n\
             \x20           accum[g].min_float[{a}] = FLOAT_MAX_VAL;\n\
             \x20           accum[g].max_float[{a}] = FLOAT_MIN_VAL;\n",
        ));
    }
    src.push_str("            accum[g].group_key = 0; accum[g].valid = 0;\n");
    src.push_str("        }\n    }\n");
    src.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");
}

fn emit_aggregate_phase(
    src: &mut String,
    aggregates: &[(AggFunc, String)],
    schema: &[ColumnInfo],
    has_group_by: bool,
    agg_count: usize,
) {
    // Per-thread local values
    src.push_str("    long local_count = row_passes ? 1 : 0;\n");
    for (a, (func, col_name)) in aggregates.iter().enumerate() {
        let col_type = column_type_constant(schema, col_name);
        let col_idx = column_index(schema, col_name);
        let read_fn = read_func_for_type(col_type);

        match func {
            AggFunc::Count => {
                src.push_str(&format!("    long local_agg_{a} = row_passes ? 1 : 0;\n"));
            }
            AggFunc::Sum | AggFunc::Avg => {
                if col_type == "COLUMN_TYPE_FLOAT32" {
                    src.push_str(&format!(
                        "    float local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : 0.0f;\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "    long local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : 0;\n",
                    ));
                }
            }
            AggFunc::Min => {
                if col_type == "COLUMN_TYPE_FLOAT32" {
                    src.push_str(&format!(
                        "    float local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : FLOAT_MAX_VAL;\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "    long local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : static_cast<long>(INT64_MAX_VAL);\n",
                    ));
                }
            }
            AggFunc::Max => {
                if col_type == "COLUMN_TYPE_FLOAT32" {
                    src.push_str(&format!(
                        "    float local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : FLOAT_MIN_VAL;\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "    long local_agg_{a} = row_passes ? {read_fn}(data, columns[{col_idx}], tid) : static_cast<long>(INT64_MIN_VAL);\n",
                    ));
                }
            }
        }
    }
    src.push('\n');

    // Merge into threadgroup accumulators
    if !has_group_by {
        // Strategy A: simd reduction for single-bucket
        emit_simd_merge(src, aggregates, schema, agg_count);
    } else {
        // Strategy B: serial per-thread accumulation for GROUP BY
        emit_serial_merge(src, aggregates, schema, agg_count);
    }
}

fn emit_simd_merge(
    src: &mut String,
    aggregates: &[(AggFunc, String)],
    schema: &[ColumnInfo],
    _agg_count: usize,
) {
    src.push_str("    // SIMD reduction (no GROUP BY, single bucket)\n");
    src.push_str("    long simd_count = jit_simd_sum_int64(local_count, simd_lane);\n");

    for (a, (func, col_name)) in aggregates.iter().enumerate() {
        let col_type = column_type_constant(schema, col_name);
        let is_float = col_type == "COLUMN_TYPE_FLOAT32";

        match func {
            AggFunc::Count => {
                src.push_str(&format!(
                    "    long simd_agg_{a} = jit_simd_sum_int64(local_agg_{a}, simd_lane);\n"
                ));
            }
            AggFunc::Sum | AggFunc::Avg => {
                if is_float {
                    src.push_str(&format!(
                        "    float simd_agg_{a} = simd_sum(local_agg_{a});\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "    long simd_agg_{a} = jit_simd_sum_int64(local_agg_{a}, simd_lane);\n"
                    ));
                }
            }
            AggFunc::Min => {
                if is_float {
                    src.push_str(&format!(
                        "    float simd_agg_{a} = simd_min(local_agg_{a});\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "    long simd_agg_{a} = jit_simd_min_int64(local_agg_{a}, simd_lane);\n"
                    ));
                }
            }
            AggFunc::Max => {
                if is_float {
                    src.push_str(&format!(
                        "    float simd_agg_{a} = simd_max(local_agg_{a});\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "    long simd_agg_{a} = jit_simd_max_int64(local_agg_{a}, simd_lane);\n"
                    ));
                }
            }
        }
    }

    // Merge simdgroups into threadgroup accumulator (serialized)
    src.push_str(
        "\n    uint num_simdgroups = (THREADGROUP_SIZE + 31) / 32;\n\
         \x20   for (uint sg = 0; sg < num_simdgroups; sg++) {\n\
         \x20       if (simd_id == sg && simd_lane == 0 && simd_count > 0) {\n\
         \x20           accum[0].count += simd_count;\n\
         \x20           accum[0].valid = 1;\n",
    );

    for (a, (func, col_name)) in aggregates.iter().enumerate() {
        let col_type = column_type_constant(schema, col_name);
        let is_float = col_type == "COLUMN_TYPE_FLOAT32";

        match func {
            AggFunc::Count => {
                src.push_str(&format!(
                    "            accum[0].sum_int[{a}] += simd_agg_{a};\n"
                ));
            }
            AggFunc::Sum | AggFunc::Avg => {
                if is_float {
                    src.push_str(&format!(
                        "            accum[0].sum_float[{a}] += simd_agg_{a};\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "            accum[0].sum_int[{a}] += simd_agg_{a};\n"
                    ));
                }
            }
            AggFunc::Min => {
                if is_float {
                    src.push_str(&format!(
                        "            if (simd_agg_{a} < accum[0].min_float[{a}]) accum[0].min_float[{a}] = simd_agg_{a};\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "            if (simd_agg_{a} < accum[0].min_int[{a}]) accum[0].min_int[{a}] = simd_agg_{a};\n",
                    ));
                }
            }
            AggFunc::Max => {
                if is_float {
                    src.push_str(&format!(
                        "            if (simd_agg_{a} > accum[0].max_float[{a}]) accum[0].max_float[{a}] = simd_agg_{a};\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "            if (simd_agg_{a} > accum[0].max_int[{a}]) accum[0].max_int[{a}] = simd_agg_{a};\n",
                    ));
                }
            }
        }
    }

    src.push_str(
        "        }\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   }\n\n",
    );
}

fn emit_serial_merge(
    src: &mut String,
    aggregates: &[(AggFunc, String)],
    schema: &[ColumnInfo],
    _agg_count: usize,
) {
    src.push_str("    // Serial per-thread accumulation (GROUP BY)\n");
    src.push_str(
        "    for (uint t = 0; t < THREADGROUP_SIZE; t++) {\n\
         \x20       if (lid == t && local_count > 0) {\n\
         \x20           uint b = bucket;\n\
         \x20           accum[b].count += 1;\n\
         \x20           accum[b].valid = 1;\n",
    );

    for (a, (func, col_name)) in aggregates.iter().enumerate() {
        let col_type = column_type_constant(schema, col_name);
        let is_float = col_type == "COLUMN_TYPE_FLOAT32";

        match func {
            AggFunc::Count => {
                src.push_str(&format!(
                    "            accum[b].sum_int[{a}] += local_agg_{a};\n"
                ));
            }
            AggFunc::Sum | AggFunc::Avg => {
                if is_float {
                    src.push_str(&format!(
                        "            accum[b].sum_float[{a}] += local_agg_{a};\n"
                    ));
                } else {
                    src.push_str(&format!(
                        "            accum[b].sum_int[{a}] += local_agg_{a};\n"
                    ));
                }
            }
            AggFunc::Min => {
                if is_float {
                    src.push_str(&format!(
                        "            if (local_agg_{a} < accum[b].min_float[{a}]) accum[b].min_float[{a}] = local_agg_{a};\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "            if (local_agg_{a} < accum[b].min_int[{a}]) accum[b].min_int[{a}] = local_agg_{a};\n",
                    ));
                }
            }
            AggFunc::Max => {
                if is_float {
                    src.push_str(&format!(
                        "            if (local_agg_{a} > accum[b].max_float[{a}]) accum[b].max_float[{a}] = local_agg_{a};\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "            if (local_agg_{a} > accum[b].max_int[{a}]) accum[b].max_int[{a}] = local_agg_{a};\n",
                    ));
                }
            }
        }
    }

    src.push_str(
        "        }\n\
         \x20       threadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \x20   }\n\n",
    );
}

fn emit_global_reduction(
    src: &mut String,
    aggregates: &[(AggFunc, String)],
    schema: &[ColumnInfo],
    _agg_count: usize,
) {
    src.push_str("    // === GLOBAL REDUCTION (device atomics) ===\n");
    src.push_str("    if (lid == 0) {\n");
    src.push_str("        for (uint g = 0; g < MAX_GROUPS; g++) {\n");
    src.push_str("            if (accum[g].valid == 0) continue;\n");
    src.push_str("            output->group_keys[g] = accum[g].group_key;\n");

    for (a, (func, col_name)) in aggregates.iter().enumerate() {
        let col_type = column_type_constant(schema, col_name);
        let is_float = col_type == "COLUMN_TYPE_FLOAT32";

        src.push_str(&format!(
            "            // Agg {a}: {func}\n\
             \x20           {{\n\
             \x20               uint off = 2080 + (g * MAX_AGGS + {a}) * 16;\n\
             \x20               device char* base = (device char*)output;\n\
             \x20               device atomic_uint* val_int_lo = (device atomic_uint*)(base + off);\n\
             \x20               device atomic_uint* val_int_hi = (device atomic_uint*)(base + off + 4);\n\
             \x20               device atomic_uint* val_float = (device atomic_uint*)(base + off + 8);\n\
             \x20               device atomic_uint* val_count = (device atomic_uint*)(base + off + 12);\n\
             \x20               atomic_fetch_add_explicit(val_count, static_cast<uint>(accum[g].count), memory_order_relaxed);\n",
        ));

        match func {
            AggFunc::Count => {
                src.push_str(&format!(
                    "                atomic_add_int64(val_int_lo, val_int_hi, accum[g].sum_int[{a}]);\n",
                ));
            }
            AggFunc::Sum | AggFunc::Avg => {
                if is_float {
                    src.push_str(&format!(
                        "                atomic_add_float(val_float, accum[g].sum_float[{a}]);\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "                atomic_add_int64(val_int_lo, val_int_hi, accum[g].sum_int[{a}]);\n",
                    ));
                }
            }
            AggFunc::Min => {
                if is_float {
                    src.push_str(&format!(
                        "                atomic_min_float(val_float, accum[g].min_float[{a}]);\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "                atomic_min_int64(val_int_lo, val_int_hi, accum[g].min_int[{a}]);\n",
                    ));
                }
            }
            AggFunc::Max => {
                if is_float {
                    src.push_str(&format!(
                        "                atomic_max_float(val_float, accum[g].max_float[{a}]);\n",
                    ));
                } else {
                    src.push_str(&format!(
                        "                atomic_max_int64(val_int_lo, val_int_hi, accum[g].max_int[{a}]);\n",
                    ));
                }
            }
        }
        src.push_str("            }\n");
    }

    src.push_str("        }\n    }\n\n");
}

fn emit_output_metadata(src: &mut String, has_group_by: bool, agg_count: usize) {
    src.push_str("    // === OUTPUT METADATA ===\n");
    src.push_str("    device atomic_uint* tg_done_counter = (device atomic_uint*)&output->error_code;\n");
    src.push_str("    uint total_tgs = (row_count + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;\n");
    src.push_str("    threadgroup_barrier(mem_flags::mem_device);\n\n");
    src.push_str("    if (lid == 0) {\n");
    src.push_str("        uint prev_done = atomic_fetch_add_explicit(tg_done_counter, 1u, memory_order_relaxed);\n");
    src.push_str("        if (prev_done + 1 == total_tgs) {\n");

    if has_group_by {
        // Count valid groups
        src.push_str("            uint group_count = 0;\n");
        src.push_str(&format!(
            "            for (uint g = 0; g < MAX_GROUPS; g++) {{\n\
             \x20               uint off = 2080 + (g * MAX_AGGS + 0) * 16;\n\
             \x20               device char* base = (device char*)output;\n\
             \x20               device atomic_uint* cnt = (device atomic_uint*)(base + off + 12);\n\
             \x20               if (atomic_load_explicit(cnt, memory_order_relaxed) > 0) group_count++;\n\
             \x20           }}\n",
        ));
    } else {
        src.push_str("            uint group_count = 1;\n");
    }

    src.push_str("            output->result_row_count = group_count;\n");
    src.push_str(&format!(
        "            output->result_col_count = {agg_count};\n"
    ));
    src.push_str("            output->sequence_id = params->sequence_id;\n");
    src.push_str("            output->error_code = 0;\n");
    src.push_str("            threadgroup_barrier(mem_flags::mem_device);\n");
    src.push_str("            output->ready_flag = 1;\n");
    src.push_str("        }\n    }\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::{AggFunc, CompareOp, LogicalOp, Value};

    /// Helper: build a scan node.
    fn scan(table: &str) -> PhysicalPlan {
        PhysicalPlan::GpuScan {
            table: table.into(),
            columns: vec!["amount".into(), "region".into()],
        }
    }

    /// Helper: build a filter node (column > value).
    fn filter_gt(column: &str, val: Value, input: PhysicalPlan) -> PhysicalPlan {
        PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Gt,
            column: column.into(),
            value: val,
            input: Box::new(input),
        }
    }

    /// Helper: build a compound AND filter.
    fn compound_and(left: PhysicalPlan, right: PhysicalPlan) -> PhysicalPlan {
        PhysicalPlan::GpuCompoundFilter {
            op: LogicalOp::And,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Helper: build an aggregate node.
    fn aggregate(
        functions: Vec<(AggFunc, &str)>,
        group_by: Vec<&str>,
        input: PhysicalPlan,
    ) -> PhysicalPlan {
        PhysicalPlan::GpuAggregate {
            functions: functions
                .into_iter()
                .map(|(f, c)| (f, c.to_string()))
                .collect(),
            group_by: group_by.into_iter().map(|s| s.to_string()).collect(),
            input: Box::new(input),
        }
    }

    // -----------------------------------------------------------------------
    // Test: deterministic -- same plan always produces the same hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_deterministic() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let h1 = plan_structure_hash(&plan);
        let h2 = plan_structure_hash(&plan);
        assert_eq!(h1, h2, "same plan must produce same hash");
    }

    // -----------------------------------------------------------------------
    // Test: structural equality -- different literals produce the same hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_structural_equality_different_literals() {
        let plan_a = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let plan_b = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(300), scan("sales")),
        );
        assert_eq!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "different literal values must produce same hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: different structure produces different hash
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_different_structure() {
        // Plan A: COUNT(*) with filter
        let plan_a = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        // Plan B: COUNT(*) without filter
        let plan_b = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "structurally different plans must produce different hashes"
        );
    }

    // -----------------------------------------------------------------------
    // Test: filter compare_op matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_filter_op_matters() {
        let plan_gt = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let plan_lt = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            PhysicalPlan::GpuFilter {
                compare_op: CompareOp::Lt,
                column: "amount".into(),
                value: Value::Int(500),
                input: Box::new(scan("sales")),
            },
        );
        assert_ne!(
            plan_structure_hash(&plan_gt),
            plan_structure_hash(&plan_lt),
            "different compare_op must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: aggregate function type matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_agg_func_matters() {
        let plan_count = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let plan_sum = aggregate(vec![(AggFunc::Sum, "amount")], vec![], scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_count),
            plan_structure_hash(&plan_sum),
            "different agg function must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: GROUP BY column matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_group_by_matters() {
        let plan_no_group = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let plan_group = aggregate(
            vec![(AggFunc::Count, "*")],
            vec!["region"],
            scan("sales"),
        );
        assert_ne!(
            plan_structure_hash(&plan_no_group),
            plan_structure_hash(&plan_group),
            "GROUP BY vs no GROUP BY must produce different hash"
        );
    }

    // -----------------------------------------------------------------------
    // Test: compound filter hashes child ops
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_compound_filter() {
        let left = filter_gt("amount", Value::Int(100), scan("sales"));
        let right = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(1000),
            input: Box::new(scan("sales")),
        };
        let compound = compound_and(left, right);
        let plan_a = aggregate(vec![(AggFunc::Count, "*")], vec![], compound);

        // Same structure, different literals
        let left2 = filter_gt("amount", Value::Int(200), scan("sales"));
        let right2 = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(2000),
            input: Box::new(scan("sales")),
        };
        let compound2 = compound_and(left2, right2);
        let plan_b = aggregate(vec![(AggFunc::Count, "*")], vec![], compound2);

        assert_eq!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "compound filters with same structure but different literals must match"
        );
    }

    // -----------------------------------------------------------------------
    // Test: filter column reference matters
    // -----------------------------------------------------------------------
    #[test]
    fn test_plan_hash_filter_column_matters() {
        let plan_a = filter_gt("amount", Value::Int(500), scan("sales"));
        let plan_b = filter_gt("price", Value::Int(500), scan("sales"));
        assert_ne!(
            plan_structure_hash(&plan_a),
            plan_structure_hash(&plan_b),
            "different filter column must produce different hash"
        );
    }

    // ===================================================================
    // Metal source generation tests
    // ===================================================================

    use crate::gpu::autonomous::loader::ColumnInfo;
    use crate::storage::schema::DataType;

    /// Helper: create a test schema with amount (INT64) and region (INT64).
    fn test_schema() -> Vec<ColumnInfo> {
        vec![
            ColumnInfo {
                name: "amount".into(),
                data_type: DataType::Int64,
            },
            ColumnInfo {
                name: "region".into(),
                data_type: DataType::Int64,
            },
        ]
    }

    /// Helper: create a test schema with amount (INT64), region (INT64), price (FLOAT64).
    #[allow(dead_code)]
    fn test_schema_with_float() -> Vec<ColumnInfo> {
        vec![
            ColumnInfo {
                name: "amount".into(),
                data_type: DataType::Int64,
            },
            ColumnInfo {
                name: "region".into(),
                data_type: DataType::Int64,
            },
            ColumnInfo {
                name: "price".into(),
                data_type: DataType::Float64,
            },
        ]
    }

    // -----------------------------------------------------------------------
    // Generate test 1: COUNT(*) no filter — kernel name, no filter code
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_count_star() {
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("fused_query_jit"),
            "must contain kernel name fused_query_jit"
        );
        assert!(
            !src.contains("FILTER PHASE"),
            "COUNT(*) without filter should not have FILTER PHASE"
        );
        assert!(
            src.contains("AGGREGATE PHASE"),
            "must have aggregate phase"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 2: SUM — contains sum accumulation
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_sum() {
        let plan = aggregate(vec![(AggFunc::Sum, "amount")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("jit_simd_sum_int64"),
            "SUM(int) must use jit_simd_sum_int64"
        );
        assert!(
            src.contains("atomic_add_int64"),
            "SUM must use atomic_add_int64 for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 3: filter GT — contains > comparison
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_gt() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("FILTER PHASE"),
            "must have FILTER PHASE when filter present"
        );
        assert!(
            src.contains("> params->filters[0].value_int"),
            "GT filter must use > operator"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 4: compound filter — contains both predicates
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_compound_filter() {
        let left = filter_gt("amount", Value::Int(100), scan("sales"));
        let right = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(1000),
            input: Box::new(scan("sales")),
        };
        let compound = compound_and(left, right);
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], compound);
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("2 predicates"),
            "compound filter must show 2 predicates"
        );
        assert!(
            src.contains("> params->filters[0].value_int"),
            "first filter must be GT"
        );
        assert!(
            src.contains("< params->filters[1].value_int"),
            "second filter must be LT"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 5: GROUP BY — contains bucket calculation
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_group_by() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec!["region"],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("% MAX_GROUPS"),
            "GROUP BY must have modular hash bucketing"
        );
        assert!(
            src.contains("Serial per-thread accumulation"),
            "GROUP BY uses serial merge strategy"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 6: headline query — filters + GROUP BY + multi-agg
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_headline_query() {
        // SELECT COUNT(*), SUM(amount) FROM sales
        // WHERE amount > 100 AND amount < 1000
        // GROUP BY region
        let left = filter_gt("amount", Value::Int(100), scan("sales"));
        let right = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(1000),
            input: Box::new(scan("sales")),
        };
        let compound = compound_and(left, right);
        let plan = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec!["region"],
            compound,
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        // Must have all components
        assert!(src.contains("FILTER PHASE"), "headline: filters");
        assert!(src.contains("% MAX_GROUPS"), "headline: GROUP BY");
        assert!(src.contains("AGGREGATE PHASE (2 aggs)"), "headline: 2 aggs");
        assert!(
            src.contains("atomic_add_int64"),
            "headline: int64 SUM atomic merge"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 7: no dead code — filter_count=0 means no filter code
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_no_dead_code() {
        // No filters, no GROUP BY
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            !src.contains("FILTER PHASE"),
            "no filter should not emit FILTER PHASE"
        );
        assert!(
            !src.contains("% MAX_GROUPS"),
            "no GROUP BY should not emit bucket code"
        );
        assert!(
            src.contains("SIMD reduction"),
            "no GROUP BY should use SIMD reduction"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 8: different plans produce different source
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_different_plans_different_source() {
        let plan_a = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let plan_b = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec!["region"],
            filter_gt("amount", Value::Int(100), scan("sales")),
        );
        let src_a = JitCompiler::generate_metal_source(&plan_a, &test_schema());
        let src_b = JitCompiler::generate_metal_source(&plan_b, &test_schema());

        assert_ne!(
            src_a, src_b,
            "different plans must produce different source"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 9: includes autonomous_types.h
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_includes_header() {
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("#include \"autonomous_types.h\""),
            "must include autonomous_types.h"
        );
        assert!(
            src.contains("#include <metal_stdlib>"),
            "must include metal_stdlib"
        );
    }

    // -----------------------------------------------------------------------
    // Generate test 10: kernel function name is fused_query_jit
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_kernel_function_name() {
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("kernel void fused_query_jit("),
            "kernel function must be named fused_query_jit"
        );
        assert!(
            !src.contains("kernel void fused_query("),
            "must NOT contain the AOT kernel name 'fused_query(' without _jit"
        );
    }

    // ===================================================================
    // JIT runtime compilation + PSO cache tests
    // ===================================================================

    use crate::gpu::device::GpuDevice;

    // -----------------------------------------------------------------------
    // Compile test 1: headline query compiles to valid PSO
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_headline() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        // Headline query: COUNT(*), SUM(amount) WHERE amount > 100 GROUP BY region
        let left = filter_gt("amount", Value::Int(100), scan("sales"));
        let right = PhysicalPlan::GpuFilter {
            compare_op: CompareOp::Lt,
            column: "amount".into(),
            value: Value::Int(1000),
            input: Box::new(scan("sales")),
        };
        let compound = compound_and(left, right);
        let plan = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec!["region"],
            compound,
        );

        let result = jit.compile(&plan, &test_schema());
        assert!(result.is_ok(), "JIT compile failed: {:?}", result.err());

        let compiled = result.unwrap();
        assert!(compiled.plan_hash != 0, "plan_hash should be non-zero");
    }

    // -----------------------------------------------------------------------
    // Compile test 2: cache hit returns same entry (no recompilation)
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_cache_hit() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));

        // First compile
        let result1 = jit.compile(&plan, &test_schema());
        assert!(result1.is_ok(), "first compile failed");
        let hash1 = result1.unwrap().plan_hash;

        // Second compile (same plan structure) — should hit cache
        let result2 = jit.compile(&plan, &test_schema());
        assert!(result2.is_ok(), "second compile failed");
        let hash2 = result2.unwrap().plan_hash;

        assert_eq!(hash1, hash2, "cache hit should return same plan_hash");
        assert_eq!(jit.cache_len(), 1, "cache should have exactly 1 entry");
    }

    // -----------------------------------------------------------------------
    // Compile test 3: cache miss for different plan structure
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_cache_miss() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        // Plan A: COUNT(*)
        let plan_a = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));

        // Plan B: COUNT(*) with filter
        let plan_b = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(100), scan("sales")),
        );

        let result_a = jit.compile(&plan_a, &test_schema());
        assert!(result_a.is_ok(), "plan A compile failed");

        let result_b = jit.compile(&plan_b, &test_schema());
        assert!(result_b.is_ok(), "plan B compile failed");

        assert_eq!(
            jit.cache_len(),
            2,
            "different plan structures should produce separate cache entries"
        );
    }

    // -----------------------------------------------------------------------
    // Compile test 4: simple COUNT(*) compiles
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_simple_count() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let result = jit.compile(&plan, &test_schema());
        assert!(
            result.is_ok(),
            "simple COUNT(*) JIT compile failed: {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------------
    // Compile test 5: CompiledPlan.source_len > 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_returns_source_len() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let compiled = jit.compile(&plan, &test_schema()).expect("compile failed");

        assert!(
            compiled.source_len > 0,
            "source_len should be > 0, got {}",
            compiled.source_len
        );
        // JIT source (with inlined header) should be substantial
        assert!(
            compiled.source_len > 1000,
            "source_len should be > 1000 (inlined header + kernel), got {}",
            compiled.source_len
        );
    }

    // -----------------------------------------------------------------------
    // Compile test 6: different literals produce cache hit (same structure hash)
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_different_literals_cache_hit() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan_a = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(500), scan("sales")),
        );
        let plan_b = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_gt("amount", Value::Int(300), scan("sales")),
        );

        jit.compile(&plan_a, &test_schema())
            .expect("plan A compile failed");
        jit.compile(&plan_b, &test_schema())
            .expect("plan B compile failed");

        assert_eq!(
            jit.cache_len(),
            1,
            "same structure with different literals should share PSO (cache hit)"
        );
    }

    // ===================================================================
    // Comprehensive source generation tests (per-pattern)
    // ===================================================================

    /// Helper: build a filter node with arbitrary op.
    fn filter_op(
        column: &str,
        op: CompareOp,
        val: Value,
        input: PhysicalPlan,
    ) -> PhysicalPlan {
        PhysicalPlan::GpuFilter {
            compare_op: op,
            column: column.into(),
            value: val,
            input: Box::new(input),
        }
    }

    // -----------------------------------------------------------------------
    // Source pattern: COUNT uses atomic_fetch_add in global reduction
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_count_uses_atomic_fetch_add() {
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("atomic_fetch_add_explicit"),
            "COUNT must use atomic_fetch_add_explicit for global count merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: SUM(int) uses jit_simd_sum_int64
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_sum_int_uses_simd_sum() {
        let plan = aggregate(vec![(AggFunc::Sum, "amount")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("jit_simd_sum_int64"),
            "SUM(int) must use jit_simd_sum_int64 for SIMD reduction"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: MIN(int) uses jit_simd_min_int64
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_min_int_uses_simd_min() {
        let plan = aggregate(vec![(AggFunc::Min, "amount")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("jit_simd_min_int64"),
            "MIN(int) must use jit_simd_min_int64 for SIMD reduction"
        );
        assert!(
            src.contains("atomic_min_int64"),
            "MIN(int) must use atomic_min_int64 for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: MAX(int) uses jit_simd_max_int64
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_max_int_uses_simd_max() {
        let plan = aggregate(vec![(AggFunc::Max, "amount")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("jit_simd_max_int64"),
            "MAX(int) must use jit_simd_max_int64 for SIMD reduction"
        );
        assert!(
            src.contains("atomic_max_int64"),
            "MAX(int) must use atomic_max_int64 for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: SUM(float) uses simd_sum built-in
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_sum_float_uses_simd_sum() {
        let plan = aggregate(vec![(AggFunc::Sum, "price")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema_with_float());

        assert!(
            src.contains("simd_sum(local_agg_0)"),
            "SUM(float) must use Metal's simd_sum"
        );
        assert!(
            src.contains("atomic_add_float"),
            "SUM(float) must use atomic_add_float for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: MIN(float) uses simd_min built-in
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_min_float_uses_simd_min() {
        let plan = aggregate(vec![(AggFunc::Min, "price")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema_with_float());

        assert!(
            src.contains("simd_min(local_agg_0)"),
            "MIN(float) must use Metal's simd_min"
        );
        assert!(
            src.contains("atomic_min_float"),
            "MIN(float) must use atomic_min_float for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: MAX(float) uses simd_max built-in
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_max_float_uses_simd_max() {
        let plan = aggregate(vec![(AggFunc::Max, "price")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source(&plan, &test_schema_with_float());

        assert!(
            src.contains("simd_max(local_agg_0)"),
            "MAX(float) must use Metal's simd_max"
        );
        assert!(
            src.contains("atomic_max_float"),
            "MAX(float) must use atomic_max_float for global merge"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: filter LT operator
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_lt_operator() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_op("amount", CompareOp::Lt, Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("< params->filters[0].value_int"),
            "LT filter must use < operator"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: filter EQ operator
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_eq_operator() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_op("amount", CompareOp::Eq, Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("== params->filters[0].value_int"),
            "EQ filter must use == operator"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: filter GE operator
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_ge_operator() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_op("amount", CompareOp::Ge, Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains(">= params->filters[0].value_int"),
            "GE filter must use >= operator"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: filter LE operator
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_le_operator() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_op("amount", CompareOp::Le, Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("<= params->filters[0].value_int"),
            "LE filter must use <= operator"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: filter NE operator
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_filter_ne_operator() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*")],
            vec![],
            filter_op("amount", CompareOp::Ne, Value::Int(500), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("!= params->filters[0].value_int"),
            "NE filter must use != operator"
        );
    }

    // -----------------------------------------------------------------------
    // Negative: no GROUP BY means no bucket/hash code emitted
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_no_groupby_no_hash_table() {
        let plan = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec![],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            !src.contains("% MAX_GROUPS"),
            "no GROUP BY must NOT emit modular hash bucketing"
        );
        assert!(
            !src.contains("Serial per-thread accumulation"),
            "no GROUP BY must NOT use serial merge"
        );
        assert!(
            src.contains("SIMD reduction"),
            "no GROUP BY should use SIMD reduction path"
        );
    }

    // -----------------------------------------------------------------------
    // Negative: no filter means no FILTER PHASE section
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_no_filter_no_filter_section() {
        let plan = aggregate(
            vec![(AggFunc::Sum, "amount")],
            vec!["region"],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            !src.contains("FILTER PHASE"),
            "no filters must NOT emit FILTER PHASE section"
        );
        assert!(
            src.contains("GROUP BY PHASE"),
            "GROUP BY should still be present"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: multi-agg without GROUP BY (all SIMD reduced)
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_multi_agg_no_groupby() {
        let plan = aggregate(
            vec![
                (AggFunc::Count, "*"),
                (AggFunc::Sum, "amount"),
                (AggFunc::Min, "amount"),
                (AggFunc::Max, "amount"),
            ],
            vec![],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("AGGREGATE PHASE (4 aggs)"),
            "must report 4 aggs in phase comment"
        );
        // Each agg should have its own SIMD reduction variable
        assert!(src.contains("simd_agg_0"), "must have simd_agg_0");
        assert!(src.contains("simd_agg_1"), "must have simd_agg_1");
        assert!(src.contains("simd_agg_2"), "must have simd_agg_2");
        assert!(src.contains("simd_agg_3"), "must have simd_agg_3");
    }

    // -----------------------------------------------------------------------
    // Source pattern: GROUP BY uses serial merge, not SIMD
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_groupby_uses_serial_merge() {
        let plan = aggregate(
            vec![(AggFunc::Sum, "amount")],
            vec!["region"],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("Serial per-thread accumulation"),
            "GROUP BY must use serial per-thread merge"
        );
        assert!(
            !src.contains("SIMD reduction (no GROUP BY"),
            "GROUP BY must NOT use SIMD reduction comment"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: JIT source inlines header (no #include types)
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_jit_source_inlines_header() {
        let plan = aggregate(vec![(AggFunc::Count, "*")], vec![], scan("sales"));
        let src = JitCompiler::generate_metal_source_for_jit(&plan, &test_schema());

        assert!(
            !src.contains("#include \"autonomous_types.h\""),
            "JIT source must NOT use #include for types (header inlined)"
        );
        assert!(
            src.contains("Inlined autonomous_types.h"),
            "JIT source must have inlined header comment"
        );
        // Verify key struct definitions are inlined
        assert!(
            src.contains("struct QueryParamsSlot"),
            "JIT source must inline QueryParamsSlot definition"
        );
        assert!(
            src.contains("struct OutputBuffer"),
            "JIT source must inline OutputBuffer definition"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: column type read functions used correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_float_column_uses_read_float32() {
        let plan = aggregate(
            vec![(AggFunc::Sum, "price")],
            vec![],
            filter_op("price", CompareOp::Gt, Value::Float(100.0), scan("sales")),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema_with_float());

        assert!(
            src.contains("read_float32"),
            "float column must use read_float32 helper"
        );
    }

    // -----------------------------------------------------------------------
    // Compile test: invalid Metal source returns Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_invalid_source_returns_err() {
        let gpu = GpuDevice::new();

        // Directly test that invalid MSL fails gracefully
        let invalid_source = "kernel void bad_kernel() { this_is_invalid_code; }";
        let ns_source = NSString::from_str(invalid_source);
        let result = gpu
            .device
            .newLibraryWithSource_options_error(&ns_source, None);

        assert!(
            result.is_err(),
            "invalid Metal source must return Err from newLibraryWithSource"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: output metadata sets correct agg count
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_output_metadata_agg_count() {
        let plan = aggregate(
            vec![
                (AggFunc::Count, "*"),
                (AggFunc::Sum, "amount"),
                (AggFunc::Min, "amount"),
            ],
            vec![],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("output->result_col_count = 3"),
            "output metadata must set result_col_count to agg count (3)"
        );
    }

    // -----------------------------------------------------------------------
    // Source pattern: single agg array size matches agg count in JIT
    // -----------------------------------------------------------------------
    #[test]
    fn test_generate_accumulator_sized_to_agg_count() {
        // 2-agg plan: accum arrays should be sized [2], not [5]
        let plan = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec![],
            scan("sales"),
        );
        let src = JitCompiler::generate_metal_source(&plan, &test_schema());

        assert!(
            src.contains("long sum_int[2]"),
            "accumulator sum_int array must be sized to exact agg_count=2"
        );
        assert!(
            !src.contains("long sum_int[5]"),
            "JIT must NOT use MAX_AGGS=5 for accumulator arrays"
        );
    }

    // -----------------------------------------------------------------------
    // Compile: GROUP BY query compiles successfully
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_group_by_query() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan = aggregate(
            vec![(AggFunc::Count, "*"), (AggFunc::Sum, "amount")],
            vec!["region"],
            filter_gt("amount", Value::Int(100), scan("sales")),
        );

        let result = jit.compile(&plan, &test_schema());
        assert!(
            result.is_ok(),
            "GROUP BY query JIT compile failed: {:?}",
            result.err()
        );
    }

    // -----------------------------------------------------------------------
    // Compile: MIN/MAX query compiles successfully
    // -----------------------------------------------------------------------
    #[test]
    fn test_compile_min_max_query() {
        let gpu = GpuDevice::new();
        let mut jit = JitCompiler::new(gpu.device.clone());

        let plan = aggregate(
            vec![(AggFunc::Min, "amount"), (AggFunc::Max, "amount")],
            vec![],
            scan("sales"),
        );

        let result = jit.compile(&plan, &test_schema());
        assert!(
            result.is_ok(),
            "MIN/MAX query JIT compile failed: {:?}",
            result.err()
        );
    }
}
