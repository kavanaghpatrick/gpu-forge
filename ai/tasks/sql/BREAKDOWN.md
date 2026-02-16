---
id: sql.BREAKDOWN
module: sql
priority: 4
status: failing
version: 1
origin: spec-workflow
dependsOn: [storage.BREAKDOWN]
tags: [gpu-query]
testRequirements:
  unit:
    required: false
    pattern: "tests/sql/**/*.test.*"
---
# SQL -- BREAKDOWN

## Context

The SQL module transforms user SQL text into a Metal compute kernel dispatch plan. It uses sqlparser-rs (the same SQL parser used by Apache DataFusion) for parsing, then implements a custom minimal query planner for the MVP SQL subset. The planner performs column pruning, predicate pushdown, and type checking, then generates a physical plan that maps logical operators to specific Metal kernel variants via MTLFunctionConstantValues [KB #202, #210]. The PSO (Pipeline State Object) cache ensures kernel compilation overhead is amortized across queries (~1ms first creation, cached after) [KB #159].

## Scope

### Rust Files to Create/Implement

- `gpu-query/src/sql/mod.rs` -- Module root, public API
- `gpu-query/src/sql/parser.rs` -- sqlparser-rs integration:
  - Parse SELECT/FROM/WHERE/GROUP BY/ORDER BY/LIMIT
  - Parse aggregates: COUNT(*), COUNT(col), SUM(col), AVG(col), MIN(col), MAX(col)
  - Parse predicates: =, <, >, <=, >=, !=, IS NULL, IS NOT NULL, BETWEEN, IN (list)
  - Parse expressions: col + col, col * literal, CAST(col AS type)
  - Error handling: user-friendly messages with caret pointing to error position
  - DESCRIBE table and dot commands (.tables, .schema, .gpu)
- `gpu-query/src/sql/logical_plan.rs` -- Logical plan nodes:
  - `LogicalPlan` enum: Scan, Filter, Aggregate, Sort, Limit, Project
  - Column pruning: only load columns referenced in SELECT, WHERE, GROUP BY, ORDER BY
  - Type checking: validate column types vs operations (e.g., SUM on numeric only)
- `gpu-query/src/sql/physical_plan.rs` -- Physical plan (kernel graph):
  - Map logical operators to Metal kernel names + function constant values
  - `KernelGraph`: ordered sequence of (kernel_name, function_constants, buffer_bindings)
  - Indirect dispatch sizing via prepare_dispatch kernel for variable-output stages
- `gpu-query/src/sql/optimizer.rs` -- Query optimizer:
  - Predicate pushdown: move WHERE clauses as close to scan as possible
  - Column pruning: propagate required columns through plan tree
  - Filter ordering: evaluate cheaper predicates first
- `gpu-query/src/sql/types.rs` -- DataType enum, schema definitions (shared with storage module)
- `gpu-query/src/sql/expressions.rs` -- Expression tree:
  - `Expr` enum: BinaryOp, Literal, ColumnRef, AggregateCall, Cast, IsNull
  - Expression evaluation planning for compound WHERE clauses
- `gpu-query/src/gpu/pipeline.rs` -- PSO creation + caching + function constants:
  - `PsoCache`: HashMap<(KernelName, FunctionConstantHash), MTLComputePipelineState>
  - `function_constants_for_filter(predicate)`: set COMPARE_OP, COLUMN_TYPE, HAS_NULL_CHECK
  - Pre-compile common PSO variants at startup (INT64 EQ, FLOAT64 GT, etc.)

### Tests to Write (~80 tests per QA Section 2.1)

- SQL parser: ~50 tests for valid SQL parsing, invalid SQL rejection, error messages
- Query planner: ~30 tests for logical plan construction, column pruning, predicate pushdown, type checking, expression evaluation
- PSO cache: verify function constant combinations produce correct PSOs, cache hits on repeated queries

## Acceptance Criteria

1. sqlparser-rs correctly parses the MVP SQL subset: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, all 6 aggregate functions, all comparison operators, BETWEEN, IN, IS NULL
2. Logical planner produces correct column pruning (only referenced columns in scan)
3. Physical planner generates correct function constant values for each predicate type and column type combination
4. PSO cache correctly caches and retrieves pipeline state objects, hitting cache on repeated queries
5. User-friendly error messages with caret pointing to error position for SQL syntax errors
6. DESCRIBE command returns correct column names, types, and statistics
7. All unit tests pass: `cargo test -p gpu-query --lib sql`

## References

- PM: Section 9 Decision 5 (Query Compilation -- sqlparser-rs + Custom Physical Planner), Risk R4 (SQL parser/planner complexity)
- UX: Section 5 (Query Input UX -- autocomplete, error feedback), Section 2.2 (First Query)
- TECH: Section 4 (Query Compilation Pipeline -- full SQL-to-Metal dispatch flow), Section 3.4 (filter kernel function constants)
- QA: Section 2.1 Level 1 (SQL parser ~50 tests, planner ~30 tests), Section 6 (correctness validation)
- KB: #159 (PSO creation ~1ms), #202 (function constants: 84% instruction reduction), #210 (function constants superior to macros)

## Technical Notes

- sqlparser-rs `GenericDialect` is sufficient for the MVP SQL subset; no need for a custom dialect
- Function constants are the key to query specialization: COMPARE_OP (0=EQ through 5=NE), COLUMN_TYPE (0=INT64 through 3=BOOL), HAS_NULL_CHECK (bool) -- these are compiled into the PSO at creation time, eliminating branches [KB #202]
- Compound predicates (AND/OR) are handled by chaining filter kernel passes: AND = bitwise AND of two selection masks, OR = bitwise OR
- The MVP does NOT support JOINs, subqueries, CTEs, or window functions [PM-Q3]. These are Phase 2 with DataFusion planner integration.
- Expression evaluation (col + col, col * literal) can be folded into the filter kernel via additional function constants or a separate expression evaluation kernel
- PSO cache should be pre-warmed at startup with the most common combinations (INT64 + each operator, FLOAT64 + each operator, VARCHAR EQ)
