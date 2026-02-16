---
id: parsers.BREAKDOWN
module: parsers
priority: 2
status: failing
version: 1
origin: spec-workflow
dependsOn: [io.BREAKDOWN]
tags: [gpu-query]
testRequirements:
  unit:
    required: false
    pattern: "tests/parsers/**/*.test.*"
---
# Parsers -- BREAKDOWN

## Context

gpu-query parses CSV, JSON, and Parquet files on the GPU using Metal compute kernels. The CSV parser uses a 2-phase approach (newline detection + field extraction) adapted from nvParse/RAPIDS cudf::io. The JSON parser uses structural indexing inspired by GpJSON (VLDB 2025) which achieves 2.9x over parallel CPU parsers [KB #472]. The Parquet decoder reads already-columnar data with GPU-side decoding of PLAIN, DICTIONARY, and RLE encodings. A schema inference kernel uses GPU-parallel type voting from sample data. These are the highest-complexity GPU kernels in the system.

## Scope

### Metal Shader Files to Implement

- `gpu-query/shaders/csv_parse.metal` -- Two kernels:
  - `csv_detect_newlines()` -- Phase 1: each thread scans 4KB chunk for '\n', accounting for quoted strings (state machine per chunk). Output: row_offsets[] via prefix scan [KB #193]
  - `csv_parse_fields()` -- Phase 2: each thread processes one row, scans for delimiters, handles quoted fields, performs type coercion (atoi, atof, dict lookup), writes to columnar SoA buffers

- `gpu-query/shaders/json_parse.metal` -- Two kernels:
  - `json_structural_index()` -- Phase 1: parallel scan for structural characters { } [ ] , : ", build bitmask, handle escape sequences
  - `json_extract_columns()` -- Phase 2: each thread extracts one field from one NDJSON record, navigates structural index, writes typed values to SoA columns

- `gpu-query/shaders/parquet_decode.metal` -- Four kernels:
  - `parquet_decode_plain_int64()` -- Plain encoding decoder for INT64
  - `parquet_decode_plain_double()` -- Plain encoding decoder for DOUBLE
  - `parquet_decode_dictionary()` -- Dictionary page decoder (index lookup into dictionary table)
  - `parquet_decode_rle()` -- Run-length encoding decoder

- `gpu-query/shaders/schema_infer.metal` -- One kernel:
  - `infer_schema()` -- Each thread examines one field from sample data, classifies type (NULL/BOOL/INT64/FLOAT64/DATE/VARCHAR), atomically votes in type_votes[col][type] matrix

### Rust Files to Implement

- `gpu-query/src/gpu/encode.rs` -- Compute encoder helpers for parser kernels (buffer binding, threadgroup sizing, dispatch)
- Extend `gpu-query/src/io/csv.rs` -- CPU-side header parsing feeds column count to GPU kernel params
- Extend `gpu-query/src/io/json.rs` -- NDJSON record boundary detection (newlines) feeds row_offsets to GPU
- Extend `gpu-query/src/io/parquet.rs` -- CPU reads Parquet footer/metadata, selects columns, mmap's relevant byte ranges for GPU decoding

### Tests to Write (GPU integration, ~45 tests per QA Section 2.2)

- CSV: row boundary detection (LF, CRLF, quoted newlines, escaped quotes, empty lines, no final newline, single row, Unicode)
- CSV: field extraction (integers, floats, strings, empty fields, NULL representations, tab/pipe delimiters, trailing whitespace)
- JSON: structural indexing correctness, field extraction for NDJSON records, escape sequences, nested objects
- Parquet: PLAIN encoding decode, DICTIONARY decode, RLE decode, null bitmap handling
- Schema inference: type voting, mixed types, all-NULL columns, promotion rules (INT->FLOAT)
- Cross-validation: CSV fields parsed by GPU == fields parsed by Rust `csv` crate

## Acceptance Criteria

1. CSV parser kernel correctly tokenizes rows and extracts typed fields for a 1000-row test file, matching Rust `csv` crate output
2. JSON parser kernel correctly extracts columns from a 1000-record NDJSON file, matching `serde_json` output
3. Parquet decoder correctly decodes PLAIN and DICTIONARY encoded columns, matching `parquet` crate output
4. Schema inference kernel correctly detects column types from sample data with >95% accuracy
5. All GPU integration tests pass: `cargo test --test gpu_csv`, `cargo test --test gpu_json`, `cargo test --test gpu_parquet`, `cargo test --test gpu_schema`
6. Edge cases handled: empty files, single-row files, all-NULL columns, Unicode strings, quoted fields with embedded delimiters

## References

- PM: Section 5 (Technical Feasibility -- GPU JSON/CSV parsing), Section 8 (MVP: CSV parser, Parquet reader must-have), Risk R5 (GPU kernel complexity)
- UX: Section 2.1 (auto-detection and schema inference at startup)
- TECH: Section 3.1 (CSV Parser Kernel), Section 3.2 (JSON Parser Kernel), Section 3.3 (Parquet Decoder Kernel), Section 3.8 (Schema Inference Kernel)
- QA: Section 3.1 (CSV parser correctness), Section 2.2 (GPU integration tests), Section 9 (fuzz testing for parsers)
- KB: #193 (prefix scan), #311 (columnar tile-based execution), #330 (GPU JSON: cuJSON), #347 (GPU string matching), #472 (GpJSON: 2.9x over CPU)

## Technical Notes

- CSV quoting with state machines is inherently sequential per row; parallelize across rows (Phase 1 finds boundaries first), then within rows -- same approach as RAPIDS cudf::io::read_csv
- NDJSON simplifies JSON parsing significantly: record boundaries are newlines, no need for nested structure navigation for top-level fields
- Parquet decompression (SNAPPY/ZSTD) runs on CPU for MVP via the `parquet` crate; GPU reads decompressed columns. GPU decompression kernels planned for Phase 2
- Schema inference type promotion rules: if column has 90% INT64 and 10% FLOAT64, promote to FLOAT64; any VARCHAR forces VARCHAR for entire column
- All parser kernels read from mmap'd Metal buffers (bytesNoCopy) created by the io module -- zero-copy from SSD to GPU
- Function constants can specialize parser kernels by delimiter character, NULL representation, and date format
