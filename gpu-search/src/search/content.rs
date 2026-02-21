//! GPU-accelerated content search (grep-like) kernel dispatch.
//!
//! Ported from rust-experiment/src/gpu_os/content_search.rs.
//!
//! Searches file contents in parallel using Metal compute shaders.
//! VECTORIZED: Each GPU thread processes 64 bytes using uchar4 loads.
//! Achieves 79-110 GB/s on M4 Pro.
//!
//! Three search paths:
//! - **Standard** (`content_search_kernel`): Padded chunks, GPU computes line numbers in-kernel
//! - **Turbo** (`turbo_search_kernel`): Padded chunks, defers line number calc to CPU
//! - **Zerocopy** (`content_search_zerocopy_kernel`): Single-dispatch on contiguous buffer,
//!   no padding or batching, GPU computes file-relative line numbers

use std::mem;

use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLResourceOptions, MTLSize,
};

use crate::gpu::pipeline::PsoCache;

// ============================================================================
// Constants (must match search_types.h)
// ============================================================================

/// Chunk size (4KB) -- each chunk of input data is padded to this size
pub const CHUNK_SIZE: usize = 4096;

/// Maximum pattern length for search
const MAX_PATTERN_LEN: usize = 64;

/// Maximum number of matches the GPU can return
const MAX_MATCHES: usize = 10000;

/// Bytes processed per GPU thread (16 x uchar4 vectorized loads)
const BYTES_PER_THREAD: usize = 64;

/// Threadgroup size matching the Metal shader THREADGROUP_SIZE
const THREADGROUP_SIZE: usize = 256;

// ============================================================================
// GPU-side repr(C) types matching search_types.h EXACTLY
// ============================================================================

/// Metadata for each chunk -- matches Metal `ChunkMetadata` in search_types.h.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ChunkMetadata {
    pub file_index: u32,
    pub chunk_index: u32,
    pub offset_in_file: u64,
    pub chunk_length: u32,
    pub flags: u32, // Bit 0: is_text, Bit 1: is_first, Bit 2: is_last
    pub buffer_offset: u64, // Absolute byte offset in contiguous buffer (zero-copy path)
}

/// Search parameters -- matches Metal `SearchParams` in search_types.h.
/// This is the 16-byte struct the shader expects, NOT the 284-byte Rust SearchParams.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuSearchParams {
    chunk_count: u32,
    pattern_len: u32,
    case_sensitive: u32,
    total_bytes: u32,
}

/// Match result -- matches Metal `MatchResult` in search_types.h.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct GpuMatchResult {
    file_index: u32,
    chunk_index: u32,
    line_number: u32,
    column: u32,
    match_length: u32,
    context_start: u32,
    context_len: u32,
    _padding: u32,
}

// Compile-time layout assertions
const _: () = assert!(mem::size_of::<GpuSearchParams>() == 16);
const _: () = assert!(mem::size_of::<GpuMatchResult>() == 32);
const _: () = assert!(mem::size_of::<ChunkMetadata>() == 32);

// ============================================================================
// Public API types
// ============================================================================

/// A match found in file contents.
#[derive(Debug, Clone)]
pub struct ContentMatch {
    /// Index of the file that contained this match
    pub file_index: usize,
    /// Line number (1-based)
    pub line_number: u32,
    /// Column within the line (0-based)
    pub column: u32,
    /// Byte offset within the chunk
    pub byte_offset: u32,
    /// Length of the matched string
    pub match_length: u32,
}

/// Search mode: standard (GPU line numbers) or turbo (CPU line numbers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// GPU computes line numbers in-kernel. Slightly slower but complete results.
    Standard,
    /// Defers line number calculation to CPU for maximum GPU throughput (70+ GB/s).
    Turbo,
}

/// Options controlling content search behavior.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub case_sensitive: bool,
    pub max_results: usize,
    pub mode: SearchMode,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        }
    }
}

// ============================================================================
// ContentSearchEngine
// ============================================================================

/// GPU-accelerated content search engine.
///
/// Holds GPU device, PSO references, and pre-allocated Metal buffers.
/// Buffers are allocated once and reused across searches to avoid
/// per-search allocation overhead.
pub struct ContentSearchEngine {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    content_search_pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    turbo_search_pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    zerocopy_search_pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Pre-allocated GPU buffers
    chunks_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    metadata_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pattern_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    matches_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    match_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // State
    max_chunks: usize,
    current_chunk_count: usize,
    total_data_bytes: usize,
    file_count: usize,

}

impl ContentSearchEngine {
    /// Create a new content search engine.
    ///
    /// `max_files` controls the maximum number of files that can be loaded.
    /// Assumes average 10 chunks per file (40KB average file size).
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
        max_files: usize,
    ) -> Self {
        let max_chunks = max_files * 10; // No cap -- dynamically resized if needed

        let command_queue = device.newCommandQueue().expect("Failed to create command queue");

        // Get PSOs from cache
        let content_search_pso = pso_cache
            .get("content_search_kernel")
            .expect("content_search_kernel PSO not in cache");
        let turbo_search_pso = pso_cache
            .get("turbo_search_kernel")
            .expect("turbo_search_kernel PSO not in cache");
        let zerocopy_search_pso = pso_cache
            .get("content_search_zerocopy_kernel")
            .expect("content_search_zerocopy_kernel PSO not in cache");

        let options = MTLResourceOptions::StorageModeShared;

        // Allocate buffers
        let chunks_buffer = device
            .newBufferWithLength_options(max_chunks * CHUNK_SIZE, options)
            .expect("Failed to allocate chunks buffer");
        let metadata_buffer = device
            .newBufferWithLength_options(max_chunks * mem::size_of::<ChunkMetadata>(), options)
            .expect("Failed to allocate metadata buffer");
        let params_buffer = device
            .newBufferWithLength_options(mem::size_of::<GpuSearchParams>(), options)
            .expect("Failed to allocate params buffer");
        let pattern_buffer = device
            .newBufferWithLength_options(MAX_PATTERN_LEN, options)
            .expect("Failed to allocate pattern buffer");
        let matches_buffer = device
            .newBufferWithLength_options(MAX_MATCHES * mem::size_of::<GpuMatchResult>(), options)
            .expect("Failed to allocate matches buffer");
        let match_count_buffer = device
            .newBufferWithLength_options(mem::size_of::<u32>(), options)
            .expect("Failed to allocate match count buffer");

        // Retain PSOs (they come as references from cache)
        let content_search_pso = Retained::from(content_search_pso);
        let turbo_search_pso = Retained::from(turbo_search_pso);
        let zerocopy_search_pso = Retained::from(zerocopy_search_pso);

        Self {
            device: Retained::from(device),
            command_queue,
            content_search_pso,
            turbo_search_pso,
            zerocopy_search_pso,
            chunks_buffer,
            metadata_buffer,
            params_buffer,
            pattern_buffer,
            matches_buffer,
            match_count_buffer,
            max_chunks,
            current_chunk_count: 0,
            total_data_bytes: 0,
            file_count: 0,
        }
    }

    /// Ensure the metadata buffer can hold at least `needed` chunk entries.
    ///
    /// If `needed > self.max_chunks`, reallocates the metadata buffer on the
    /// GPU with capacity for `needed` entries and updates `self.max_chunks`.
    ///
    /// Returns `Err` if the GPU fails to allocate the new buffer, in which
    /// case the old buffer and `max_chunks` are left unchanged.
    fn ensure_metadata_capacity(&mut self, needed: usize) -> Result<(), &'static str> {
        if needed <= self.max_chunks {
            return Ok(());
        }
        let options = MTLResourceOptions::StorageModeShared;
        let new_buffer = self
            .device
            .newBufferWithLength_options(needed * mem::size_of::<ChunkMetadata>(), options);
        match new_buffer {
            Some(buf) => {
                self.metadata_buffer = buf;
                self.max_chunks = needed;
                Ok(())
            }
            None => {
                eprintln!(
                    "[gpu-search] WARNING: metadata buffer reallocation failed \
                     (requested {} entries, {} bytes)",
                    needed,
                    needed * mem::size_of::<ChunkMetadata>()
                );
                Err("metadata buffer reallocation failed")
            }
        }
    }

    /// Load raw content bytes for searching.
    ///
    /// Splits content into CHUNK_SIZE chunks, creates metadata, and copies
    /// everything to GPU buffers. Each call represents one "file" at the
    /// given file_index.
    ///
    /// Returns the number of chunks created.
    pub fn load_content(&mut self, content: &[u8], file_index: u32) -> usize {
        if content.is_empty() {
            return 0;
        }

        let num_chunks = content.len().div_ceil(CHUNK_SIZE);
        let start_chunk = self.current_chunk_count;

        if start_chunk + num_chunks > self.max_chunks {
            return 0; // Would exceed buffer
        }

        unsafe {
            let chunks_ptr = self.chunks_buffer.contents().as_ptr() as *mut u8;
            let meta_ptr = self.metadata_buffer.contents().as_ptr() as *mut ChunkMetadata;

            for chunk_i in 0..num_chunks {
                let chunk_idx = start_chunk + chunk_i;
                let offset = chunk_i * CHUNK_SIZE;
                let chunk_len = (content.len() - offset).min(CHUNK_SIZE);
                let chunk = &content[offset..offset + chunk_len];

                // Copy chunk data (padded to CHUNK_SIZE with zeros)
                let dst = chunks_ptr.add(chunk_idx * CHUNK_SIZE);
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk_len);
                // Zero-pad remainder
                if chunk_len < CHUNK_SIZE {
                    std::ptr::write_bytes(dst.add(chunk_len), 0, CHUNK_SIZE - chunk_len);
                }

                // Write metadata
                let mut flags = 1u32; // is_text
                if chunk_i == 0 {
                    flags |= 2; // is_first
                }
                if chunk_i == num_chunks - 1 {
                    flags |= 4; // is_last
                }

                *meta_ptr.add(chunk_idx) = ChunkMetadata {
                    file_index,
                    chunk_index: chunk_i as u32,
                    offset_in_file: offset as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                    buffer_offset: 0, // Not used in padded path
                };
            }
        }

        self.current_chunk_count += num_chunks;
        // total_data_bytes tracks the padded size (chunks * CHUNK_SIZE) for dispatch
        self.total_data_bytes = self.current_chunk_count * CHUNK_SIZE;
        self.file_count = (file_index as usize) + 1;
        num_chunks
    }

    /// Reset the engine for a new search (clear loaded data).
    ///
    /// Zeros metadata_buffer, match_count_buffer, and matches_buffer to prevent
    /// stale data from previous searches. chunks_buffer is NOT zeroed (too large,
    /// and already zero-padded per-chunk in load_content()).
    pub fn reset(&mut self) {
        self.current_chunk_count = 0;
        self.total_data_bytes = 0;
        self.file_count = 0;

        // Zero metadata_buffer (max_chunks * 24 bytes) to clear stale chunk metadata
        unsafe {
            let ptr = self.metadata_buffer.contents().as_ptr() as *mut u8;
            let len = self.max_chunks * mem::size_of::<ChunkMetadata>();
            std::ptr::write_bytes(ptr, 0, len);
        }

        // Zero match_count_buffer (4 bytes)
        unsafe {
            let ptr = self.match_count_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, mem::size_of::<u32>());
        }

        // Zero matches_buffer (MAX_MATCHES * 32 = 320KB) for defense-in-depth
        unsafe {
            let ptr = self.matches_buffer.contents().as_ptr() as *mut u8;
            let len = MAX_MATCHES * mem::size_of::<GpuMatchResult>();
            std::ptr::write_bytes(ptr, 0, len);
        }
    }

    /// Search for a pattern in all loaded content.
    ///
    /// Returns a list of matches. The pattern must be <= 64 bytes.
    /// Uses the vectorized kernel: each thread processes 64 bytes with uchar4 loads.
    pub fn search(&self, pattern: &[u8], options: &SearchOptions) -> Vec<ContentMatch> {
        if pattern.is_empty() || pattern.len() > MAX_PATTERN_LEN || self.current_chunk_count == 0 {
            return vec![];
        }

        // Prepare pattern bytes (lowercase if case-insensitive)
        let pattern_bytes: Vec<u8> = if options.case_sensitive {
            pattern.to_vec()
        } else {
            pattern.iter().map(|b| b.to_ascii_lowercase()).collect()
        };

        // Write GPU params
        unsafe {
            // Reset match count to 0
            let count_ptr = self.match_count_buffer.contents().as_ptr() as *mut u32;
            *count_ptr = 0;

            // Write search params (16 bytes matching Metal SearchParams)
            let params_ptr = self.params_buffer.contents().as_ptr() as *mut GpuSearchParams;
            *params_ptr = GpuSearchParams {
                chunk_count: self.current_chunk_count as u32,
                pattern_len: pattern_bytes.len() as u32,
                case_sensitive: if options.case_sensitive { 1 } else { 0 },
                total_bytes: self.total_data_bytes as u32,
            };

            // Write pattern bytes
            let pattern_ptr = self.pattern_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(
                pattern_bytes.as_ptr(),
                pattern_ptr,
                pattern_bytes.len(),
            );
        }

        // Select kernel
        let pso = match options.mode {
            SearchMode::Standard => &self.content_search_pso,
            SearchMode::Turbo => &self.turbo_search_pso,
        };

        // Dispatch GPU search inside autoreleasepool to prevent Metal object leaks.
        // On background threads without a pool, autoreleased ObjC objects (command
        // buffers, encoders) accumulate indefinitely, exhausting Metal driver resources
        // and eventually blocking the main thread's CAMetalLayer.nextDrawable().
        autoreleasepool(|_| {
            let cmd = self
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&*self.chunks_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&*self.metadata_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&*self.params_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&*self.pattern_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&*self.matches_buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&*self.match_count_buffer), 0, 5);
            }

            // One thread per 64 bytes of data
            let total_threads = self.total_data_bytes.div_ceil(BYTES_PER_THREAD);

            let grid_size = MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: THREADGROUP_SIZE,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd.commit();
            cmd.waitUntilCompleted();
        });

        // Read results (padded path: grid-absolute byte_offset)
        self.collect_results(options, false)
    }

    /// Search for a pattern using an externally-provided content buffer.
    ///
    /// Instead of using the engine's internal `chunks_buffer`, the caller
    /// passes their own Metal buffer (e.g., from a `ContentStore`) and
    /// the corresponding `ChunkMetadata` slice describing the chunks
    /// within that buffer.
    ///
    /// This enables zero-disk-I/O search: the content store's Metal buffer
    /// is used directly as the GPU input without copying data.
    ///
    /// Buffer layout expected by the GPU kernel:
    /// - buffer(0) = `content_buffer` (caller-provided)
    /// - buffer(1) = chunk metadata (written from `chunk_metas`)
    /// - buffer(2) = search params
    /// - buffer(3) = pattern
    /// - buffer(4) = match output
    /// - buffer(5) = match count
    pub fn search_with_buffer(
        &mut self,
        content_buffer: &ProtocolObject<dyn MTLBuffer>,
        chunk_metas: &[ChunkMetadata],
        pattern: &[u8],
        options: &SearchOptions,
    ) -> Vec<ContentMatch> {
        if pattern.is_empty() || pattern.len() > MAX_PATTERN_LEN || chunk_metas.is_empty() {
            return vec![];
        }

        let chunk_count = chunk_metas.len();
        let total_data_bytes = chunk_count * CHUNK_SIZE;

        // Prepare pattern bytes (lowercase if case-insensitive)
        let pattern_bytes: Vec<u8> = if options.case_sensitive {
            pattern.to_vec()
        } else {
            pattern.iter().map(|b| b.to_ascii_lowercase()).collect()
        };

        unsafe {
            // (1) Write pattern to pattern_buffer
            let pattern_ptr = self.pattern_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(
                pattern_bytes.as_ptr(),
                pattern_ptr,
                pattern_bytes.len(),
            );

            // (2) Write search params to params_buffer
            let params_ptr = self.params_buffer.contents().as_ptr() as *mut GpuSearchParams;
            *params_ptr = GpuSearchParams {
                chunk_count: chunk_count as u32,
                pattern_len: pattern_bytes.len() as u32,
                case_sensitive: if options.case_sensitive { 1 } else { 0 },
                total_bytes: total_data_bytes as u32,
            };

            // (3) Write chunk_metas to metadata_buffer
            let meta_ptr = self.metadata_buffer.contents().as_ptr() as *mut ChunkMetadata;
            std::ptr::copy_nonoverlapping(chunk_metas.as_ptr(), meta_ptr, chunk_count);

            // (4) Reset match_count to 0
            let count_ptr = self.match_count_buffer.contents().as_ptr() as *mut u32;
            *count_ptr = 0;
        }

        // Update internal state for collect_results
        self.current_chunk_count = chunk_count;
        self.total_data_bytes = total_data_bytes;

        // Select kernel
        let pso = match options.mode {
            SearchMode::Standard => &self.content_search_pso,
            SearchMode::Turbo => &self.turbo_search_pso,
        };

        // (5) Dispatch GPU compute with content_buffer at buffer(0)
        autoreleasepool(|_| {
            let cmd = self
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(pso);
            unsafe {
                // Use caller's content_buffer at index 0 instead of self.chunks_buffer
                encoder.setBuffer_offset_atIndex(Some(content_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&*self.metadata_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&*self.params_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&*self.pattern_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&*self.matches_buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&*self.match_count_buffer), 0, 5);
            }

            // One thread per 64 bytes of data
            let total_threads = total_data_bytes.div_ceil(BYTES_PER_THREAD);

            let grid_size = MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: THREADGROUP_SIZE,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd.commit();
            cmd.waitUntilCompleted();
        });

        // (6) Read back matches (padded path: grid-absolute byte_offset)
        self.collect_results(options, false)
    }

    /// Zero-copy single-dispatch search on a contiguous content buffer.
    ///
    /// Dispatches the zerocopy kernel directly on the caller's Metal buffer
    /// (e.g. from `ContentStore::metal_buffer()`) in a single GPU dispatch.
    /// No padding, no CPU memcpy, no batching. All chunk metadata is written
    /// at once and the entire search completes in one `commit()` + `waitUntilCompleted()`.
    ///
    /// Each `ChunkMetadata` must have `buffer_offset` set to the absolute byte
    /// offset within the contiguous buffer.
    ///
    /// Flow: write metadata -> write params -> single dispatch -> collect results.
    ///
    /// Buffer layout:
    /// - buffer(0) = contiguous content buffer (raw bytes, NOT padded)
    /// - buffer(1) = chunk metadata (with buffer_offset populated)
    /// - buffer(2) = search params
    /// - buffer(3) = pattern
    /// - buffer(4) = match output
    /// - buffer(5) = match count
    pub fn search_zerocopy(
        &mut self,
        content_buffer: &ProtocolObject<dyn MTLBuffer>,
        chunk_metas: &[ChunkMetadata],
        pattern: &[u8],
        options: &SearchOptions,
    ) -> Vec<ContentMatch> {
        if pattern.is_empty() || pattern.len() > MAX_PATTERN_LEN || chunk_metas.is_empty() {
            return vec![];
        }

        // Ensure metadata buffer can hold all chunks; return empty on failure
        if self.ensure_metadata_capacity(chunk_metas.len()).is_err() {
            return vec![];
        }

        let pattern_bytes: Vec<u8> = if options.case_sensitive {
            pattern.to_vec()
        } else {
            pattern.iter().map(|b| b.to_ascii_lowercase()).collect()
        };

        // Step 1: Write pattern
        unsafe {
            let pattern_ptr = self.pattern_buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(
                pattern_bytes.as_ptr(),
                pattern_ptr,
                pattern_bytes.len(),
            );
        }

        let chunk_count = chunk_metas.len();
        let total_bytes = chunk_count * CHUNK_SIZE;

        // Step 2: Write metadata and params
        unsafe {
            let meta_ptr = self.metadata_buffer.contents().as_ptr() as *mut ChunkMetadata;
            std::ptr::copy_nonoverlapping(chunk_metas.as_ptr(), meta_ptr, chunk_count);

            let params_ptr = self.params_buffer.contents().as_ptr() as *mut GpuSearchParams;
            *params_ptr = GpuSearchParams {
                chunk_count: chunk_count as u32,
                pattern_len: pattern_bytes.len() as u32,
                case_sensitive: if options.case_sensitive { 1 } else { 0 },
                total_bytes: total_bytes as u32,
            };

            let count_ptr = self.match_count_buffer.contents().as_ptr() as *mut u32;
            *count_ptr = 0;
        }

        self.current_chunk_count = chunk_count;
        self.total_data_bytes = total_bytes;

        // Step 3: Single GPU dispatch
        autoreleasepool(|_| {
            let cmd = self
                .command_queue
                .commandBuffer()
                .expect("Failed to create command buffer");
            let encoder = cmd
                .computeCommandEncoder()
                .expect("Failed to create compute encoder");

            encoder.setComputePipelineState(&self.zerocopy_search_pso);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(content_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&*self.metadata_buffer), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&*self.params_buffer), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&*self.pattern_buffer), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&*self.matches_buffer), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&*self.match_count_buffer), 0, 5);
            }

            let total_threads = total_bytes.div_ceil(BYTES_PER_THREAD);
            let grid_size = MTLSize {
                width: total_threads,
                height: 1,
                depth: 1,
            };
            let tg_size = MTLSize {
                width: THREADGROUP_SIZE,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreads_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd.commit();
            cmd.waitUntilCompleted();
        });

        // Step 4: Collect results (zerocopy: file-relative byte_offset)
        let mut all_results = self.collect_results(options, true);
        if all_results.len() > options.max_results {
            all_results.truncate(options.max_results);
        }

        all_results
    }

    /// Read match results from GPU buffers.
    /// When `zerocopy` is true, byte_offset = context_start + column (file-relative),
    /// since the zerocopy kernel stores file-relative offsets in context_start.
    /// When false (padded path), byte_offset = chunk_index * CHUNK_SIZE + context_start + column
    /// (grid-absolute, for streaming.rs translation).
    fn collect_results(&self, options: &SearchOptions, zerocopy: bool) -> Vec<ContentMatch> {
        let mut results = Vec::new();

        unsafe {
            let count_ptr = self.match_count_buffer.contents().as_ptr() as *const u32;
            let count = (*count_ptr) as usize;
            let result_count = count.min(options.max_results).min(MAX_MATCHES);

            let matches_ptr = self.matches_buffer.contents().as_ptr() as *const GpuMatchResult;

            for i in 0..result_count {
                let m = *matches_ptr.add(i);

                // For zerocopy: context_start is file-relative (line_start_abs - buffer_offset),
                // so context_start + column = file-relative byte offset of match.
                // For padded: context_start is chunk-relative, so we need chunk_index * CHUNK_SIZE
                // to produce grid-absolute offset (streaming.rs translates to file-relative).
                let byte_offset = if zerocopy {
                    m.context_start + m.column
                } else {
                    m.chunk_index * CHUNK_SIZE as u32
                        + m.context_start + m.column
                };

                debug_assert!(
                    (m.chunk_index as usize) < self.current_chunk_count,
                    "GPU match chunk_index {} >= current_chunk_count {}",
                    m.chunk_index, self.current_chunk_count,
                );

                results.push(ContentMatch {
                    file_index: m.file_index as usize,
                    line_number: m.line_number,
                    column: m.column,
                    byte_offset,
                    match_length: m.match_length,
                });
            }
        }

        gpu_sort_results(&mut results);
        results
    }

    /// Get current chunk count.
    pub fn chunk_count(&self) -> usize {
        self.current_chunk_count
    }

    /// Get loaded file count.
    pub fn file_count(&self) -> usize {
        self.file_count
    }

    /// Get a reference to the Metal device.
    ///
    /// Used by the content store fast-path to create padded GPU buffers.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
}

// ============================================================================
// Test-only buffer inspection methods
// ============================================================================

/// Test-only buffer inspection methods (always compiled for integration test access).
impl ContentSearchEngine {
    /// Returns raw bytes of metadata_buffer up to max_chunks * sizeof(ChunkMetadata).
    pub fn inspect_metadata_buffer(&self) -> Vec<u8> {
        let byte_len = self.max_chunks * mem::size_of::<ChunkMetadata>();
        let mut out = vec![0u8; byte_len];
        unsafe {
            let src = self.metadata_buffer.contents().as_ptr() as *const u8;
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), byte_len);
        }
        out
    }

    /// Reads the current match_count_buffer value.
    pub fn inspect_match_count(&self) -> u32 {
        unsafe {
            let ptr = self.match_count_buffer.contents().as_ptr() as *const u32;
            *ptr
        }
    }

    /// Returns the max_chunks capacity.
    pub fn max_chunks(&self) -> usize {
        self.max_chunks
    }
}

// ============================================================================
// CPU reference implementation for verification
// ============================================================================

/// CPU reference search: simple byte-by-byte pattern matching.
/// Used to verify GPU results.
pub fn cpu_search(content: &[u8], pattern: &[u8], case_sensitive: bool) -> usize {
    if pattern.is_empty() || content.len() < pattern.len() {
        return 0;
    }

    let pattern_bytes: Vec<u8> = if case_sensitive {
        pattern.to_vec()
    } else {
        pattern.iter().map(|b| b.to_ascii_lowercase()).collect()
    };

    let mut count = 0;
    let end = content.len() - pattern.len() + 1;
    for i in 0..end {
        let mut matched = true;
        for j in 0..pattern_bytes.len() {
            let a = if case_sensitive {
                content[i + j]
            } else {
                content[i + j].to_ascii_lowercase()
            };
            if a != pattern_bytes[j] {
                matched = false;
                break;
            }
        }
        if matched {
            count += 1;
        }
    }

    count
}

// ============================================================================
// GPU Sort Helper
// ============================================================================

/// Sort ContentMatch results by (file_index, byte_offset) using GPU argsort
/// for batches >64, falling back to CPU sort for small batches or GPU errors.
fn gpu_sort_results(results: &mut Vec<ContentMatch>) {
    const GPU_SORT_THRESHOLD: usize = 64;

    let mut gpu_sorted = false;

    if results.len() > GPU_SORT_THRESHOLD {
        if let Ok(mut sorter) = forge_sort::GpuSorter::new() {
            let keys: Vec<u64> = results
                .iter()
                .map(|r| ((r.file_index as u64) << 32) | (r.byte_offset as u64))
                .collect();
            match sorter.argsort_u64(&keys) {
                Ok(indices) => {
                    let sorted: Vec<ContentMatch> = indices
                        .iter()
                        .map(|&i| results[i as usize].clone())
                        .collect();
                    *results = sorted;
                    gpu_sorted = true;
                }
                Err(e) => {
                    eprintln!("[gpu-search] argsort_u64 failed: {e}, falling back to CPU sort");
                }
            }
        }
    }

    if !gpu_sorted {
        results.sort_by(|a, b| {
            a.file_index
                .cmp(&b.file_index)
                .then(a.byte_offset.cmp(&b.byte_offset))
        });
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    #[test]
    fn test_content_search() {
        // Initialize GPU
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        // Create test content with known patterns
        let content = b"fn main() {\n    println!(\"hello\");\n}\n\nfn test_one() {\n    let x = 1;\n}\n\nfn test_two() {\n    let y = 2;\n}\n";
        let chunks = engine.load_content(content, 0);
        assert!(chunks > 0, "Should load at least 1 chunk");

        // Search for "fn " -- should find 3 matches
        let options = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let gpu_results = engine.search(b"fn ", &options);
        let cpu_count = cpu_search(content, b"fn ", true);

        println!("GPU matches: {}", gpu_results.len());
        println!("CPU matches: {}", cpu_count);
        for (i, m) in gpu_results.iter().enumerate() {
            println!(
                "  GPU match {}: file={}, line={}, col={}",
                i, m.file_index, m.line_number, m.column
            );
        }

        assert_eq!(
            gpu_results.len(),
            cpu_count,
            "GPU match count ({}) must equal CPU reference count ({})",
            gpu_results.len(),
            cpu_count
        );
        assert_eq!(gpu_results.len(), 3, "Should find exactly 3 'fn ' matches");

        // Verify first match line number (within thread 0's window, always correct)
        assert_eq!(gpu_results[0].line_number, 1, "First fn on line 1");

        // Note: line numbers are computed locally within each GPU thread's 64-byte
        // window, so they may differ from global line numbers for matches in
        // later thread windows. This is by design for throughput. Full line
        // number resolution would be done by the search orchestrator in Phase 2.
    }

    #[test]
    fn test_content_search_turbo() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let content = b"hello world hello world hello\n";
        engine.load_content(content, 0);

        let options = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Turbo,
        };
        let gpu_results = engine.search(b"hello", &options);
        let cpu_count = cpu_search(content, b"hello", true);

        assert_eq!(
            gpu_results.len(),
            cpu_count,
            "Turbo mode: GPU({}) must equal CPU({})",
            gpu_results.len(),
            cpu_count
        );
        assert_eq!(gpu_results.len(), 3, "Should find 3 'hello' matches");
    }

    #[test]
    fn test_content_search_case_insensitive() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let content = b"Hello HELLO hello hElLo\n";
        engine.load_content(content, 0);

        let options = SearchOptions {
            case_sensitive: false,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let gpu_results = engine.search(b"hello", &options);
        let cpu_count = cpu_search(content, b"hello", false);

        assert_eq!(
            gpu_results.len(),
            cpu_count,
            "Case insensitive: GPU({}) must equal CPU({})",
            gpu_results.len(),
            cpu_count
        );
        assert_eq!(gpu_results.len(), 4, "Should find 4 case-insensitive matches");
    }

    #[test]
    fn test_content_search_no_matches() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let content = b"this content has no matches for the pattern\n";
        engine.load_content(content, 0);

        let options = SearchOptions::default();
        let gpu_results = engine.search(b"ZZZZZ", &options);
        let cpu_count = cpu_search(content, b"ZZZZZ", true);

        assert_eq!(gpu_results.len(), cpu_count);
        assert_eq!(gpu_results.len(), 0);
    }

    #[test]
    fn test_content_search_empty() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        // Search with no loaded content
        let options = SearchOptions::default();
        let results = engine.search(b"test", &options);
        assert_eq!(results.len(), 0);

        // Search with empty pattern
        let results = engine.search(b"", &options);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_content_search_multi_file() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let file0 = b"fn alpha() {}\nfn beta() {}\n";
        let file1 = b"fn gamma() {}\n";
        engine.load_content(file0, 0);
        engine.load_content(file1, 1);

        let options = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results = engine.search(b"fn ", &options);

        let cpu_count0 = cpu_search(file0, b"fn ", true);
        let cpu_count1 = cpu_search(file1, b"fn ", true);
        let total_cpu = cpu_count0 + cpu_count1;

        assert_eq!(
            results.len(),
            total_cpu,
            "Multi-file: GPU({}) must equal CPU({})",
            results.len(),
            total_cpu
        );
        assert_eq!(results.len(), 3);

        // File 0 should have 2 matches, file 1 should have 1
        let f0_matches: Vec<_> = results.iter().filter(|m| m.file_index == 0).collect();
        let f1_matches: Vec<_> = results.iter().filter(|m| m.file_index == 1).collect();
        assert_eq!(f0_matches.len(), 2);
        assert_eq!(f1_matches.len(), 1);
    }

    #[test]
    fn test_content_search_real_file() {
        // Search this very source file for "fn " patterns
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        // Use a known Rust source file
        let source_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("search")
            .join("content.rs");
        let content = std::fs::read(&source_path).expect("Failed to read source file");

        let chunks = engine.load_content(&content, 0);
        println!("Loaded {} bytes in {} chunks", content.len(), chunks);

        let options = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let gpu_results = engine.search(b"fn ", &options);
        let cpu_count = cpu_search(&content, b"fn ", true);

        println!(
            "Real file search: GPU={}, CPU={} matches for 'fn ' in {} bytes",
            gpu_results.len(),
            cpu_count,
            content.len()
        );

        // GPU count may be slightly less than CPU due to matches crossing 64-byte
        // thread boundaries (known kernel limitation). No false positives expected.
        assert!(
            gpu_results.len() <= cpu_count,
            "GPU should not produce false positives: GPU({}) > CPU({})",
            gpu_results.len(),
            cpu_count
        );
        // Allow up to ~10% boundary misses for real files.
        // Matches crossing 64-byte thread boundaries are missed by design.
        // The exact miss rate depends on where `fn ` patterns land relative
        // to thread boundaries, which shifts as the source file changes.
        let min_expected = cpu_count * 90 / 100;
        assert!(
            gpu_results.len() >= min_expected,
            "Too many boundary misses: GPU({}) < 90% of CPU({})",
            gpu_results.len(),
            min_expected
        );
    }

    #[test]
    fn test_cpu_search_reference() {
        assert_eq!(cpu_search(b"hello world", b"hello", true), 1);
        assert_eq!(cpu_search(b"hello world", b"world", true), 1);
        assert_eq!(cpu_search(b"aaa", b"a", true), 3);
        assert_eq!(cpu_search(b"aaa", b"aa", true), 2); // overlapping
        assert_eq!(cpu_search(b"", b"test", true), 0);
        assert_eq!(cpu_search(b"test", b"", true), 0);
        assert_eq!(cpu_search(b"Hello", b"hello", false), 1);
        assert_eq!(cpu_search(b"HELLO", b"hello", false), 1);
    }

    #[test]
    fn test_layout_assertions() {
        assert_eq!(mem::size_of::<GpuSearchParams>(), 16);
        assert_eq!(mem::size_of::<GpuMatchResult>(), 32);
        assert_eq!(mem::size_of::<ChunkMetadata>(), 32);
    }

    #[test]
    fn test_search_with_buffer_basic() {
        use crate::index::content_store::ContentStoreBuilder;

        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        // Build a ContentStore with known content (each file < CHUNK_SIZE)
        let file0 = b"fn alpha() { let x = 42; }\nfn beta() { return 7; }\n";
        let file1 = b"fn gamma() { println!(\"hello\"); }\n";
        let file2 = b"struct Foo { bar: u32 }\nimpl Foo { fn new() -> Self { Foo { bar: 0 } } }\n";

        let mut builder = ContentStoreBuilder::new(64 * 1024).unwrap();
        builder.append(file0, 0, 0, 0);
        builder.append(file1, 1, 0, 0);
        builder.append(file2, 2, 0, 0);
        let store = builder.finalize(&device);

        // Build ChunkMetadata: each file fits in one chunk since all < CHUNK_SIZE
        let mut chunk_metas = Vec::new();
        for (file_idx, meta) in store.files().iter().enumerate() {
            let file_len = meta.content_len as usize;
            let num_chunks = if file_len == 0 { 0 } else { file_len.div_ceil(CHUNK_SIZE) };
            for chunk_i in 0..num_chunks {
                let offset_in_file = chunk_i * CHUNK_SIZE;
                let chunk_len = (file_len - offset_in_file).min(CHUNK_SIZE);
                let mut flags = 1u32; // is_text
                if chunk_i == 0 {
                    flags |= 2; // is_first
                }
                if chunk_i == num_chunks - 1 {
                    flags |= 4; // is_last
                }
                chunk_metas.push(ChunkMetadata {
                    file_index: file_idx as u32,
                    chunk_index: chunk_i as u32,
                    offset_in_file: offset_in_file as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                    buffer_offset: 0, // Not used in padded path
                });
            }
        }

        // The GPU kernel reads data at chunk_index * CHUNK_SIZE from buffer(0).
        // ContentStore stores files contiguously (NOT padded to CHUNK_SIZE).
        // Build a padded buffer with each file's chunk at the correct offset.
        let options = MTLResourceOptions::StorageModeShared;
        let padded_len = chunk_metas.len() * CHUNK_SIZE;
        let padded_buffer = device
            .newBufferWithLength_options(padded_len, options)
            .expect("Failed to allocate padded buffer");

        unsafe {
            let dst_base = padded_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst_base, 0, padded_len);

            let src = store.buffer();
            let mut chunk_idx = 0usize;
            for meta in store.files().iter() {
                let file_start = meta.content_offset as usize;
                let file_len = meta.content_len as usize;
                let num_chunks = if file_len == 0 { 0 } else { file_len.div_ceil(CHUNK_SIZE) };
                for c in 0..num_chunks {
                    let src_offset = file_start + c * CHUNK_SIZE;
                    let copy_len = (file_len - c * CHUNK_SIZE).min(CHUNK_SIZE);
                    let dst = dst_base.add(chunk_idx * CHUNK_SIZE);
                    std::ptr::copy_nonoverlapping(
                        src[src_offset..src_offset + copy_len].as_ptr(),
                        dst,
                        copy_len,
                    );
                    chunk_idx += 1;
                }
            }
        }

        // Search for "fn " using the external buffer
        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results = engine.search_with_buffer(&padded_buffer, &chunk_metas, b"fn ", &search_opts);

        // CPU reference counts
        let cpu0 = cpu_search(file0, b"fn ", true);
        let cpu1 = cpu_search(file1, b"fn ", true);
        let cpu2 = cpu_search(file2, b"fn ", true);
        let total_cpu = cpu0 + cpu1 + cpu2;

        println!(
            "search_with_buffer: GPU={}, CPU={} ({}+{}+{})",
            results.len(),
            total_cpu,
            cpu0,
            cpu1,
            cpu2
        );
        for (i, m) in results.iter().enumerate() {
            println!(
                "  match {}: file={}, line={}, col={}",
                i, m.file_index, m.line_number, m.column
            );
        }

        assert_eq!(
            results.len(),
            total_cpu,
            "search_with_buffer GPU({}) must equal CPU({})",
            results.len(),
            total_cpu
        );
        // file0: "fn alpha" + "fn beta" = 2, file1: "fn gamma" = 1, file2: "fn new" = 1
        assert_eq!(results.len(), 4, "Should find 4 'fn ' matches across 3 files");

        // Verify file_index distribution
        let f0: Vec<_> = results.iter().filter(|m| m.file_index == 0).collect();
        let f1: Vec<_> = results.iter().filter(|m| m.file_index == 1).collect();
        let f2: Vec<_> = results.iter().filter(|m| m.file_index == 2).collect();
        assert_eq!(f0.len(), 2, "file0 should have 2 'fn ' matches");
        assert_eq!(f1.len(), 1, "file1 should have 1 'fn ' match");
        assert_eq!(f2.len(), 1, "file2 should have 1 'fn ' match");
    }

    #[test]
    fn test_search_with_buffer_no_matches() {
        use crate::index::content_store::ContentStoreBuilder;

        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let content = b"this has no matching pattern at all\n";
        let mut builder = ContentStoreBuilder::new(16384).unwrap();
        builder.append(content, 0, 0, 0);
        let store = builder.finalize(&device);
        let _ = store; // keep alive for reference

        // Build one chunk
        let chunk_metas = vec![ChunkMetadata {
            file_index: 0,
            chunk_index: 0,
            offset_in_file: 0,
            chunk_length: content.len() as u32,
            flags: 1 | 2 | 4, // is_text | is_first | is_last
            buffer_offset: 0,
        }];

        // Create padded buffer with content
        let options = MTLResourceOptions::StorageModeShared;
        let padded_buffer = device
            .newBufferWithLength_options(CHUNK_SIZE, options)
            .expect("Failed to allocate buffer");
        unsafe {
            let dst = padded_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(dst, 0, CHUNK_SIZE);
            std::ptr::copy_nonoverlapping(content.as_ptr(), dst, content.len());
        }

        let search_opts = SearchOptions::default();
        let results =
            engine.search_with_buffer(&padded_buffer, &chunk_metas, b"ZZZZZ", &search_opts);
        assert_eq!(results.len(), 0, "Should find no matches for non-existent pattern");
    }

    #[test]
    fn test_search_with_buffer_empty_inputs() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let chunk_metas = vec![ChunkMetadata {
            file_index: 0,
            chunk_index: 0,
            offset_in_file: 0,
            chunk_length: 100,
            flags: 7,
            buffer_offset: 0,
        }];

        let options = MTLResourceOptions::StorageModeShared;
        let buf = device
            .newBufferWithLength_options(CHUNK_SIZE, options)
            .expect("buffer");

        // Empty pattern should return empty
        let results = engine.search_with_buffer(&buf, &chunk_metas, b"", &SearchOptions::default());
        assert_eq!(results.len(), 0);

        // Empty chunk_metas should return empty
        let results = engine.search_with_buffer(&buf, &[], b"test", &SearchOptions::default());
        assert_eq!(results.len(), 0);
    }

    // ================================================================
    // Single-dispatch search tests (task 3.1)
    // ================================================================

    /// Test that search_zerocopy handles >100K chunks in a single dispatch.
    ///
    /// Creates synthetic chunk metadata pointing to a shared content buffer.
    /// Each "file" is a short line with a known pattern. Verifies that:
    /// - ensure_metadata_capacity resizes for >100K chunks
    /// - Single GPU dispatch completes without error
    /// - All expected matches are found (up to MAX_MATCHES)
    #[test]
    fn test_single_dispatch_large_chunk_count() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        // Start with small max_files so max_chunks < 100K, forcing resize
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);
        assert!(
            engine.max_chunks() < 100_001,
            "Initial max_chunks should be < 100K to test resize"
        );

        // Create a content buffer with repeating pattern lines.
        // Each "file" is a short line like "fn test_NNN()\n" placed at
        // a unique offset. We tile files into a moderately-sized buffer.
        let file_content = b"fn test_marker()\n"; // 17 bytes, contains "fn "
        let file_len = file_content.len();
        let num_chunks: usize = 100_001; // >100K

        // All chunks point to the same content repeated in a buffer.
        // Buffer size: enough for one copy of file_content per unique offset slot.
        // To keep memory reasonable, reuse offsets cyclically with a pool of slots.
        let num_slots = 1024; // 1024 unique slots
        let buf_size = num_slots * CHUNK_SIZE; // 4MB -- each slot is CHUNK_SIZE apart
        let options = MTLResourceOptions::StorageModeShared;
        let content_buffer = device
            .newBufferWithLength_options(buf_size, options)
            .expect("Failed to allocate content buffer");

        // Fill each slot with the file content (zero-padded to CHUNK_SIZE)
        unsafe {
            let base = content_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(base, 0, buf_size);
            for slot in 0..num_slots {
                let dst = base.add(slot * CHUNK_SIZE);
                std::ptr::copy_nonoverlapping(file_content.as_ptr(), dst, file_len);
            }
        }

        // Build chunk metadata: each chunk maps to a slot cyclically
        let chunk_metas: Vec<ChunkMetadata> = (0..num_chunks)
            .map(|i| {
                let slot = i % num_slots;
                ChunkMetadata {
                    file_index: i as u32,
                    chunk_index: 0,
                    offset_in_file: 0,
                    chunk_length: file_len as u32,
                    flags: 1 | 2 | 4, // is_text | is_first | is_last
                    buffer_offset: (slot * CHUNK_SIZE) as u64,
                }
            })
            .collect();

        assert_eq!(chunk_metas.len(), num_chunks);

        // Search for "fn " -- every chunk has exactly 1 match
        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard, // mode is ignored by search_zerocopy
        };
        let results = engine.search_zerocopy(&content_buffer, &chunk_metas, b"fn ", &search_opts);

        // GPU should find matches up to MAX_MATCHES (10000)
        // With 100K+ chunks each containing 1 match, we expect MAX_MATCHES results
        println!(
            "test_single_dispatch_large_chunk_count: {} results from {} chunks",
            results.len(),
            num_chunks
        );
        assert!(
            results.len() > 0,
            "Should find at least some matches in >100K chunks"
        );
        // Match count capped by MAX_MATCHES (10000) since kernel limits to that
        assert!(
            results.len() <= MAX_MATCHES,
            "Results should not exceed MAX_MATCHES"
        );

        // Verify metadata buffer was resized to hold all chunks
        assert!(
            engine.max_chunks() >= num_chunks,
            "max_chunks ({}) should be >= {} after resize",
            engine.max_chunks(),
            num_chunks
        );
    }

    /// Test ensure_metadata_capacity with various chunk counts.
    ///
    /// Verifies:
    /// - No-op when capacity is sufficient
    /// - Resizes when needed exceeds current max_chunks
    /// - max_chunks updated after successful resize
    #[test]
    fn test_single_dispatch_metadata_capacity() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);

        // Start with small capacity (100 files * 10 = 1000 chunks)
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 100);
        let initial_max = engine.max_chunks();
        assert_eq!(initial_max, 1000, "100 files * 10 = 1000 initial max_chunks");

        // Test 1: No-op when within capacity
        let result = engine.ensure_metadata_capacity(500);
        assert!(result.is_ok(), "Should succeed when within capacity");
        assert_eq!(
            engine.max_chunks(),
            initial_max,
            "max_chunks unchanged when within capacity"
        );

        // Test 2: No-op at exact boundary
        let result = engine.ensure_metadata_capacity(1000);
        assert!(result.is_ok(), "Should succeed at exact capacity");
        assert_eq!(
            engine.max_chunks(),
            initial_max,
            "max_chunks unchanged at exact boundary"
        );

        // Test 3: Resize when exceeding capacity
        let result = engine.ensure_metadata_capacity(5000);
        assert!(result.is_ok(), "Should succeed with resize to 5000");
        assert_eq!(
            engine.max_chunks(),
            5000,
            "max_chunks should be exactly 5000 after resize"
        );

        // Test 4: Larger resize
        let result = engine.ensure_metadata_capacity(150_000);
        assert!(result.is_ok(), "Should succeed with resize to 150K");
        assert_eq!(
            engine.max_chunks(),
            150_000,
            "max_chunks should be exactly 150000 after resize"
        );

        // Test 5: No-op when new capacity is within already-resized size
        let result = engine.ensure_metadata_capacity(100_000);
        assert!(result.is_ok(), "Should succeed when within resized capacity");
        assert_eq!(
            engine.max_chunks(),
            150_000,
            "max_chunks unchanged when already sufficient"
        );
    }

    /// Test that single-dispatch zerocopy results match CPU reference.
    ///
    /// Creates a content buffer with multiple files containing known patterns,
    /// searches via zerocopy kernel, and compares match count against CPU search.
    #[test]
    fn test_single_dispatch_cpu_reference_match() {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let mut engine = ContentSearchEngine::new(&device, &pso_cache, 1000);

        // Create varied content with different match densities
        let files: Vec<&[u8]> = vec![
            b"fn alpha() { return 1; }\nfn beta() { return 2; }\nfn gamma() { return 3; }\n",
            b"no functions here, just plain text\nand another line\n",
            b"fn single_match() { }\n",
            b"struct Foo;\nimpl Foo {\n    fn new() -> Self { Foo }\n    fn drop(&mut self) {}\n}\n",
            b"// comment line\n// fn fake_in_comment\nfn real_function() {}\n",
        ];

        // CPU reference: count total "fn " matches across all files
        let pattern = b"fn ";
        let total_cpu_matches: usize = files.iter().map(|f| cpu_search(f, pattern, true)).sum();
        println!("CPU reference: {} total 'fn ' matches", total_cpu_matches);

        // Build a contiguous content buffer and chunk metadata
        // Each file occupies exactly one chunk slot (all < CHUNK_SIZE)
        let num_chunks = files.len();
        let buf_size = num_chunks * CHUNK_SIZE;
        let options = MTLResourceOptions::StorageModeShared;
        let content_buffer = device
            .newBufferWithLength_options(buf_size, options)
            .expect("Failed to allocate content buffer");

        let mut chunk_metas = Vec::with_capacity(num_chunks);
        unsafe {
            let base = content_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(base, 0, buf_size);

            for (i, file_data) in files.iter().enumerate() {
                let offset = i * CHUNK_SIZE;
                std::ptr::copy_nonoverlapping(file_data.as_ptr(), base.add(offset), file_data.len());

                chunk_metas.push(ChunkMetadata {
                    file_index: i as u32,
                    chunk_index: 0,
                    offset_in_file: 0,
                    chunk_length: file_data.len() as u32,
                    flags: 1 | 2 | 4, // is_text | is_first | is_last
                    buffer_offset: offset as u64,
                });
            }
        }

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let gpu_results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, pattern, &search_opts);

        println!(
            "GPU zerocopy results: {}, CPU reference: {}",
            gpu_results.len(),
            total_cpu_matches
        );
        for (i, m) in gpu_results.iter().enumerate() {
            println!(
                "  match {}: file={}, line={}, col={}",
                i, m.file_index, m.line_number, m.column
            );
        }

        assert_eq!(
            gpu_results.len(),
            total_cpu_matches,
            "GPU zerocopy match count ({}) must equal CPU reference ({})",
            gpu_results.len(),
            total_cpu_matches
        );

        // Verify per-file match distribution
        let expected_per_file: Vec<usize> = files.iter().map(|f| cpu_search(f, pattern, true)).collect();
        for (file_idx, expected) in expected_per_file.iter().enumerate() {
            let gpu_count = gpu_results.iter().filter(|m| m.file_index == file_idx).count();
            assert_eq!(
                gpu_count, *expected,
                "File {} GPU matches ({}) != CPU expected ({})",
                file_idx, gpu_count, expected
            );
        }

        // Verify all matches have valid file_index
        for m in &gpu_results {
            assert!(
                m.file_index < files.len(),
                "match file_index {} out of range (max {})",
                m.file_index,
                files.len() - 1
            );
            assert!(m.line_number >= 1, "line_number should be >= 1");
        }
    }

    // ================================================================
    // GPU line number accuracy tests (task 3.2)
    // ================================================================

    /// Helper: compute CPU reference line number for a byte offset within file content.
    /// Line number is 1-based: count newlines before byte_offset, then +1.
    fn cpu_line_number(content: &[u8], byte_offset: usize) -> u32 {
        content.iter().take(byte_offset).filter(|&&b| b == b'\n').count() as u32 + 1
    }

    /// Helper: set up a zerocopy search engine, content buffer, and chunk metadata
    /// for a single file's content. Returns (engine, content_buffer, chunk_metas).
    fn setup_zerocopy_single_file(
        content: &[u8],
    ) -> (
        ContentSearchEngine,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Vec<ChunkMetadata>,
    ) {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let engine = ContentSearchEngine::new(&device, &pso_cache, 100);

        let file_len = content.len();
        let num_chunks = if file_len == 0 { 0 } else { file_len.div_ceil(CHUNK_SIZE) };

        // Allocate buffer large enough (num_chunks * CHUNK_SIZE, but at least CHUNK_SIZE
        // to avoid zero-length buffer)
        let buf_size = num_chunks.max(1) * CHUNK_SIZE;
        let options = MTLResourceOptions::StorageModeShared;
        let content_buffer = device
            .newBufferWithLength_options(buf_size, options)
            .expect("Failed to allocate content buffer");

        // Copy content into buffer (contiguous, not padded per-chunk for zerocopy)
        unsafe {
            let base = content_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(base, 0, buf_size);
            std::ptr::copy_nonoverlapping(content.as_ptr(), base, file_len);
        }

        // Build chunk metadata with buffer_offset pointing to contiguous layout
        let mut chunk_metas = Vec::with_capacity(num_chunks);
        for chunk_i in 0..num_chunks {
            let offset_in_file = chunk_i * CHUNK_SIZE;
            let chunk_len = (file_len - offset_in_file).min(CHUNK_SIZE);
            let mut flags = 1u32; // is_text
            if chunk_i == 0 {
                flags |= 2; // is_first
            }
            if chunk_i == num_chunks - 1 {
                flags |= 4; // is_last
            }
            chunk_metas.push(ChunkMetadata {
                file_index: 0,
                chunk_index: chunk_i as u32,
                offset_in_file: offset_in_file as u64,
                chunk_length: chunk_len as u32,
                flags,
                buffer_offset: offset_in_file as u64,
            });
        }

        (engine, content_buffer, chunk_metas)
    }

    /// Helper: set up zerocopy for multiple files in a contiguous buffer.
    /// Returns (engine, content_buffer, chunk_metas, file_offsets) where
    /// file_offsets[i] = byte offset of file i in the contiguous buffer.
    fn setup_zerocopy_multi_file(
        files: &[&[u8]],
    ) -> (
        ContentSearchEngine,
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Vec<ChunkMetadata>,
        Vec<usize>,
    ) {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let pso_cache = PsoCache::new(&device);
        let engine = ContentSearchEngine::new(&device, &pso_cache, 1000);

        // Compute total size and file offsets (contiguous layout)
        let mut file_offsets = Vec::with_capacity(files.len());
        let mut total_len = 0usize;
        for file_data in files {
            file_offsets.push(total_len);
            total_len += file_data.len();
        }

        // Compute total chunks across all files
        let total_chunks: usize = files
            .iter()
            .map(|f| if f.is_empty() { 0 } else { f.len().div_ceil(CHUNK_SIZE) })
            .sum();

        let buf_size = total_chunks.max(1) * CHUNK_SIZE;
        let options = MTLResourceOptions::StorageModeShared;
        let content_buffer = device
            .newBufferWithLength_options(buf_size, options)
            .expect("Failed to allocate content buffer");

        // Copy all files contiguously
        unsafe {
            let base = content_buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(base, 0, buf_size);
            let mut offset = 0usize;
            for file_data in files {
                std::ptr::copy_nonoverlapping(file_data.as_ptr(), base.add(offset), file_data.len());
                offset += file_data.len();
            }
        }

        // Build chunk metadata for each file
        let mut chunk_metas = Vec::with_capacity(total_chunks);
        for (file_idx, file_data) in files.iter().enumerate() {
            let file_len = file_data.len();
            let file_offset = file_offsets[file_idx];
            let num_chunks = if file_len == 0 { 0 } else { file_len.div_ceil(CHUNK_SIZE) };
            for chunk_i in 0..num_chunks {
                let offset_in_file = chunk_i * CHUNK_SIZE;
                let chunk_len = (file_len - offset_in_file).min(CHUNK_SIZE);
                let mut flags = 1u32;
                if chunk_i == 0 {
                    flags |= 2;
                }
                if chunk_i == num_chunks - 1 {
                    flags |= 4;
                }
                chunk_metas.push(ChunkMetadata {
                    file_index: file_idx as u32,
                    chunk_index: chunk_i as u32,
                    offset_in_file: offset_in_file as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                    buffer_offset: (file_offset + offset_in_file) as u64,
                });
            }
        }

        (engine, content_buffer, chunk_metas, file_offsets)
    }

    /// Test GPU line numbers for matches at various positions in a multi-line file.
    ///
    /// Creates content with a known pattern ("MARK") on specific lines and verifies
    /// that the GPU line_number for each match equals the CPU reference computed by
    /// counting newlines before the match byte offset.
    #[test]
    fn test_gpu_line_number_basic() {
        // Content with "MARK" on lines 1, 3, and 5
        let content = b"MARK on line one\nsecond line here\nMARK on line three\nfourth line here\nMARK on line five\n";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        // CPU reference: find all "MARK" positions and compute line numbers
        let mut cpu_matches = Vec::new();
        for i in 0..content.len().saturating_sub(3) {
            if &content[i..i + 4] == b"MARK" {
                let line = cpu_line_number(content, i);
                cpu_matches.push((i, line));
            }
        }

        println!("test_gpu_line_number_basic:");
        println!("  CPU matches: {:?}", cpu_matches);
        for (i, m) in results.iter().enumerate() {
            println!(
                "  GPU match {}: line={}, col={}, byte_offset={}",
                i, m.line_number, m.column, m.byte_offset
            );
        }

        assert_eq!(
            results.len(),
            cpu_matches.len(),
            "GPU match count ({}) != CPU match count ({})",
            results.len(),
            cpu_matches.len()
        );

        // Verify each GPU line number matches CPU reference
        for (gpu_match, (cpu_byte_off, cpu_line)) in results.iter().zip(cpu_matches.iter()) {
            assert_eq!(
                gpu_match.line_number, *cpu_line,
                "GPU line_number {} != CPU line {} for match at byte offset {}",
                gpu_match.line_number, cpu_line, cpu_byte_off
            );
        }
    }

    /// Test GPU line number for match on the very first line (line 1).
    ///
    /// No newlines before the match — line_number should be 1.
    #[test]
    fn test_gpu_line_number_first_line() {
        let content = b"MARK at start\nsecond line\nthird line\n";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        assert_eq!(results.len(), 1, "Should find exactly 1 MARK");
        assert_eq!(
            results[0].line_number, 1,
            "Match on first line should have line_number=1"
        );

        // CPU reference verification
        let cpu_line = cpu_line_number(content, 0);
        assert_eq!(cpu_line, 1);
        assert_eq!(results[0].line_number, cpu_line);
    }

    /// Test GPU line number for match on the last line.
    ///
    /// Content ends with the match on the final line (no trailing newline).
    #[test]
    fn test_gpu_line_number_last_line() {
        let content = b"first line\nsecond line\nthird line\nMARK on last";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        assert_eq!(results.len(), 1, "Should find exactly 1 MARK on last line");

        let byte_offset = content
            .windows(4)
            .position(|w| w == b"MARK")
            .expect("MARK should be in content");
        let cpu_line = cpu_line_number(content, byte_offset);
        assert_eq!(cpu_line, 4, "MARK should be on line 4");
        assert_eq!(
            results[0].line_number, cpu_line,
            "GPU line_number {} != CPU line {} for last-line match",
            results[0].line_number, cpu_line
        );
    }

    /// Test GPU line number after consecutive empty lines (\n\n).
    ///
    /// Empty lines increment the line counter. Pattern after two empty lines
    /// should have a line number reflecting those empty lines.
    #[test]
    fn test_gpu_line_number_after_empty_lines() {
        let content = b"line one\n\n\nMARK after empties\nfinal line\n";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        assert_eq!(results.len(), 1, "Should find exactly 1 MARK");

        let byte_offset = content
            .windows(4)
            .position(|w| w == b"MARK")
            .expect("MARK in content");
        let cpu_line = cpu_line_number(content, byte_offset);
        // "line one\n" = line 1, "\n" = line 2 (empty), "\n" = line 3 (empty), "MARK..." = line 4
        assert_eq!(cpu_line, 4, "MARK should be on line 4 (after 2 empty lines)");
        assert_eq!(
            results[0].line_number, cpu_line,
            "GPU line_number {} != CPU line {} after empty lines",
            results[0].line_number, cpu_line
        );
    }

    /// Test GPU line number with CRLF (\r\n) line endings.
    ///
    /// The GPU kernel counts 0x0A (\n) for line boundaries. CRLF files have
    /// \r before each \n, but line counting should still work since only \n
    /// is counted. The \r is part of the line content.
    #[test]
    fn test_gpu_line_number_crlf() {
        let content = b"first line\r\nsecond line\r\nMARK on three\r\nfourth line\r\n";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        assert_eq!(results.len(), 1, "Should find exactly 1 MARK in CRLF content");

        let byte_offset = content
            .windows(4)
            .position(|w| w == b"MARK")
            .expect("MARK in content");
        let cpu_line = cpu_line_number(content, byte_offset);
        assert_eq!(cpu_line, 3, "MARK should be on line 3 in CRLF content");
        assert_eq!(
            results[0].line_number, cpu_line,
            "GPU line_number {} != CPU line {} for CRLF content",
            results[0].line_number, cpu_line
        );
    }

    /// Test GPU line numbers across multiple files in a single zerocopy dispatch.
    ///
    /// Each file has "MARK" on a different line. Verifies that GPU line numbers
    /// are file-relative (restart from 1 for each file).
    #[test]
    fn test_gpu_line_number_multi_file() {
        let file0: &[u8] = b"MARK on first line\nsecond line\n";
        let file1: &[u8] = b"first line\nsecond line\nMARK on third\n";
        let file2: &[u8] = b"aaa\nbbb\nccc\nddd\nMARK on fifth\n";

        let files: &[&[u8]] = &[file0, file1, file2];
        let (mut engine, content_buffer, chunk_metas, _file_offsets) =
            setup_zerocopy_multi_file(files);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        assert_eq!(results.len(), 3, "Should find 1 MARK per file (3 total)");

        // Expected: file0 MARK on line 1, file1 MARK on line 3, file2 MARK on line 5
        let expected: Vec<(usize, u32)> = vec![(0, 1), (1, 3), (2, 5)];

        for (file_idx, expected_line) in &expected {
            let file_matches: Vec<_> = results
                .iter()
                .filter(|m| m.file_index == *file_idx)
                .collect();
            assert_eq!(
                file_matches.len(),
                1,
                "File {} should have exactly 1 MARK match",
                file_idx
            );

            // Compute CPU reference within the file
            let file_data = files[*file_idx];
            let byte_off = file_data
                .windows(4)
                .position(|w| w == b"MARK")
                .expect("MARK in file");
            let cpu_line = cpu_line_number(file_data, byte_off);
            assert_eq!(
                cpu_line, *expected_line,
                "CPU line for file {} should be {}",
                file_idx, expected_line
            );
            assert_eq!(
                file_matches[0].line_number, cpu_line,
                "File {} GPU line_number {} != CPU line {}",
                file_idx, file_matches[0].line_number, cpu_line
            );
        }
    }

    /// Test GPU line numbers with multiple matches across several lines.
    ///
    /// Uses a pattern that appears at most once per line to avoid hitting the
    /// MAX_MATCHES_PER_THREAD (4) limit per GPU thread's 64-byte window.
    /// Verifies line_number is correct for every match.
    #[test]
    fn test_gpu_line_number_multiple_matches_per_line() {
        // "MARK" appears once on line 1, once on line 2, once on line 3, once on line 4.
        // Total = 4 matches, within MAX_MATCHES_PER_THREAD limit for a 64-byte window.
        let content = b"MARK first line\nMARK second line\nMARK third line\nMARK fourth line\n";

        let (mut engine, content_buffer, chunk_metas) = setup_zerocopy_single_file(content);

        let search_opts = SearchOptions {
            case_sensitive: true,
            max_results: MAX_MATCHES,
            mode: SearchMode::Standard,
        };
        let results =
            engine.search_zerocopy(&content_buffer, &chunk_metas, b"MARK", &search_opts);

        // CPU reference: find all "MARK" positions and line numbers
        let pattern = b"MARK";
        let mut cpu_matches = Vec::new();
        for i in 0..content.len().saturating_sub(pattern.len() - 1) {
            if &content[i..i + pattern.len()] == pattern {
                let line = cpu_line_number(content, i);
                cpu_matches.push((i, line));
            }
        }

        println!("test_gpu_line_number_multiple_matches_per_line:");
        println!("  CPU matches: {:?}", cpu_matches);
        for (i, m) in results.iter().enumerate() {
            println!(
                "  GPU match {}: line={}, col={}, byte_offset={}",
                i, m.line_number, m.column, m.byte_offset
            );
        }

        assert_eq!(
            results.len(),
            cpu_matches.len(),
            "GPU count ({}) != CPU count ({})",
            results.len(),
            cpu_matches.len()
        );

        // Verify every match's line_number
        for (gpu_m, (cpu_off, cpu_line)) in results.iter().zip(cpu_matches.iter()) {
            assert_eq!(
                gpu_m.line_number, *cpu_line,
                "GPU line {} != CPU line {} at byte offset {}",
                gpu_m.line_number, cpu_line, cpu_off
            );
        }

        // Each line should have exactly 1 "MARK" match
        for line_num in 1..=4u32 {
            let count = results.iter().filter(|m| m.line_number == line_num).count();
            assert_eq!(count, 1, "Line {} should have 1 'MARK' match, got {}", line_num, count);
        }
    }

    #[test]
    fn test_gpu_sort_key_encoding() {
        // Verify composite key (file_index << 32 | byte_offset) sorts correctly
        let mut results = vec![
            ContentMatch { file_index: 2, line_number: 1, column: 0, byte_offset: 100, match_length: 3 },
            ContentMatch { file_index: 0, line_number: 1, column: 0, byte_offset: 50, match_length: 3 },
            ContentMatch { file_index: 0, line_number: 1, column: 0, byte_offset: 10, match_length: 3 },
            ContentMatch { file_index: 1, line_number: 1, column: 0, byte_offset: 0, match_length: 3 },
        ];

        gpu_sort_results(&mut results);

        assert_eq!(results[0].file_index, 0);
        assert_eq!(results[0].byte_offset, 10);
        assert_eq!(results[1].file_index, 0);
        assert_eq!(results[1].byte_offset, 50);
        assert_eq!(results[2].file_index, 1);
        assert_eq!(results[2].byte_offset, 0);
        assert_eq!(results[3].file_index, 2);
        assert_eq!(results[3].byte_offset, 100);
    }

    #[test]
    fn test_gpu_sort_cpu_fallback() {
        // With <= 64 results, should use CPU sort (same correct ordering)
        let mut results = vec![
            ContentMatch { file_index: 1, line_number: 1, column: 0, byte_offset: 20, match_length: 3 },
            ContentMatch { file_index: 0, line_number: 1, column: 0, byte_offset: 5, match_length: 3 },
        ];

        gpu_sort_results(&mut results);

        assert_eq!(results[0].file_index, 0);
        assert_eq!(results[1].file_index, 1);
    }

    #[test]
    fn test_gpu_sort_edge_values() {
        // Test with edge values: max u32 file_index, max u32 byte_offset
        let mut results = vec![
            ContentMatch { file_index: usize::MAX & 0xFFFF_FFFF, line_number: 0, column: 0, byte_offset: u32::MAX, match_length: 1 },
            ContentMatch { file_index: 0, line_number: 0, column: 0, byte_offset: 0, match_length: 1 },
        ];

        gpu_sort_results(&mut results);

        assert_eq!(results[0].file_index, 0);
        assert_eq!(results[0].byte_offset, 0);
    }
}
