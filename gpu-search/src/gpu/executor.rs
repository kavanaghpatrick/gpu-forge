//! SearchExecutor with completion-handler re-dispatch chain.
//!
//! Persistent GPU search engine that stays alive between searches. Uses
//! a dedicated background thread with Metal command buffer dispatch for each
//! search, MTLSharedEvent for idle/wake signaling, and a condvar-based
//! wake mechanism for the re-dispatch chain.
//!
//! Architecture:
//! ```text
//! CPU: submit_search(pattern, files) -> sequence_id
//!      poll_results(sequence_id) -> Option<SearchResults>
//!
//! Background thread:
//!   loop {
//!     wait_for_work_or_timeout()
//!     if has_pending_search:
//!       encode GPU command buffer -> commit -> waitUntilCompleted
//!       store results
//!     if idle_timeout:
//!       signal SharedEvent(idle) -> park thread
//!   }
//!
//! GPU: [search_cmd_0 ~~~~] [search_cmd_1 ~~~~] ... idle
//! ```
//!
//! Adapted from gpu-query's `AutonomousExecutor` + `RedispatchChain` pattern.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLResourceOptions, MTLSharedEvent, MTLSize,
};

use crate::gpu::pipeline::PsoCache;

// ============================================================================
// Constants
// ============================================================================

/// Chunk size matching content_search kernel.
const CHUNK_SIZE: usize = 4096;

/// Maximum pattern length for search.
const MAX_PATTERN_LEN: usize = 64;

/// Maximum matches the GPU can return per dispatch.
const MAX_MATCHES: usize = 10_000;

/// Bytes processed per GPU thread (vectorized uchar4).
const BYTES_PER_THREAD: usize = 64;

/// Threadgroup size matching the Metal shader.
const THREADGROUP_SIZE: usize = 256;

/// Idle timeout in milliseconds. If no new searches arrive within this
/// window, the re-dispatch thread parks and the executor goes idle.
const IDLE_TIMEOUT_MS: u64 = 500;

// ============================================================================
// GPU-side repr(C) types (must match search_types.h)
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ChunkMetadata {
    file_index: u32,
    chunk_index: u32,
    offset_in_file: u64,
    chunk_length: u32,
    flags: u32,
    buffer_offset: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuSearchParams {
    chunk_count: u32,
    pattern_len: u32,
    case_sensitive: u32,
    total_bytes: u32,
}

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

const _: () = assert!(std::mem::size_of::<GpuSearchParams>() == 16);
const _: () = assert!(std::mem::size_of::<GpuMatchResult>() == 32);
const _: () = assert!(std::mem::size_of::<ChunkMetadata>() == 32);

// ============================================================================
// Engine state
// ============================================================================

/// Executor lifecycle state.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorState {
    /// Not yet started.
    Created = 0,
    /// Actively running (re-dispatch thread is live).
    Active = 1,
    /// Idle -- no work for > IDLE_TIMEOUT_MS, thread parked.
    Idle = 2,
    /// Permanently stopped.
    Shutdown = 3,
}

impl ExecutorState {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => ExecutorState::Created,
            1 => ExecutorState::Active,
            2 => ExecutorState::Idle,
            3 => ExecutorState::Shutdown,
            _ => ExecutorState::Shutdown,
        }
    }
}

// ============================================================================
// Search results
// ============================================================================

/// A single match from the GPU search.
#[derive(Debug, Clone)]
pub struct SearchMatch {
    /// Index of the file (in the order submitted).
    pub file_index: usize,
    /// Line number (GPU-local, may need refinement).
    pub line_number: u32,
    /// Column within the line.
    pub column: u32,
    /// Byte offset within the file chunk.
    pub byte_offset: u32,
    /// Length of the matched text.
    pub match_length: u32,
}

/// Results from a completed search.
#[derive(Debug, Clone)]
pub struct SearchResults {
    /// Sequence ID of the search request.
    pub sequence_id: u32,
    /// Matches found.
    pub matches: Vec<SearchMatch>,
    /// Total match count (may exceed matches.len() if capped).
    pub total_match_count: u32,
    /// GPU execution time in microseconds (approximate).
    pub elapsed_us: u64,
}

// ============================================================================
// Pending search request (CPU-side, queued for GPU)
// ============================================================================

struct PendingSearch {
    sequence_id: u32,
    pattern: Vec<u8>,
    file_contents: Vec<Vec<u8>>,
    case_sensitive: bool,
}

// ============================================================================
// Shared state between main thread and dispatch thread
// ============================================================================

struct SharedState {
    /// Current executor state.
    state: AtomicU8,
    /// Number of re-dispatches completed.
    redispatch_count: AtomicU64,
    /// Error count.
    error_count: AtomicU64,
    /// Pending search queue (protected by condvar for wake signaling).
    pending: Mutex<Vec<PendingSearch>>,
    /// Condvar to wake the dispatch thread when new work arrives or shutdown.
    wake_signal: Condvar,
    /// Completed results (sequence_id -> results).
    completed: Mutex<HashMap<u32, SearchResults>>,
}

/// GPU resources used by the dispatch thread. NOT Send by default (ObjC types)
/// so we wrap them carefully.
struct GpuResources {
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    chunks_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    metadata_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    params_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pattern_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    matches_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    match_count_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    event_counter: AtomicU64,
    max_chunks: usize,
}

// SAFETY: Metal objects are reference-counted Objective-C objects, thread-safe for
// retain/release. Metal command queues and buffers are documented as safe from any thread.
unsafe impl Send for GpuResources {}

// ============================================================================
// SearchExecutor
// ============================================================================

/// GPU search executor with persistent re-dispatch chain.
///
/// Lifecycle: `new()` -> `start()` -> `submit_search()` / `poll_results()`
/// -> `stop()`.
///
/// The executor owns a separate MTLCommandQueue for search compute, isolated
/// from any UI rendering queue. When started, it spawns a background thread
/// that processes search requests. When no work arrives for > 500ms, it goes
/// idle. New work via `submit_search()` wakes it.
pub struct SearchExecutor {
    #[allow(dead_code)]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    shared: Arc<SharedState>,
    /// Handle to the dispatch thread (joined on drop).
    thread_handle: Option<std::thread::JoinHandle<()>>,
    /// Monotonically increasing sequence ID for submitted searches.
    next_sequence_id: AtomicU32,
    /// SharedEvent value accessor (lives on main thread too).
    shared_event_value: Arc<AtomicU64>,
}

// SAFETY: SharedState is all atomics/mutexes (Send+Sync). Retained<MTLDevice> is
// thread-safe for retain/release. JoinHandle is Send. Atomics are Send+Sync.
unsafe impl Send for SearchExecutor {}
unsafe impl Sync for SearchExecutor {}

impl SearchExecutor {
    /// Create a new SearchExecutor (does NOT start the dispatch thread).
    ///
    /// Call `start()` to begin processing.
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        pso_cache: &PsoCache,
    ) -> Self {
        let shared = Arc::new(SharedState {
            state: AtomicU8::new(ExecutorState::Created as u8),
            redispatch_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            pending: Mutex::new(Vec::new()),
            wake_signal: Condvar::new(),
            completed: Mutex::new(HashMap::new()),
        });

        let max_chunks = 100_000; // ~400MB data capacity
        let options = MTLResourceOptions::StorageModeShared;

        // Separate command queue for search compute
        let command_queue = device
            .newCommandQueue()
            .expect("Failed to create search command queue");

        // Get content search PSO
        let pso = pso_cache
            .get("content_search_kernel")
            .expect("content_search_kernel PSO not in cache");
        let pso = Retained::from(pso);

        // Pre-allocate GPU buffers
        let chunks_buffer = device
            .newBufferWithLength_options(max_chunks * CHUNK_SIZE, options)
            .expect("Failed to allocate chunks buffer");
        let metadata_buffer = device
            .newBufferWithLength_options(
                max_chunks * std::mem::size_of::<ChunkMetadata>(),
                options,
            )
            .expect("Failed to allocate metadata buffer");
        let params_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<GpuSearchParams>(), options)
            .expect("Failed to allocate params buffer");
        let pattern_buffer = device
            .newBufferWithLength_options(MAX_PATTERN_LEN, options)
            .expect("Failed to allocate pattern buffer");
        let matches_buffer = device
            .newBufferWithLength_options(
                MAX_MATCHES * std::mem::size_of::<GpuMatchResult>(),
                options,
            )
            .expect("Failed to allocate matches buffer");
        let match_count_buffer = device
            .newBufferWithLength_options(std::mem::size_of::<u32>(), options)
            .expect("Failed to allocate match count buffer");

        // MTLSharedEvent for idle/wake
        let shared_event = device
            .newSharedEvent()
            .expect("Failed to create MTLSharedEvent");
        shared_event.setSignaledValue(0);

        let shared_event_value = Arc::new(AtomicU64::new(0));

        // Store GPU resources for the dispatch thread (created but not started yet)
        // We store them in the thread_handle closure when start() is called.
        // For now, we need to keep them around. Use a mutex to pass them to the thread.
        let gpu_resources = GpuResources {
            command_queue,
            pso,
            chunks_buffer,
            metadata_buffer,
            params_buffer,
            pattern_buffer,
            matches_buffer,
            match_count_buffer,
            shared_event,
            event_counter: AtomicU64::new(0),
            max_chunks,
        };

        // Store gpu_resources in a Mutex so we can move them into the thread later
        // Actually, let's just start the thread right away and use the Created state
        // to prevent it from processing work until start() is called.
        let shared_clone = shared.clone();
        let event_val_clone = shared_event_value.clone();

        let thread_handle = std::thread::Builder::new()
            .name("gpu-search-executor".into())
            .spawn(move || {
                dispatch_thread_main(shared_clone, gpu_resources, event_val_clone);
            })
            .expect("Failed to spawn search executor thread");

        Self {
            device: Retained::from(device),
            shared,
            thread_handle: Some(thread_handle),
            next_sequence_id: AtomicU32::new(1),
            shared_event_value,
        }
    }

    /// Start the executor. The dispatch thread begins watching for
    /// pending searches and dispatching them on the GPU.
    pub fn start(&self) {
        // Transition Created -> Active
        let _ = self.shared.state.compare_exchange(
            ExecutorState::Created as u8,
            ExecutorState::Active as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        // Wake the thread in case it's waiting
        self.shared.wake_signal.notify_one();
    }

    /// Submit a search request. Returns a sequence_id to poll for results.
    ///
    /// `pattern`: the literal search pattern bytes.
    /// `files`: file contents to search through.
    /// `case_sensitive`: whether the search is case-sensitive.
    pub fn submit_search(
        &self,
        pattern: &[u8],
        files: Vec<Vec<u8>>,
        case_sensitive: bool,
    ) -> u32 {
        let seq = self.next_sequence_id.fetch_add(1, Ordering::Relaxed);

        let pending = PendingSearch {
            sequence_id: seq,
            pattern: pattern.to_vec(),
            file_contents: files,
            case_sensitive,
        };

        // Enqueue
        {
            let mut queue = self.shared.pending.lock().unwrap();
            queue.push(pending);
        }

        // Wake if idle -- transition Idle -> Active
        let _ = self.shared.state.compare_exchange(
            ExecutorState::Idle as u8,
            ExecutorState::Active as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        // Wake the dispatch thread
        self.shared.wake_signal.notify_one();

        seq
    }

    /// Poll for results of a previously submitted search.
    ///
    /// Returns `Some(results)` if the search is complete, `None` if still pending.
    /// Results are consumed (removed from the internal map) on first poll.
    pub fn poll_results(&self, sequence_id: u32) -> Option<SearchResults> {
        self.shared
            .completed
            .lock()
            .unwrap()
            .remove(&sequence_id)
    }

    /// Stop the executor. The dispatch thread will exit after the
    /// current in-flight command buffer completes.
    pub fn stop(&self) {
        self.shared
            .state
            .store(ExecutorState::Shutdown as u8, Ordering::Release);
        self.shared.wake_signal.notify_one();
    }

    /// Get the current executor state.
    pub fn state(&self) -> ExecutorState {
        ExecutorState::from_u8(self.shared.state.load(Ordering::Acquire))
    }

    /// Get the number of re-dispatches (search commands) completed.
    pub fn redispatch_count(&self) -> u64 {
        self.shared.redispatch_count.load(Ordering::Relaxed)
    }

    /// Get the number of GPU errors encountered.
    pub fn error_count(&self) -> u64 {
        self.shared.error_count.load(Ordering::Relaxed)
    }

    /// Get the current MTLSharedEvent signaled value.
    /// Even = idle transition, Odd = active/wake.
    pub fn shared_event_value(&self) -> u64 {
        self.shared_event_value.load(Ordering::Acquire)
    }

    /// Get the number of pending (unprocessed) searches.
    pub fn pending_count(&self) -> usize {
        self.shared.pending.lock().unwrap().len()
    }

    /// Get the number of completed (uncollected) results.
    pub fn completed_count(&self) -> usize {
        self.shared.completed.lock().unwrap().len()
    }
}

impl Drop for SearchExecutor {
    fn drop(&mut self) {
        self.stop();
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }
}

// ============================================================================
// Dispatch thread
// ============================================================================

/// Main loop for the dispatch thread.
fn dispatch_thread_main(
    shared: Arc<SharedState>,
    gpu: GpuResources,
    event_value: Arc<AtomicU64>,
) {
    // Wait until started (state transitions from Created to Active)
    loop {
        let state = ExecutorState::from_u8(shared.state.load(Ordering::Acquire));
        match state {
            ExecutorState::Created => {
                // Wait for start() signal
                let guard = shared.pending.lock().unwrap();
                let _ = shared
                    .wake_signal
                    .wait_timeout(guard, Duration::from_millis(50))
                    .unwrap();
                continue;
            }
            ExecutorState::Shutdown => return,
            _ => break,
        }
    }

    // Signal active via SharedEvent (odd = active)
    let val = gpu.event_counter.fetch_add(1, Ordering::AcqRel) + 1;
    gpu.shared_event.setSignaledValue(val);
    event_value.store(val, Ordering::Release);

    let mut last_work_time = Instant::now();

    // Main dispatch loop
    loop {
        let state = ExecutorState::from_u8(shared.state.load(Ordering::Acquire));
        if state == ExecutorState::Shutdown {
            return;
        }

        // Try to pop a pending search
        let pending = {
            let mut queue = shared.pending.lock().unwrap();
            if queue.is_empty() {
                None
            } else {
                Some(queue.remove(0))
            }
        };

        if let Some(search) = pending {
            // Process the search
            let results = execute_search_on_gpu(&gpu, &search, &shared);
            shared
                .completed
                .lock()
                .unwrap()
                .insert(search.sequence_id, results);
            shared.redispatch_count.fetch_add(1, Ordering::Relaxed);
            last_work_time = Instant::now();
            continue;
        }

        // No work available -- check idle timeout
        let elapsed = last_work_time.elapsed();
        if elapsed > Duration::from_millis(IDLE_TIMEOUT_MS) {
            // Go idle
            shared
                .state
                .store(ExecutorState::Idle as u8, Ordering::Release);

            // Signal SharedEvent with even value (idle)
            let val = gpu.event_counter.fetch_add(1, Ordering::AcqRel) + 1;
            gpu.shared_event.setSignaledValue(val);
            event_value.store(val, Ordering::Release);

            // Park until woken by submit_search or stop
            let guard = shared.pending.lock().unwrap();
            let _ = shared
                .wake_signal
                .wait_timeout(guard, Duration::from_secs(60))
                .unwrap();

            // Check if we were woken for shutdown
            let state = ExecutorState::from_u8(shared.state.load(Ordering::Acquire));
            if state == ExecutorState::Shutdown {
                return;
            }

            // Woken for new work -- transition back to Active
            if state == ExecutorState::Active {
                let val = gpu.event_counter.fetch_add(1, Ordering::AcqRel) + 1;
                gpu.shared_event.setSignaledValue(val);
                event_value.store(val, Ordering::Release);
                last_work_time = Instant::now();
            }
            continue;
        }

        // Not timed out yet -- short wait for new work
        let remaining = Duration::from_millis(IDLE_TIMEOUT_MS) - elapsed;
        let guard = shared.pending.lock().unwrap();
        let _ = shared
            .wake_signal
            .wait_timeout(guard, remaining.min(Duration::from_millis(10)))
            .unwrap();
    }
}

/// Execute a single search request on the GPU via Metal command buffer.
fn execute_search_on_gpu(
    gpu: &GpuResources,
    search: &PendingSearch,
    shared: &SharedState,
) -> SearchResults {
    let start = Instant::now();

    // Load file contents into GPU buffers
    let mut chunk_count = 0usize;

    unsafe {
        let chunks_ptr = gpu.chunks_buffer.contents().as_ptr() as *mut u8;
        let meta_ptr = gpu.metadata_buffer.contents().as_ptr() as *mut ChunkMetadata;

        for (file_idx, content) in search.file_contents.iter().enumerate() {
            if content.is_empty() {
                continue;
            }
            let num_file_chunks = content.len().div_ceil(CHUNK_SIZE);

            for chunk_i in 0..num_file_chunks {
                if chunk_count >= gpu.max_chunks {
                    break;
                }

                let offset = chunk_i * CHUNK_SIZE;
                let chunk_len = (content.len() - offset).min(CHUNK_SIZE);
                let chunk = &content[offset..offset + chunk_len];

                // Copy chunk data (padded to CHUNK_SIZE)
                let dst = chunks_ptr.add(chunk_count * CHUNK_SIZE);
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), dst, chunk_len);
                if chunk_len < CHUNK_SIZE {
                    std::ptr::write_bytes(dst.add(chunk_len), 0, CHUNK_SIZE - chunk_len);
                }

                // Write metadata
                let mut flags = 1u32; // is_text
                if chunk_i == 0 {
                    flags |= 2; // is_first
                }
                if chunk_i == num_file_chunks - 1 {
                    flags |= 4; // is_last
                }

                *meta_ptr.add(chunk_count) = ChunkMetadata {
                    file_index: file_idx as u32,
                    chunk_index: chunk_i as u32,
                    offset_in_file: offset as u64,
                    chunk_length: chunk_len as u32,
                    flags,
                    buffer_offset: 0,
                };

                chunk_count += 1;
            }
        }
    }

    if chunk_count == 0 {
        return SearchResults {
            sequence_id: search.sequence_id,
            matches: vec![],
            total_match_count: 0,
            elapsed_us: 0,
        };
    }

    let total_data_bytes = chunk_count * CHUNK_SIZE;

    // Prepare pattern
    let pattern_bytes: Vec<u8> = if search.case_sensitive {
        search.pattern.clone()
    } else {
        search
            .pattern
            .iter()
            .map(|b| b.to_ascii_lowercase())
            .collect()
    };

    // Write GPU params
    unsafe {
        // Reset match count
        let count_ptr = gpu.match_count_buffer.contents().as_ptr() as *mut u32;
        *count_ptr = 0;

        // Write search params
        let params_ptr = gpu.params_buffer.contents().as_ptr() as *mut GpuSearchParams;
        *params_ptr = GpuSearchParams {
            chunk_count: chunk_count as u32,
            pattern_len: pattern_bytes.len() as u32,
            case_sensitive: if search.case_sensitive { 1 } else { 0 },
            total_bytes: total_data_bytes as u32,
        };

        // Write pattern bytes
        let pattern_ptr = gpu.pattern_buffer.contents().as_ptr() as *mut u8;
        std::ptr::write_bytes(pattern_ptr, 0, MAX_PATTERN_LEN);
        std::ptr::copy_nonoverlapping(
            pattern_bytes.as_ptr(),
            pattern_ptr,
            pattern_bytes.len().min(MAX_PATTERN_LEN),
        );
    }

    // Create and dispatch command buffer
    let cmd = match gpu.command_queue.commandBuffer() {
        Some(cb) => cb,
        None => {
            shared.error_count.fetch_add(1, Ordering::Relaxed);
            return SearchResults {
                sequence_id: search.sequence_id,
                matches: vec![],
                total_match_count: 0,
                elapsed_us: start.elapsed().as_micros() as u64,
            };
        }
    };

    let encoder = match cmd.computeCommandEncoder() {
        Some(enc) => enc,
        None => {
            shared.error_count.fetch_add(1, Ordering::Relaxed);
            return SearchResults {
                sequence_id: search.sequence_id,
                matches: vec![],
                total_match_count: 0,
                elapsed_us: start.elapsed().as_micros() as u64,
            };
        }
    };

    encoder.setComputePipelineState(&gpu.pso);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&*gpu.chunks_buffer), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&*gpu.metadata_buffer), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&*gpu.params_buffer), 0, 2);
        encoder.setBuffer_offset_atIndex(Some(&*gpu.pattern_buffer), 0, 3);
        encoder.setBuffer_offset_atIndex(Some(&*gpu.matches_buffer), 0, 4);
        encoder.setBuffer_offset_atIndex(Some(&*gpu.match_count_buffer), 0, 5);
    }

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

    // Commit and wait for GPU completion (re-dispatch chain serializes searches)
    cmd.commit();
    cmd.waitUntilCompleted();

    // Collect results
    let elapsed_us = start.elapsed().as_micros() as u64;
    let mut matches = Vec::new();

    unsafe {
        let count_ptr = gpu.match_count_buffer.contents().as_ptr() as *const u32;
        let total_count = *count_ptr;
        let result_count = (total_count as usize).min(MAX_MATCHES);
        let matches_ptr = gpu.matches_buffer.contents().as_ptr() as *const GpuMatchResult;

        for i in 0..result_count {
            let m = *matches_ptr.add(i);
            matches.push(SearchMatch {
                file_index: m.file_index as usize,
                line_number: m.line_number,
                column: m.column,
                byte_offset: m.chunk_index * CHUNK_SIZE as u32 + m.context_start,
                match_length: m.match_length,
            });
        }

        SearchResults {
            sequence_id: search.sequence_id,
            matches,
            total_match_count: total_count,
            elapsed_us,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::pipeline::PsoCache;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    #[test]
    fn test_executor_lifecycle() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);

        // 1. Create executor
        let executor = SearchExecutor::new(&device, &pso_cache);
        assert_eq!(executor.state(), ExecutorState::Created);

        // 2. Start
        executor.start();
        assert_eq!(executor.state(), ExecutorState::Active);

        // 3. Submit a search
        let content =
            b"fn main() {\n    println!(\"hello\");\n}\n\nfn test() {\n    let x = 1;\n}\n";
        let seq = executor.submit_search(b"fn ", vec![content.to_vec()], true);
        assert!(seq >= 1, "Should get valid sequence_id");

        // 4. Poll for results (may need to wait)
        let mut results = None;
        for _ in 0..200 {
            results = executor.poll_results(seq);
            if results.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let results = results.expect("Search should complete within 2 seconds");
        assert_eq!(results.sequence_id, seq);
        // Content has 2 "fn " occurrences
        assert_eq!(
            results.matches.len(),
            2,
            "Should find 2 'fn ' matches, got {}",
            results.matches.len()
        );

        // 5. Wait for idle (no more work -> should idle after IDLE_TIMEOUT_MS)
        std::thread::sleep(std::time::Duration::from_millis(IDLE_TIMEOUT_MS + 200));
        let state = executor.state();
        assert_eq!(state, ExecutorState::Idle, "Should be idle after timeout");

        // 6. Wake with new search
        let seq2 = executor.submit_search(b"let", vec![content.to_vec()], true);
        // submit_search auto-wakes if idle
        assert_eq!(executor.state(), ExecutorState::Active);

        let mut results2 = None;
        for _ in 0..200 {
            results2 = executor.poll_results(seq2);
            if results2.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let results2 = results2.expect("Second search should complete");
        assert_eq!(results2.sequence_id, seq2);
        assert!(results2.matches.len() >= 1, "Should find 'let' matches");

        // 7. Stop
        executor.stop();
        assert_eq!(executor.state(), ExecutorState::Shutdown);

        // 8. Verify no errors
        assert_eq!(executor.error_count(), 0, "Should have no GPU errors");

        println!("Lifecycle test passed:");
        println!(
            "  First search: {} matches in {}us",
            results.matches.len(),
            results.elapsed_us
        );
        println!(
            "  Second search: {} matches in {}us",
            results2.matches.len(),
            results2.elapsed_us
        );
        println!("  Re-dispatches: {}", executor.redispatch_count());
        println!("  SharedEvent value: {}", executor.shared_event_value());
    }

    #[test]
    fn test_executor_multi_file_search() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);
        let executor = SearchExecutor::new(&device, &pso_cache);
        executor.start();

        let file0 = b"fn alpha() {}\nfn beta() {}\n";
        let file1 = b"fn gamma() {}\n";
        let file2 = b"no matches here\n";

        let seq = executor.submit_search(
            b"fn ",
            vec![file0.to_vec(), file1.to_vec(), file2.to_vec()],
            true,
        );

        let mut results = None;
        for _ in 0..200 {
            results = executor.poll_results(seq);
            if results.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let results = results.expect("Multi-file search should complete");
        assert_eq!(
            results.matches.len(),
            3,
            "Should find 3 'fn ' matches across files"
        );

        let f0: Vec<_> = results
            .matches
            .iter()
            .filter(|m| m.file_index == 0)
            .collect();
        let f1: Vec<_> = results
            .matches
            .iter()
            .filter(|m| m.file_index == 1)
            .collect();
        let f2: Vec<_> = results
            .matches
            .iter()
            .filter(|m| m.file_index == 2)
            .collect();
        assert_eq!(f0.len(), 2, "File 0 should have 2 matches");
        assert_eq!(f1.len(), 1, "File 1 should have 1 match");
        assert_eq!(f2.len(), 0, "File 2 should have 0 matches");

        executor.stop();
    }

    #[test]
    fn test_executor_empty_search() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);
        let executor = SearchExecutor::new(&device, &pso_cache);
        executor.start();

        let seq = executor.submit_search(b"test", vec![vec![]], true);

        let mut results = None;
        for _ in 0..200 {
            results = executor.poll_results(seq);
            if results.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let results = results.expect("Empty search should complete");
        assert_eq!(results.matches.len(), 0, "No matches in empty files");
        assert_eq!(results.total_match_count, 0);

        executor.stop();
    }

    #[test]
    fn test_executor_case_insensitive() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);
        let executor = SearchExecutor::new(&device, &pso_cache);
        executor.start();

        let content = b"Hello HELLO hello hElLo\n";
        let seq = executor.submit_search(b"hello", vec![content.to_vec()], false);

        let mut results = None;
        for _ in 0..200 {
            results = executor.poll_results(seq);
            if results.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let results = results.expect("Case-insensitive search should complete");
        assert_eq!(
            results.matches.len(),
            4,
            "Should find 4 case-insensitive 'hello' matches"
        );

        executor.stop();
    }

    #[test]
    fn test_executor_shared_event() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);
        let executor = SearchExecutor::new(&device, &pso_cache);

        // Before start: event value is 0
        assert_eq!(executor.shared_event_value(), 0);

        // After start: event value should be odd (active)
        executor.start();
        // Give the thread a moment to start and signal
        std::thread::sleep(std::time::Duration::from_millis(50));
        let val_after_start = executor.shared_event_value();
        assert!(
            val_after_start % 2 == 1,
            "Active event value should be odd, got {}",
            val_after_start
        );

        // Wait for idle
        std::thread::sleep(std::time::Duration::from_millis(IDLE_TIMEOUT_MS + 200));
        let val_after_idle = executor.shared_event_value();
        assert!(
            val_after_idle % 2 == 0,
            "Idle event value should be even, got {}",
            val_after_idle
        );
        assert!(
            val_after_idle > val_after_start,
            "Event value should increase"
        );

        executor.stop();
    }

    #[test]
    fn test_executor_sequential_searches() {
        let device = get_device();
        let pso_cache = PsoCache::new(&device);
        let executor = SearchExecutor::new(&device, &pso_cache);
        executor.start();

        let content = b"fn main() { fn test() { fn helper() {} } }\n";
        let mut sequence_ids = Vec::new();

        for _ in 0..5 {
            let seq = executor.submit_search(b"fn ", vec![content.to_vec()], true);
            sequence_ids.push(seq);
        }

        // Collect all results
        for (i, &seq) in sequence_ids.iter().enumerate() {
            let mut results = None;
            for _ in 0..200 {
                results = executor.poll_results(seq);
                if results.is_some() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            let results = results.unwrap_or_else(|| panic!("Search {} should complete", i));
            assert_eq!(results.sequence_id, seq);
            assert_eq!(
                results.matches.len(),
                3,
                "Each search should find 3 'fn ' matches"
            );
        }

        executor.stop();
    }
}
