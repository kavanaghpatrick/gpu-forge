//! Triple-buffered work queue for CPU->GPU search request handoff.
//!
//! Adapted from gpu-query's `WorkQueue` pattern. Three `SearchRequestSlot`
//! slots in a single Metal buffer (StorageModeShared). The CPU writes search
//! parameters to successive slots with Release ordering on `sequence_id`;
//! the GPU reads with Acquire ordering and processes only when it observes
//! a new (higher) sequence_id.
//!
//! Memory ordering: Acquire-Release on `sequence_id`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

/// Maximum pattern length in a search request slot.
pub const MAX_PATTERN_LEN: usize = 256;

/// Search request status values (written by GPU to indicate processing state).
pub mod status {
    /// Slot is empty / not yet submitted.
    pub const EMPTY: u32 = 0;
    /// CPU has written a new request, awaiting GPU pickup.
    pub const PENDING: u32 = 1;
    /// GPU is currently processing this request.
    pub const PROCESSING: u32 = 2;
    /// GPU has completed processing; results are available.
    pub const COMPLETE: u32 = 3;
    /// GPU encountered an error processing this request.
    pub const ERROR: u32 = 4;
}

/// Search request flags (bitfield).
pub mod request_flags {
    /// Case-sensitive search (default).
    pub const CASE_SENSITIVE: u32 = 1 << 0;
    /// Content search mode (search file contents).
    pub const CONTENT_SEARCH: u32 = 1 << 1;
    /// Filename search mode (search path index).
    pub const FILENAME_SEARCH: u32 = 1 << 2;
    /// Use turbo mode (defers line-number resolution to CPU).
    pub const TURBO_MODE: u32 = 1 << 3;
}

/// A single work queue slot for a search request.
///
/// Layout: 512 bytes, 4-byte aligned. `sequence_id` is at offset 0 so the
/// GPU can atomically read it to detect new work.
///
/// The CPU writes all fields EXCEPT `sequence_id` first, issues a Release
/// fence, then writes `sequence_id` last. The GPU issues an Acquire fence
/// after reading `sequence_id` to ensure it sees all other fields.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SearchRequestSlot {
    /// Monotonically increasing sequence number (offset 0, 4 bytes).
    /// Written LAST by CPU with Release ordering.
    /// Read FIRST by GPU with Acquire ordering.
    pub sequence_id: u32,

    /// Processing status (offset 4, 4 bytes).
    /// Written by GPU to signal completion.
    pub status: u32,

    /// Request flags bitfield (offset 8, 4 bytes).
    /// See `request_flags` module.
    pub flags: u32,

    /// Length of the search pattern in bytes (offset 12, 4 bytes).
    pub pattern_len: u32,

    /// Search pattern bytes, null-padded (offset 16, 256 bytes).
    pub pattern: [u8; MAX_PATTERN_LEN],

    /// Total bytes in the search input buffer (offset 272, 4 bytes).
    pub total_bytes: u32,

    /// Maximum number of matches to return (offset 276, 4 bytes).
    pub max_matches: u32,

    /// Number of files in the batch (offset 280, 4 bytes).
    pub file_count: u32,

    /// Number of matches found by GPU (offset 284, 4 bytes).
    /// Written by GPU after processing.
    pub match_count: u32,

    /// Elapsed GPU time in microseconds (offset 288, 4 bytes).
    /// Written by GPU after processing.
    pub elapsed_us: u32,

    /// Error code if status == ERROR (offset 292, 4 bytes).
    pub error_code: u32,

    /// Reserved padding to reach 512 bytes total (offset 296, 216 bytes = 54 x u32).
    pub _reserved: [u32; 54],
}

// Compile-time layout assertions
const _: () = assert!(std::mem::size_of::<SearchRequestSlot>() == 512);
const _: () = assert!(std::mem::offset_of!(SearchRequestSlot, sequence_id) == 0);

impl Default for SearchRequestSlot {
    fn default() -> Self {
        Self {
            sequence_id: 0,
            status: status::EMPTY,
            flags: 0,
            pattern_len: 0,
            pattern: [0u8; MAX_PATTERN_LEN],
            total_bytes: 0,
            max_matches: 0,
            file_count: 0,
            match_count: 0,
            elapsed_us: 0,
            error_code: 0,
            _reserved: [0u32; 54],
        }
    }
}

impl SearchRequestSlot {
    /// Create a new search request slot from a pattern and parameters.
    pub fn new(
        pattern: &[u8],
        total_bytes: u32,
        max_matches: u32,
        file_count: u32,
        flags: u32,
    ) -> Self {
        let mut slot = Self::default();
        let copy_len = pattern.len().min(MAX_PATTERN_LEN);
        slot.pattern[..copy_len].copy_from_slice(&pattern[..copy_len]);
        slot.pattern_len = copy_len as u32;
        slot.total_bytes = total_bytes;
        slot.max_matches = max_matches;
        slot.file_count = file_count;
        slot.flags = flags;
        slot.status = status::PENDING;
        slot
    }

    /// Get the pattern as a byte slice.
    pub fn pattern_bytes(&self) -> &[u8] {
        &self.pattern[..self.pattern_len as usize]
    }
}

/// Size of a single work queue slot in bytes.
pub const SLOT_SIZE: usize = std::mem::size_of::<SearchRequestSlot>();

/// Number of slots in the triple buffer.
pub const SLOT_COUNT: usize = 3;

/// Total size of the work queue buffer in bytes (3 x 512 = 1536).
pub const BUFFER_SIZE: usize = SLOT_SIZE * SLOT_COUNT;

/// Triple-buffered work queue wrapping a Metal buffer for CPU->GPU search
/// request handoff.
///
/// The CPU writes search parameters to successive slots (0, 1, 2, 0, 1, ...),
/// bumping the `sequence_id` with Release ordering so the GPU can detect new
/// work by reading the sequence_id with Acquire ordering.
pub struct WorkQueue {
    /// Metal buffer backing the triple-buffer (1536 bytes, StorageModeShared).
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Index of the next slot to write (cycles 0 -> 1 -> 2 -> 0).
    write_idx: u32,
    /// Next sequence ID to assign (monotonically increasing, starts at 1).
    next_sequence_id: u32,
}

impl WorkQueue {
    /// Allocate a new triple-buffered work queue on the given Metal device.
    ///
    /// The buffer is 1536 bytes (3 x 512B) with StorageModeShared for
    /// unified memory access by both CPU and GPU.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        let options = MTLResourceOptions::StorageModeShared;
        let buffer = device
            .newBufferWithLength_options(BUFFER_SIZE, options)
            .expect("Failed to allocate work queue Metal buffer");

        // Zero-initialize the buffer
        unsafe {
            let ptr = buffer.contents().as_ptr() as *mut u8;
            std::ptr::write_bytes(ptr, 0, BUFFER_SIZE);
        }

        Self {
            buffer,
            write_idx: 0,
            next_sequence_id: 1,
        }
    }

    /// Submit a search request to the work queue.
    ///
    /// Steps:
    /// 1. Copy all fields EXCEPT `sequence_id` to the current slot
    /// 2. Release fence (ensures GPU sees all fields before the new sequence_id)
    /// 3. Write `sequence_id` last
    /// 4. Advance `write_idx` mod 3
    /// 5. Increment `next_sequence_id`
    ///
    /// Returns the sequence_id assigned to this request.
    pub fn submit_request(&mut self, request: &SearchRequestSlot) -> u32 {
        let seq = self.next_sequence_id;
        let slot_offset = self.write_idx as usize * SLOT_SIZE;

        unsafe {
            let base = self.buffer.contents().as_ptr() as *mut u8;
            let slot_ptr = base.add(slot_offset);

            // Copy everything after sequence_id (offset 4 onward, 508 bytes).
            // sequence_id is at offset 0, 4 bytes.
            let src = request as *const SearchRequestSlot as *const u8;
            std::ptr::copy_nonoverlapping(src.add(4), slot_ptr.add(4), SLOT_SIZE - 4);

            // Release fence: ensure all field writes are visible before sequence_id
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

            // Write sequence_id last so GPU sees consistent data when it reads the new ID
            let seq_ptr = slot_ptr as *mut u32;
            std::ptr::write_volatile(seq_ptr, seq);
        }

        // Advance write index (0 -> 1 -> 2 -> 0)
        self.write_idx = (self.write_idx + 1) % SLOT_COUNT as u32;
        self.next_sequence_id += 1;

        seq
    }

    /// Read the latest written sequence ID (CPU-side diagnostic).
    ///
    /// Returns the sequence_id of the most recently written slot.
    /// Uses Acquire ordering to pair with the Release in `submit_request`.
    pub fn read_latest_sequence_id(&self) -> u32 {
        let last_idx = if self.write_idx == 0 {
            SLOT_COUNT as u32 - 1
        } else {
            self.write_idx - 1
        };
        let slot_offset = last_idx as usize * SLOT_SIZE;

        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let base = self.buffer.contents().as_ptr() as *const u8;
            let seq_ptr = base.add(slot_offset) as *const u32;
            std::ptr::read_volatile(seq_ptr)
        }
    }

    /// Read the latest completed result from the work queue.
    ///
    /// Scans all slots for the highest sequence_id with status == COMPLETE.
    /// Returns None if no completed results exist.
    pub fn get_latest(&self) -> Option<SearchRequestSlot> {
        let mut best: Option<(u32, SearchRequestSlot)> = None;

        for i in 0..SLOT_COUNT {
            let slot = self.read_slot(i);
            if slot.status == status::COMPLETE {
                match &best {
                    Some((best_seq, _)) if slot.sequence_id <= *best_seq => {}
                    _ => {
                        best = Some((slot.sequence_id, slot));
                    }
                }
            }
        }

        best.map(|(_, slot)| slot)
    }

    /// Get a reference to the underlying Metal buffer (for binding to GPU commands).
    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Get the current write index (for testing/diagnostics).
    pub fn write_idx(&self) -> u32 {
        self.write_idx
    }

    /// Get the next sequence ID that will be assigned (for testing/diagnostics).
    pub fn next_sequence_id(&self) -> u32 {
        self.next_sequence_id
    }

    /// Read back the SearchRequestSlot at a given slot index (for testing).
    ///
    /// # Panics
    /// Panics if `slot_idx >= SLOT_COUNT`.
    pub fn read_slot(&self, slot_idx: usize) -> SearchRequestSlot {
        assert!(slot_idx < SLOT_COUNT, "slot index out of range");
        let slot_offset = slot_idx * SLOT_SIZE;

        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let base = self.buffer.contents().as_ptr() as *const u8;
            let slot_ptr = base.add(slot_offset) as *const SearchRequestSlot;
            std::ptr::read(slot_ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    // ================================================================
    // Buffer allocation tests
    // ================================================================

    #[test]
    fn test_work_queue_buffer_size() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert_eq!(
            wq.buffer().length() as usize,
            BUFFER_SIZE,
            "Work queue buffer must be 3 x 512 = 1536 bytes"
        );
    }

    #[test]
    fn test_work_queue_buffer_shared_mode() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        let ptr = wq.buffer().contents().as_ptr();
        assert!(!ptr.is_null(), "StorageModeShared buffer must have CPU-visible pointer");
    }

    #[test]
    fn test_work_queue_slot_size() {
        assert_eq!(SLOT_SIZE, 512, "SearchRequestSlot must be 512 bytes");
    }

    #[test]
    fn test_work_queue_buffer_size_constant() {
        assert_eq!(BUFFER_SIZE, 1536, "Triple buffer = 3 x 512 = 1536");
    }

    // ================================================================
    // Write index cycling tests
    // ================================================================

    #[test]
    fn test_work_queue_initial_state() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert_eq!(wq.write_idx(), 0, "Initial write_idx must be 0");
        assert_eq!(wq.next_sequence_id(), 1, "First sequence_id must be 1");
    }

    #[test]
    fn test_work_queue_write_idx_cycles() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let request = SearchRequestSlot::default();

        assert_eq!(wq.write_idx(), 0);
        wq.submit_request(&request);
        assert_eq!(wq.write_idx(), 1, "After 1st write: idx=1");
        wq.submit_request(&request);
        assert_eq!(wq.write_idx(), 2, "After 2nd write: idx=2");
        wq.submit_request(&request);
        assert_eq!(wq.write_idx(), 0, "After 3rd write: idx=0 (wrap)");
        wq.submit_request(&request);
        assert_eq!(wq.write_idx(), 1, "After 4th write: idx=1");
    }

    // ================================================================
    // Sequence ID tests
    // ================================================================

    #[test]
    fn test_work_queue_sequence_monotonic() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let request = SearchRequestSlot::default();

        let mut prev_seq = 0u32;
        for i in 0..10 {
            let seq = wq.submit_request(&request);
            assert!(
                seq > prev_seq,
                "sequence_id must increase: iteration {}, prev={}, curr={}",
                i, prev_seq, seq
            );
            prev_seq = seq;
        }
    }

    #[test]
    fn test_work_queue_sequence_values() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let request = SearchRequestSlot::default();

        for expected in 1..=6u32 {
            let seq = wq.submit_request(&request);
            assert_eq!(seq, expected, "sequence_id should be {}", expected);
            let read_seq = wq.read_latest_sequence_id();
            assert_eq!(read_seq, expected, "read_latest should be {}", expected);
        }
    }

    // ================================================================
    // Slot population correctness tests
    // ================================================================

    #[test]
    fn test_work_queue_preserves_fields() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let request = SearchRequestSlot::new(
            b"test_pattern",
            1024 * 1024,
            10_000,
            42,
            request_flags::CASE_SENSITIVE | request_flags::CONTENT_SEARCH,
        );

        wq.submit_request(&request);

        let readback = wq.read_slot(0);
        assert_eq!(readback.sequence_id, 1, "sequence_id");
        assert_eq!(readback.status, status::PENDING, "status");
        assert_eq!(
            readback.flags,
            request_flags::CASE_SENSITIVE | request_flags::CONTENT_SEARCH,
            "flags"
        );
        assert_eq!(readback.pattern_len, 12, "pattern_len");
        assert_eq!(&readback.pattern[..12], b"test_pattern", "pattern");
        assert_eq!(readback.pattern[12], 0, "pattern null-padded");
        assert_eq!(readback.total_bytes, 1024 * 1024, "total_bytes");
        assert_eq!(readback.max_matches, 10_000, "max_matches");
        assert_eq!(readback.file_count, 42, "file_count");
    }

    #[test]
    fn test_work_queue_writes_correct_slots() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        for i in 0..3u32 {
            let request = SearchRequestSlot::new(
                format!("pat{}", i).as_bytes(),
                (i + 1) * 1000,
                100,
                i + 1,
                0,
            );
            wq.submit_request(&request);
        }

        for i in 0..3usize {
            let slot = wq.read_slot(i);
            assert_eq!(slot.sequence_id, (i + 1) as u32, "slot {} sequence_id", i);
            assert_eq!(slot.total_bytes, ((i as u32) + 1) * 1000, "slot {} total_bytes", i);
            assert_eq!(slot.file_count, (i as u32) + 1, "slot {} file_count", i);
        }
    }

    #[test]
    fn test_work_queue_wraparound() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        // Fill all 3 slots
        for i in 0..3u32 {
            let request = SearchRequestSlot::new(b"x", (i + 1) * 100, 10, i + 1, 0);
            wq.submit_request(&request);
        }

        // 4th write overwrites slot 0
        let request = SearchRequestSlot::new(b"overwrite", 9999, 10, 99, 0);
        wq.submit_request(&request);

        let slot0 = wq.read_slot(0);
        assert_eq!(slot0.total_bytes, 9999, "Slot 0 overwritten");
        assert_eq!(slot0.sequence_id, 4, "Slot 0 has seq=4 after wraparound");
        assert_eq!(slot0.file_count, 99, "Slot 0 has correct file_count");

        // Slots 1, 2 untouched
        let slot1 = wq.read_slot(1);
        assert_eq!(slot1.total_bytes, 200, "Slot 1 unchanged");
        assert_eq!(slot1.sequence_id, 2);

        let slot2 = wq.read_slot(2);
        assert_eq!(slot2.total_bytes, 300, "Slot 2 unchanged");
        assert_eq!(slot2.sequence_id, 3);
    }

    // ================================================================
    // Zero-initialization and search request helpers
    // ================================================================

    #[test]
    fn test_work_queue_zero_initialized() {
        let device = get_device();
        let wq = WorkQueue::new(&device);

        for i in 0..SLOT_COUNT {
            let slot = wq.read_slot(i);
            assert_eq!(slot.sequence_id, 0, "slot {} sequence_id should be 0", i);
            assert_eq!(slot.status, status::EMPTY, "slot {} status should be EMPTY", i);
            assert_eq!(slot.flags, 0, "slot {} flags should be 0", i);
            assert_eq!(slot.pattern_len, 0, "slot {} pattern_len should be 0", i);
            assert_eq!(slot.total_bytes, 0, "slot {} total_bytes should be 0", i);
        }
    }

    #[test]
    fn test_search_request_slot_new() {
        let slot = SearchRequestSlot::new(
            b"hello world",
            4096,
            1000,
            5,
            request_flags::CASE_SENSITIVE | request_flags::TURBO_MODE,
        );
        assert_eq!(slot.pattern_len, 11);
        assert_eq!(&slot.pattern[..11], b"hello world");
        assert_eq!(slot.pattern[11], 0);
        assert_eq!(slot.total_bytes, 4096);
        assert_eq!(slot.max_matches, 1000);
        assert_eq!(slot.file_count, 5);
        assert_eq!(slot.flags, request_flags::CASE_SENSITIVE | request_flags::TURBO_MODE);
        assert_eq!(slot.status, status::PENDING);
        assert_eq!(slot.sequence_id, 0); // Not assigned until submit
    }

    #[test]
    fn test_search_request_slot_pattern_bytes() {
        let slot = SearchRequestSlot::new(b"fn main", 0, 0, 0, 0);
        assert_eq!(slot.pattern_bytes(), b"fn main");
    }

    #[test]
    fn test_search_request_slot_long_pattern() {
        let long = vec![b'A'; 300];
        let slot = SearchRequestSlot::new(&long, 0, 0, 0, 0);
        assert_eq!(slot.pattern_len, 256);
        assert_eq!(slot.pattern[255], b'A');
    }

    // ================================================================
    // get_latest tests
    // ================================================================

    #[test]
    fn test_work_queue_get_latest_none_when_empty() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert!(wq.get_latest().is_none(), "No completed results in empty queue");
    }

    #[test]
    fn test_work_queue_get_latest_none_when_pending() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let request = SearchRequestSlot::new(b"test", 1024, 100, 1, 0);
        wq.submit_request(&request);
        // Status is PENDING, not COMPLETE
        assert!(wq.get_latest().is_none(), "PENDING requests not returned by get_latest");
    }

    #[test]
    fn test_work_queue_get_latest_returns_completed() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let request = SearchRequestSlot::new(b"search_me", 2048, 500, 10, 0);
        wq.submit_request(&request);

        // Simulate GPU completing the request by writing status
        unsafe {
            let base = wq.buffer().contents().as_ptr() as *mut u8;
            // status is at offset 4 in slot 0
            let status_ptr = base.add(4) as *mut u32;
            std::ptr::write_volatile(status_ptr, status::COMPLETE);
            // Write match_count
            let match_count_ptr = base.add(284) as *mut u32;
            std::ptr::write_volatile(match_count_ptr, 42);
        }

        let result = wq.get_latest();
        assert!(result.is_some(), "Should find completed result");
        let result = result.unwrap();
        assert_eq!(result.sequence_id, 1);
        assert_eq!(result.status, status::COMPLETE);
        assert_eq!(result.match_count, 42);
        assert_eq!(&result.pattern[..9], b"search_me");
    }

    // ================================================================
    // Release/Acquire ordering verification
    // ================================================================

    #[test]
    fn test_work_queue_sequence_id_at_offset_zero() {
        assert_eq!(
            std::mem::offset_of!(SearchRequestSlot, sequence_id),
            0,
            "sequence_id must be at offset 0 for atomic read by GPU"
        );
    }

    #[test]
    fn test_work_queue_release_acquire_fields_visible() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let request = SearchRequestSlot::new(
            b"release_acquire_test",
            999_999,
            50_000,
            77,
            request_flags::CONTENT_SEARCH | request_flags::FILENAME_SEARCH,
        );
        wq.submit_request(&request);

        let slot = wq.read_slot(0);
        // If sequence_id is observed (1), all other fields must be visible
        assert_eq!(slot.sequence_id, 1, "sequence_id observed");
        assert_eq!(slot.pattern_len, 20, "pattern_len visible");
        assert_eq!(&slot.pattern[..20], b"release_acquire_test", "pattern visible");
        assert_eq!(slot.total_bytes, 999_999, "total_bytes visible");
        assert_eq!(slot.max_matches, 50_000, "max_matches visible");
        assert_eq!(slot.file_count, 77, "file_count visible");
        assert_eq!(
            slot.flags,
            request_flags::CONTENT_SEARCH | request_flags::FILENAME_SEARCH,
            "flags visible"
        );
    }

    // ================================================================
    // Extended wrap-around and interleaved tests
    // ================================================================

    #[test]
    fn test_work_queue_100_writes() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        for i in 0..100u32 {
            let request = SearchRequestSlot::new(
                format!("q{}", i).as_bytes(),
                i * 10,
                100,
                i,
                0,
            );
            wq.submit_request(&request);
        }

        assert_eq!(wq.write_idx(), 100 % SLOT_COUNT as u32);
        assert_eq!(wq.next_sequence_id(), 101);
        assert_eq!(wq.read_latest_sequence_id(), 100);
    }

    #[test]
    fn test_work_queue_interleaved_write_read() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        for round in 0..30u32 {
            let request = SearchRequestSlot::new(
                format!("round_{}", round).as_bytes(),
                round * 100 + 42,
                100,
                round,
                0,
            );
            wq.submit_request(&request);

            let last_slot_idx = if wq.write_idx() == 0 {
                SLOT_COUNT - 1
            } else {
                (wq.write_idx() - 1) as usize
            };
            let readback = wq.read_slot(last_slot_idx);
            assert_eq!(readback.total_bytes, round * 100 + 42, "round {} total_bytes", round);
            assert_eq!(readback.file_count, round, "round {} file_count", round);
            assert_eq!(readback.sequence_id, round + 1, "round {} sequence_id", round);
        }
    }

    // ================================================================
    // Stale sequence detection
    // ================================================================

    #[test]
    fn test_work_queue_stale_detection() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let mut gpu_last_seen: u32 = 0;
        let mut queries_detected = 0u32;

        for i in 1..=10u32 {
            let request = SearchRequestSlot::new(b"query", i * 100, 100, i, 0);
            wq.submit_request(&request);

            let current_seq = wq.read_latest_sequence_id();
            if current_seq > gpu_last_seen {
                queries_detected += 1;
                gpu_last_seen = current_seq;
            }
        }

        assert_eq!(queries_detected, 10, "all 10 queries detected as new");
        assert_eq!(gpu_last_seen, 10);
    }

    #[test]
    fn test_work_queue_no_spurious_wakeup() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let request = SearchRequestSlot::default();

        wq.submit_request(&request);
        let seq_after = wq.read_latest_sequence_id();
        assert_eq!(seq_after, 1);

        // Poll without writing -- sequence must stay stable
        for _ in 0..10 {
            assert_eq!(wq.read_latest_sequence_id(), 1, "no spurious change");
        }

        wq.submit_request(&request);
        assert_eq!(wq.read_latest_sequence_id(), 2, "advances only on write");
    }

    // ================================================================
    // Layout / offset assertions
    // ================================================================

    #[test]
    fn test_work_queue_layout() {
        let slot = SearchRequestSlot::default();
        let base = &slot as *const _ as usize;

        assert_eq!(&slot.sequence_id as *const _ as usize - base, 0, "sequence_id at 0");
        assert_eq!(&slot.status as *const _ as usize - base, 4, "status at 4");
        assert_eq!(&slot.flags as *const _ as usize - base, 8, "flags at 8");
        assert_eq!(&slot.pattern_len as *const _ as usize - base, 12, "pattern_len at 12");
        assert_eq!(&slot.pattern as *const _ as usize - base, 16, "pattern at 16");
        assert_eq!(&slot.total_bytes as *const _ as usize - base, 272, "total_bytes at 272");
        assert_eq!(&slot.max_matches as *const _ as usize - base, 276, "max_matches at 276");
        assert_eq!(&slot.file_count as *const _ as usize - base, 280, "file_count at 280");
        assert_eq!(&slot.match_count as *const _ as usize - base, 284, "match_count at 284");
        assert_eq!(&slot.elapsed_us as *const _ as usize - base, 288, "elapsed_us at 288");
        assert_eq!(&slot.error_code as *const _ as usize - base, 292, "error_code at 292");
        assert_eq!(&slot._reserved as *const _ as usize - base, 296, "_reserved at 296");
    }
}
