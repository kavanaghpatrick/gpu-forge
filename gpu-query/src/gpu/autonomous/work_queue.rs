//! Triple-buffered work queue for CPU->GPU parameter handoff.
//!
//! The work queue uses three 512-byte `QueryParamsSlot` slots in a single
//! Metal buffer (1536 bytes total, StorageModeShared). The CPU writes to one
//! slot at a time, advancing `write_idx` mod 3 after each write. A monotonically
//! increasing `sequence_id` with Release ordering ensures the GPU sees all fields
//! before it observes the new sequence number.
//!
//! Memory ordering: Acquire-Release on `sequence_id` [KB #154].

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::types::QueryParamsSlot;

/// Size of a single work queue slot in bytes.
pub const SLOT_SIZE: usize = std::mem::size_of::<QueryParamsSlot>();

/// Number of slots in the triple buffer.
pub const SLOT_COUNT: usize = 3;

/// Total size of the work queue buffer in bytes (3 x 512 = 1536).
pub const BUFFER_SIZE: usize = SLOT_SIZE * SLOT_COUNT;

/// Triple-buffered work queue wrapping a Metal buffer for CPU->GPU parameter handoff.
///
/// The CPU writes query parameters to successive slots (0, 1, 2, 0, 1, ...),
/// bumping the `sequence_id` with Release ordering so the GPU can detect new work
/// by reading the sequence_id with Acquire ordering.
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

    /// Write query parameters to the current slot.
    ///
    /// Steps:
    /// 1. Copy all fields EXCEPT `sequence_id` to the current slot
    /// 2. Release fence (ensures GPU sees all fields before the new sequence_id)
    /// 3. Write `sequence_id` last
    /// 4. Advance `write_idx` mod 3
    /// 5. Increment `next_sequence_id`
    pub fn write_params(&mut self, params: &QueryParamsSlot) {
        let seq = self.next_sequence_id;
        let slot_offset = self.write_idx as usize * SLOT_SIZE;

        unsafe {
            let base = self.buffer.contents().as_ptr() as *mut u8;
            let slot_ptr = base.add(slot_offset);

            // Copy the entire QueryParamsSlot first (including caller's sequence_id, which
            // we will overwrite). Using copy_nonoverlapping for the bulk of the struct.
            let src = params as *const QueryParamsSlot as *const u8;

            // Copy everything after sequence_id (offset 4 onward, 508 bytes)
            // sequence_id is at offset 0, 4 bytes
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
    }

    /// Read the latest written sequence ID (CPU-side debug/diagnostic).
    ///
    /// Returns the sequence_id of the most recently written slot.
    /// Uses Acquire ordering to pair with the Release in `write_params`.
    pub fn read_latest_sequence_id(&self) -> u32 {
        // The most recently written slot is at (write_idx - 1 + 3) % 3
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

    /// Read back the QueryParamsSlot at a given slot index (for testing).
    ///
    /// # Panics
    /// Panics if `slot_idx >= SLOT_COUNT`.
    pub fn read_slot(&self, slot_idx: usize) -> QueryParamsSlot {
        assert!(slot_idx < SLOT_COUNT, "slot index out of range");
        let slot_offset = slot_idx * SLOT_SIZE;

        unsafe {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let base = self.buffer.contents().as_ptr() as *const u8;
            let slot_ptr = base.add(slot_offset) as *const QueryParamsSlot;
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
    fn buffer_size_is_1536() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert_eq!(
            wq.buffer().length() as usize,
            BUFFER_SIZE,
            "Work queue buffer must be 3 x 512 = 1536 bytes"
        );
    }

    #[test]
    fn buffer_is_shared_mode() {
        // StorageModeShared means CPU and GPU can both access.
        // We verify by checking that contents() returns a non-null pointer,
        // which is only valid for Shared/Managed modes.
        let device = get_device();
        let wq = WorkQueue::new(&device);
        let ptr = wq.buffer().contents().as_ptr();
        assert!(!ptr.is_null(), "StorageModeShared buffer must have CPU-visible pointer");
    }

    #[test]
    fn slot_size_is_512() {
        assert_eq!(SLOT_SIZE, 512, "QueryParamsSlot must be 512 bytes");
    }

    #[test]
    fn buffer_size_constant() {
        assert_eq!(BUFFER_SIZE, 1536, "Triple buffer = 3 x 512 = 1536");
    }

    // ================================================================
    // Write index cycling tests
    // ================================================================

    #[test]
    fn initial_write_idx_is_zero() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert_eq!(wq.write_idx(), 0, "Initial write_idx must be 0");
    }

    #[test]
    fn write_idx_cycles_0_1_2_0() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let params = QueryParamsSlot::default();

        assert_eq!(wq.write_idx(), 0);
        wq.write_params(&params);
        assert_eq!(wq.write_idx(), 1, "After 1st write: idx=1");
        wq.write_params(&params);
        assert_eq!(wq.write_idx(), 2, "After 2nd write: idx=2");
        wq.write_params(&params);
        assert_eq!(wq.write_idx(), 0, "After 3rd write: idx=0 (wrap)");
        wq.write_params(&params);
        assert_eq!(wq.write_idx(), 1, "After 4th write: idx=1");
    }

    // ================================================================
    // Sequence ID tests
    // ================================================================

    #[test]
    fn initial_sequence_id_is_one() {
        let device = get_device();
        let wq = WorkQueue::new(&device);
        assert_eq!(wq.next_sequence_id(), 1, "First sequence_id must be 1");
    }

    #[test]
    fn sequence_id_monotonically_increases() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let params = QueryParamsSlot::default();

        let mut prev_seq = 0u32;
        for i in 0..10 {
            wq.write_params(&params);
            let seq = wq.read_latest_sequence_id();
            assert!(
                seq > prev_seq,
                "sequence_id must increase: iteration {}, prev={}, curr={}",
                i,
                prev_seq,
                seq
            );
            prev_seq = seq;
        }
    }

    #[test]
    fn sequence_id_values_are_1_through_n() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let params = QueryParamsSlot::default();

        for expected in 1..=6u32 {
            wq.write_params(&params);
            let actual = wq.read_latest_sequence_id();
            assert_eq!(actual, expected, "sequence_id should be {}", expected);
        }
    }

    // ================================================================
    // Slot population correctness tests
    // ================================================================

    #[test]
    fn slot_population_preserves_fields() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let mut params = QueryParamsSlot::default();
        params.query_hash = 0xDEAD_BEEF_CAFE_BABE;
        params.filter_count = 2;
        params.filters[0].column_idx = 1;
        params.filters[0].compare_op = 4; // GT
        params.filters[0].value_int = 500;
        params.agg_count = 1;
        params.aggs[0].agg_func = 0; // COUNT
        params.group_by_col = 3;
        params.has_group_by = 1;
        params.row_count = 1_000_000;

        wq.write_params(&params);

        // Read back slot 0 (the one we just wrote)
        let readback = wq.read_slot(0);

        assert_eq!(readback.sequence_id, 1, "sequence_id should be 1");
        assert_eq!(readback.query_hash, 0xDEAD_BEEF_CAFE_BABE, "query_hash");
        assert_eq!(readback.filter_count, 2, "filter_count");
        assert_eq!(readback.filters[0].column_idx, 1, "filters[0].column_idx");
        assert_eq!(readback.filters[0].compare_op, 4, "filters[0].compare_op");
        assert_eq!(readback.filters[0].value_int, 500, "filters[0].value_int");
        assert_eq!(readback.agg_count, 1, "agg_count");
        assert_eq!(readback.aggs[0].agg_func, 0, "aggs[0].agg_func");
        assert_eq!(readback.group_by_col, 3, "group_by_col");
        assert_eq!(readback.has_group_by, 1, "has_group_by");
        assert_eq!(readback.row_count, 1_000_000, "row_count");
    }

    #[test]
    fn writes_go_to_correct_slots() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        // Write 3 different params to fill all 3 slots
        for i in 0..3u32 {
            let mut params = QueryParamsSlot::default();
            params.row_count = (i + 1) * 1000;
            params.filter_count = i;
            wq.write_params(&params);
        }

        // Verify each slot has the right data
        for i in 0..3usize {
            let slot = wq.read_slot(i);
            assert_eq!(
                slot.sequence_id,
                (i + 1) as u32,
                "slot {} sequence_id",
                i
            );
            assert_eq!(
                slot.row_count,
                ((i as u32) + 1) * 1000,
                "slot {} row_count",
                i
            );
            assert_eq!(
                slot.filter_count, i as u32,
                "slot {} filter_count",
                i
            );
        }
    }

    #[test]
    fn wraparound_overwrites_correct_slot() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let mut params = QueryParamsSlot::default();

        // Write 3 params (fills all slots)
        for i in 0..3u32 {
            params.row_count = (i + 1) * 100;
            wq.write_params(&params);
        }

        // Write a 4th -- should overwrite slot 0
        params.row_count = 9999;
        wq.write_params(&params);

        let slot0 = wq.read_slot(0);
        assert_eq!(slot0.row_count, 9999, "Slot 0 should be overwritten with 4th write");
        assert_eq!(slot0.sequence_id, 4, "Slot 0 should have sequence_id=4 after wraparound");

        // Slots 1 and 2 should still have their old data
        let slot1 = wq.read_slot(1);
        assert_eq!(slot1.row_count, 200, "Slot 1 unchanged");
        assert_eq!(slot1.sequence_id, 2, "Slot 1 sequence_id unchanged");

        let slot2 = wq.read_slot(2);
        assert_eq!(slot2.row_count, 300, "Slot 2 unchanged");
        assert_eq!(slot2.sequence_id, 3, "Slot 2 sequence_id unchanged");
    }

    #[test]
    fn zero_initialized_buffer() {
        let device = get_device();
        let wq = WorkQueue::new(&device);

        // All slots should be zero on creation
        for i in 0..SLOT_COUNT {
            let slot = wq.read_slot(i);
            assert_eq!(slot.sequence_id, 0, "slot {} sequence_id should be 0", i);
            assert_eq!(slot.query_hash, 0, "slot {} query_hash should be 0", i);
            assert_eq!(slot.filter_count, 0, "slot {} filter_count should be 0", i);
            assert_eq!(slot.agg_count, 0, "slot {} agg_count should be 0", i);
            assert_eq!(slot.row_count, 0, "slot {} row_count should be 0", i);
        }
    }
}
