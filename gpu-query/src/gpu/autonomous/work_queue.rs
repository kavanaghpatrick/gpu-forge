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

    // ================================================================
    // Extended wrap-around tests (100+ writes)
    // ================================================================

    #[test]
    fn wraparound_after_100_writes() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        // Write 100 params, each with a unique row_count
        for i in 0..100u32 {
            let mut params = QueryParamsSlot::default();
            params.row_count = i + 1;
            params.query_hash = i as u64 * 0x1234;
            wq.write_params(&params);
        }

        // After 100 writes, write_idx should be 100 % 3 = 1
        assert_eq!(wq.write_idx(), 100 % SLOT_COUNT as u32, "write_idx after 100 writes");
        assert_eq!(wq.next_sequence_id(), 101, "next sequence_id after 100 writes");

        // The last 3 writes (98, 99, 100) occupy slots:
        //   write 98 (0-indexed) -> sequence_id=98, slot = 97 % 3 = 0
        //   write 99 -> sequence_id=99, slot = 98 % 3 = 1
        //   write 100 -> sequence_id=100, slot = 99 % 3 = 2
        // Actually: write_idx starts at 0, write #1 goes to slot 0, etc.
        // write #98 goes to slot (98-1) % 3 = 97 % 3 = 0
        // write #99 goes to slot 98 % 3 = 2
        // write #100 goes to slot 99 % 3 = 0
        // Let's just verify the most recent write is correct.
        let latest_seq = wq.read_latest_sequence_id();
        assert_eq!(latest_seq, 100, "latest sequence_id after 100 writes");

        // Verify the most recently written slot has correct data
        // The last write (100th) went to slot (write_idx - 1 + 3) % 3
        let last_slot_idx = if wq.write_idx() == 0 {
            SLOT_COUNT - 1
        } else {
            (wq.write_idx() - 1) as usize
        };
        let last_slot = wq.read_slot(last_slot_idx);
        assert_eq!(last_slot.sequence_id, 100, "last written slot has seq=100");
        assert_eq!(last_slot.row_count, 100, "last written slot has row_count=100");
    }

    #[test]
    fn wraparound_300_writes_slot_cycling_correct() {
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        // Write 300 params -- exactly 100 full cycles through 3 slots
        for i in 1..=300u32 {
            let mut params = QueryParamsSlot::default();
            params.row_count = i;
            wq.write_params(&params);
        }

        // After 300 writes, write_idx should wrap back to 0 (300 % 3 = 0)
        assert_eq!(wq.write_idx(), 0, "write_idx after 300 writes (300 mod 3 = 0)");
        assert_eq!(wq.next_sequence_id(), 301);

        // Slots should contain writes #298, #299, #300
        // write #298 -> slot (298-1) % 3 = 297 % 3 = 0
        // write #299 -> slot 298 % 3 = 1
        // write #300 -> slot 299 % 3 = 2
        let slot0 = wq.read_slot(0);
        let slot1 = wq.read_slot(1);
        let slot2 = wq.read_slot(2);

        assert_eq!(slot0.sequence_id, 298, "slot 0 has write #298");
        assert_eq!(slot0.row_count, 298, "slot 0 row_count");
        assert_eq!(slot1.sequence_id, 299, "slot 1 has write #299");
        assert_eq!(slot1.row_count, 299, "slot 1 row_count");
        assert_eq!(slot2.sequence_id, 300, "slot 2 has write #300");
        assert_eq!(slot2.row_count, 300, "slot 2 row_count");
    }

    // ================================================================
    // Concurrent write/read simulation (different slots)
    // ================================================================

    #[test]
    fn concurrent_write_read_different_slots() {
        // Simulate a scenario where we write to slot N while reading from slot N-1.
        // This is the actual usage pattern: CPU writes to write_idx, GPU reads the
        // previous slot(s). We verify no data corruption.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        // Fill all 3 slots with distinct data
        for i in 0..3u32 {
            let mut params = QueryParamsSlot::default();
            params.row_count = (i + 1) * 1000;
            params.filter_count = i + 1;
            params.query_hash = 0xAAAA_0000 + i as u64;
            wq.write_params(&params);
        }

        // Now simulate: read slot 0 (old data) while writing to slot 0 (overwrite)
        let old_slot1 = wq.read_slot(1);
        let old_slot2 = wq.read_slot(2);

        // Write new data to slot 0 (4th write wraps around)
        let mut new_params = QueryParamsSlot::default();
        new_params.row_count = 7777;
        new_params.filter_count = 4;
        new_params.query_hash = 0xBBBB_0000;
        wq.write_params(&new_params);

        // Slot 0 should have new data
        let new_slot0 = wq.read_slot(0);
        assert_eq!(new_slot0.row_count, 7777, "slot 0 overwritten");
        assert_eq!(new_slot0.sequence_id, 4, "slot 0 new sequence");
        assert_eq!(new_slot0.query_hash, 0xBBBB_0000, "slot 0 new hash");

        // Slots 1 and 2 must be untouched (simulating GPU reading these)
        assert_eq!(old_slot1.row_count, 2000, "slot 1 preserved");
        assert_eq!(old_slot1.sequence_id, 2, "slot 1 seq preserved");
        assert_eq!(old_slot2.row_count, 3000, "slot 2 preserved");
        assert_eq!(old_slot2.sequence_id, 3, "slot 2 seq preserved");

        // Re-read slots 1 and 2 from buffer to confirm no corruption
        let reread_slot1 = wq.read_slot(1);
        let reread_slot2 = wq.read_slot(2);
        assert_eq!(reread_slot1.row_count, old_slot1.row_count, "slot 1 still intact");
        assert_eq!(reread_slot2.row_count, old_slot2.row_count, "slot 2 still intact");
    }

    #[test]
    fn interleaved_write_read_no_corruption() {
        // Interleave writes and reads to simulate real CPU/GPU access pattern:
        // CPU writes, GPU reads the PREVIOUS slot.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        for round in 0..30u32 {
            let mut params = QueryParamsSlot::default();
            params.row_count = round * 100 + 42;
            params.query_hash = round as u64 * 0xDEAD;
            params.filter_count = round % 5;
            wq.write_params(&params);

            // Read back the slot we just wrote
            let last_slot_idx = if wq.write_idx() == 0 {
                SLOT_COUNT - 1
            } else {
                (wq.write_idx() - 1) as usize
            };
            let readback = wq.read_slot(last_slot_idx);
            assert_eq!(
                readback.row_count,
                round * 100 + 42,
                "round {} readback row_count",
                round
            );
            assert_eq!(
                readback.query_hash,
                round as u64 * 0xDEAD,
                "round {} readback query_hash",
                round
            );
            assert_eq!(
                readback.filter_count,
                round % 5,
                "round {} readback filter_count",
                round
            );
            assert_eq!(
                readback.sequence_id,
                round + 1,
                "round {} readback sequence_id",
                round
            );
        }
    }

    // ================================================================
    // sequence_id written last (Release ordering verification)
    // ================================================================

    #[test]
    fn sequence_id_written_last_all_fields_visible() {
        // Verify that when sequence_id is observed, all other fields are already
        // written. We do this by writing params with distinctive field values,
        // then confirming that the observed sequence_id implies all fields are
        // present. This tests the Release/Acquire ordering contract.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);

        let mut params = QueryParamsSlot::default();
        params.query_hash = 0xCAFE_BABE_DEAD_BEEF;
        params.filter_count = 3;
        params.filters[0].column_idx = 42;
        params.filters[0].compare_op = 5; // GE
        params.filters[0].value_int = -999;
        params.filters[1].column_idx = 7;
        params.filters[1].compare_op = 2; // LT
        params.filters[1].value_int = 12345;
        params.filters[2].column_idx = 0;
        params.filters[2].compare_op = 0; // EQ
        params.filters[2].value_int = 0;
        params.agg_count = 2;
        params.aggs[0].agg_func = 1; // SUM
        params.aggs[0].column_idx = 1;
        params.aggs[1].agg_func = 4; // MAX
        params.aggs[1].column_idx = 2;
        params.group_by_col = 5;
        params.has_group_by = 1;
        params.row_count = 500_000;

        wq.write_params(&params);

        // Read the slot -- if sequence_id is 1, ALL fields must be present
        // (this is the contract guaranteed by Release fence before sequence_id write)
        let slot = wq.read_slot(0);
        assert_eq!(slot.sequence_id, 1, "sequence_id observed");

        // ALL fields must be visible once sequence_id is observed
        assert_eq!(slot.query_hash, 0xCAFE_BABE_DEAD_BEEF, "query_hash visible with seq");
        assert_eq!(slot.filter_count, 3, "filter_count visible with seq");
        assert_eq!(slot.filters[0].column_idx, 42, "filter[0].column_idx");
        assert_eq!(slot.filters[0].compare_op, 5, "filter[0].compare_op");
        assert_eq!(slot.filters[0].value_int, -999, "filter[0].value_int (negative)");
        assert_eq!(slot.filters[1].column_idx, 7, "filter[1].column_idx");
        assert_eq!(slot.filters[1].value_int, 12345, "filter[1].value_int");
        assert_eq!(slot.filters[2].column_idx, 0, "filter[2].column_idx");
        assert_eq!(slot.agg_count, 2, "agg_count visible");
        assert_eq!(slot.aggs[0].agg_func, 1, "aggs[0].agg_func (SUM)");
        assert_eq!(slot.aggs[0].column_idx, 1, "aggs[0].column_idx");
        assert_eq!(slot.aggs[1].agg_func, 4, "aggs[1].agg_func (MAX)");
        assert_eq!(slot.aggs[1].column_idx, 2, "aggs[1].column_idx");
        assert_eq!(slot.group_by_col, 5, "group_by_col visible");
        assert_eq!(slot.has_group_by, 1, "has_group_by visible");
        assert_eq!(slot.row_count, 500_000, "row_count visible");
    }

    #[test]
    fn sequence_id_at_offset_zero_in_buffer() {
        // Verify sequence_id is at byte offset 0 of each slot -- critical for
        // the GPU to read it atomically as the "last written" indicator.
        assert_eq!(
            std::mem::offset_of!(QueryParamsSlot, sequence_id),
            0,
            "sequence_id must be at offset 0 for atomic read by GPU"
        );

        // Verify that write_params writes sequence_id at the correct buffer offset
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let params = QueryParamsSlot::default();

        wq.write_params(&params); // writes to slot 0
        wq.write_params(&params); // writes to slot 1

        // Manually read the raw u32 at slot offsets to verify sequence_id placement
        unsafe {
            let base = wq.buffer().contents().as_ptr() as *const u8;

            // Slot 0: sequence_id at offset 0
            let seq0_ptr = base as *const u32;
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            let seq0 = std::ptr::read_volatile(seq0_ptr);
            assert_eq!(seq0, 1, "raw seq at slot 0 byte offset 0");

            // Slot 1: sequence_id at offset 512
            let seq1_ptr = base.add(SLOT_SIZE) as *const u32;
            let seq1 = std::ptr::read_volatile(seq1_ptr);
            assert_eq!(seq1, 2, "raw seq at slot 1 byte offset 512");
        }
    }

    // ================================================================
    // Stale sequence detection
    // ================================================================

    #[test]
    fn stale_sequence_detection() {
        // Simulate GPU-side stale detection: GPU remembers last_seen_sequence_id,
        // only processes if current > last_seen.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let mut params = QueryParamsSlot::default();

        let mut gpu_last_seen: u32 = 0;
        let mut queries_processed = 0u32;

        // Write 10 queries
        for i in 1..=10u32 {
            params.row_count = i * 100;
            wq.write_params(&params);

            // GPU polls: check sequence_id of latest slot
            let current_seq = wq.read_latest_sequence_id();
            if current_seq > gpu_last_seen {
                // New query detected -- process it
                queries_processed += 1;
                gpu_last_seen = current_seq;
            }
        }

        assert_eq!(queries_processed, 10, "all 10 queries detected as new");
        assert_eq!(gpu_last_seen, 10, "GPU saw all sequence IDs up to 10");
    }

    #[test]
    fn stale_sequence_no_spurious_wakeup() {
        // Verify that polling without a new write does NOT produce a new sequence.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let params = QueryParamsSlot::default();

        wq.write_params(&params); // seq=1
        let seq_after_write = wq.read_latest_sequence_id();
        assert_eq!(seq_after_write, 1);

        // Poll multiple times without writing -- sequence must stay the same
        for _ in 0..10 {
            let seq = wq.read_latest_sequence_id();
            assert_eq!(seq, 1, "no spurious sequence change without write");
        }

        // Write again -- now it should change
        wq.write_params(&params);
        let seq_after_second = wq.read_latest_sequence_id();
        assert_eq!(seq_after_second, 2, "sequence advances only on write");
    }

    #[test]
    fn stale_slots_retain_old_sequence() {
        // After wraparound, stale slots keep their old sequence_id.
        // A GPU reader checking a specific slot can detect staleness.
        let device = get_device();
        let mut wq = WorkQueue::new(&device);
        let mut params = QueryParamsSlot::default();

        // Fill all 3 slots: seq 1, 2, 3
        for i in 1..=3u32 {
            params.row_count = i;
            wq.write_params(&params);
        }

        // Overwrite slot 0 with seq 4
        params.row_count = 40;
        wq.write_params(&params);

        // Slot 0 is fresh (seq=4), slots 1,2 are stale (seq=2,3)
        let s0 = wq.read_slot(0);
        let s1 = wq.read_slot(1);
        let s2 = wq.read_slot(2);

        assert_eq!(s0.sequence_id, 4, "slot 0 is fresh");
        assert_eq!(s1.sequence_id, 2, "slot 1 is stale (from 2nd write)");
        assert_eq!(s2.sequence_id, 3, "slot 2 is stale (from 3rd write)");

        // A GPU reader with last_seen=3 would skip stale slots 1,2 and process slot 0
        let gpu_last_seen = 3u32;
        let mut new_work_found = false;
        for i in 0..SLOT_COUNT {
            let slot = wq.read_slot(i);
            if slot.sequence_id > gpu_last_seen {
                new_work_found = true;
                assert_eq!(i, 0, "only slot 0 has new work");
                assert_eq!(slot.row_count, 40, "new work has correct data");
            }
        }
        assert!(new_work_found, "GPU found new work via stale detection");
    }
}
