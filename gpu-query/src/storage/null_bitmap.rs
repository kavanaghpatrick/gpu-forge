//! Null bitmap for tracking NULL values in columnar data.
//!
//! Each bit represents one row: 1 = NULL, 0 = not null.
//! Stored as Vec<u32> (32 rows per word) for efficient GPU transfer
//! (matches the bitmask format used by filter kernels).

/// A bitmap tracking NULL values, one bit per row.
///
/// Bit = 1 means the row's value is NULL.
/// Stored as u32 words for alignment with GPU bitmask buffers.
#[derive(Debug, Clone)]
pub struct NullBitmap {
    /// Packed bits: bit `i` of word `i/32` = null flag for row `i`.
    words: Vec<u32>,
    /// Total number of rows this bitmap covers.
    row_count: usize,
}

impl NullBitmap {
    /// Create a new null bitmap for `row_count` rows, all initially non-null.
    pub fn new(row_count: usize) -> Self {
        let word_count = (row_count + 31) / 32;
        Self {
            words: vec![0u32; word_count],
            row_count,
        }
    }

    /// Mark row `row` as NULL.
    ///
    /// # Panics
    /// Panics if `row >= row_count`.
    pub fn set_null(&mut self, row: usize) {
        assert!(row < self.row_count, "row {} out of bounds ({})", row, self.row_count);
        let word_idx = row / 32;
        let bit_idx = row % 32;
        self.words[word_idx] |= 1u32 << bit_idx;
    }

    /// Clear the null flag for row `row` (mark as non-null).
    ///
    /// # Panics
    /// Panics if `row >= row_count`.
    pub fn clear_null(&mut self, row: usize) {
        assert!(row < self.row_count, "row {} out of bounds ({})", row, self.row_count);
        let word_idx = row / 32;
        let bit_idx = row % 32;
        self.words[word_idx] &= !(1u32 << bit_idx);
    }

    /// Check if row `row` is NULL.
    ///
    /// # Panics
    /// Panics if `row >= row_count`.
    pub fn is_null(&self, row: usize) -> bool {
        assert!(row < self.row_count, "row {} out of bounds ({})", row, self.row_count);
        let word_idx = row / 32;
        let bit_idx = row % 32;
        (self.words[word_idx] >> bit_idx) & 1 == 1
    }

    /// Count total NULL rows in the bitmap.
    pub fn null_count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Total number of rows this bitmap covers.
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Number of u32 words in the backing storage.
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    /// Access the underlying u32 words (for GPU buffer upload).
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    /// Mutable access to the underlying u32 words.
    pub fn as_words_mut(&mut self) -> &mut [u32] {
        &mut self.words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_all_non_null() {
        let bm = NullBitmap::new(100);
        assert_eq!(bm.row_count(), 100);
        assert_eq!(bm.null_count(), 0);
        for i in 0..100 {
            assert!(!bm.is_null(i));
        }
    }

    #[test]
    fn test_set_and_check_null() {
        let mut bm = NullBitmap::new(64);
        bm.set_null(0);
        bm.set_null(31);
        bm.set_null(32);
        bm.set_null(63);

        assert!(bm.is_null(0));
        assert!(bm.is_null(31));
        assert!(bm.is_null(32));
        assert!(bm.is_null(63));
        assert!(!bm.is_null(1));
        assert!(!bm.is_null(30));
        assert!(!bm.is_null(33));
        assert_eq!(bm.null_count(), 4);
    }

    #[test]
    fn test_clear_null() {
        let mut bm = NullBitmap::new(10);
        bm.set_null(5);
        assert!(bm.is_null(5));
        assert_eq!(bm.null_count(), 1);

        bm.clear_null(5);
        assert!(!bm.is_null(5));
        assert_eq!(bm.null_count(), 0);
    }

    #[test]
    fn test_word_count() {
        assert_eq!(NullBitmap::new(0).word_count(), 0);
        assert_eq!(NullBitmap::new(1).word_count(), 1);
        assert_eq!(NullBitmap::new(32).word_count(), 1);
        assert_eq!(NullBitmap::new(33).word_count(), 2);
        assert_eq!(NullBitmap::new(64).word_count(), 2);
        assert_eq!(NullBitmap::new(65).word_count(), 3);
    }

    #[test]
    fn test_null_count_multiple_words() {
        let mut bm = NullBitmap::new(128);
        // Set every 10th row null
        for i in (0..128).step_by(10) {
            bm.set_null(i);
        }
        // 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120 = 13
        assert_eq!(bm.null_count(), 13);
    }

    #[test]
    fn test_as_words() {
        let mut bm = NullBitmap::new(32);
        bm.set_null(0);
        bm.set_null(31);
        let words = bm.as_words();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0], (1u32 << 0) | (1u32 << 31));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_set_null_out_of_bounds() {
        let mut bm = NullBitmap::new(10);
        bm.set_null(10);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_is_null_out_of_bounds() {
        let bm = NullBitmap::new(10);
        bm.is_null(10);
    }

    #[test]
    fn test_empty_bitmap() {
        let bm = NullBitmap::new(0);
        assert_eq!(bm.row_count(), 0);
        assert_eq!(bm.null_count(), 0);
        assert_eq!(bm.word_count(), 0);
        assert!(bm.as_words().is_empty());
    }

    #[test]
    fn test_idempotent_set_null() {
        let mut bm = NullBitmap::new(10);
        bm.set_null(5);
        bm.set_null(5);
        assert_eq!(bm.null_count(), 1);
    }
}
