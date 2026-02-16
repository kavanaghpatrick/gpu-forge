// Shared repr(C) types matching search_types.h for CPU<->GPU data exchange.
//
// IMPORTANT: These structs MUST match the Metal shader definitions in
// shaders/search_types.h exactly. Any layout mismatch will cause silent
// data corruption on the GPU side.

/// Search parameters passed from CPU to GPU.
/// Matches `SearchParams` in search_types.h.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SearchParams {
    /// Search pattern bytes (null-padded)
    pub pattern: [u8; 256],
    /// Length of the pattern
    pub pattern_len: u32,
    /// Total bytes in the input buffer
    pub total_bytes: u32,
    /// 1 = case sensitive, 0 = case insensitive
    pub case_sensitive: u32,
    /// Maximum number of matches to return
    pub max_matches: u32,
    /// Number of files in the batch
    pub file_count: u32,
    /// Padding for alignment
    pub _reserved: [u32; 2],
}

impl SearchParams {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create SearchParams from a pattern string.
    pub fn new(pattern: &[u8], total_bytes: u32, case_sensitive: bool, max_matches: u32) -> Self {
        let mut params = Self {
            pattern: [0u8; 256],
            pattern_len: pattern.len().min(256) as u32,
            total_bytes,
            case_sensitive: if case_sensitive { 1 } else { 0 },
            max_matches,
            file_count: 0,
            _reserved: [0; 2],
        };
        let copy_len = pattern.len().min(256);
        params.pattern[..copy_len].copy_from_slice(&pattern[..copy_len]);
        params
    }
}

/// A single match result written by the GPU.
/// Matches `GpuMatchResult` in search_types.h.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuMatchResult {
    /// Index of the file containing the match
    pub file_index: u32,
    /// Byte offset of the match in the file
    pub byte_offset: u32,
    /// Line number (computed post-search or in kernel)
    pub line_number: u32,
    /// Column offset within the line
    pub column: u32,
    /// Length of the matched region
    pub match_length: u32,
    /// Start offset of surrounding context
    pub context_start: u32,
    /// Length of surrounding context
    pub context_len: u32,
    /// Padding for alignment
    pub _reserved: u32,
}

impl GpuMatchResult {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Maximum path length in GpuPathEntry (bytes).
pub const GPU_PATH_MAX_LEN: usize = 224;

/// A filesystem path entry for the GPU-resident index.
/// MUST be exactly 256 bytes. Matches `GpuPathEntry` in search_types.h.
///
/// Note: size is split into size_lo/size_hi (two u32s) to match the Metal
/// shader layout which uses `uint` (32-bit) types. Reconstruct full size
/// with `((size_hi as u64) << 32) | (size_lo as u64)`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuPathEntry {
    /// UTF-8 encoded path bytes (null-padded)
    pub path: [u8; GPU_PATH_MAX_LEN],
    /// Actual length of the path
    pub path_len: u32,
    /// Bitflags: is_dir, is_hidden, is_symlink, etc.
    pub flags: u32,
    /// Index of parent directory in the index
    pub parent_idx: u32,
    /// File size (low 32 bits)
    pub size_lo: u32,
    /// File size (high 32 bits)
    pub size_hi: u32,
    /// Last modification time (unix timestamp, truncated to u32)
    pub mtime: u32,
    /// Padding to reach 256 bytes total
    pub _reserved: [u32; 2],
}

impl GpuPathEntry {
    pub const SIZE: usize = 256;

    /// Create a new empty path entry.
    pub fn new() -> Self {
        Self {
            path: [0u8; GPU_PATH_MAX_LEN],
            path_len: 0,
            flags: 0,
            parent_idx: u32::MAX,
            size_lo: 0,
            size_hi: 0,
            mtime: 0,
            _reserved: [0; 2],
        }
    }

    /// Set the path from a byte slice.
    pub fn set_path(&mut self, p: &[u8]) {
        let copy_len = p.len().min(GPU_PATH_MAX_LEN);
        self.path[..copy_len].copy_from_slice(&p[..copy_len]);
        if copy_len < GPU_PATH_MAX_LEN {
            self.path[copy_len..].fill(0);
        }
        self.path_len = copy_len as u32;
    }

    /// Set the file size (split into lo/hi u32 pair).
    pub fn set_size(&mut self, size: u64) {
        self.size_lo = size as u32;
        self.size_hi = (size >> 32) as u32;
    }

    /// Get the file size (reconstructed from lo/hi).
    pub fn size(&self) -> u64 {
        ((self.size_hi as u64) << 32) | (self.size_lo as u64)
    }
}

impl Default for GpuPathEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// GpuPathEntry flag constants.
pub mod path_flags {
    pub const IS_DIR: u32 = 1 << 0;
    pub const IS_HIDDEN: u32 = 1 << 1;
    pub const IS_SYMLINK: u32 = 1 << 2;
    pub const IS_EXECUTABLE: u32 = 1 << 3;
    pub const IS_DELETED: u32 = 1 << 4;
}

// ============================================================================
// Compile-time layout assertions
// ============================================================================

// GpuPathEntry MUST be exactly 256 bytes to match Metal shader expectations.
const _: () = assert!(std::mem::size_of::<GpuPathEntry>() == 256);

// GpuMatchResult should be 32 bytes (8 x u32).
const _: () = assert!(std::mem::size_of::<GpuMatchResult>() == 32);

// SearchParams: 256 (pattern) + 5*4 (fields) + 2*4 (reserved) = 284 bytes.
const _: () = assert!(std::mem::size_of::<SearchParams>() == 284);

// ============================================================================
// Runtime tests with offset verification
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_types_layout() {
        // --- GpuPathEntry: exactly 256 bytes ---
        assert_eq!(mem::size_of::<GpuPathEntry>(), 256, "GpuPathEntry must be 256 bytes");
        assert_eq!(mem::align_of::<GpuPathEntry>(), 4, "GpuPathEntry alignment must be 4");

        // Field offset checks (cumulative byte offsets matching search_types.h)
        let entry = GpuPathEntry::new();
        let base = &entry as *const _ as usize;

        let path_offset = &entry.path as *const _ as usize - base;
        let path_len_offset = &entry.path_len as *const _ as usize - base;
        let flags_offset = &entry.flags as *const _ as usize - base;
        let parent_idx_offset = &entry.parent_idx as *const _ as usize - base;
        let size_lo_offset = &entry.size_lo as *const _ as usize - base;
        let size_hi_offset = &entry.size_hi as *const _ as usize - base;
        let mtime_offset = &entry.mtime as *const _ as usize - base;
        let reserved_offset = &entry._reserved as *const _ as usize - base;

        assert_eq!(path_offset, 0, "path at offset 0");
        assert_eq!(path_len_offset, 224, "path_len at offset 224");
        assert_eq!(flags_offset, 228, "flags at offset 228");
        assert_eq!(parent_idx_offset, 232, "parent_idx at offset 232");
        assert_eq!(size_lo_offset, 236, "size_lo at offset 236");
        assert_eq!(size_hi_offset, 240, "size_hi at offset 240");
        assert_eq!(mtime_offset, 244, "mtime at offset 244");
        assert_eq!(reserved_offset, 248, "reserved at offset 248");

        // --- GpuMatchResult: 32 bytes ---
        assert_eq!(mem::size_of::<GpuMatchResult>(), 32, "GpuMatchResult must be 32 bytes");

        let result = GpuMatchResult::default();
        let base = &result as *const _ as usize;

        assert_eq!(&result.file_index as *const _ as usize - base, 0);
        assert_eq!(&result.byte_offset as *const _ as usize - base, 4);
        assert_eq!(&result.line_number as *const _ as usize - base, 8);
        assert_eq!(&result.column as *const _ as usize - base, 12);
        assert_eq!(&result.match_length as *const _ as usize - base, 16);
        assert_eq!(&result.context_start as *const _ as usize - base, 20);
        assert_eq!(&result.context_len as *const _ as usize - base, 24);
        assert_eq!(&result._reserved as *const _ as usize - base, 28);

        // --- SearchParams: 284 bytes ---
        assert_eq!(mem::size_of::<SearchParams>(), 284, "SearchParams must be 284 bytes");

        let params = SearchParams::new(b"test", 1024, true, 10000);
        let base = &params as *const _ as usize;

        assert_eq!(&params.pattern as *const _ as usize - base, 0, "pattern at offset 0");
        assert_eq!(&params.pattern_len as *const _ as usize - base, 256, "pattern_len at offset 256");
        assert_eq!(&params.total_bytes as *const _ as usize - base, 260, "total_bytes at offset 260");
        assert_eq!(&params.case_sensitive as *const _ as usize - base, 264, "case_sensitive at offset 264");
        assert_eq!(&params.max_matches as *const _ as usize - base, 268, "max_matches at offset 268");
        assert_eq!(&params.file_count as *const _ as usize - base, 272, "file_count at offset 272");
        assert_eq!(&params._reserved as *const _ as usize - base, 276, "reserved at offset 276");
    }

    #[test]
    fn test_search_params_new() {
        let params = SearchParams::new(b"hello", 4096, false, 1000);
        assert_eq!(&params.pattern[..5], b"hello");
        assert_eq!(params.pattern[5], 0); // null-padded
        assert_eq!(params.pattern_len, 5);
        assert_eq!(params.total_bytes, 4096);
        assert_eq!(params.case_sensitive, 0);
        assert_eq!(params.max_matches, 1000);
    }

    #[test]
    fn test_search_params_long_pattern() {
        // Pattern longer than 256 bytes should be truncated
        let long = vec![b'x'; 300];
        let params = SearchParams::new(&long, 0, true, 0);
        assert_eq!(params.pattern_len, 256);
        assert_eq!(params.pattern[255], b'x');
    }

    #[test]
    fn test_gpu_path_entry_helpers() {
        let mut entry = GpuPathEntry::new();
        entry.set_path(b"/usr/local/bin/test");
        assert_eq!(entry.path_len, 19);
        assert_eq!(&entry.path[..19], b"/usr/local/bin/test");
        assert_eq!(entry.path[19], 0);

        entry.set_size(0x1_FFFF_FFFF); // > 4GB
        assert_eq!(entry.size_lo, 0xFFFF_FFFF);
        assert_eq!(entry.size_hi, 1);
        assert_eq!(entry.size(), 0x1_FFFF_FFFF);
    }

    #[test]
    fn test_gpu_path_entry_flags() {
        let mut entry = GpuPathEntry::new();
        entry.flags = path_flags::IS_DIR | path_flags::IS_HIDDEN;
        assert_ne!(entry.flags & path_flags::IS_DIR, 0);
        assert_ne!(entry.flags & path_flags::IS_HIDDEN, 0);
        assert_eq!(entry.flags & path_flags::IS_SYMLINK, 0);
    }

    #[test]
    fn test_gpu_match_result_default() {
        let result = GpuMatchResult::default();
        assert_eq!(result.file_index, 0);
        assert_eq!(result.byte_offset, 0);
        assert_eq!(result.line_number, 0);
        assert_eq!(result.column, 0);
        assert_eq!(result.match_length, 0);
        assert_eq!(result.context_start, 0);
        assert_eq!(result.context_len, 0);
    }
}
