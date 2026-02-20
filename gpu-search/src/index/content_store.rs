// Content store: in-memory file content index for zero-disk-I/O search
//
// FileContentMeta is a 32-byte #[repr(C)] struct describing one file's
// location within the contiguous content buffer.  ContentStore holds
// the buffer + metadata table and supports insert / lookup.
//
// ContentStoreBuilder allocates page-aligned anonymous mmap for the
// content buffer.  On finalize, the mmap is made read-only and wrapped
// in a Metal buffer via bytesNoCopy for zero-copy GPU access.

use std::path::PathBuf;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::io::mmap::{align_to_page, PAGE_SIZE};
use crate::search::content::ChunkMetadata;

/// Metadata for a single file stored in the content buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FileContentMeta {
    /// Byte offset into ContentStore::buffer where this file's content begins.
    pub content_offset: u64,
    /// Length of the file content in bytes.
    pub content_len: u32,
    /// Identifier linking to the path index (external).
    pub path_id: u32,
    /// CRC32 hash of the file content.
    pub content_hash: u32,
    /// Last-modified time (seconds since epoch, truncated to u32).
    pub mtime: u32,
    /// Number of trigrams in the file (reserved for Phase 2).
    pub trigram_count: u32,
    /// Bit flags (reserved: bit 0 = deleted, others TBD).
    pub flags: u32,
}

// Static assert: FileContentMeta must be exactly 32 bytes.
const _: () = assert!(
    std::mem::size_of::<FileContentMeta>() == 32,
    "FileContentMeta must be exactly 32 bytes"
);

/// Backing storage for content data -- either a Vec (simple/test path),
/// an anonymous mmap region (production path from ContentStoreBuilder),
/// or a file-backed mmap (GCIX load path).
enum ContentBacking {
    /// Heap-allocated Vec (used by ContentStore::new/with_capacity/insert).
    Vec(Vec<u8>),
    /// Anonymous mmap region with page-aligned pointer and length.
    /// The mmap is read-only after finalization.
    Mmap {
        ptr: *mut std::ffi::c_void,
        /// Page-aligned mapped length (>= data_len).
        mapped_len: usize,
        /// Actual data bytes written (may be < mapped_len).
        data_len: usize,
    },
    /// File-backed mmap (GCIX load path). The MmapBuffer owns the entire
    /// file mapping; content_offset/content_len identify the sub-region
    /// within it that holds actual content data.
    FileMmap {
        /// Owns the file mmap -- keeps it alive for the lifetime of ContentStore.
        mmap: crate::io::mmap::MmapBuffer,
        /// Byte offset within the mmap where content data begins.
        content_offset: usize,
        /// Length of the content data region.
        content_len: usize,
    },
}

// SAFETY: The mmap region is read-only after finalization and the pointer
// is valid for the lifetime of the ContentStore. Vec is inherently Send+Sync.
unsafe impl Send for ContentBacking {}
unsafe impl Sync for ContentBacking {}

impl ContentBacking {
    fn as_slice(&self) -> &[u8] {
        match self {
            ContentBacking::Vec(v) => v.as_slice(),
            ContentBacking::Mmap { ptr, data_len, .. } => {
                if *data_len == 0 {
                    return &[];
                }
                // SAFETY: ptr is valid for mapped_len bytes (>= data_len),
                // region is read-only, valid for lifetime of self.
                unsafe { std::slice::from_raw_parts(*ptr as *const u8, *data_len) }
            }
            ContentBacking::FileMmap {
                mmap,
                content_offset,
                content_len,
            } => {
                if *content_len == 0 {
                    return &[];
                }
                // Return the content sub-region of the file mmap.
                &mmap.as_slice()[*content_offset..*content_offset + *content_len]
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            ContentBacking::Vec(v) => v.len(),
            ContentBacking::Mmap { data_len, .. } => *data_len,
            ContentBacking::FileMmap { content_len, .. } => *content_len,
        }
    }

    /// Get the raw pointer to the start of the data (for Metal buffer creation).
    #[allow(dead_code)]
    fn as_ptr(&self) -> *const u8 {
        match self {
            ContentBacking::Vec(v) => v.as_ptr(),
            ContentBacking::Mmap { ptr, .. } => *ptr as *const u8,
            ContentBacking::FileMmap {
                mmap,
                content_offset,
                ..
            } => {
                // SAFETY: mmap.as_ptr() is valid for mmap.file_size() bytes,
                // content_offset is within bounds (validated at construction).
                unsafe { mmap.as_ptr().add(*content_offset) }
            }
        }
    }

    /// Get the page-aligned mapped length (for mmap), or data len (for Vec).
    #[allow(dead_code)]
    fn mapped_len(&self) -> usize {
        match self {
            ContentBacking::Vec(v) => v.len(),
            ContentBacking::Mmap { mapped_len, .. } => *mapped_len,
            ContentBacking::FileMmap {
                mmap,
                content_offset,
                ..
            } => {
                // The mapped region from content_offset to end of mmap.
                // This is page-aligned because content_offset is page-aligned
                // and the mmap itself is page-aligned.
                mmap.mapped_len() - *content_offset
            }
        }
    }
}

impl Drop for ContentBacking {
    fn drop(&mut self) {
        if let ContentBacking::Mmap { ptr, mapped_len, .. } = self {
            if *mapped_len > 0 {
                // SAFETY: ptr was returned by a successful mmap() call with
                // mapped_len bytes. We only call munmap once (in Drop).
                unsafe {
                    libc::munmap(*ptr, *mapped_len);
                }
            }
        }
    }
}

/// In-memory content store holding all indexed file contents in a
/// contiguous byte buffer with per-file metadata.
///
/// # Drop Order
///
/// `metal_buffer` is declared before `backing`, so the Metal buffer is
/// released before the backing memory is freed. This guarantees the GPU
/// buffer never references freed memory.
pub struct ContentStore {
    // -- Metal buffer MUST be dropped before backing --
    /// Optional Metal buffer wrapping the content data (zero-copy via bytesNoCopy).
    metal_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,

    // -- backing keeps the memory alive --
    /// Contiguous content data (all files concatenated).
    backing: ContentBacking,
    /// Per-file metadata table (index = file_id).
    files: Vec<FileContentMeta>,
    /// Path table: file_id -> original file path (for match resolution).
    paths: Vec<PathBuf>,
    /// Running total of content bytes stored.
    total_bytes: u64,
    /// Number of files stored.
    file_count: u32,
    /// Bytes wasted by updates (old content that is no longer referenced).
    /// Tracked for future compaction support.
    dead_bytes: u64,
    /// Pre-built chunk metadata for GPU dispatch (cached from index build).
    chunk_metadata: Option<Vec<ChunkMetadata>>,
}

// SAFETY: ContentBacking is Send+Sync. Retained<MTLBuffer> is thread-safe
// for read access on Apple platforms. Vec<FileContentMeta> and scalars are
// inherently Send+Sync.
unsafe impl Send for ContentStore {}
unsafe impl Sync for ContentStore {}

impl ContentStore {
    /// Create an empty content store (Vec-backed, no Metal buffer).
    pub fn new() -> Self {
        Self {
            metal_buffer: None,
            backing: ContentBacking::Vec(Vec::new()),
            files: Vec::new(),
            paths: Vec::new(),
            total_bytes: 0,
            file_count: 0,
            dead_bytes: 0,
            chunk_metadata: None,
        }
    }

    /// Create a content store with pre-allocated capacity (Vec-backed).
    pub fn with_capacity(estimated_bytes: usize) -> Self {
        Self {
            metal_buffer: None,
            backing: ContentBacking::Vec(Vec::with_capacity(estimated_bytes)),
            files: Vec::new(),
            paths: Vec::new(),
            total_bytes: 0,
            file_count: 0,
            dead_bytes: 0,
            chunk_metadata: None,
        }
    }

    /// Insert file content into the store (Vec-backed path only).
    ///
    /// Returns the file_id (index into the files table) which can be
    /// used with `content_for()` to retrieve the content later.
    ///
    /// # Panics
    /// Panics if the store is mmap-backed (finalized from ContentStoreBuilder).
    /// Use ContentStoreBuilder for mmap-backed stores.
    pub fn insert(
        &mut self,
        content: &[u8],
        path_id: u32,
        content_hash: u32,
        mtime: u32,
    ) -> u32 {
        let vec = match &mut self.backing {
            ContentBacking::Vec(v) => v,
            ContentBacking::Mmap { .. } | ContentBacking::FileMmap { .. } => {
                panic!("Cannot insert into mmap-backed ContentStore; use ContentStoreBuilder")
            }
        };

        let file_id = self.file_count;
        let offset = vec.len() as u64;

        let meta = FileContentMeta {
            content_offset: offset,
            content_len: content.len() as u32,
            path_id,
            content_hash,
            mtime,
            trigram_count: 0,
            flags: 0,
        };

        vec.extend_from_slice(content);
        self.files.push(meta);
        self.total_bytes += content.len() as u64;
        self.file_count += 1;

        file_id
    }

    /// Retrieve file content by file_id.
    ///
    /// Returns `None` if the file_id is out of bounds, the entry is marked
    /// as deleted (flag bit 0), or the content range is invalid.
    pub fn content_for(&self, file_id: u32) -> Option<&[u8]> {
        let meta = self.files.get(file_id as usize)?;
        // Skip deleted entries (bit 0 = deleted flag)
        if meta.flags & 1 != 0 {
            return None;
        }
        let start = meta.content_offset as usize;
        let end = start + meta.content_len as usize;
        if end > self.backing.len() {
            return None;
        }
        let buf = self.backing.as_slice();
        Some(&buf[start..end])
    }

    /// Total bytes of content stored.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Number of files stored.
    pub fn file_count(&self) -> u32 {
        self.file_count
    }

    /// Access the files metadata table.
    pub fn files(&self) -> &[FileContentMeta] {
        &self.files
    }

    /// Access the raw content buffer.
    pub fn buffer(&self) -> &[u8] {
        self.backing.as_slice()
    }

    /// Get a reference to the Metal buffer, if available.
    #[inline]
    pub fn metal_buffer(&self) -> Option<&ProtocolObject<dyn MTLBuffer>> {
        self.metal_buffer.as_deref()
    }

    /// Check whether this store has a GPU-accessible Metal buffer.
    #[inline]
    pub fn has_metal_buffer(&self) -> bool {
        self.metal_buffer.is_some()
    }

    /// Set the path table for file_id -> path resolution.
    ///
    /// Called after building the content store to associate file IDs with
    /// their original file paths. The paths Vec must have the same length
    /// as the files table (one path per file_id).
    pub fn set_paths(&mut self, paths: Vec<PathBuf>) {
        self.paths = paths;
    }

    /// Resolve a file_id to its original file path.
    ///
    /// Returns `None` if no path table has been set or the file_id is
    /// out of bounds.
    pub fn path_for(&self, file_id: u32) -> Option<&PathBuf> {
        self.paths.get(file_id as usize)
    }

    /// Access the path table.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Dead bytes: wasted space from update_file operations (old content
    /// that is no longer referenced). Useful for deciding when to compact.
    pub fn dead_bytes(&self) -> u64 {
        self.dead_bytes
    }

    /// Access the pre-built chunk metadata, if available.
    pub fn chunk_metadata(&self) -> Option<&[ChunkMetadata]> {
        self.chunk_metadata.as_deref()
    }

    /// Set pre-built chunk metadata for GPU dispatch caching.
    pub fn set_chunk_metadata(&mut self, chunks: Vec<ChunkMetadata>) {
        self.chunk_metadata = Some(chunks);
    }

    // ---------------------------------------------------------------
    // Incremental update methods (Vec-backed stores only)
    // ---------------------------------------------------------------

    /// Update an existing file's content (append-only strategy).
    ///
    /// Appends `new_content` to the end of the buffer and updates the
    /// file's metadata entry to point at the new location. The old content
    /// becomes dead space tracked by `dead_bytes`.
    ///
    /// # Panics
    /// Panics if the store is mmap-backed or `file_id` is out of bounds.
    pub fn update_file(
        &mut self,
        file_id: u32,
        new_content: &[u8],
        new_hash: u32,
        new_mtime: u32,
    ) {
        let vec = match &mut self.backing {
            ContentBacking::Vec(v) => v,
            ContentBacking::Mmap { .. } | ContentBacking::FileMmap { .. } => {
                panic!(
                    "Cannot mutate mmap-backed ContentStore; \
                     create a new ContentStoreBuilder for mmap-backed stores"
                )
            }
        };

        // Append new content at the end of the buffer
        let new_offset = vec.len() as u64;
        vec.extend_from_slice(new_content);

        let meta = self
            .files
            .get_mut(file_id as usize)
            .unwrap_or_else(|| panic!("update_file: file_id {} out of bounds", file_id));

        // Track dead bytes from the old content
        self.dead_bytes += meta.content_len as u64;

        // Update metadata to point at new content
        meta.content_offset = new_offset;
        meta.content_len = new_content.len() as u32;
        meta.content_hash = new_hash;
        meta.mtime = new_mtime;

        // Update total_bytes: add new, but don't subtract old (old is in dead_bytes)
        self.total_bytes += new_content.len() as u64;
    }

    /// Mark a file as deleted (sets flag bit 0).
    ///
    /// The file's content becomes dead space. `content_for()` will return
    /// `None` for deleted entries. The file_id slot is NOT reused.
    ///
    /// # Panics
    /// Panics if the store is mmap-backed or `file_id` is out of bounds.
    pub fn remove_file(&mut self, file_id: u32) {
        // Verify Vec-backed (panics on mmap)
        match &self.backing {
            ContentBacking::Vec(_) => {}
            ContentBacking::Mmap { .. } | ContentBacking::FileMmap { .. } => {
                panic!(
                    "Cannot mutate mmap-backed ContentStore; \
                     create a new ContentStoreBuilder for mmap-backed stores"
                )
            }
        }

        let meta = self
            .files
            .get_mut(file_id as usize)
            .unwrap_or_else(|| panic!("remove_file: file_id {} out of bounds", file_id));

        // Only mark if not already deleted
        if meta.flags & 1 == 0 {
            meta.flags |= 1; // set deleted bit
            self.dead_bytes += meta.content_len as u64;
        }
    }

    /// Add a new file to the store (Vec-backed only).
    ///
    /// Functionally identical to `insert()` but explicitly named for
    /// FSEvents use. Returns the new file_id.
    ///
    /// # Panics
    /// Panics if the store is mmap-backed.
    pub fn add_file(
        &mut self,
        content: &[u8],
        path_id: u32,
        hash: u32,
        mtime: u32,
    ) -> u32 {
        self.insert(content, path_id, hash, mtime)
    }

    /// Add a new file to the store with an associated path (Vec-backed only).
    ///
    /// Like `add_file()` but also records the file path for match resolution.
    /// Returns the new file_id.
    ///
    /// # Panics
    /// Panics if the store is mmap-backed.
    pub fn add_file_with_path(
        &mut self,
        content: &[u8],
        path: PathBuf,
        path_id: u32,
        hash: u32,
        mtime: u32,
    ) -> u32 {
        let file_id = self.insert(content, path_id, hash, mtime);
        self.paths.push(path);
        file_id
    }

    /// Look up a file_id by path (linear scan).
    ///
    /// Returns `None` if the path is not in the store or no path table is set.
    pub fn find_file_by_path(&self, path: &std::path::Path) -> Option<u32> {
        self.paths
            .iter()
            .position(|p| p == path)
            .map(|i| i as u32)
    }

    /// Construct a ContentStore from a file-backed mmap (GCIX load path).
    ///
    /// The MmapBuffer owns the entire GCIX file mapping. The `content_offset`
    /// and `content_len` identify the content data sub-region within it.
    /// The `files` metadata table is pre-parsed from the mmap.
    ///
    /// If `device` is provided, creates a Metal buffer via bytesNoCopy from the
    /// content region (which must be page-aligned within the mmap).
    pub fn from_gcix_mmap(
        mmap: crate::io::mmap::MmapBuffer,
        files: Vec<FileContentMeta>,
        content_offset: usize,
        content_len: usize,
        device: Option<&ProtocolObject<dyn MTLDevice>>,
        paths: Vec<PathBuf>,
    ) -> Self {
        let file_count = files.len() as u32;
        let total_bytes = content_len as u64;

        // Create Metal buffer from the content region via bytesNoCopy if device provided.
        // content_offset is page-aligned (guaranteed by GCIX format), and the mmap
        // pointer is page-aligned (guaranteed by OS), so content_ptr is page-aligned.
        let metal_buffer = if let Some(device) = device {
            if content_len > 0 {
                let content_ptr = unsafe { mmap.as_ptr().add(content_offset) };
                // The mapped length from content_offset to end of mmap (page-aligned).
                let available_len = mmap.mapped_len() - content_offset;

                let options = MTLResourceOptions::StorageModeShared;

                // Verify page alignment for bytesNoCopy
                let ptr_aligned = (content_ptr as usize).is_multiple_of(PAGE_SIZE);
                let len_aligned = available_len.is_multiple_of(PAGE_SIZE);

                if ptr_aligned && len_aligned {
                    // SAFETY: content_ptr is page-aligned, available_len is page-aligned,
                    // the mmap region is valid and read-only, no deallocator because
                    // ContentBacking::FileMmap owns the MmapBuffer.
                    let buf = unsafe {
                        let nn = NonNull::new(content_ptr as *mut u8)
                            .expect("content pointer should be non-null");
                        let nn_void = nn.cast::<std::ffi::c_void>();
                        device.newBufferWithBytesNoCopy_length_options_deallocator(
                            nn_void,
                            available_len,
                            options,
                            None,
                        )
                    };
                    if buf.is_some() {
                        buf
                    } else {
                        // Fallback: copy content data into a new Metal buffer
                        unsafe {
                            let nn = NonNull::new(content_ptr as *mut u8)
                                .expect("content pointer should be non-null");
                            let nn_void = nn.cast::<std::ffi::c_void>();
                            device.newBufferWithBytes_length_options(
                                nn_void,
                                content_len,
                                options,
                            )
                        }
                    }
                } else {
                    // Not page-aligned: copy fallback
                    unsafe {
                        let nn = NonNull::new(content_ptr as *mut u8)
                            .expect("content pointer should be non-null");
                        let nn_void = nn.cast::<std::ffi::c_void>();
                        device.newBufferWithBytes_length_options(
                            nn_void,
                            content_len,
                            options,
                        )
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        Self {
            metal_buffer,
            backing: ContentBacking::FileMmap {
                mmap,
                content_offset,
                content_len,
            },
            files,
            paths, // Loaded from GCIX v2 path table
            total_bytes,
            file_count,
            dead_bytes: 0,
            chunk_metadata: None,
        }
    }
}

impl Default for ContentStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ContentStoreBuilder: mmap-backed builder with Metal buffer finalization
// ---------------------------------------------------------------------------

/// Builder for constructing a ContentStore backed by anonymous mmap.
///
/// Allocates a page-aligned anonymous mmap region, appends file contents
/// sequentially, then finalizes to a read-only ContentStore with an
/// optional Metal buffer via `bytesNoCopy`.
///
/// # Example
/// ```ignore
/// let mut builder = ContentStoreBuilder::new(1024 * 1024)?; // 1MB capacity
/// builder.append(b"file content", 0, 0xABCD, 12345);
/// builder.append(b"more content", 1, 0xEF01, 12346);
/// let store = builder.finalize(&device);
/// assert!(store.has_metal_buffer());
/// ```
pub struct ContentStoreBuilder {
    /// Pointer to the anonymous mmap region (PROT_READ|PROT_WRITE).
    ptr: *mut std::ffi::c_void,
    /// Page-aligned capacity of the mmap region.
    capacity: usize,
    /// Write cursor: next byte offset to write at.
    cursor: usize,
    /// Per-file metadata accumulated during build.
    files: Vec<FileContentMeta>,
    /// Path table: file_id -> original file path.
    paths: Vec<PathBuf>,
    /// Running total of content bytes appended.
    total_bytes: u64,
    /// Number of files appended.
    file_count: u32,
}

// SAFETY: The mmap region is private (MAP_PRIVATE) and only accessed by
// the owning thread during build. After finalize, ownership transfers
// to ContentStore which is Send+Sync.
unsafe impl Send for ContentStoreBuilder {}

impl ContentStoreBuilder {
    /// Allocate a new builder with the given capacity.
    ///
    /// The capacity is rounded up to the next page boundary (16KB on
    /// Apple Silicon). The actual mmap region may be larger than requested.
    ///
    /// # Errors
    /// Returns `io::Error` if `mmap` fails (e.g., out of virtual address space).
    pub fn new(capacity: usize) -> std::io::Result<Self> {
        // Must allocate at least one page
        let capacity = if capacity == 0 {
            PAGE_SIZE
        } else {
            align_to_page(capacity)
        };

        // SAFETY: MAP_ANON | MAP_PRIVATE allocates anonymous memory with no
        // file descriptor. The kernel returns a page-aligned pointer.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        // Verify page alignment (should always be true for mmap, but be safe)
        debug_assert!(
            (ptr as usize).is_multiple_of(PAGE_SIZE),
            "mmap returned non-page-aligned pointer: {:p}",
            ptr
        );

        Ok(Self {
            ptr,
            capacity,
            cursor: 0,
            files: Vec::new(),
            paths: Vec::new(),
            total_bytes: 0,
            file_count: 0,
        })
    }

    /// Append file content to the mmap region.
    ///
    /// Returns the file_id (index into the metadata table).
    ///
    /// # Panics
    /// Panics if the content would exceed the allocated capacity.
    /// Callers should check `remaining_capacity()` before appending.
    pub fn append(
        &mut self,
        content: &[u8],
        path_id: u32,
        content_hash: u32,
        mtime: u32,
    ) -> u32 {
        let new_end = self.cursor + content.len();
        assert!(
            new_end <= self.capacity,
            "ContentStoreBuilder: append would exceed capacity ({} + {} > {})",
            self.cursor,
            content.len(),
            self.capacity
        );

        let file_id = self.file_count;
        let offset = self.cursor as u64;

        let meta = FileContentMeta {
            content_offset: offset,
            content_len: content.len() as u32,
            path_id,
            content_hash,
            mtime,
            trigram_count: 0,
            flags: 0,
        };

        // Copy content into the mmap region at the cursor position
        if !content.is_empty() {
            // SAFETY: ptr + cursor is within the mmap region (checked above),
            // content is a valid slice, and the regions don't overlap.
            unsafe {
                let dst = (self.ptr as *mut u8).add(self.cursor);
                std::ptr::copy_nonoverlapping(content.as_ptr(), dst, content.len());
            }
        }

        self.cursor = new_end;
        self.files.push(meta);
        self.total_bytes += content.len() as u64;
        self.file_count += 1;

        file_id
    }

    /// Append file content with an associated file path for match resolution.
    ///
    /// Like `append()` but also records the file path in the path table.
    /// Returns the file_id (index into the metadata and path tables).
    pub fn append_with_path(
        &mut self,
        content: &[u8],
        path: PathBuf,
        path_id: u32,
        content_hash: u32,
        mtime: u32,
    ) -> u32 {
        let file_id = self.append(content, path_id, content_hash, mtime);
        self.paths.push(path);
        file_id
    }

    /// Remaining capacity in bytes.
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.cursor
    }

    /// Current write position (total bytes appended).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Total capacity of the mmap region.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of files appended so far.
    pub fn file_count(&self) -> u32 {
        self.file_count
    }

    /// Finalize the builder into a ContentStore with a Metal buffer.
    ///
    /// 1. Makes the mmap region read-only via `mprotect(PROT_READ)`
    /// 2. Creates a Metal buffer via `newBufferWithBytesNoCopy` (zero-copy)
    /// 3. Falls back to `newBufferWithBytes` if bytesNoCopy fails
    /// 4. Transfers ownership of mmap and metadata to ContentStore
    ///
    /// Consumes the builder.
    pub fn finalize(
        self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> ContentStore {
        // Wrap in ManuallyDrop to prevent Drop from running, then
        // read fields out. We take ownership of ptr/files.
        let mut me = std::mem::ManuallyDrop::new(self);
        let ptr = me.ptr;
        let capacity = me.capacity;
        let data_len = me.cursor;
        // SAFETY: we will not use `me.files` or `me.paths` again after this take.
        let files = std::mem::take(&mut me.files);
        let paths = std::mem::take(&mut me.paths);
        let total_bytes = me.total_bytes;
        let file_count = me.file_count;

        // Make the mmap region read-only
        // SAFETY: ptr is a valid mmap'd region of `capacity` bytes.
        unsafe {
            libc::mprotect(ptr, capacity, libc::PROT_READ);
        }

        // Create Metal buffer via bytesNoCopy (zero-copy path)
        let options = MTLResourceOptions::StorageModeShared;
        let metal_buffer = if data_len > 0 {
            // The mmap length for bytesNoCopy must be page-aligned.
            // `capacity` is already page-aligned from new().
            let aligned_len = capacity;

            // SAFETY: ptr is page-aligned (from mmap), valid for capacity bytes,
            // read-only, and will outlive the Metal buffer (ContentStore owns it).
            // No deallocator because ContentStore manages the memory.
            let buf = unsafe {
                let nn = NonNull::new(ptr as *mut u8)
                    .expect("mmap pointer should be non-null");
                let nn_void = nn.cast::<std::ffi::c_void>();
                device.newBufferWithBytesNoCopy_length_options_deallocator(
                    nn_void,
                    aligned_len,
                    options,
                    None,
                )
            };

            if buf.is_some() {
                buf
            } else {
                // Fallback: copy data into a new Metal buffer
                // SAFETY: ptr is valid for data_len bytes.
                unsafe {
                    let nn = NonNull::new(ptr as *mut u8)
                        .expect("mmap pointer should be non-null");
                    let nn_void = nn.cast::<std::ffi::c_void>();
                    device.newBufferWithBytes_length_options(nn_void, data_len, options)
                }
            }
        } else {
            None
        };

        ContentStore {
            metal_buffer,
            backing: ContentBacking::Mmap {
                ptr,
                mapped_len: capacity,
                data_len,
            },
            files,
            paths,
            total_bytes,
            file_count,
            dead_bytes: 0,
            chunk_metadata: None,
        }
    }

    /// Finalize the builder into a ContentStore WITHOUT a Metal buffer.
    ///
    /// Useful for testing or when no GPU device is available.
    /// Makes the mmap region read-only and transfers ownership.
    pub fn finalize_without_metal(self) -> ContentStore {
        let mut me = std::mem::ManuallyDrop::new(self);
        let ptr = me.ptr;
        let capacity = me.capacity;
        let data_len = me.cursor;
        let files = std::mem::take(&mut me.files);
        let paths = std::mem::take(&mut me.paths);
        let total_bytes = me.total_bytes;
        let file_count = me.file_count;

        // Make the mmap region read-only
        // SAFETY: ptr is a valid mmap'd region of `capacity` bytes.
        unsafe {
            libc::mprotect(ptr, capacity, libc::PROT_READ);
        }

        ContentStore {
            metal_buffer: None,
            backing: ContentBacking::Mmap {
                ptr,
                mapped_len: capacity,
                data_len,
            },
            files,
            paths,
            total_bytes,
            file_count,
            dead_bytes: 0,
            chunk_metadata: None,
        }
    }
}

impl Drop for ContentStoreBuilder {
    fn drop(&mut self) {
        if self.capacity > 0 {
            // SAFETY: ptr was returned by a successful mmap() call with
            // capacity bytes. We only call munmap once (in Drop).
            unsafe {
                libc::munmap(self.ptr, self.capacity);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ChunkMetadata builder: maps ContentStore into GPU chunk format
// ---------------------------------------------------------------------------

use crate::search::content::CHUNK_SIZE;

/// Build GPU-compatible ChunkMetadata from a ContentStore.
///
/// Splits each file's content into CHUNK_SIZE (4096 byte) chunks and produces
/// a ChunkMetadata entry for each chunk. The resulting Vec can be written
/// directly to the GPU metadata buffer for dispatch.
///
/// Chunk offsets (`offset_in_file`) are byte offsets within the original file.
/// `file_index` maps to the file_id in the ContentStore's files table.
///
/// Flags:
/// - Bit 0 (1): is_text (always set)
/// - Bit 1 (2): is_first_chunk
/// - Bit 2 (4): is_last_chunk
pub fn build_chunk_metadata(store: &ContentStore) -> Vec<ChunkMetadata> {
    let mut chunks = Vec::new();

    for file_id in 0..store.file_count() {
        let meta = &store.files()[file_id as usize];
        let content_len = meta.content_len as usize;

        // Empty files produce no chunks (same as ContentSearchEngine::load_content)
        if content_len == 0 {
            continue;
        }

        let num_chunks = content_len.div_ceil(CHUNK_SIZE);

        for chunk_i in 0..num_chunks {
            let offset = chunk_i * CHUNK_SIZE;
            let chunk_len = (content_len - offset).min(CHUNK_SIZE);

            let mut flags = 1u32; // is_text
            if chunk_i == 0 {
                flags |= 2; // is_first
            }
            if chunk_i == num_chunks - 1 {
                flags |= 4; // is_last
            }

            chunks.push(ChunkMetadata {
                file_index: file_id,
                chunk_index: chunk_i as u32,
                offset_in_file: offset as u64,
                chunk_length: chunk_len as u32,
                flags,
                buffer_offset: (meta.content_offset as usize + offset) as u64,
            });
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn get_device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        MTLCreateSystemDefaultDevice().expect("No Metal device (test requires Apple Silicon)")
    }

    // ================================================================
    // Original ContentStore (Vec-backed) tests
    // ================================================================

    #[test]
    fn test_file_content_meta_size() {
        assert_eq!(
            std::mem::size_of::<FileContentMeta>(),
            32,
            "FileContentMeta must be exactly 32 bytes"
        );
    }

    #[test]
    fn test_new_empty() {
        let store = ContentStore::new();
        assert_eq!(store.total_bytes(), 0);
        assert_eq!(store.file_count(), 0);
        assert!(store.content_for(0).is_none());
        assert!(!store.has_metal_buffer());
    }

    #[test]
    fn test_with_capacity() {
        let store = ContentStore::with_capacity(1024);
        assert_eq!(store.total_bytes(), 0);
        assert_eq!(store.file_count(), 0);
        match &store.backing {
            ContentBacking::Vec(v) => assert!(v.capacity() >= 1024),
            _ => panic!("Expected Vec backing"),
        }
    }

    #[test]
    fn test_insert_and_content_for_roundtrip() {
        let mut store = ContentStore::new();

        let content_a = b"Hello, world!";
        let content_b = b"fn main() { println!(\"GPU search\"); }";
        let content_c = b"";

        let id_a = store.insert(content_a, 10, 0xAABB, 1000);
        let id_b = store.insert(content_b, 20, 0xCCDD, 2000);
        let id_c = store.insert(content_c, 30, 0x0000, 3000);

        assert_eq!(id_a, 0);
        assert_eq!(id_b, 1);
        assert_eq!(id_c, 2);

        assert_eq!(store.content_for(id_a).unwrap(), content_a);
        assert_eq!(store.content_for(id_b).unwrap(), content_b);
        assert_eq!(store.content_for(id_c).unwrap(), content_c);

        assert_eq!(store.file_count(), 3);
        assert_eq!(
            store.total_bytes(),
            (content_a.len() + content_b.len() + content_c.len()) as u64
        );
    }

    #[test]
    fn test_content_for_out_of_bounds() {
        let store = ContentStore::new();
        assert!(store.content_for(0).is_none());
        assert!(store.content_for(999).is_none());
    }

    #[test]
    fn test_metadata_fields() {
        let mut store = ContentStore::new();
        let content = b"test data";
        let id = store.insert(content, 42, 0xDEAD, 9999);

        let meta = &store.files()[id as usize];
        assert_eq!(meta.content_offset, 0);
        assert_eq!(meta.content_len, content.len() as u32);
        assert_eq!(meta.path_id, 42);
        assert_eq!(meta.content_hash, 0xDEAD);
        assert_eq!(meta.mtime, 9999);
        assert_eq!(meta.trigram_count, 0);
        assert_eq!(meta.flags, 0);
    }

    #[test]
    fn test_multiple_inserts_contiguous() {
        let mut store = ContentStore::new();

        let c1 = b"AAAA";
        let c2 = b"BBBBBB";
        let c3 = b"CC";

        store.insert(c1, 0, 0, 0);
        store.insert(c2, 1, 0, 0);
        store.insert(c3, 2, 0, 0);

        // Verify contiguous layout in buffer
        let buf = store.buffer();
        assert_eq!(&buf[0..4], b"AAAA");
        assert_eq!(&buf[4..10], b"BBBBBB");
        assert_eq!(&buf[10..12], b"CC");
        assert_eq!(buf.len(), 12);
    }

    #[test]
    fn test_default() {
        let store = ContentStore::default();
        assert_eq!(store.file_count(), 0);
        assert_eq!(store.total_bytes(), 0);
    }

    // ================================================================
    // ContentStoreBuilder (mmap-backed) tests
    // ================================================================

    #[test]
    fn test_builder_new_allocates_page_aligned() {
        let builder = ContentStoreBuilder::new(100).expect("builder allocation failed");
        // Capacity should be rounded up to PAGE_SIZE
        assert_eq!(builder.capacity(), PAGE_SIZE);
        assert_eq!(builder.cursor(), 0);
        assert_eq!(builder.file_count(), 0);
        // Verify the mmap pointer is page-aligned (16KB on Apple Silicon)
        assert!(
            (builder.ptr as usize).is_multiple_of(PAGE_SIZE),
            "mmap pointer should be page-aligned: {:p}",
            builder.ptr
        );
    }

    #[test]
    fn test_builder_append_and_roundtrip() {
        let mut builder = ContentStoreBuilder::new(4096).unwrap();

        let c1 = b"Hello from mmap!";
        let c2 = b"GPU search content";
        let c3 = b"";

        let id0 = builder.append(c1, 10, 0xAA, 1000);
        let id1 = builder.append(c2, 20, 0xBB, 2000);
        let id2 = builder.append(c3, 30, 0xCC, 3000);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(builder.file_count(), 3);
        assert_eq!(builder.cursor(), c1.len() + c2.len() + c3.len());

        // Finalize without Metal to test mmap backing
        let store = builder.finalize_without_metal();

        assert_eq!(store.content_for(id0).unwrap(), c1.as_slice());
        assert_eq!(store.content_for(id1).unwrap(), c2.as_slice());
        assert_eq!(store.content_for(id2).unwrap(), c3.as_slice());
        assert_eq!(store.file_count(), 3);
        assert_eq!(
            store.total_bytes(),
            (c1.len() + c2.len() + c3.len()) as u64
        );
        assert!(!store.has_metal_buffer());
    }

    #[test]
    fn test_builder_page_alignment_after_finalize() {
        let mut builder = ContentStoreBuilder::new(8192).unwrap();
        builder.append(b"some content", 0, 0, 0);

        let store = builder.finalize_without_metal();

        // The backing mmap pointer should be page-aligned
        let ptr = store.backing.as_ptr();
        assert!(
            (ptr as usize).is_multiple_of(PAGE_SIZE),
            "Content buffer pointer should be 16KB page-aligned: {:p}",
            ptr
        );

        // The mapped length should be page-aligned
        let mapped = store.backing.mapped_len();
        assert!(
            mapped.is_multiple_of(PAGE_SIZE),
            "Mapped length should be page-aligned: {}",
            mapped
        );
    }

    #[test]
    fn test_builder_finalize_with_metal_buffer() {
        let device = get_device();
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();

        // Insert several files
        for i in 0..10 {
            let content = format!("File {} content for GPU search testing", i);
            builder.append(content.as_bytes(), i, 0, 0);
        }

        let store = builder.finalize(&device);

        assert!(
            store.has_metal_buffer(),
            "Metal buffer should be created via bytesNoCopy"
        );

        // Verify Metal buffer length covers the content
        let metal_buf = store.metal_buffer().unwrap();
        assert!(
            metal_buf.length() >= store.total_bytes() as usize,
            "Metal buffer length ({}) must cover total content bytes ({})",
            metal_buf.length(),
            store.total_bytes()
        );

        // Verify content roundtrip through the finalized store
        for i in 0..10u32 {
            let expected = format!("File {} content for GPU search testing", i);
            let actual = store.content_for(i).unwrap();
            assert_eq!(
                actual,
                expected.as_bytes(),
                "Content mismatch for file_id {}",
                i
            );
        }
    }

    #[test]
    fn test_builder_metal_buffer_contents_match() {
        let device = get_device();
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();

        let test_data = b"DEADBEEF_CONTENT_FOR_GPU_VERIFICATION";
        builder.append(test_data, 0, 0, 0);

        let store = builder.finalize(&device);
        assert!(store.has_metal_buffer());

        let metal_buf = store.metal_buffer().unwrap();

        // Verify Metal buffer contents match the mmap'd data byte-for-byte
        unsafe {
            let buf_ptr = metal_buf.contents().as_ptr() as *const u8;
            let buf_slice = std::slice::from_raw_parts(buf_ptr, test_data.len());
            assert_eq!(
                buf_slice, test_data,
                "Metal buffer contents should match source data"
            );
        }
    }

    #[test]
    fn test_builder_remaining_capacity() {
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();
        assert_eq!(builder.remaining_capacity(), PAGE_SIZE);

        builder.append(b"hello", 0, 0, 0);
        assert_eq!(builder.remaining_capacity(), PAGE_SIZE - 5);
    }

    #[test]
    #[should_panic(expected = "exceed capacity")]
    fn test_builder_append_exceeds_capacity() {
        let mut builder = ContentStoreBuilder::new(32).unwrap();
        // Capacity is rounded to PAGE_SIZE, so this won't actually panic
        // unless we fill it. Fill the whole capacity + 1.
        let data = vec![0u8; builder.capacity() + 1];
        builder.append(&data, 0, 0, 0);
    }

    #[test]
    fn test_builder_zero_capacity_gets_one_page() {
        let builder = ContentStoreBuilder::new(0).unwrap();
        assert_eq!(builder.capacity(), PAGE_SIZE);
    }

    #[test]
    fn test_builder_drop_without_finalize() {
        // Ensure Drop works correctly when builder is not finalized
        {
            let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();
            builder.append(b"test", 0, 0, 0);
            // builder dropped here -- should munmap cleanly
        }
        // If we get here without a crash, Drop worked
    }

    #[test]
    fn test_builder_large_content() {
        let device = get_device();
        // Allocate enough for 1MB of content
        let capacity = 1024 * 1024;
        let mut builder = ContentStoreBuilder::new(capacity).unwrap();

        // Insert a large file
        let large_content = vec![0xABu8; 512 * 1024]; // 512KB
        let id = builder.append(&large_content, 0, 0xDEAD, 9999);

        let store = builder.finalize(&device);

        assert!(store.has_metal_buffer());
        assert_eq!(store.content_for(id).unwrap(), large_content.as_slice());
    }

    // ================================================================
    // Phase 1 Checkpoint: end-to-end integration test
    // ================================================================

    #[test]
    fn test_checkpoint_end_to_end() {
        use crate::index::content_index_store::ContentIndexStore;
        use crate::index::content_snapshot::ContentSnapshot;

        let device = get_device();

        // ----------------------------------------------------------
        // Step 1: Create a ContentStoreBuilder with enough capacity
        // ----------------------------------------------------------
        // 100 files, each ~50-100 bytes. 64KB is more than enough.
        let mut builder = ContentStoreBuilder::new(64 * 1024)
            .expect("ContentStoreBuilder allocation failed");

        // ----------------------------------------------------------
        // Step 2: Insert 100 files with known, deterministic content
        // ----------------------------------------------------------
        let file_count = 100u32;
        let mut expected_contents: Vec<Vec<u8>> = Vec::with_capacity(file_count as usize);

        for i in 0..file_count {
            let content = format!(
                "// file_{:04}.rs\nfn gpu_search_content_{}() -> u32 {{ {} }}\n",
                i, i, i * 7 + 13
            );
            let bytes = content.into_bytes();
            let hash = i.wrapping_mul(0x9E37_79B9); // deterministic hash
            let mtime = 1700000000 + i;
            let file_id = builder.append(&bytes, i, hash, mtime);
            assert_eq!(file_id, i, "file_id should equal insertion order index");
            expected_contents.push(bytes);
        }

        assert_eq!(builder.file_count(), file_count);

        // ----------------------------------------------------------
        // Step 3: Finalize to ContentStore with Metal buffer
        // ----------------------------------------------------------
        let store = builder.finalize(&device);

        assert_eq!(store.file_count(), file_count);
        assert!(
            store.total_bytes() > 0,
            "total_bytes should be positive after inserting 100 files"
        );

        // ----------------------------------------------------------
        // Step 4: Verify content_for() returns exact bytes for each file
        // ----------------------------------------------------------
        for i in 0..file_count {
            let actual = store
                .content_for(i)
                .unwrap_or_else(|| panic!("content_for({}) returned None", i));
            assert_eq!(
                actual, &expected_contents[i as usize],
                "content mismatch for file_id {}",
                i
            );
        }

        // Out-of-bounds file_id returns None
        assert!(store.content_for(file_count).is_none());
        assert!(store.content_for(u32::MAX).is_none());

        // ----------------------------------------------------------
        // Step 5: Verify Metal buffer is valid and page-aligned
        // ----------------------------------------------------------
        let metal_buf = store
            .metal_buffer()
            .expect("Metal buffer should be present after finalize with device");

        assert!(
            metal_buf.length() >= store.total_bytes() as usize,
            "Metal buffer length ({}) must be >= total_bytes ({})",
            metal_buf.length(),
            store.total_bytes()
        );

        // Verify page alignment of the Metal buffer's contents pointer
        let buf_ptr = metal_buf.contents().as_ptr() as usize;
        assert!(
            buf_ptr.is_multiple_of(PAGE_SIZE),
            "Metal buffer contents pointer should be page-aligned (16KB), got 0x{:x}",
            buf_ptr
        );

        // Verify Metal buffer data matches the content buffer
        unsafe {
            let metal_ptr = metal_buf.contents().as_ptr() as *const u8;
            let metal_slice =
                std::slice::from_raw_parts(metal_ptr, store.total_bytes() as usize);
            assert_eq!(
                metal_slice,
                store.buffer(),
                "Metal buffer contents must match content store buffer byte-for-byte"
            );
        }

        // ----------------------------------------------------------
        // Step 6: Wrap in ContentSnapshot, store in ContentIndexStore
        // ----------------------------------------------------------
        let timestamp = 1700000099u64;
        let snapshot = ContentSnapshot::new(store, timestamp);
        assert_eq!(snapshot.file_count(), file_count);
        assert_eq!(snapshot.build_timestamp(), timestamp);

        let index_store = ContentIndexStore::new();
        assert!(
            !index_store.is_available(),
            "ContentIndexStore should start empty"
        );
        index_store.swap(snapshot);
        assert!(
            index_store.is_available(),
            "ContentIndexStore should be available after swap"
        );

        // ----------------------------------------------------------
        // Step 7: Read back via arc-swap guard
        // ----------------------------------------------------------
        let guard = index_store.snapshot();
        let snap_ref = guard
            .as_ref()
            .as_ref()
            .expect("snapshot guard should contain Some(ContentSnapshot)");

        assert_eq!(snap_ref.file_count(), file_count);
        assert_eq!(snap_ref.build_timestamp(), timestamp);

        // ----------------------------------------------------------
        // Step 8: Verify all content still accessible through snapshot
        // ----------------------------------------------------------
        let cs = snap_ref.content_store();
        assert_eq!(cs.file_count(), file_count);

        for i in 0..file_count {
            let actual = cs
                .content_for(i)
                .unwrap_or_else(|| panic!("content_for({}) via snapshot returned None", i));
            assert_eq!(
                actual, &expected_contents[i as usize],
                "content mismatch via snapshot for file_id {}",
                i
            );
        }

        // Verify Metal buffer still accessible through the snapshot
        assert!(
            cs.has_metal_buffer(),
            "Metal buffer should still be accessible via snapshot"
        );

        // Verify metadata integrity through the snapshot
        let files = cs.files();
        assert_eq!(files.len(), file_count as usize);
        for i in 0..file_count {
            let meta = &files[i as usize];
            assert_eq!(meta.path_id, i, "path_id mismatch for file {}", i);
            assert_eq!(
                meta.content_hash,
                i.wrapping_mul(0x9E37_79B9),
                "content_hash mismatch for file {}",
                i
            );
            assert_eq!(
                meta.mtime,
                1700000000 + i,
                "mtime mismatch for file {}",
                i
            );
            assert_eq!(
                meta.content_len,
                expected_contents[i as usize].len() as u32,
                "content_len mismatch for file {}",
                i
            );
        }
    }

    // ================================================================
    // ChunkMetadata builder tests
    // ================================================================

    #[test]
    fn test_chunk_metadata_empty_store() {
        let store = ContentStore::new();
        let chunks = build_chunk_metadata(&store);
        assert!(chunks.is_empty(), "Empty store should produce no chunks");
    }

    #[test]
    fn test_chunk_metadata_single_small_file() {
        let mut store = ContentStore::new();
        store.insert(b"hello world", 0, 0, 0); // 11 bytes < CHUNK_SIZE

        let chunks = build_chunk_metadata(&store);
        assert_eq!(chunks.len(), 1, "Small file should produce exactly 1 chunk");

        let c = &chunks[0];
        assert_eq!(c.file_index, 0);
        assert_eq!(c.chunk_index, 0);
        assert_eq!(c.offset_in_file, 0);
        assert_eq!(c.chunk_length, 11);
        // is_text | is_first | is_last
        assert_eq!(c.flags, 1 | 2 | 4);
    }

    #[test]
    fn test_chunk_metadata_exact_chunk_boundary() {
        use crate::search::content::CHUNK_SIZE;

        let mut store = ContentStore::new();
        let data = vec![0x41u8; CHUNK_SIZE]; // exactly 4096 bytes
        store.insert(&data, 0, 0, 0);

        let chunks = build_chunk_metadata(&store);
        assert_eq!(chunks.len(), 1, "Exact CHUNK_SIZE should produce 1 chunk");
        assert_eq!(chunks[0].chunk_length, CHUNK_SIZE as u32);
        assert_eq!(chunks[0].flags, 1 | 2 | 4); // single chunk = first + last
    }

    #[test]
    fn test_chunk_metadata_multi_chunk_file() {
        use crate::search::content::CHUNK_SIZE;

        let mut store = ContentStore::new();
        // 2.5 chunks worth of data = 10240 bytes
        let data = vec![0x42u8; CHUNK_SIZE * 2 + CHUNK_SIZE / 2];
        store.insert(&data, 0, 0, 0);

        let chunks = build_chunk_metadata(&store);
        assert_eq!(chunks.len(), 3, "10240 bytes should produce 3 chunks");

        // First chunk
        assert_eq!(chunks[0].file_index, 0);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].offset_in_file, 0);
        assert_eq!(chunks[0].chunk_length, CHUNK_SIZE as u32);
        assert_eq!(chunks[0].flags, 1 | 2); // is_text + is_first

        // Middle chunk
        assert_eq!(chunks[1].chunk_index, 1);
        assert_eq!(chunks[1].offset_in_file, CHUNK_SIZE as u64);
        assert_eq!(chunks[1].chunk_length, CHUNK_SIZE as u32);
        assert_eq!(chunks[1].flags, 1); // is_text only

        // Last chunk (partial)
        assert_eq!(chunks[2].chunk_index, 2);
        assert_eq!(chunks[2].offset_in_file, (CHUNK_SIZE * 2) as u64);
        assert_eq!(chunks[2].chunk_length, (CHUNK_SIZE / 2) as u32);
        assert_eq!(chunks[2].flags, 1 | 4); // is_text + is_last
    }

    #[test]
    fn test_chunk_metadata_empty_file_skipped() {
        let mut store = ContentStore::new();
        store.insert(b"", 0, 0, 0); // empty file
        store.insert(b"data", 1, 0, 0); // non-empty file

        let chunks = build_chunk_metadata(&store);
        assert_eq!(chunks.len(), 1, "Empty file should produce no chunks");
        assert_eq!(chunks[0].file_index, 1, "Only non-empty file should appear");
    }

    #[test]
    fn test_chunk_metadata_10_files_varying_sizes() {
        use crate::search::content::CHUNK_SIZE;

        let mut store = ContentStore::new();
        // 10 files with varying sizes
        let sizes: Vec<usize> = vec![
            100,              // 1 chunk
            CHUNK_SIZE,       // 1 chunk (exact boundary)
            CHUNK_SIZE + 1,   // 2 chunks
            CHUNK_SIZE * 3,   // 3 chunks (exact)
            CHUNK_SIZE * 2 + 500, // 3 chunks
            0,                // 0 chunks (empty, skipped)
            1,                // 1 chunk
            CHUNK_SIZE - 1,   // 1 chunk
            CHUNK_SIZE * 5 + 2048, // 6 chunks
            50,               // 1 chunk
        ];

        for (i, &size) in sizes.iter().enumerate() {
            let data = vec![(i as u8).wrapping_add(0x30); size];
            store.insert(&data, i as u32, 0, 0);
        }

        let chunks = build_chunk_metadata(&store);

        // Expected chunk counts: 1, 1, 2, 3, 3, 0, 1, 1, 6, 1 = 19
        let expected_total: usize = sizes
            .iter()
            .map(|&s| if s == 0 { 0 } else { s.div_ceil(CHUNK_SIZE) })
            .sum();
        assert_eq!(expected_total, 19);
        assert_eq!(
            chunks.len(),
            expected_total,
            "Total chunk count mismatch: expected {}, got {}",
            expected_total,
            chunks.len()
        );

        // Verify offsets are correct for each file
        let mut chunk_idx = 0;
        for (file_id, &size) in sizes.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let num_chunks = size.div_ceil(CHUNK_SIZE);

            for ci in 0..num_chunks {
                let c = &chunks[chunk_idx];
                assert_eq!(
                    c.file_index, file_id as u32,
                    "chunk {} should belong to file {}",
                    chunk_idx, file_id
                );
                assert_eq!(c.chunk_index, ci as u32);
                assert_eq!(c.offset_in_file, (ci * CHUNK_SIZE) as u64);

                let expected_len = (size - ci * CHUNK_SIZE).min(CHUNK_SIZE);
                assert_eq!(
                    c.chunk_length, expected_len as u32,
                    "chunk {} length mismatch",
                    chunk_idx
                );

                // Verify flags
                let is_first = ci == 0;
                let is_last = ci == num_chunks - 1;
                let expected_flags = 1u32
                    | if is_first { 2 } else { 0 }
                    | if is_last { 4 } else { 0 };
                assert_eq!(
                    c.flags, expected_flags,
                    "chunk {} flags mismatch (file {}, ci {})",
                    chunk_idx, file_id, ci
                );

                chunk_idx += 1;
            }
        }
        assert_eq!(chunk_idx, chunks.len());
    }

    // ================================================================
    // Incremental update tests
    // ================================================================

    #[test]
    fn test_update_file_replaces_content() {
        let mut store = ContentStore::new();

        let original = b"original content";
        let id = store.insert(original, 0, 0xAAAA, 1000);

        assert_eq!(store.content_for(id).unwrap(), original);
        assert_eq!(store.dead_bytes(), 0);

        // Update with new content
        let updated = b"updated content -- longer than before!";
        store.update_file(id, updated, 0xBBBB, 2000);

        // content_for should return the new content
        assert_eq!(store.content_for(id).unwrap(), updated);

        // Metadata should be updated
        let meta = &store.files()[id as usize];
        assert_eq!(meta.content_hash, 0xBBBB);
        assert_eq!(meta.mtime, 2000);
        assert_eq!(meta.content_len, updated.len() as u32);

        // Dead bytes should track old content
        assert_eq!(store.dead_bytes(), original.len() as u64);
    }

    #[test]
    fn test_update_file_multiple_updates_accumulate_dead_bytes() {
        let mut store = ContentStore::new();
        let id = store.insert(b"v1", 0, 1, 100);

        store.update_file(id, b"version2", 2, 200);
        assert_eq!(store.dead_bytes(), 2); // "v1" = 2 bytes

        store.update_file(id, b"version_three", 3, 300);
        assert_eq!(store.dead_bytes(), 2 + 8); // "v1" + "version2"

        assert_eq!(store.content_for(id).unwrap(), b"version_three");
    }

    #[test]
    fn test_update_file_preserves_other_files() {
        let mut store = ContentStore::new();
        let id_a = store.insert(b"file_a", 0, 0, 0);
        let id_b = store.insert(b"file_b", 1, 0, 0);
        let id_c = store.insert(b"file_c", 2, 0, 0);

        // Update only file B
        store.update_file(id_b, b"file_b_updated", 0xFF, 999);

        // A and C should be unchanged
        assert_eq!(store.content_for(id_a).unwrap(), b"file_a");
        assert_eq!(store.content_for(id_b).unwrap(), b"file_b_updated");
        assert_eq!(store.content_for(id_c).unwrap(), b"file_c");
    }

    #[test]
    fn test_remove_file_marks_deleted() {
        let mut store = ContentStore::new();
        let id_a = store.insert(b"keep_this", 0, 0, 0);
        let id_b = store.insert(b"delete_this", 1, 0, 0);

        assert!(store.content_for(id_b).is_some());

        store.remove_file(id_b);

        // Deleted file returns None
        assert!(store.content_for(id_b).is_none());

        // Other file still accessible
        assert_eq!(store.content_for(id_a).unwrap(), b"keep_this");

        // Dead bytes tracked
        assert_eq!(store.dead_bytes(), b"delete_this".len() as u64);

        // Flag bit 0 is set
        assert_eq!(store.files()[id_b as usize].flags & 1, 1);
    }

    #[test]
    fn test_remove_file_idempotent() {
        let mut store = ContentStore::new();
        let id = store.insert(b"data", 0, 0, 0);

        store.remove_file(id);
        let dead_after_first = store.dead_bytes();

        // Second remove should not double-count dead bytes
        store.remove_file(id);
        assert_eq!(store.dead_bytes(), dead_after_first);
    }

    #[test]
    fn test_add_file_works_like_insert() {
        let mut store = ContentStore::new();
        let id_insert = store.insert(b"via insert", 0, 0xAA, 100);
        let id_add = store.add_file(b"via add_file", 1, 0xBB, 200);

        assert_eq!(id_insert, 0);
        assert_eq!(id_add, 1);

        assert_eq!(store.content_for(id_insert).unwrap(), b"via insert");
        assert_eq!(store.content_for(id_add).unwrap(), b"via add_file");
        assert_eq!(store.file_count(), 2);
    }

    #[test]
    fn test_update_then_remove() {
        let mut store = ContentStore::new();
        let id = store.insert(b"original", 0, 0, 0);

        // Update the file
        store.update_file(id, b"updated", 1, 1);
        assert_eq!(store.content_for(id).unwrap(), b"updated");
        let dead_after_update = store.dead_bytes();
        assert_eq!(dead_after_update, b"original".len() as u64);

        // Then remove it
        store.remove_file(id);
        assert!(store.content_for(id).is_none());
        // Dead bytes should include both old content AND the updated content
        assert_eq!(
            store.dead_bytes(),
            dead_after_update + b"updated".len() as u64
        );
    }

    #[test]
    fn test_update_insert_remove_interleaved() {
        let mut store = ContentStore::new();

        // Insert 3 files
        let id0 = store.insert(b"alpha", 0, 0, 0);
        let id1 = store.insert(b"beta", 1, 0, 0);
        let id2 = store.insert(b"gamma", 2, 0, 0);

        // Update file 1
        store.update_file(id1, b"beta_v2", 1, 1);

        // Add a new file
        let id3 = store.add_file(b"delta", 3, 0, 0);

        // Remove file 0
        store.remove_file(id0);

        // Verify state
        assert!(store.content_for(id0).is_none()); // removed
        assert_eq!(store.content_for(id1).unwrap(), b"beta_v2"); // updated
        assert_eq!(store.content_for(id2).unwrap(), b"gamma"); // untouched
        assert_eq!(store.content_for(id3).unwrap(), b"delta"); // added

        assert_eq!(store.file_count(), 4);
        // Dead: "beta" (4 bytes from update) + "alpha" (5 bytes from remove)
        assert_eq!(store.dead_bytes(), 4 + 5);
    }

    #[test]
    #[should_panic(expected = "Cannot mutate mmap-backed")]
    fn test_update_file_panics_on_mmap_backed() {
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();
        builder.append(b"content", 0, 0, 0);
        let mut store = builder.finalize_without_metal();
        store.update_file(0, b"new", 0, 0);
    }

    #[test]
    #[should_panic(expected = "Cannot mutate mmap-backed")]
    fn test_remove_file_panics_on_mmap_backed() {
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();
        builder.append(b"content", 0, 0, 0);
        let mut store = builder.finalize_without_metal();
        store.remove_file(0);
    }

    #[test]
    #[should_panic(expected = "Cannot insert into mmap-backed")]
    fn test_add_file_panics_on_mmap_backed() {
        let mut builder = ContentStoreBuilder::new(PAGE_SIZE).unwrap();
        builder.append(b"content", 0, 0, 0);
        let mut store = builder.finalize_without_metal();
        store.add_file(b"new", 1, 0, 0);
    }

    // ================================================================
    // Phase 6 unit tests (task 6.1)
    // ================================================================

    #[test]
    fn test_insert_retrieve_roundtrip() {
        // Insert 10 files with distinct content, verify byte-exact retrieval.
        let mut store = ContentStore::new();
        let mut expected: Vec<Vec<u8>> = Vec::new();

        for i in 0u32..10 {
            let content = format!("File-{}-content-{}", i, "x".repeat(i as usize * 7 + 3));
            expected.push(content.as_bytes().to_vec());
            store.insert(content.as_bytes(), i, i.wrapping_mul(0x1234), i * 100);
        }

        assert_eq!(store.file_count(), 10);
        for i in 0u32..10 {
            let actual = store.content_for(i).expect("content_for should return Some");
            assert_eq!(actual, &expected[i as usize], "byte-exact mismatch for file_id {}", i);
        }
    }

    #[test]
    fn test_empty_file() {
        // Inserting empty content should not panic and should roundtrip correctly.
        let mut store = ContentStore::new();

        let id = store.insert(b"", 0, 0, 0);
        assert_eq!(id, 0);

        let content = store.content_for(id).expect("empty file should return Some");
        assert!(content.is_empty(), "empty file should return empty slice");

        // Metadata should reflect zero length
        let meta = &store.files()[id as usize];
        assert_eq!(meta.content_len, 0);
        assert_eq!(meta.content_offset, 0);
    }

    #[test]
    fn test_large_file() {
        // Insert 10MB content, verify byte-exact roundtrip.
        let size = 10 * 1024 * 1024; // 10 MB
        let large_content: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();

        let mut store = ContentStore::new();
        let id = store.insert(&large_content, 0, 0xDEAD, 42);

        let retrieved = store.content_for(id).expect("large file should return Some");
        assert_eq!(retrieved.len(), size);
        assert_eq!(retrieved, &large_content[..], "10MB content roundtrip mismatch");
        assert_eq!(store.total_bytes(), size as u64);
    }

    #[test]
    fn test_binary_content() {
        // Insert content containing all 256 byte values, verify roundtrip.
        let all_bytes: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
        assert_eq!(all_bytes.len(), 256);

        let mut store = ContentStore::new();
        let id = store.insert(&all_bytes, 0, 0xFF, 0);

        let retrieved = store.content_for(id).expect("binary content should return Some");
        assert_eq!(retrieved.len(), 256);
        assert_eq!(retrieved, &all_bytes[..], "all-256-byte-values roundtrip mismatch");

        // Double-check every individual byte
        for (i, &byte) in retrieved.iter().enumerate() {
            assert_eq!(byte, i as u8, "byte value mismatch at index {}", i);
        }
    }
}
