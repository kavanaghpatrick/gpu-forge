// Binary file detection for gpu-search
//
// Two-layer detection:
// 1. Extension-based: skip known binary extensions (fast, no I/O)
// 2. Content-based: NUL byte heuristic in first 8KB (accurate, needs read)
//
// Configurable via BinaryDetector.include_binary flag.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Maximum bytes to read for NUL byte heuristic.
const BINARY_CHECK_SIZE: usize = 8192;

/// Known binary file extensions that should always be skipped.
const BINARY_EXTENSIONS: &[&str] = &[
    // Compiled objects / executables
    "exe", "o", "obj", "dylib", "so", "a", "lib",
    // Metal GPU artifacts
    "metallib", "air",
    // Images
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg",
    // Audio
    "mp3", "wav",
    // Video
    "mp4", "avi", "mov",
    // Archives
    "zip", "gz", "tar",
    // Documents (binary formats)
    "pdf", "doc", "docx", "xls", "xlsx",
];

/// Binary file detector with configurable behavior.
#[derive(Debug, Clone)]
pub struct BinaryDetector {
    /// If true, skip binary detection and include all files.
    pub include_binary: bool,
}

impl BinaryDetector {
    /// Create a new BinaryDetector that skips binary files by default.
    pub fn new() -> Self {
        Self {
            include_binary: false,
        }
    }

    /// Create a BinaryDetector that includes binary files (no filtering).
    pub fn include_all() -> Self {
        Self {
            include_binary: true,
        }
    }

    /// Returns true if the file should be skipped (is binary).
    ///
    /// When `include_binary` is true, always returns false (never skip).
    /// Otherwise checks extension first (fast), then content (accurate).
    pub fn should_skip(&self, path: &Path) -> bool {
        if self.include_binary {
            return false;
        }
        if is_binary_path(path) {
            return true;
        }
        is_binary_file(path)
    }
}

impl Default for BinaryDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a file path has a known binary extension.
///
/// Case-insensitive comparison against BINARY_EXTENSIONS list.
pub fn is_binary_path(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        BINARY_EXTENSIONS.contains(&ext_lower.as_str())
    } else {
        false
    }
}

/// Check if file content contains NUL bytes in the first 8KB.
///
/// Returns true if NUL byte found (binary), false if all bytes are non-NUL (text).
/// Returns false on read errors (assume text, let downstream handle errors).
pub fn is_binary_content(data: &[u8]) -> bool {
    let check_len = data.len().min(BINARY_CHECK_SIZE);
    data[..check_len].contains(&0)
}

/// Check if a file on disk is binary by reading its first 8KB.
///
/// Returns false on read errors (assume text).
fn is_binary_file(path: &Path) -> bool {
    let mut buf = [0u8; BINARY_CHECK_SIZE];
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let Ok(n) = file.read(&mut buf) else {
        return false;
    };
    if n == 0 {
        return false; // empty files are not binary
    }
    is_binary_content(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_binary_detection_extensions() {
        // Known binary extensions
        assert!(is_binary_path(Path::new("foo.exe")));
        assert!(is_binary_path(Path::new("bar.o")));
        assert!(is_binary_path(Path::new("lib.dylib")));
        assert!(is_binary_path(Path::new("shader.metallib")));
        assert!(is_binary_path(Path::new("shader.air")));
        assert!(is_binary_path(Path::new("image.png")));
        assert!(is_binary_path(Path::new("photo.jpg")));
        assert!(is_binary_path(Path::new("photo.JPEG")));
        assert!(is_binary_path(Path::new("archive.zip")));
        assert!(is_binary_path(Path::new("archive.gz")));
        assert!(is_binary_path(Path::new("archive.tar")));
        assert!(is_binary_path(Path::new("doc.pdf")));
        assert!(is_binary_path(Path::new("video.mp4")));
        assert!(is_binary_path(Path::new("audio.mp3")));
        assert!(is_binary_path(Path::new("data.xlsx")));

        // Text extensions should not match
        assert!(!is_binary_path(Path::new("main.rs")));
        assert!(!is_binary_path(Path::new("Cargo.toml")));
        assert!(!is_binary_path(Path::new("README.md")));
        assert!(!is_binary_path(Path::new("script.py")));
        assert!(!is_binary_path(Path::new("index.html")));

        // No extension
        assert!(!is_binary_path(Path::new("Makefile")));
        assert!(!is_binary_path(Path::new("LICENSE")));
    }

    #[test]
    fn test_binary_detection_content_nul() {
        // Text content -- no NUL bytes
        let text = b"Hello, world!\nThis is a text file.\n";
        assert!(!is_binary_content(text));

        // Binary content -- contains NUL byte
        let binary = b"Hello\x00world";
        assert!(is_binary_content(binary));

        // NUL at start
        let nul_start = b"\x00binary data here";
        assert!(is_binary_content(nul_start));

        // Empty content
        let empty: &[u8] = &[];
        assert!(!is_binary_content(empty));

        // All NUL
        let all_nul = [0u8; 100];
        assert!(is_binary_content(&all_nul));
    }

    #[test]
    fn test_binary_detection_content_large() {
        // NUL byte beyond 8KB should not be detected
        let mut data = vec![b'A'; 16384];
        data[10000] = 0; // NUL at 10KB -- beyond check window
        assert!(!is_binary_content(&data));

        // NUL byte within 8KB should be detected
        let mut data2 = vec![b'A'; 16384];
        data2[4000] = 0; // NUL at 4KB -- within check window
        assert!(is_binary_content(&data2));
    }

    #[test]
    fn test_binary_detection_real_files() {
        let dir = TempDir::new().unwrap();

        // Text file
        let text_path = dir.path().join("hello.txt");
        {
            let mut f = File::create(&text_path).unwrap();
            f.write_all(b"Hello, world!\nThis is text.\n").unwrap();
        }
        assert!(!is_binary_file(&text_path));

        // Binary file (contains NUL)
        let bin_path = dir.path().join("data.bin");
        {
            let mut f = File::create(&bin_path).unwrap();
            f.write_all(b"ELF\x00\x01\x02\x03").unwrap();
        }
        assert!(is_binary_file(&bin_path));

        // Empty file
        let empty_path = dir.path().join("empty.txt");
        File::create(&empty_path).unwrap();
        assert!(!is_binary_file(&empty_path));

        // Nonexistent file -- returns false (assume text)
        assert!(!is_binary_file(Path::new("/nonexistent/file.txt")));
    }

    #[test]
    fn test_binary_detection_detector_struct() {
        let dir = TempDir::new().unwrap();

        // Create a text file
        let text_path = dir.path().join("code.rs");
        {
            let mut f = File::create(&text_path).unwrap();
            f.write_all(b"fn main() {}\n").unwrap();
        }

        // Create a binary file by extension
        let bin_ext_path = dir.path().join("lib.dylib");
        {
            let mut f = File::create(&bin_ext_path).unwrap();
            f.write_all(b"not really a dylib").unwrap();
        }

        // Create a binary file by content
        let bin_content_path = dir.path().join("mystery.dat");
        {
            let mut f = File::create(&bin_content_path).unwrap();
            f.write_all(b"\x00\x01\x02\x03").unwrap();
        }

        // Default detector: skip binary
        let detector = BinaryDetector::new();
        assert!(!detector.should_skip(&text_path));
        assert!(detector.should_skip(&bin_ext_path));
        assert!(detector.should_skip(&bin_content_path));

        // Include-all detector: skip nothing
        let include_all = BinaryDetector::include_all();
        assert!(!include_all.should_skip(&text_path));
        assert!(!include_all.should_skip(&bin_ext_path));
        assert!(!include_all.should_skip(&bin_content_path));
    }

    #[test]
    fn test_binary_detection_case_insensitive_ext() {
        assert!(is_binary_path(Path::new("file.PNG")));
        assert!(is_binary_path(Path::new("file.Exe")));
        assert!(is_binary_path(Path::new("file.DyLib")));
        assert!(is_binary_path(Path::new("file.METALLIB")));
    }
}
