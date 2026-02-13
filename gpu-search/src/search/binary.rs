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
    "exe", "o", "obj", "dylib", "so", "a", "lib", "dll", "bin",
    "class", "pyc", "pyo", "wasm", "dSYM", "elc", "eln",
    // Rust build artifacts
    "rmeta", "rlib", "crate", "d",
    // Metal GPU artifacts
    "metallib", "air",
    // Images
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "tiff", "tif",
    "webp", "heic", "heif", "raw", "cr2", "nef", "icns",
    // Fonts
    "ttf", "otf", "woff", "woff2", "eot",
    // Audio
    "mp3", "wav", "aac", "flac", "ogg", "m4a", "aiff",
    // Video
    "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm",
    // Archives
    "zip", "gz", "tar", "bz2", "xz", "7z", "rar", "zst", "lz4", "tgz",
    "ltar",
    // Documents (binary formats)
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    // Database / data
    "db", "sqlite", "sqlite3", "mdb",
    // Disk images / packages
    "dmg", "iso", "pkg", "deb", "rpm",
    // macOS specific
    "plist", "nib", "storyboardc", "car", "pf_fragment",
    "pbxproj", "xcworkspacedata", "xcscheme", "xcuserstate",
    "mom", "momd", "ipa",
    // macOS data stores / caches (text format but not source code)
    "emlx", "mbox", "olk15message", "olk16message",
    "spotlight", "mdimporter",
    "savedSearch", "webarchive", "download",
    // Misc binary / data
    "dat", "swp", "swo", "bak", "orig", "tmp",
    "cache", "idx", "pack", "bitmap",
    "min.js", "min.css", "bundle.js", "chunk.js",
];

/// Known text file extensions that should never trigger the content-based
/// binary check. Avoids reading 8KB per file for common text formats.
const TEXT_EXTENSIONS: &[&str] = &[
    // Programming languages
    "rs", "c", "h", "cpp", "cc", "cxx", "hpp", "hh", "cs", "java",
    "kt", "kts", "scala", "go", "py", "rb", "pl", "pm", "lua",
    "js", "jsx", "ts", "tsx", "mjs", "cjs", "mts", "cts",
    "swift", "m", "mm", "zig", "nim", "dart", "r", "jl",
    "hs", "ml", "mli", "fs", "fsx", "fsi", "clj", "cljs", "cljc",
    "ex", "exs", "erl", "hrl", "elm", "v", "sv", "vhd", "vhdl",
    "lean", "olean",
    // Shell / scripting
    "sh", "bash", "zsh", "fish", "ps1", "psm1", "bat", "cmd",
    // Web
    "html", "htm", "css", "scss", "sass", "less",
    // Data / config
    "json", "yaml", "yml", "toml", "ini", "cfg", "conf",
    "xml", "csv", "tsv", "env",
    // Documentation
    "md", "markdown", "rst", "txt", "adoc", "tex", "org",
    // Build / project
    "mk", "cmake", "gradle", "sbt", "cabal",
    "lock", "sum",
    // Metal / GPU
    "metal", "msl", "glsl", "hlsl", "wgsl", "cl",
    // Misc text
    "log", "diff", "patch", "sql", "graphql", "gql",
    "proto", "thrift", "avsc",
    "dockerfile", "editorconfig", "gitignore", "gitattributes",
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
    /// Three-layer check (fastest first):
    /// 1. Known binary extension -> skip immediately (no I/O)
    /// 2. Known text extension -> keep immediately (no I/O)
    /// 3. Unknown extension -> read first 8KB for NUL byte heuristic
    pub fn should_skip(&self, path: &Path) -> bool {
        if self.include_binary {
            return false;
        }
        if is_binary_path(path) {
            return true;
        }
        // Known text extension -> definitely not binary, skip I/O
        if is_known_text(path) {
            return false;
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

/// Check if a file path has a known text extension.
///
/// If the extension matches TEXT_EXTENSIONS, we know it's text and can
/// skip the expensive 8KB content read. This is the key optimization
/// for large-scale searches from system root.
pub fn is_known_text(path: &Path) -> bool {
    if let Some(ext) = path.extension() {
        let ext_lower = ext.to_string_lossy().to_lowercase();
        TEXT_EXTENSIONS.contains(&ext_lower.as_str())
    } else {
        // Files with no extension (Makefile, LICENSE, Dockerfile, etc.)
        // Check common extensionless text files by filename
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            matches!(
                name,
                "Makefile" | "makefile" | "GNUmakefile"
                | "Dockerfile" | "Containerfile"
                | "LICENSE" | "LICENCE" | "COPYING"
                | "README" | "CHANGES" | "CHANGELOG" | "AUTHORS"
                | "INSTALL" | "NEWS" | "TODO" | "THANKS"
                | "Rakefile" | "Gemfile" | "Brewfile"
                | "Procfile" | "Vagrantfile"
                | ".gitignore" | ".gitattributes" | ".editorconfig"
                | ".dockerignore" | ".eslintrc" | ".prettierrc"
            )
        } else {
            false
        }
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

    #[test]
    fn test_known_text_extensions() {
        // Programming languages
        assert!(is_known_text(Path::new("main.rs")));
        assert!(is_known_text(Path::new("app.py")));
        assert!(is_known_text(Path::new("index.js")));
        assert!(is_known_text(Path::new("component.tsx")));
        assert!(is_known_text(Path::new("main.go")));
        assert!(is_known_text(Path::new("App.swift")));
        assert!(is_known_text(Path::new("shader.metal")));

        // Config / data
        assert!(is_known_text(Path::new("Cargo.toml")));
        assert!(is_known_text(Path::new("config.json")));
        assert!(is_known_text(Path::new("config.yaml")));
        assert!(is_known_text(Path::new("style.css")));

        // Documentation
        assert!(is_known_text(Path::new("README.md")));
        assert!(is_known_text(Path::new("notes.txt")));

        // Extensionless text files
        assert!(is_known_text(Path::new("Makefile")));
        assert!(is_known_text(Path::new("Dockerfile")));
        assert!(is_known_text(Path::new("LICENSE")));

        // Case insensitive
        assert!(is_known_text(Path::new("code.RS")));
        assert!(is_known_text(Path::new("page.HTML")));

        // Unknown extensions should NOT match
        assert!(!is_known_text(Path::new("data.xyz")));
        assert!(!is_known_text(Path::new("file.unknown")));
        // Extensionless unknown files
        assert!(!is_known_text(Path::new("randomfile")));
    }

    #[test]
    fn test_text_extension_skips_content_check() {
        // A .rs file with NUL bytes should NOT be skipped by the detector
        // because the text extension fast-path returns false before content check
        let dir = TempDir::new().unwrap();

        let rs_with_nul = dir.path().join("weird.rs");
        {
            let mut f = File::create(&rs_with_nul).unwrap();
            f.write_all(b"fn main() { \x00 }").unwrap();
        }

        let detector = BinaryDetector::new();
        // .rs is a known text extension, so should_skip returns false
        // even though content has NUL bytes (text extension fast-path)
        assert!(!detector.should_skip(&rs_with_nul));
    }

    #[test]
    fn test_detector_unknown_ext_checks_content() {
        let dir = TempDir::new().unwrap();

        // Unknown extension with text content -> not skipped
        let text_unk = dir.path().join("data.xyz");
        {
            let mut f = File::create(&text_unk).unwrap();
            f.write_all(b"this is plain text\n").unwrap();
        }
        let detector = BinaryDetector::new();
        assert!(!detector.should_skip(&text_unk));

        // Unknown extension with binary content -> skipped
        let bin_unk = dir.path().join("data.qqq");
        {
            let mut f = File::create(&bin_unk).unwrap();
            f.write_all(b"\x00\x01\x02binary").unwrap();
        }
        assert!(detector.should_skip(&bin_unk));
    }
}
