//! Centralized error types and recovery strategies for gpu-search.
//!
//! Every failure mode in the GPU search pipeline has a defined recovery action:
//!
//! | Error                 | Recovery                              |
//! |-----------------------|---------------------------------------|
//! | DeviceNotFound        | Show error to user, exit gracefully   |
//! | ShaderCompilation     | Fatal: show error, cannot proceed     |
//! | IoQueue               | Fall back to CPU reads (std::fs)      |
//! | FileRead              | Skip file, log warning                |
//! | Utf8Error             | Skip file, log warning                |
//! | PermissionDenied      | Skip file, log warning                |
//! | WatchdogTimeout       | Restart GPU command chain             |
//! | OutOfMemory           | Reduce batch size by half, retry      |
//! | SearchCancelled       | Discard results, no error to user     |

use std::fmt;
use std::path::PathBuf;

/// Central error type for all gpu-search operations.
#[derive(Debug)]
pub enum GpuSearchError {
    /// No Metal GPU device found (e.g. no Apple Silicon).
    DeviceNotFound,

    /// Metal shader compilation or PSO creation failed.
    ShaderCompilation(String),

    /// MTLIOCommandQueue creation or operation failed.
    IoQueue(String),

    /// Failed to read a file from disk.
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    /// File contains invalid UTF-8 (binary or corrupt text).
    Utf8Error {
        path: PathBuf,
    },

    /// Permission denied when accessing a file or directory.
    PermissionDenied {
        path: PathBuf,
    },

    /// GPU command buffer hit Metal watchdog timeout (>5s on macOS).
    WatchdogTimeout,

    /// GPU buffer allocation failed -- not enough VRAM for batch.
    OutOfMemory,

    /// Search was cancelled by user (new keystroke arrived).
    SearchCancelled,
}

impl fmt::Display for GpuSearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuSearchError::DeviceNotFound => {
                write!(f, "No Metal GPU device found -- Apple Silicon required")
            }
            GpuSearchError::ShaderCompilation(msg) => {
                write!(f, "Metal shader compilation failed: {}", msg)
            }
            GpuSearchError::IoQueue(msg) => {
                write!(f, "MTLIOCommandQueue failed: {}", msg)
            }
            GpuSearchError::FileRead { path, source } => {
                write!(f, "Failed to read file {:?}: {}", path, source)
            }
            GpuSearchError::Utf8Error { path } => {
                write!(f, "Invalid UTF-8 in file {:?}", path)
            }
            GpuSearchError::PermissionDenied { path } => {
                write!(f, "Permission denied: {:?}", path)
            }
            GpuSearchError::WatchdogTimeout => {
                write!(f, "GPU watchdog timeout -- command buffer exceeded time limit")
            }
            GpuSearchError::OutOfMemory => {
                write!(f, "GPU out of memory -- batch size too large")
            }
            GpuSearchError::SearchCancelled => {
                write!(f, "Search cancelled")
            }
        }
    }
}

impl std::error::Error for GpuSearchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuSearchError::FileRead { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Type alias for Results using GpuSearchError.
pub type Result<T> = std::result::Result<T, GpuSearchError>;

// ============================================================================
// Recovery actions
// ============================================================================

/// Describes how the system should recover from a given error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Fatal error -- show message to user, cannot continue.
    Fatal,
    /// Fall back to CPU-based alternative (e.g. std::fs reads instead of MTLIOCommandQueue).
    FallbackToCpu,
    /// Skip this file and continue searching remaining files.
    SkipFile,
    /// Restart the GPU command chain (re-create command buffer).
    RestartChain,
    /// Reduce batch size by half and retry the operation.
    ReduceBatchSize,
    /// Silently discard -- not a real error (user cancelled).
    Discard,
}

impl GpuSearchError {
    /// Determine the appropriate recovery action for this error.
    pub fn recovery_action(&self) -> RecoveryAction {
        match self {
            GpuSearchError::DeviceNotFound => RecoveryAction::Fatal,
            GpuSearchError::ShaderCompilation(_) => RecoveryAction::Fatal,
            GpuSearchError::IoQueue(_) => RecoveryAction::FallbackToCpu,
            GpuSearchError::FileRead { .. } => RecoveryAction::SkipFile,
            GpuSearchError::Utf8Error { .. } => RecoveryAction::SkipFile,
            GpuSearchError::PermissionDenied { .. } => RecoveryAction::SkipFile,
            GpuSearchError::WatchdogTimeout => RecoveryAction::RestartChain,
            GpuSearchError::OutOfMemory => RecoveryAction::ReduceBatchSize,
            GpuSearchError::SearchCancelled => RecoveryAction::Discard,
        }
    }

    /// Whether this error should be logged (vs silently handled).
    pub fn should_log(&self) -> bool {
        match self {
            GpuSearchError::SearchCancelled => false,
            _ => true,
        }
    }

    /// Whether this error is fatal (no recovery possible).
    pub fn is_fatal(&self) -> bool {
        matches!(
            self.recovery_action(),
            RecoveryAction::Fatal
        )
    }

    /// Whether this error means "skip this file, continue with rest".
    pub fn is_skippable(&self) -> bool {
        matches!(
            self.recovery_action(),
            RecoveryAction::SkipFile
        )
    }
}

// ============================================================================
// From impls for common error sources
// ============================================================================

impl From<std::io::Error> for GpuSearchError {
    fn from(err: std::io::Error) -> Self {
        match err.kind() {
            std::io::ErrorKind::PermissionDenied => GpuSearchError::PermissionDenied {
                path: PathBuf::new(), // caller should use FileRead with path instead
            },
            std::io::ErrorKind::OutOfMemory => GpuSearchError::OutOfMemory,
            _ => GpuSearchError::FileRead {
                path: PathBuf::new(),
                source: err,
            },
        }
    }
}

impl From<std::str::Utf8Error> for GpuSearchError {
    fn from(_: std::str::Utf8Error) -> Self {
        GpuSearchError::Utf8Error {
            path: PathBuf::new(),
        }
    }
}

impl From<std::string::FromUtf8Error> for GpuSearchError {
    fn from(_: std::string::FromUtf8Error) -> Self {
        GpuSearchError::Utf8Error {
            path: PathBuf::new(),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::path::PathBuf;

    #[test]
    fn test_error_display() {
        // Each variant should produce a non-empty, meaningful message
        let errors: Vec<GpuSearchError> = vec![
            GpuSearchError::DeviceNotFound,
            GpuSearchError::ShaderCompilation("missing function".into()),
            GpuSearchError::IoQueue("descriptor invalid".into()),
            GpuSearchError::FileRead {
                path: PathBuf::from("/tmp/test.rs"),
                source: io::Error::new(io::ErrorKind::NotFound, "file not found"),
            },
            GpuSearchError::Utf8Error {
                path: PathBuf::from("/tmp/binary.bin"),
            },
            GpuSearchError::PermissionDenied {
                path: PathBuf::from("/root/secret"),
            },
            GpuSearchError::WatchdogTimeout,
            GpuSearchError::OutOfMemory,
            GpuSearchError::SearchCancelled,
        ];

        for err in &errors {
            let msg = format!("{}", err);
            assert!(!msg.is_empty(), "Display for {:?} should not be empty", err);
            // Verify specific content in each message
            match err {
                GpuSearchError::DeviceNotFound => {
                    assert!(msg.contains("Metal"), "DeviceNotFound: {}", msg);
                    assert!(msg.contains("Apple Silicon"), "DeviceNotFound: {}", msg);
                }
                GpuSearchError::ShaderCompilation(_) => {
                    assert!(msg.contains("shader"), "ShaderCompilation: {}", msg);
                    assert!(msg.contains("missing function"), "ShaderCompilation: {}", msg);
                }
                GpuSearchError::IoQueue(_) => {
                    assert!(msg.contains("MTLIOCommandQueue"), "IoQueue: {}", msg);
                    assert!(msg.contains("descriptor invalid"), "IoQueue: {}", msg);
                }
                GpuSearchError::FileRead { .. } => {
                    assert!(msg.contains("test.rs"), "FileRead: {}", msg);
                    assert!(msg.contains("file not found"), "FileRead: {}", msg);
                }
                GpuSearchError::Utf8Error { .. } => {
                    assert!(msg.contains("UTF-8"), "Utf8Error: {}", msg);
                    assert!(msg.contains("binary.bin"), "Utf8Error: {}", msg);
                }
                GpuSearchError::PermissionDenied { .. } => {
                    assert!(msg.contains("Permission denied"), "PermissionDenied: {}", msg);
                    assert!(msg.contains("secret"), "PermissionDenied: {}", msg);
                }
                GpuSearchError::WatchdogTimeout => {
                    assert!(msg.contains("watchdog"), "WatchdogTimeout: {}", msg);
                    assert!(msg.contains("timeout"), "WatchdogTimeout: {}", msg);
                }
                GpuSearchError::OutOfMemory => {
                    assert!(msg.contains("memory"), "OutOfMemory: {}", msg);
                    assert!(msg.contains("batch"), "OutOfMemory: {}", msg);
                }
                GpuSearchError::SearchCancelled => {
                    assert!(msg.contains("cancelled"), "SearchCancelled: {}", msg);
                }
            }
        }
    }

    #[test]
    fn test_error_recovery() {
        // DeviceNotFound -> Fatal
        assert_eq!(
            GpuSearchError::DeviceNotFound.recovery_action(),
            RecoveryAction::Fatal,
        );
        assert!(GpuSearchError::DeviceNotFound.is_fatal());

        // ShaderCompilation -> Fatal
        assert_eq!(
            GpuSearchError::ShaderCompilation("bad".into()).recovery_action(),
            RecoveryAction::Fatal,
        );
        assert!(GpuSearchError::ShaderCompilation("bad".into()).is_fatal());

        // IoQueue -> FallbackToCpu
        assert_eq!(
            GpuSearchError::IoQueue("fail".into()).recovery_action(),
            RecoveryAction::FallbackToCpu,
        );
        assert!(!GpuSearchError::IoQueue("fail".into()).is_fatal());

        // FileRead -> SkipFile
        let file_err = GpuSearchError::FileRead {
            path: PathBuf::from("/tmp/x"),
            source: io::Error::new(io::ErrorKind::NotFound, "gone"),
        };
        assert_eq!(file_err.recovery_action(), RecoveryAction::SkipFile);
        assert!(file_err.is_skippable());

        // Utf8Error -> SkipFile
        let utf8_err = GpuSearchError::Utf8Error {
            path: PathBuf::from("/tmp/x"),
        };
        assert_eq!(utf8_err.recovery_action(), RecoveryAction::SkipFile);
        assert!(utf8_err.is_skippable());

        // PermissionDenied -> SkipFile
        let perm_err = GpuSearchError::PermissionDenied {
            path: PathBuf::from("/root"),
        };
        assert_eq!(perm_err.recovery_action(), RecoveryAction::SkipFile);
        assert!(perm_err.is_skippable());

        // WatchdogTimeout -> RestartChain
        assert_eq!(
            GpuSearchError::WatchdogTimeout.recovery_action(),
            RecoveryAction::RestartChain,
        );
        assert!(!GpuSearchError::WatchdogTimeout.is_fatal());

        // OutOfMemory -> ReduceBatchSize
        assert_eq!(
            GpuSearchError::OutOfMemory.recovery_action(),
            RecoveryAction::ReduceBatchSize,
        );
        assert!(!GpuSearchError::OutOfMemory.is_fatal());

        // SearchCancelled -> Discard
        assert_eq!(
            GpuSearchError::SearchCancelled.recovery_action(),
            RecoveryAction::Discard,
        );
        assert!(!GpuSearchError::SearchCancelled.is_fatal());
        assert!(!GpuSearchError::SearchCancelled.should_log());
    }

    #[test]
    fn test_error_source_chain() {
        // FileRead should expose source io::Error
        let io_err = io::Error::new(io::ErrorKind::NotFound, "test source");
        let err = GpuSearchError::FileRead {
            path: PathBuf::from("/tmp/x"),
            source: io_err,
        };
        assert!(
            std::error::Error::source(&err).is_some(),
            "FileRead should have a source error"
        );

        // Other variants should NOT have a source
        assert!(std::error::Error::source(&GpuSearchError::DeviceNotFound).is_none());
        assert!(std::error::Error::source(&GpuSearchError::WatchdogTimeout).is_none());
        assert!(std::error::Error::source(&GpuSearchError::OutOfMemory).is_none());
        assert!(std::error::Error::source(&GpuSearchError::SearchCancelled).is_none());
    }

    #[test]
    fn test_from_io_error() {
        // PermissionDenied io::Error -> PermissionDenied variant
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let err: GpuSearchError = io_err.into();
        assert!(matches!(err, GpuSearchError::PermissionDenied { .. }));
        assert_eq!(err.recovery_action(), RecoveryAction::SkipFile);

        // OutOfMemory io::Error -> OutOfMemory variant
        let io_err = io::Error::new(io::ErrorKind::OutOfMemory, "oom");
        let err: GpuSearchError = io_err.into();
        assert!(matches!(err, GpuSearchError::OutOfMemory));

        // Generic io::Error -> FileRead variant
        let io_err = io::Error::new(io::ErrorKind::NotFound, "gone");
        let err: GpuSearchError = io_err.into();
        assert!(matches!(err, GpuSearchError::FileRead { .. }));
    }

    #[test]
    fn test_from_utf8_error() {
        // std::str::Utf8Error -> Utf8Error variant
        let bad_bytes = vec![0xFF, 0xFE];
        let utf8_err = std::str::from_utf8(&bad_bytes).unwrap_err();
        let err: GpuSearchError = utf8_err.into();
        assert!(matches!(err, GpuSearchError::Utf8Error { .. }));
        assert!(err.is_skippable());

        // std::string::FromUtf8Error -> Utf8Error variant
        let from_utf8_err = String::from_utf8(bad_bytes).unwrap_err();
        let err: GpuSearchError = from_utf8_err.into();
        assert!(matches!(err, GpuSearchError::Utf8Error { .. }));
    }

    #[test]
    fn test_should_log() {
        // All errors should be logged except SearchCancelled
        assert!(GpuSearchError::DeviceNotFound.should_log());
        assert!(GpuSearchError::ShaderCompilation("x".into()).should_log());
        assert!(GpuSearchError::IoQueue("x".into()).should_log());
        assert!(GpuSearchError::WatchdogTimeout.should_log());
        assert!(GpuSearchError::OutOfMemory.should_log());
        assert!(!GpuSearchError::SearchCancelled.should_log());
    }

    #[test]
    fn test_error_is_send_sync() {
        // GpuSearchError must be Send for cross-thread error propagation
        fn assert_send<T: Send>() {}
        assert_send::<GpuSearchError>();
    }

    #[test]
    fn test_recovery_action_debug() {
        // RecoveryAction should be Debug-printable for logging
        let action = RecoveryAction::Fatal;
        let s = format!("{:?}", action);
        assert!(s.contains("Fatal"));
    }
}
