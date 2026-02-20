//! Search cancellation and generation tracking.
//!
//! Provides cooperative cancellation for GPU search operations and generation
//! IDs to discard stale results from superseded searches.
//!
//! ## Cancellation
//!
//! `CancellationToken` / `CancellationHandle` pair: the token is passed into
//! the search pipeline (checked between GPU chunk dispatches), while the handle
//! is held by the caller to trigger cancellation.
//!
//! ## Generation Tracking
//!
//! `SearchGeneration` assigns monotonically increasing IDs to searches.
//! When a new keystroke arrives, the generation increments. Results tagged
//! with an older generation are stale and discarded.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// CancellationToken + CancellationHandle
// ============================================================================

/// Token checked by the search pipeline to detect cancellation.
///
/// Passed into GPU dispatch loops. Between chunk dispatches, the pipeline
/// calls `is_cancelled()` and bails early if true. Uses `Relaxed` ordering
/// for maximum throughput -- the worst case is processing one extra chunk
/// before noticing cancellation.
#[derive(Clone)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

/// Handle held by the caller to cancel an in-flight search.
///
/// Calling `cancel()` sets the shared flag. The associated
/// `CancellationToken` will observe the cancellation on its next check.
#[derive(Clone)]
pub struct CancellationHandle {
    flag: Arc<AtomicBool>,
}

/// Create a new cancellation pair (token, handle).
///
/// The token is given to the search pipeline; the handle stays with the caller.
pub fn cancellation_pair() -> (CancellationToken, CancellationHandle) {
    let flag = Arc::new(AtomicBool::new(false));
    (
        CancellationToken {
            flag: Arc::clone(&flag),
        },
        CancellationHandle { flag },
    )
}

impl CancellationToken {
    /// Check if cancellation has been requested.
    ///
    /// Uses `Relaxed` ordering -- the pipeline may process one extra chunk
    /// before observing the flag, which is acceptable for search latency.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

impl CancellationHandle {
    /// Signal cancellation to the associated token.
    ///
    /// Uses `Relaxed` ordering for symmetry with the token check.
    /// Idempotent: calling cancel() multiple times is harmless.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Relaxed);
    }

    /// Check if this handle has already been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Relaxed)
    }
}

// ============================================================================
// SearchGeneration
// ============================================================================

/// Monotonically increasing generation counter for search sessions.
///
/// Each new search (triggered by a keystroke) increments the generation.
/// Results arriving from an older generation are stale and should be dropped.
///
/// Thread-safe: can be shared across the UI thread and search dispatch threads.
#[derive(Debug)]
pub struct SearchGeneration {
    current: Arc<AtomicU64>,
}

impl SearchGeneration {
    /// Create a new generation counter starting at 0.
    pub fn new() -> Self {
        Self {
            current: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start a new search generation. Returns a guard tied to this generation.
    ///
    /// The generation counter is atomically incremented. The returned guard
    /// captures the new generation ID so results can be checked for staleness.
    pub fn next(&self) -> SearchGenerationGuard {
        let id = self.current.fetch_add(1, Ordering::AcqRel) + 1;
        SearchGenerationGuard {
            generation_id: id,
            current: Arc::clone(&self.current),
        }
    }

    /// Get the current generation ID (most recent search).
    pub fn current_id(&self) -> u64 {
        self.current.load(Ordering::Acquire)
    }
}

impl Default for SearchGeneration {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SearchGeneration {
    fn clone(&self) -> Self {
        Self {
            current: Arc::clone(&self.current),
        }
    }
}

// ============================================================================
// SearchGenerationGuard
// ============================================================================

/// Guard associated with a specific search generation.
///
/// Holds the generation ID assigned when the search started. Call `is_stale()`
/// to check whether a newer search has superseded this one.
#[derive(Debug, Clone)]
pub struct SearchGenerationGuard {
    /// The generation ID assigned to this search.
    generation_id: u64,
    /// Shared reference to the global generation counter.
    current: Arc<AtomicU64>,
}

impl SearchGenerationGuard {
    /// Check if this search generation is stale (superseded by a newer search).
    ///
    /// Returns `true` if the global generation has advanced past this guard's ID.
    pub fn is_stale(&self) -> bool {
        self.current.load(Ordering::Acquire) > self.generation_id
    }

    /// Get this guard's generation ID.
    pub fn generation_id(&self) -> u64 {
        self.generation_id
    }
}

// ============================================================================
// Convenience: combined cancellation + generation
// ============================================================================

/// A search session combining cancellation token and generation guard.
///
/// Convenience struct for passing both cancellation and staleness checks
/// into the search pipeline.
#[derive(Clone)]
pub struct SearchSession {
    /// Cancellation token checked between GPU dispatches.
    pub token: CancellationToken,
    /// Generation guard for discarding stale results.
    pub guard: SearchGenerationGuard,
}

impl SearchSession {
    /// Check if this search should stop (cancelled or stale).
    #[inline]
    pub fn should_stop(&self) -> bool {
        self.token.is_cancelled() || self.guard.is_stale()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_cancel_token_basic() {
        let (token, handle) = cancellation_pair();

        // Initially not cancelled
        assert!(!token.is_cancelled(), "Token should not be cancelled initially");
        assert!(!handle.is_cancelled(), "Handle should not report cancelled initially");

        // Cancel
        handle.cancel();

        // Now cancelled
        assert!(token.is_cancelled(), "Token should be cancelled after cancel()");
        assert!(handle.is_cancelled(), "Handle should report cancelled after cancel()");

        // Idempotent: cancel again is fine
        handle.cancel();
        assert!(token.is_cancelled(), "Token should still be cancelled");
    }

    #[test]
    fn test_cancel_across_threads() {
        let (token, handle) = cancellation_pair();

        // Clone token for the reader thread
        let token_clone = token.clone();

        // Spawn a thread that waits for cancellation
        let reader = thread::spawn(move || {
            let mut iterations = 0u64;
            while !token_clone.is_cancelled() {
                iterations += 1;
                if iterations > 10_000_000 {
                    panic!("Cancellation not observed after 10M iterations");
                }
                // Tight spin -- Relaxed ordering means we may loop a few times
                std::hint::spin_loop();
            }
            iterations
        });

        // Give the reader thread a moment to start spinning
        thread::sleep(Duration::from_millis(1));

        // Cancel from this thread
        handle.cancel();

        // Reader should observe cancellation and exit
        let iterations = reader.join().expect("Reader thread panicked");
        assert!(iterations > 0, "Reader should have spun at least once");
        assert!(token.is_cancelled(), "Token should be cancelled");
    }

    #[test]
    fn test_generation_tracking() {
        let gen = SearchGeneration::new();

        // Initially at generation 0
        assert_eq!(gen.current_id(), 0, "Should start at generation 0");

        // First search -> generation 1
        let guard1 = gen.next();
        assert_eq!(guard1.generation_id(), 1);
        assert_eq!(gen.current_id(), 1);
        assert!(!guard1.is_stale(), "Guard1 should not be stale (current)");

        // Second search -> generation 2
        let guard2 = gen.next();
        assert_eq!(guard2.generation_id(), 2);
        assert_eq!(gen.current_id(), 2);

        // Guard1 is now stale, guard2 is current
        assert!(guard1.is_stale(), "Guard1 should be stale after gen advanced");
        assert!(!guard2.is_stale(), "Guard2 should be current");

        // Third search -> generation 3
        let guard3 = gen.next();
        assert_eq!(guard3.generation_id(), 3);

        // Both guard1 and guard2 are stale
        assert!(guard1.is_stale());
        assert!(guard2.is_stale());
        assert!(!guard3.is_stale());
    }

    #[test]
    fn test_rapid_cancellation() {
        // Simulate rapid sequential searches: cancel old, start new, repeat
        let gen = SearchGeneration::new();
        let mut handles: Vec<CancellationHandle> = Vec::new();
        let mut guards: Vec<SearchGenerationGuard> = Vec::new();

        for i in 0..100 {
            // Cancel previous search
            if let Some(prev_handle) = handles.last() {
                prev_handle.cancel();
            }

            // Start new search
            let (token, handle) = cancellation_pair();
            let guard = gen.next();

            // Previous guards should all be stale
            for (j, old_guard) in guards.iter().enumerate() {
                assert!(
                    old_guard.is_stale(),
                    "Guard {} should be stale at iteration {}",
                    j, i
                );
            }

            // Current should not be stale
            assert!(
                !guard.is_stale(),
                "Current guard should not be stale at iteration {}",
                i
            );

            // Token for current search should not be cancelled
            assert!(
                !token.is_cancelled(),
                "New token should not be cancelled at iteration {}",
                i
            );

            // Previous handles should all be cancelled
            for (j, old_handle) in handles.iter().enumerate() {
                assert!(
                    old_handle.is_cancelled(),
                    "Handle {} should be cancelled at iteration {}",
                    j, i
                );
            }

            handles.push(handle);
            guards.push(guard);
        }

        // Final state: generation 100, only last guard is current
        assert_eq!(gen.current_id(), 100);
        assert!(!guards.last().unwrap().is_stale());
        for guard in &guards[..guards.len() - 1] {
            assert!(guard.is_stale());
        }
    }

    #[test]
    fn test_search_session_should_stop() {
        let gen = SearchGeneration::new();
        let (token, handle) = cancellation_pair();
        let guard = gen.next();

        let session = SearchSession {
            token: token.clone(),
            guard: guard.clone(),
        };

        // Not cancelled, not stale -> should not stop
        assert!(!session.should_stop(), "Fresh session should not stop");

        // Cancel -> should stop
        handle.cancel();
        assert!(session.should_stop(), "Cancelled session should stop");
    }

    #[test]
    fn test_search_session_stale_stops() {
        let gen = SearchGeneration::new();
        let (token, _handle) = cancellation_pair();
        let guard = gen.next();

        let session = SearchSession {
            token,
            guard,
        };

        // Not stale yet
        assert!(!session.should_stop());

        // Advance generation -> session becomes stale
        let _guard2 = gen.next();
        assert!(session.should_stop(), "Stale session should stop");
    }

    #[test]
    fn test_generation_clone_shares_state() {
        let gen1 = SearchGeneration::new();
        let gen2 = gen1.clone();

        let guard1 = gen1.next();
        assert_eq!(gen2.current_id(), 1, "Clone should see same generation");

        let guard2 = gen2.next();
        assert_eq!(gen1.current_id(), 2, "Original should see clone's advance");

        assert!(guard1.is_stale());
        assert!(!guard2.is_stale());
    }

    #[test]
    fn test_generation_concurrent_increment() {
        let gen = Arc::new(SearchGeneration::new());
        let mut threads = Vec::new();

        // 10 threads each increment 100 times
        for _ in 0..10 {
            let gen_clone = Arc::clone(&gen);
            threads.push(thread::spawn(move || {
                for _ in 0..100 {
                    gen_clone.next();
                }
            }));
        }

        for t in threads {
            t.join().unwrap();
        }

        // Should have exactly 1000 increments
        assert_eq!(
            gen.current_id(),
            1000,
            "10 threads x 100 increments = 1000"
        );
    }
}
