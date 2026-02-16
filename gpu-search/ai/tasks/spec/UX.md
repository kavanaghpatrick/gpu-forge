# gpu-search Persistent File Index -- UX/UI Designer Analysis

**Author**: UX/UI Designer Agent
**Date**: 2026-02-14
**Status**: Final
**Feature**: Persistent GPU-Friendly File Index with FSEvents Background Updates
**Upstream**: Requirement spec (persistent mmap index, always from `/`, FSEvents incremental updates)

---

## 1. Prior Art Research

### 1.1 macOS Spotlight

Spotlight has evolved its indexing UX significantly over macOS releases. In earlier OS X versions, clicking the Spotlight icon displayed a progress indicator and written time estimate. The magnifying glass icon gained a center dot to indicate active indexing. Modern macOS (Mojave+) shows only a sparse progress bar when actively searching during indexing -- Apple removed the time estimate, reasoning that SSD storage makes indexing fast enough that an ETA calculation is unnecessary.

Key takeaway: Spotlight relies on a **persistent system-level dot indicator** in the menu bar. It does not show progress percentages because first-time indexing is infrequent and re-indexing is invisible. Users complained about the loss of detailed progress info, suggesting minimal indicators frustrate power users. ([Apple Community](https://discussions.apple.com/thread/255513567), [Macworld](https://www.macworld.com/article/190259/spotindex.html))

### 1.2 Everything (voidtools)

Everything indexes NTFS volumes by reading the Master File Table directly, producing a 2MB index nearly instantly. Its UX approach:

- **Status bar progress**: A progress bar appears in the status bar during initial indexing and rescanning. The status bar shows "Ready" when indexing is complete and "Updating" during incremental updates.
- **Right-click to cancel**: Users can right-click the progress bar to pause or cancel indexing.
- **Percentage display**: Shows `xx%` complete, though the percentage can be inaccurate on first scan because file count is unknown. After the initial scan, re-indexing shows more accurate progress.
- **Real-time updates**: NTFS change journal integration means the index updates in real-time with zero user-visible delay.

Key takeaway: Everything puts index status **in the status bar** as a single source of truth. The status bar always answers "is the index ready?" at a glance. This is the correct placement for gpu-search. ([voidtools forum](https://www.voidtools.com/forum/viewtopic.php?t=8549), [voidtools docs](https://www.voidtools.com/support/everything/using_everything/))

### 1.3 VS Code

VS Code shows indexing status in its status bar and notification system:

- Extensions display "Indexing in progress..." with real-time file count progress.
- The Copilot status dashboard shows local index status and indexing type.
- Some extensions offer configurable notifications: toast vs. status bar display.
- Large workspaces that take minutes to index prompted [feature requests for better status indicators](https://github.com/microsoft/vscode-cpptools/issues/3813), showing that silent indexing frustrates users.

Key takeaway: VS Code relies on **status bar items** for persistent index state and **toast notifications** for transient events (index complete, errors). The notification approach is wrong for a TUI (no toast system), but the status bar pattern maps directly. ([VS Code UX Guidelines](https://code.visualstudio.com/api/ux-guidelines/status-bar))

### 1.4 Sublime Text

Sublime Text shows indexing progress in the status bar during file indexing. The `Help > Indexing Status` menu entry reveals a detailed window with progress bar and log messages. For large projects, indexing takes only a few seconds and is described as "unobtrusive."

Key takeaway: Sublime Text uses a **two-tier approach** -- brief status bar indicator for normal operation, detailed status window accessible on demand. This maps well to gpu-search: status bar for at-a-glance state, with verbose detail available if the user wants it. ([Sublime Text Blog](https://www.sublimetext.com/blog/articles/file-indexing), [Sublime Text Docs](https://www.sublimetext.com/docs/indexing.html))

### 1.5 TUI Progress Best Practices

Research on CLI/TUI progress patterns identifies three core approaches: spinners, X-of-Y counters, and progress bars. Key principles:

- **Never leave the user staring at a blank screen.** If an operation takes time, show meaningful status updates.
- **Tick the spinner on real progress events**, not on a timer. This shows the process is actively working, not just alive.
- **Show step counts instead of percentages** when the total is unknown. "42,000 files indexed" is more informative than a stuck "12%" bar.
- **The progress bar should never stop moving.** A frozen bar suggests the application has hung.

([Evil Martians CLI UX](https://evilmartians.com/chronicles/cli-ux-best-practices-3-patterns-for-improving-progress-displays), [Usersnap Progress Indicators](https://usersnap.com/blog/progress-indicators/))

---

## 2. Index Lifecycle States

The persistent file index has a well-defined state machine. Every UX decision must map to one of these states.

### 2.1 State Machine

```
                  +-----------+
     App launch   |           |
    (no index) -->| BUILDING  |----> index file created ----> READY
                  |           |
                  +-----------+
                       ^
                       | corrupt/missing
                       |
  +-------+       +---------+        +----------+
  | READY |<------| LOADING |<-------| App boot |
  +-------+       +---------+        +----------+
       |               |
       |               +---> load failed ---> BUILDING
       |
       +----> FSEvent ----> UPDATING ----> READY
       |
       +----> age > max_age ----> STALE (visual hint only, still usable)
       |
       +----> permissions error / disk full ----> ERROR (degraded, fallback to walk)
```

### 2.2 State Definitions

| State | Description | Duration | User Impact |
|-------|-------------|----------|-------------|
| **LOADING** | App startup, mmap-ing existing index from `~/.gpu-search/index/` | <1ms (mmap) | None -- imperceptible |
| **BUILDING** | First run or index corrupt. Full filesystem scan from `/` | 30-120s depending on disk | Search works via walk fallback. Index build is background-only. |
| **READY** | Index loaded and current. FSEvents watcher active. | Persistent | Full speed search via index |
| **UPDATING** | FSEvents triggered re-scan. Index still usable during update. | 1-5s typical | Search uses current index; new index replaces atomically on completion |
| **STALE** | Index exists but `saved_at` exceeds `DEFAULT_MAX_AGE` (1 hour) | Until next FSEvent update | Search still works; visual hint that results may be incomplete |
| **ERROR** | Index corrupt, permissions denied, disk full | Until resolved | Fallback to `walk_and_filter()`. Show error in status bar. |

---

## 3. Index Status Indicators

### 3.1 Design Principle: Status Bar as Single Source of Truth

Following Everything, VS Code, and Sublime Text, the index status belongs **in the existing status bar**. It occupies a dedicated segment on the left side, before the search-specific segments. The status bar is always visible, always current, and does not require user action to check.

The status bar currently renders:
```
{Searching... | } {count} matches {in X.Xms} | {root} | {filters}
```

With the index status prepended, it becomes:
```
{index_status} | {Searching... | } {count} matches {in X.Xms} | / | {filters}
```

Note: The root is always `/` now (hardcoded), so the root segment becomes static.

### 3.2 Index Status Segment Design

The index status segment uses a compact format with color-coded state:

| State | Text | Color | Icon/Symbol |
|-------|------|-------|-------------|
| LOADING | `Index: loading` | TEXT_MUTED | -- |
| BUILDING | `Indexing: 42,000 files` | ACCENT | -- |
| READY | `Index: 1.3M files` | SUCCESS (#9ECE6A) | -- |
| UPDATING | `Index: updating` | ACCENT | -- |
| STALE | `Index: 1.3M files (stale)` | ACCENT | -- |
| ERROR | `Index: error` | ERROR (#F7768E) | -- |

The READY state is the steady-state that users see 99% of the time. It shows the file count for confidence ("yes, the index covers the full system"). The SUCCESS color (#9ECE6A, Tokyo Night green) provides a subtle positive signal without demanding attention.

### 3.3 Wireframe: Status Bar with Index Status

**Steady state (index ready, search complete):**
```
Index: 1.3M files | 42 matches in 0.8ms | / | .rs
```

**During initial build (searching simultaneously via walk fallback):**
```
Indexing: 142,000 files | Searching... | 7 matches | 2.1s | / | .rs
```

**During initial build (no active search):**
```
Indexing: 142,000 files | 0 matches | /
```

**Index updating via FSEvents:**
```
Index: updating | 42 matches in 0.8ms | / | .rs
```

**Error state:**
```
Index: error | Searching... | 3 matches | 5.2s | /
```

### 3.4 Implementation: StatusBar Struct Extension

The existing `StatusBar` struct in `status_bar.rs` gains a new field:

```rust
pub enum IndexState {
    Loading,
    Building { files_indexed: usize },
    Ready { file_count: usize },
    Updating,
    Stale { file_count: usize },
    Error { message: String },
}

pub struct StatusBar {
    // ... existing fields ...
    pub index_state: IndexState,
}
```

The `render()` method prepends the index status segment before the existing search status segments.

---

## 4. First-Run Experience

### 4.1 Problem

On first launch, no index file exists at `~/.gpu-search/index/`. The `SharedIndexManager::load()` returns `SharedIndexError::NotFound`. The app must build a full index from `/`, which takes 30-120 seconds for a typical macOS system with 1-2M files.

During this time, the user must not be blocked from searching.

### 4.2 Design: Non-Blocking Build with Progressive Feedback

The first-run experience follows the "progressive enhancement" principle. Search works immediately via the existing `walk_and_filter()` pipeline. The index build runs entirely in the background. The user sees:

**Frame 1 (app opens):**
```
+-- gpu-search ------------------------------------------ 720x400 --+
| [Search]  [___search input________________________________] [Gear] |
|--------------------------------------------------------------------|
|                                                                    |
|                  Type to search files and content                   |
|                                                                    |
|--------------------------------------------------------------------|
| Indexing: 0 files | 0 matches | /                                  |
+--------------------------------------------------------------------+
```

**After 5 seconds (user has not typed):**
```
+-- gpu-search ------------------------------------------ 720x400 --+
| [Search]  [___search input________________________________] [Gear] |
|--------------------------------------------------------------------|
|                                                                    |
|                  Type to search files and content                   |
|                                                                    |
|--------------------------------------------------------------------|
| Indexing: 84,000 files | 0 matches | /                             |
+--------------------------------------------------------------------+
```

**User types a query during indexing:**
```
+-- gpu-search ------------------------------------------ 720x400 --+
| [Search]  [orchestrator_______________________________]    [Gear]  |
|--------------------------------------------------------------------|
| FILENAME MATCHES (2)                                               |
|   orchestrator.rs        src/search/                               |
|   orchestrator.md        docs/                                     |
|                                                                    |
| CONTENT MATCHES (5 in 2 files)                                     |
|   orchestrator.rs  -- 3 matches                                    |
|     :142  let orchestrator = SearchOrchestrator::new(...);         |
|     ...                                                            |
|--------------------------------------------------------------------|
| Indexing: 420,000 files | Searching... | 7 matches | 3.1s | /     |
+--------------------------------------------------------------------+
```

The search uses `walk_and_filter()` (the existing 29-second pipeline) while the index builds in parallel. This is slower than indexed search but functional. The user gets results -- they are not blocked.

**Index build completes (user may or may not be searching):**
```
| Index: 1,342,000 files | 42 matches in 0.8ms | / | .rs            |
```

The status bar transitions from `Indexing: N files` (ACCENT) to `Index: N files` (SUCCESS). No toast, no modal, no interruption. The next search the user performs will automatically use the index and be dramatically faster.

### 4.3 Design Decision: No Time Estimate

Following Spotlight's modern approach and the TUI best practices research, the first-run experience does **not** show an estimated time remaining. Reasons:

1. **Total file count is unknown** until the scan completes. Percentage-based progress would be inaccurate (Everything has the same problem on first scan).
2. **Monotonically increasing file count is more honest** than a potentially stuck percentage bar. "420,000 files" is concrete -- the user can see it ticking up and know the system is working.
3. **The user is not waiting.** Search works immediately via walk fallback. The index build is a background optimization, not a blocking operation.

### 4.4 Design Decision: No Dismissible Banner

Some apps show a first-run banner like "Building search index for the first time. This may take a few minutes." This is unnecessary for gpu-search because:

1. The status bar already communicates the state clearly via `Indexing: N files`.
2. A banner would consume vertical space in a compact 400px window.
3. The user can search immediately -- there is nothing to "wait for."

If the index build takes an unusually long time (>2 minutes), the file count in the status bar provides implicit reassurance. The user sees `Indexing: 1,200,000 files` and understands the system is processing a large filesystem.

---

## 5. Background Update UX

### 5.1 Problem

After the initial index build, the `IndexWatcher` monitors `/` via FSEvents (macOS `RecommendedWatcher` with 500ms debounce). When files change, the watcher re-scans, rebuilds the `GpuResidentIndex`, and persists it atomically. This happens silently in the background.

Should the user see anything when the index updates?

### 5.2 Design: Subtle Transient Indicator

Background updates should be **nearly invisible** in the steady state. The user should not be distracted by routine filesystem changes. However, the state should be **discoverable** if the user looks.

**During update (1-5 seconds):**
```
Index: updating | 42 matches in 0.8ms | / | .rs
```

The status bar shows `Index: updating` in ACCENT color, replacing the file count temporarily. This is:
- **Non-intrusive**: No animation, no flashing, no modal. Just a text change in the status bar.
- **Informative**: If the user is watching the status bar during a slow update, they see it.
- **Transient**: Reverts to `Index: 1.3M files` in SUCCESS when the update completes.

For typical FSEvents updates (file save triggers re-scan), the update completes in 1-5 seconds. On most frames, the user will not notice the transition because they are focused on the search input or results.

### 5.3 Design Decision: No Notification on Update Completion

Unlike VS Code's toast notifications, gpu-search does not notify the user when a background update completes. The reasoning:

1. TUI apps have no toast system. Printing a transient message would require either a dedicated notification area (wasted space) or interrupting the results display.
2. Background updates are frequent (every file save in monitored directories). Notifying on each would be noise.
3. The status bar file count update (`1,342,000` -> `1,342,001`) serves as a passive confirmation.

### 5.4 Index Freshness During Active Search

If a background index update completes while a search is in progress, the current search should **not** be interrupted. The new index is persisted to disk and will be used by the next search. The in-flight search continues with the previous index state. This prevents result flickering and mid-search data races.

---

## 6. Startup Experience

### 6.1 Problem

On subsequent launches (index already exists), `SharedIndexManager::load()` reads the GSIX binary via mmap in <1ms. The user launches the app and the index is immediately available. How should this be communicated?

### 6.2 Design: Instant Ready State

The status bar shows the READY state immediately on launch:

```
Index: 1,342,000 files | 0 matches | /
```

There is no loading animation, no splash screen, no "Loading index..." transient state. The mmap load is too fast (<1ms) for any visual transition to be perceptible. Attempting to show a "loading" state would require artificial delay, which violates the principle of honest feedback.

### 6.3 Design Decision: No "Last Updated" Timestamp

The status bar does **not** show when the index was last updated (e.g., "Updated 5 min ago"). Reasons:

1. **Space constraint**: The status bar is already dense. Adding a timestamp would push other segments off-screen on narrower terminals.
2. **FSEvents keeps it fresh**: With the watcher active, the index is continuously updated. A timestamp would almost always show "just now" or "seconds ago," providing no useful information.
3. **Stale state covers the edge case**: If the index is genuinely old (>1 hour, FSEvents watcher not running), the STALE state communicates this via `Index: 1.3M files (stale)` in ACCENT color. This is more actionable than a raw timestamp.

### 6.4 Stale Index Detection on Startup

On launch, the app calls `SharedIndexManager::is_stale(root, DEFAULT_MAX_AGE)`. If the index is stale (saved_at > 1 hour ago, or root directory mtime newer than index):

1. Status bar shows `Index: 1.3M files (stale)` in ACCENT color.
2. The FSEvents watcher starts and immediately triggers a background re-scan.
3. Search works using the stale index (results may be incomplete but are still fast).
4. When the re-scan completes, status transitions to `Index: 1.3M files` in SUCCESS.

This is transparent to the user. They get fast (slightly outdated) results immediately, and fresh results within seconds as the background update completes.

---

## 7. Error States

### 7.1 Error Taxonomy

| Error | Cause | Frequency | Severity |
|-------|-------|-----------|----------|
| Index corrupt | Interrupted write, disk error, format version mismatch | Rare | Medium |
| Permissions denied | SIP-protected dirs, unreadable home folders | Every scan (partial) | Low |
| Disk full | Cannot write index to `~/.gpu-search/index/` | Rare | Medium |
| FSEvents failure | Watcher creation fails, path unwatchable | Rare | Low |
| Index too large | >4GB index file on 32-bit offset format | Never (current format handles it) | N/A |

### 7.2 Error UX by Type

#### 7.2.1 Index Corrupt

**Detection**: `SharedIndexManager::load()` returns `SharedIndexError::InvalidFormat` (bad magic, version mismatch, data too short).

**UX Response**:
1. Status bar shows `Index: rebuilding` in ACCENT.
2. The corrupt index file is deleted via `SharedIndexManager::delete()`.
3. A fresh full scan starts in the background.
4. Search falls back to `walk_and_filter()` during rebuild.
5. No error dialog or modal -- the user sees the same experience as first-run.

**Rationale**: Index corruption is recoverable. The user does not need to know the index was corrupt -- they just need search to work. Silent rebuild is the correct response.

#### 7.2.2 Permissions Denied on Some Directories

**Detection**: `FilesystemScanner::scan()` silently skips unreadable entries (`Err(_) => return ignore::WalkState::Continue`). The scanner already handles this gracefully.

**UX Response**: None. Permissions errors on individual directories are expected when scanning from `/` (e.g., `/System`, `/private/var`, SIP-protected paths). The index simply does not contain those paths. This is identical to how Spotlight handles it -- SIP-protected directories are excluded from the index without notification.

If the user searches for a file they know exists but cannot find it, the "no results" state does not explain why. This is acceptable because:
1. System-protected files are rarely searched.
2. Adding a "some directories were inaccessible" warning would appear on every build and become noise.

#### 7.2.3 Disk Full

**Detection**: `SharedIndexManager::save()` returns `SharedIndexError::Io` when writing the index file fails.

**UX Response**:
1. Status bar shows `Index: error` in ERROR color (#F7768E).
2. The error is logged to stderr: `[IndexWatcher] failed to persist index: No space left on device`.
3. Search falls back to `walk_and_filter()` (functional but slow).
4. On the next FSEvents trigger, the watcher retries the save.

**Rationale**: Disk full is rare and external to the app. The status bar error indicator alerts the user without interrupting their workflow. The app degrades gracefully to the walk-based search.

#### 7.2.4 FSEvents Watcher Failure

**Detection**: `IndexWatcher::start()` returns `WatcherError::Notify` or `WatcherError::Io`.

**UX Response**:
1. Status bar shows the index state as READY (if index was loaded) but without live updates.
2. The watcher failure is logged to stderr.
3. The index becomes stale over time (no FSEvents updates), eventually triggering the STALE visual indicator.

**Rationale**: Watcher failure means the index will not auto-update but remains usable. This is a degraded-but-functional state, similar to Spotlight when mds is temporarily unavailable.

### 7.3 Error State Wireframe

```
+-- gpu-search ------------------------------------------ 720x400 --+
| [Search]  [orchestrator_______________________________]    [Gear]  |
|--------------------------------------------------------------------|
|                                                                    |
| FILENAME MATCHES (2)                                               |
|   orchestrator.rs        src/search/                               |
|   orchestrator.md        docs/                                     |
|                                                                    |
|--------------------------------------------------------------------|
| Index: error | Searching... | 7 matches | 5.2s | /                |
+--------------------------------------------------------------------+
```

The `Index: error` segment is in ERROR red. The search still works (via walk fallback). Results are correct but slower. The user can investigate via the stderr log if they choose.

---

## 8. Status Bar Integration

### 8.1 Current Layout

The existing status bar renders a horizontal row of segments separated by `|`:

```rust
// When searching:
"Searching... | {count} matches | {elapsed} | {root} | {filters}"

// When idle:
"{count} matches in {elapsed} | {root} | {filters}"
```

### 8.2 Proposed Layout with Index Status

The index status is prepended as the first segment. The root segment changes from the dynamic `search_root` to the static `/` (since the index always covers root).

**When searching (index ready):**
```
1.3M files | Searching... | 42 matches | 2.1s | / | .rs
```

**When idle (index ready):**
```
1.3M files | 42 matches in 0.8ms | / | .rs
```

**When building index (searching via walk):**
```
Indexing: 420K files | Searching... | 7 matches | 3.1s | /
```

**When building index (idle):**
```
Indexing: 420K files | 0 matches | /
```

**When updating index:**
```
Updating index | 42 matches in 0.8ms | / | .rs
```

**When index error:**
```
Index error | Searching... | 7 matches | 5.2s | /
```

### 8.3 Color Mapping

| Segment | Condition | Color |
|---------|-----------|-------|
| `1.3M files` | Index READY | SUCCESS (#9ECE6A) |
| `Indexing: 420K files` | Index BUILDING | ACCENT (#E0AF68) |
| `Updating index` | Index UPDATING | ACCENT (#E0AF68) |
| `Index error` | Index ERROR | ERROR (#F7768E) |
| `1.3M files (stale)` | Index STALE | ACCENT (#E0AF68) |
| `Searching...` | Active search | ACCENT (#E0AF68) |
| Match count/elapsed | Always | TEXT_PRIMARY / TEXT_MUTED |
| Root `/` | Always | TEXT_PRIMARY |
| Filters | When active | ACCENT (#E0AF68) |

### 8.4 Number Formatting

File counts use abbreviated formatting for readability:
- `< 1,000`: show exact (`842 files`)
- `1,000 - 999,999`: show with K (`420K files`)
- `>= 1,000,000`: show with M (`1.3M files`)

This keeps the segment compact. The exact count is not useful -- `1,342,847` is harder to read than `1.3M` and conveys the same confidence.

### 8.5 Segment Priority on Narrow Displays

If the terminal is narrow enough that segments would overlap, segments are hidden in this priority (last hidden first):

1. Filters (already conditional on `active_filter_count > 0`)
2. Root `/` (static, adds little value)
3. Elapsed time
4. Match count
5. Index status (never hidden -- most important)

This ensures the index status is always visible, even on narrow terminals.

---

## 9. Progressive Enhancement: Search Without Index

### 9.1 Core Principle

The index is a performance optimization, not a functional requirement. Search must work without it. The user should never be told "please wait for the index to build before searching." This matches how macOS Spotlight works -- you can search immediately after a fresh OS install, even while Spotlight indexes in the background.

### 9.2 Fallback Behavior

| Index State | Search Method | User Experience |
|-------------|--------------|-----------------|
| READY | Index lookup + GPU dispatch | Sub-millisecond results |
| BUILDING | `walk_and_filter()` + streaming GPU dispatch | 5-29s progressive results (identical to current behavior) |
| UPDATING | Current index (pre-update) | Sub-millisecond results (possibly missing very recent files) |
| STALE | Stale index | Sub-millisecond results (possibly missing recent files) |
| ERROR | `walk_and_filter()` fallback | 5-29s progressive results |

### 9.3 Transition Handling

When the index build completes during a session, the next search automatically uses it. There is no "switch over" moment -- the orchestrator simply checks `SharedIndexManager::load()` at the start of each search. If the index is available and fresh, it uses it. If not, it falls back to walking.

This means the user may notice a dramatic speed improvement mid-session: their first few searches take 3-5 seconds (walk-based), then suddenly searches return in <1ms. The status bar transition from `Indexing: N files` to `1.3M files` in SUCCESS green explains this change without requiring a tooltip or notification.

### 9.4 No Explicit Mode Toggle

There is no user-facing toggle to "enable" or "disable" the index. The index is always built, always updated, always used when available. The only user action that could affect the index is deleting `~/.gpu-search/index/`, which triggers a rebuild on next launch. This simplicity is intentional -- the index should be invisible infrastructure, not a feature the user manages.

---

## 10. First-Run Progress Detail Design

### 10.1 Progress Counter Update Frequency

The `FilesystemScanner::scan()` runs on a background thread. To feed the status bar with a live file count during the initial build, the scanner needs to report progress periodically. Two options:

**Option A: Atomic counter** -- The scanner increments an `AtomicUsize` for each file processed. The UI reads it every frame (~16ms at 60fps). This adds negligible overhead to the scanner and provides smooth, real-time progress.

**Option B: Channel-based batched updates** -- The scanner sends periodic count updates via a channel (every 1,000 files or every 100ms). This is coarser but avoids atomic contention on the hot path.

**Recommendation**: Option A (atomic counter). The scanner's hot path is filesystem I/O, not CPU-bound. An atomic increment per file is ~1ns overhead on top of ~1-10us per file stat, making it negligible. The UI gets frame-accurate progress without channel overhead.

### 10.2 Progress Display During Build

The file count in the status bar updates every frame during indexing. To avoid visual jitter, the count is formatted with abbreviated units (K/M) and only the significant digits change:

```
Frame 1:    Indexing: 0 files
Frame 100:  Indexing: 12K files
Frame 500:  Indexing: 84K files
Frame 1000: Indexing: 420K files
Frame 2000: Indexing: 1.2M files
Complete:   1.3M files
```

The transition from `Indexing: N` to the final `N files` state is immediate on build completion. No fade, no animation -- the color changes from ACCENT to SUCCESS and the "Indexing:" prefix drops.

### 10.3 Request Repaint During Build

While the index is building, the app must call `ctx.request_repaint()` every frame to keep the progress counter updating. This is the same pattern already used for the `is_searching` state. The condition becomes:

```rust
if self.is_searching || matches!(self.index_state, IndexState::Building { .. }) {
    ctx.request_repaint();
}
```

---

## 11. Interaction During Index Build

### 11.1 Search Input

The search input is fully interactive during index build. Typing, debounce, and dispatch all work normally. The search dispatches to `walk_and_filter()` instead of the index, but the user-facing behavior is identical to the current app (progressive streaming results).

### 11.2 Filter Bar

Filters work normally. They are applied to the search request regardless of whether the index or walk is used.

### 11.3 Keyboard Navigation

All keybinds work normally. Results from walk-based search are navigable, selectable, and actionable (open file, open in editor, copy path).

### 11.4 No Disabled States

Nothing in the UI is disabled during index build. There are no grayed-out elements, no "please wait" messages, no blocked interactions. The app is fully functional from the moment it opens. The index is purely additive performance.

---

## 12. Visual Summary: State Transitions

```
App Launch (first time)
  |
  v
+------------------------------------------------------------------+
| Status bar: Indexing: 0 files | 0 matches | /                    |
| Center:     "Type to search files and content"                   |
+------------------------------------------------------------------+
  |
  | User types query (search via walk_and_filter)
  v
+------------------------------------------------------------------+
| Status bar: Indexing: 420K files | Searching... | 7 matches | /  |
| Results:    Progressive streaming results                        |
+------------------------------------------------------------------+
  |
  | Background: index build completes
  v
+------------------------------------------------------------------+
| Status bar: 1.3M files | 42 matches in 0.8ms | / | .rs          |
| Results:    Search results (now index-backed, sub-ms)            |
+------------------------------------------------------------------+
  |
  | FSEvent: file saved
  v
+------------------------------------------------------------------+
| Status bar: Updating index | 42 matches in 0.8ms | / | .rs      |
+------------------------------------------------------------------+
  |
  | Background: update completes (1-5s)
  v
+------------------------------------------------------------------+
| Status bar: 1.3M files | 42 matches in 0.8ms | / | .rs          |
+------------------------------------------------------------------+

App Launch (subsequent, index exists)
  |
  v
+------------------------------------------------------------------+
| Status bar: 1.3M files | 0 matches | /                          |
| Center:     "Type to search files and content"                   |
+------------------------------------------------------------------+
```

---

## 13. Accessibility Considerations

### 13.1 Color Is Not the Only Indicator

The index status uses both color AND text to communicate state:
- BUILDING: ACCENT color + "Indexing:" prefix
- READY: SUCCESS color + file count (no prefix)
- ERROR: ERROR color + "Index error" text

A colorblind user can distinguish states by reading the text alone.

### 13.2 WCAG Contrast

All index status text uses colors already verified for WCAG AA compliance in `theme.rs`:
- SUCCESS (#9ECE6A) on BG_BASE (#1A1B26): contrast ratio ~6.2:1 (passes AA)
- ACCENT (#E0AF68) on BG_BASE (#1A1B26): contrast ratio ~7.5:1 (passes AA)
- ERROR (#F7768E) on BG_BASE (#1A1B26): contrast ratio ~5.9:1 (passes AA)

### 13.3 Screen Reader Semantics

egui's `colored_label` does not expose semantic roles to screen readers. If screen reader support is added in the future, the index status segment should be marked as a live region (`aria-live="polite"`) so state changes are announced without interrupting the user.

---

## 14. Implementation Checklist

| ID | Component | Effort | Files | Dependencies |
|----|-----------|--------|-------|--------------|
| IX-UX-1 | Add `IndexState` enum to `status_bar.rs` | S | `status_bar.rs` | None |
| IX-UX-2 | Render index status as first segment in `StatusBar::render()` | S | `status_bar.rs` | IX-UX-1 |
| IX-UX-3 | Add `index_state` field to `GpuSearchApp` | S | `app.rs` | IX-UX-1 |
| IX-UX-4 | Wire `AtomicUsize` progress counter from scanner to UI | M | `scanner.rs`, `app.rs` | IX-UX-3 |
| IX-UX-5 | Poll index state in `eframe::App::update()` | S | `app.rs` | IX-UX-3, IX-UX-4 |
| IX-UX-6 | `request_repaint()` during BUILDING state | S | `app.rs` | IX-UX-5 |
| IX-UX-7 | Format file count with K/M abbreviation | S | `status_bar.rs` | IX-UX-2 |
| IX-UX-8 | Handle STALE detection on startup | S | `app.rs` | IX-UX-3 |
| IX-UX-9 | Handle ERROR state from `SharedIndexManager::load()` | S | `app.rs` | IX-UX-3 |
| IX-UX-10 | Change `search_root` display to static `/` | S | `status_bar.rs` | None |
| IX-UX-11 | Unit tests: status bar renders each IndexState variant | M | `status_bar.rs` tests | IX-UX-2 |
| IX-UX-12 | Integration test: status transitions during simulated build | M | integration tests | IX-UX-5 |

Estimated total UX effort: 3-4 engineering days (small to medium tasks, no major layout changes).

---

## 15. Non-Goals

- **Progress bar widget**: A dedicated thin progress bar (as sketched in the previous UX spec wireframes for search progress) is not needed for index status. The file count in the status bar provides sufficient feedback. A progress bar requires knowing the total, which is unknown during first build.
- **Index settings UI**: No UI for configuring index location, exclude patterns, or max age. These are compile-time constants in `shared_index.rs`.
- **Index rebuild button**: No manual "rebuild index" action. The FSEvents watcher handles all updates. If the user deletes the index file, it rebuilds automatically on next launch.
- **Multiple index profiles**: The app always indexes from `/`. No per-project or per-directory index selection.
- **Index size display**: The index file size (e.g., "342MB on disk") is not shown. The file count is the user-relevant metric.
- **Notification system**: No toast, banner, or popup for index events. The status bar is sufficient.
