//! TUI dashboard module for gpu-query.
//!
//! Provides an interactive terminal dashboard with gradient-colored rendering,
//! query editor, results display, and GPU metrics. Built on ratatui + crossterm.
//!
//! Layout composition is handled by `ui::render_ui` with responsive breakpoints:
//! - >= 120 cols: full three-panel (catalog | editor+results | GPU dashboard)
//! - 80-119 cols: two-panel (editor+results | GPU dashboard)
//! - < 80 cols: minimal REPL (editor + results)

pub mod app;
pub mod autocomplete;
pub mod catalog;
pub mod dashboard;
pub mod editor;
pub mod event;
pub mod gradient;
pub mod results;
pub mod themes;
pub mod ui;

use app::{AppState, EngineStatus, QueryCompatibility, QueryState, SqlValidity};
use event::{handle_key, poll_event, AppEvent};

use crossterm::{
    event::{KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags},
    execute,
    terminal::{
        disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;
use std::path::PathBuf;
use std::time::Duration;

/// Run the interactive TUI dashboard.
/// This takes over the terminal until the user quits (q/Ctrl+C).
pub fn run_dashboard(data_dir: PathBuf, theme_name: &str) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;

    // Enable Kitty keyboard protocol for terminals that support it.
    // This makes Ctrl+Enter work properly in kitty, WezTerm, ghostty, foot.
    let has_keyboard_enhancement = supports_keyboard_enhancement().unwrap_or(false);
    if has_keyboard_enhancement {
        execute!(
            stdout,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
    }

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Initialize app state
    let mut app = AppState::new(data_dir.clone(), theme_name);

    // Scan for tables in data directory and populate catalog tree
    if let Ok(catalog_entries) = app.catalog_cache.get_or_refresh().map(|s| s.to_vec()) {
        app.tables = catalog_entries.iter().map(|e| e.name.clone()).collect();

        let tree_entries: Vec<catalog::CatalogEntry> = catalog_entries
            .iter()
            .map(catalog::CatalogEntry::from_table_entry)
            .collect();

        let dir_name = data_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("data")
            .to_string();

        app.catalog_state.load(tree_entries, dir_name);
    }
    app.status_message = format!(
        "gpu-query dashboard | {} tables loaded | Type SQL + F5 to execute{}",
        app.tables.len(),
        if has_keyboard_enhancement {
            " (Ctrl+Enter also works)"
        } else {
            ""
        }
    );

    // Main event loop (~60fps)
    let tick_rate = Duration::from_millis(app.tick_rate_ms);

    loop {
        // Render using the responsive layout system
        let metrics_snapshot = app.gpu_metrics.clone();
        terminal.draw(|f| ui::render_ui(f, &mut app, &metrics_snapshot))?;

        // Poll autonomous executor for ready results on every iteration
        poll_autonomous_result(&mut app);

        // Handle events
        match poll_event(tick_rate)? {
            AppEvent::Quit => break,
            AppEvent::Key(key) => {
                // Track whether this is an editor-modifying key for live mode
                let text_before = app.editor_state.text();
                handle_key(&key, &mut app);
                let text_after = app.editor_state.text();
                let text_changed = text_before != text_after;

                // In live mode, submit autonomous query on every keystroke that changes text
                if text_changed && app.live_mode {
                    handle_live_mode_keystroke(&mut app);
                }
            }
            AppEvent::Resize(_, _) => {
                // ratatui handles resize automatically; layout recalculates
            }
            AppEvent::Tick => {
                app.tick();
            }
        }

        if !app.running {
            break;
        }
    }

    // Restore terminal
    if has_keyboard_enhancement {
        execute!(terminal.backend_mut(), PopKeyboardEnhancementFlags)?;
    }
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

/// Poll the autonomous executor for ready results and update app state.
///
/// Called on every tick/iteration of the event loop. If the autonomous executor
/// has a result ready (GPU set ready_flag=1), reads it, converts to `QueryResult`,
/// and updates the app state with the result and latency.
fn poll_autonomous_result(app: &mut AppState) {
    // Only poll if we're in AutonomousSubmitted state
    if app.query_state != QueryState::AutonomousSubmitted {
        return;
    }

    let is_ready = match &app.autonomous_executor {
        Some(executor) => executor.poll_ready(),
        None => false,
    };

    if !is_ready {
        return;
    }

    // Read the result from unified memory
    let executor = app.autonomous_executor.as_ref().unwrap();
    let output = executor.read_result();

    // Convert OutputBuffer to QueryResult
    let result = convert_autonomous_output(&output, app);

    // Update latency tracking
    let latency_us = output.latency_ns / 1000; // ns -> us
    app.last_autonomous_us = Some(latency_us);
    app.last_exec_us = Some(latency_us);

    // Update autonomous stats
    app.autonomous_stats.total_queries += 1;
    if latency_us < 1000 {
        app.autonomous_stats.consecutive_sub_1ms += 1;
    } else {
        app.autonomous_stats.consecutive_sub_1ms = 0;
    }

    // Update engine status
    app.engine_status = EngineStatus::Live;

    // Update status message
    app.status_message = format!(
        "Autonomous query: {} rows in {:.1}us | [auto]",
        result.row_count,
        latency_us as f64,
    );

    app.set_result(result);
}

/// Convert an `OutputBuffer` from the autonomous executor into a `QueryResult`.
///
/// Reads the aggregate results from the output buffer and formats them into
/// column headers and string rows suitable for the TUI results panel.
fn convert_autonomous_output(
    output: &crate::gpu::autonomous::types::OutputBuffer,
    app: &AppState,
) -> crate::gpu::executor::QueryResult {
    let row_count = output.result_row_count as usize;
    let col_count = output.result_col_count as usize;

    // Build column headers from the cached plan's aggregate functions
    let mut columns: Vec<String> = Vec::new();
    let mut agg_funcs: Vec<(u32, u32)> = Vec::new(); // (agg_func, column_type)

    if let Some(plan) = &app.cached_plan {
        // Extract aggregate info from plan
        if let crate::sql::physical_plan::PhysicalPlan::GpuAggregate {
            functions,
            group_by,
            ..
        } = plan
        {
            // Add GROUP BY column header first
            if !group_by.is_empty() {
                columns.push(group_by[0].clone());
            }
            // Add aggregate column headers
            for (func, col_name) in functions {
                let label = match func {
                    crate::sql::types::AggFunc::Count => format!("COUNT({})", col_name),
                    crate::sql::types::AggFunc::Sum => format!("SUM({})", col_name),
                    crate::sql::types::AggFunc::Avg => format!("AVG({})", col_name),
                    crate::sql::types::AggFunc::Min => format!("MIN({})", col_name),
                    crate::sql::types::AggFunc::Max => format!("MAX({})", col_name),
                };
                columns.push(label);
                // Track agg func code and column type for formatting
                let func_code = func.to_gpu_code();
                // Determine column type from plan (0=INT64, 1=FLOAT32)
                let col_type = if col_name == "*" {
                    0u32 // COUNT(*) always integer
                } else {
                    0u32 // Default to int, sufficient for display
                };
                agg_funcs.push((func_code, col_type));
            }
        }
    }

    // Fallback: generate generic column headers if plan extraction failed
    if columns.is_empty() {
        for i in 0..col_count {
            columns.push(format!("col_{}", i));
            agg_funcs.push((0, 0));
        }
    }

    // Build rows from output buffer
    let mut rows: Vec<Vec<String>> = Vec::new();
    let has_group_by = columns.len() > agg_funcs.len();

    for g in 0..row_count {
        let mut row: Vec<String> = Vec::new();

        // Add GROUP BY key value if present
        if has_group_by {
            row.push(output.group_keys[g].to_string());
        }

        // Add aggregate values
        for (a, (func_code, _col_type)) in agg_funcs.iter().enumerate() {
            let agg_res = &output.agg_results[g][a];
            let formatted = match *func_code {
                0 => agg_res.value_int.to_string(),          // COUNT
                1 => agg_res.value_int.to_string(),          // SUM (int)
                2 => format!("{:.2}", agg_res.value_float),  // AVG
                3 => agg_res.value_int.to_string(),          // MIN
                4 => agg_res.value_int.to_string(),          // MAX
                _ => agg_res.value_int.to_string(),
            };
            row.push(formatted);
        }

        rows.push(row);
    }

    let result_row_count = rows.len();
    crate::gpu::executor::QueryResult {
        columns,
        rows,
        row_count: result_row_count,
    }
}

/// Handle a keystroke in live mode: update SQL validity, check compatibility,
/// and submit the query to the appropriate executor.
///
/// Called on every keystroke that modifies editor text when live mode is active.
/// - If SQL is valid and autonomous-compatible: submit via autonomous executor
/// - If SQL is valid but fallback-only: use standard `execute_editor_query()`
/// - If SQL is empty/incomplete/invalid: do nothing (wait for more input)
fn handle_live_mode_keystroke(app: &mut AppState) {
    // 1. Update SQL validity and compatibility
    app::update_sql_validity(app);
    app::update_query_compatibility(app);

    // 2. Only proceed if SQL is valid
    if app.sql_validity != SqlValidity::Valid {
        return;
    }

    // 3. Route based on compatibility
    match app.query_compatibility {
        QueryCompatibility::Autonomous => {
            // Submit to autonomous executor (0ms debounce -- every keystroke)
            submit_autonomous_live_query(app);
        }
        QueryCompatibility::Fallback => {
            // Use standard executor for unsupported patterns
            app.engine_status = EngineStatus::Fallback;
            let _ = ui::execute_editor_query(app);
        }
        QueryCompatibility::Unknown | QueryCompatibility::Invalid => {
            // Not ready to execute
        }
    }
}

/// Submit the current SQL to the autonomous executor for live mode execution.
///
/// Extracts the plan, resolves the table name from the plan, and calls
/// `submit_query()` on the autonomous executor. Sets query state to
/// `AutonomousSubmitted` so the event loop polls for the result.
fn submit_autonomous_live_query(app: &mut AppState) {
    // Need both a cached plan and an autonomous executor
    let plan = match &app.cached_plan {
        Some(p) => p.clone(),
        None => return,
    };

    // Extract table name from plan
    let table_name = extract_table_name(&plan);

    // Get schema from the executor's resident table
    let schema: Vec<crate::gpu::autonomous::loader::ColumnInfo> = {
        match &app.autonomous_executor {
            Some(_) => {
                // Build schema from the cached plan -- use column info from resident table
                // For now, use the plan to extract needed info
                extract_schema_from_plan(&plan)
            }
            None => return, // No executor, can't submit
        }
    };

    if schema.is_empty() {
        return;
    }

    // Submit query to autonomous executor
    let executor = match &mut app.autonomous_executor {
        Some(e) => e,
        None => return,
    };

    match executor.submit_query(&plan, &schema, &table_name) {
        Ok(_seq_id) => {
            app.query_state = QueryState::AutonomousSubmitted;
            app.engine_status = EngineStatus::Live;
        }
        Err(e) => {
            // Submission failed -- fall back to standard executor
            app.engine_status = EngineStatus::Fallback;
            app.status_message = format!("Autonomous submit failed: {}", e);
        }
    }
}

/// Extract the table name from a physical plan by walking down to the GpuScan node.
fn extract_table_name(plan: &crate::sql::physical_plan::PhysicalPlan) -> String {
    match plan {
        crate::sql::physical_plan::PhysicalPlan::GpuScan { table, .. } => table.clone(),
        crate::sql::physical_plan::PhysicalPlan::GpuFilter { input, .. } => {
            extract_table_name(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuCompoundFilter { left, .. } => {
            extract_table_name(left)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuAggregate { input, .. } => {
            extract_table_name(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuSort { input, .. } => {
            extract_table_name(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuLimit { input, .. } => {
            extract_table_name(input)
        }
    }
}

/// Extract a basic schema from a physical plan's column references.
///
/// This provides column names and inferred types from the plan's scan columns.
/// In production, this would come from the ResidentTable's stored schema.
fn extract_schema_from_plan(
    plan: &crate::sql::physical_plan::PhysicalPlan,
) -> Vec<crate::gpu::autonomous::loader::ColumnInfo> {
    match plan {
        crate::sql::physical_plan::PhysicalPlan::GpuScan { columns, .. } => columns
            .iter()
            .map(|c| crate::gpu::autonomous::loader::ColumnInfo {
                name: c.clone(),
                data_type: crate::storage::schema::DataType::Int64, // default inference
            })
            .collect(),
        crate::sql::physical_plan::PhysicalPlan::GpuFilter { input, .. } => {
            extract_schema_from_plan(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuCompoundFilter { left, .. } => {
            extract_schema_from_plan(left)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuAggregate { input, .. } => {
            extract_schema_from_plan(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuSort { input, .. } => {
            extract_schema_from_plan(input)
        }
        crate::sql::physical_plan::PhysicalPlan::GpuLimit { input, .. } => {
            extract_schema_from_plan(input)
        }
    }
}
