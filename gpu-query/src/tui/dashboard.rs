//! GPU status dashboard panel.
//!
//! Renders GPU utilization bar (gradient green->yellow->red), memory bar,
//! scan throughput display, and sparkline history of recent queries.
//! Uses ratatui Sparkline widget for history charts.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Sparkline},
    Frame,
};

use crate::gpu::metrics::{format_bytes, format_throughput, format_time, GpuMetricsCollector};
use crate::tui::app::{AppState, EngineStatus};
use crate::tui::gradient::Gradient;
use crate::tui::themes::Theme;

/// Render the GPU status dashboard panel.
///
/// Shows: utilization bar, memory bar, throughput, query count, and sparkline history.
pub fn render_gpu_dashboard(
    f: &mut Frame,
    area: Rect,
    metrics: &GpuMetricsCollector,
    theme: &Theme,
    app: &AppState,
) {
    let block = Block::default()
        .title(Span::styled(
            " GPU Status ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(theme.border_style);

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 3 || inner.width < 10 {
        return; // Too small to render anything useful
    }

    // Layout: utilization bar + memory bar + stats + sparkline + engine section
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // GPU utilization bar
            Constraint::Length(1), // Memory bar
            Constraint::Length(1), // Throughput + query count
            Constraint::Length(1), // Average stats
            Constraint::Length(1), // Spacer
            Constraint::Length(1), // Engine header
            Constraint::Length(1), // Engine latency
            Constraint::Length(1), // Engine queries + JIT
            Constraint::Length(1), // Spacer
            Constraint::Min(2),    // Sparkline history
        ])
        .split(inner);

    // 1. GPU utilization bar (gradient green->yellow->red)
    render_utilization_bar(f, chunks[0], metrics, &theme.thermal, "GPU");

    // 2. Memory bar
    render_utilization_bar(f, chunks[1], metrics, &theme.glow, "MEM");

    // 3. Throughput + query count line
    render_stats_line(f, chunks[2], metrics, theme);

    // 4. Average stats
    render_avg_line(f, chunks[3], metrics, theme);

    // 5-7. Engine status section
    render_engine_status(f, chunks[5], app, theme);
    render_engine_latency(f, chunks[6], app, theme);
    render_engine_jit_stats(f, chunks[7], app, theme);

    // 8. Sparkline history (throughput over time)
    if chunks[9].height >= 2 {
        render_sparkline(f, chunks[9], metrics, theme);
    }
}

/// Render a utilization bar with gradient coloring.
///
/// Format: `LABEL [=========>        ] 45%`
fn render_utilization_bar(
    f: &mut Frame,
    area: Rect,
    metrics: &GpuMetricsCollector,
    gradient: &Gradient,
    label: &str,
) {
    if area.width < 15 {
        return;
    }

    let util = match label {
        "GPU" => metrics.utilization_estimate(),
        "MEM" => metrics.memory_utilization(),
        _ => 0.0,
    };

    let label_width = 5; // "GPU: " or "MEM: "
    let pct_width = 5; // " 100%"
    let bar_width = area.width.saturating_sub(label_width + pct_width) as usize;

    if bar_width < 3 {
        return;
    }

    let filled = (util * bar_width as f32) as usize;
    let empty = bar_width.saturating_sub(filled);

    // Build gradient-colored filled portion
    let mut spans = Vec::with_capacity(bar_width + 4);

    // Label
    spans.push(Span::styled(
        format!("{}: ", label),
        Style::default().fg(ratatui::style::Color::Rgb(180, 180, 190)),
    ));

    // Filled bars with gradient color per character
    for i in 0..filled {
        let t = if filled > 1 {
            i as f32 / (filled - 1) as f32
        } else {
            util
        };
        let color = gradient.at(t * util);
        spans.push(Span::styled(
            "\u{2588}", // Full block character
            Style::default().fg(color),
        ));
    }

    // Empty portion
    if empty > 0 {
        spans.push(Span::styled(
            "\u{2591}".repeat(empty), // Light shade
            Style::default().fg(ratatui::style::Color::Rgb(50, 50, 60)),
        ));
    }

    // Percentage
    let pct_color = gradient.at(util);
    spans.push(Span::styled(
        format!(" {:3.0}%", util * 100.0),
        Style::default().fg(pct_color).add_modifier(Modifier::BOLD),
    ));

    let line = Line::from(spans);
    f.render_widget(Paragraph::new(line), area);
}

/// Render the throughput and query count stats line.
fn render_stats_line(f: &mut Frame, area: Rect, metrics: &GpuMetricsCollector, theme: &Theme) {
    let throughput_str = format_throughput(metrics.latest.scan_throughput_gbps);
    let mem_str = format_bytes(metrics.latest.memory_used_bytes);
    let time_str = format_time(metrics.latest.gpu_time_ms);

    let line = Line::from(vec![
        Span::styled(
            throughput_str.to_string(),
            Style::default().fg(theme.accent),
        ),
        Span::styled(" | ", Style::default().fg(theme.muted)),
        Span::styled(time_str.to_string(), Style::default().fg(theme.text)),
        Span::styled(" | ", Style::default().fg(theme.muted)),
        Span::styled(mem_str.to_string(), Style::default().fg(theme.text)),
        Span::styled(" | ", Style::default().fg(theme.muted)),
        Span::styled(
            format!("Q#{}", metrics.query_count),
            Style::default().fg(theme.muted),
        ),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

/// Render average statistics line.
fn render_avg_line(f: &mut Frame, area: Rect, metrics: &GpuMetricsCollector, theme: &Theme) {
    if metrics.query_count == 0 {
        let line = Line::from(Span::styled(
            "No queries yet",
            Style::default().fg(theme.muted),
        ));
        f.render_widget(Paragraph::new(line), area);
        return;
    }

    let avg_time = format_time(metrics.avg_time_ms());
    let avg_throughput = format_throughput(metrics.avg_throughput_gbps());
    let peak_mem = format_bytes(metrics.peak_memory_bytes);

    let line = Line::from(vec![
        Span::styled("avg: ", Style::default().fg(theme.muted)),
        Span::styled(avg_time.to_string(), Style::default().fg(theme.text)),
        Span::styled(" | ", Style::default().fg(theme.muted)),
        Span::styled(avg_throughput.to_string(), Style::default().fg(theme.text)),
        Span::styled(" | peak: ", Style::default().fg(theme.muted)),
        Span::styled(peak_mem.to_string(), Style::default().fg(theme.text)),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

/// Render engine status badge line.
///
/// Format: `ENGINE [LIVE]` or `ENGINE [OFF]` etc.
fn render_engine_status(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let (badge_text, badge_color) = match app.engine_status {
        EngineStatus::Live => ("[LIVE]", ratatui::style::Color::Green),
        EngineStatus::WarmingUp => ("[WARMING]", ratatui::style::Color::Yellow),
        EngineStatus::Compiling => ("[COMPILING]", ratatui::style::Color::Yellow),
        EngineStatus::Idle => ("[IDLE]", ratatui::style::Color::Rgb(100, 100, 120)),
        EngineStatus::Fallback => ("[FALLBACK]", ratatui::style::Color::Rgb(255, 165, 0)),
        EngineStatus::Off => ("[OFF]", ratatui::style::Color::Rgb(80, 80, 90)),
        EngineStatus::Error => ("[ERROR]", ratatui::style::Color::Red),
    };

    let line = Line::from(vec![
        Span::styled(
            "ENGINE ",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            badge_text,
            Style::default()
                .fg(badge_color)
                .add_modifier(Modifier::BOLD),
        ),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

/// Render engine latency line.
///
/// Format: `last: 420us | avg: 380us`
fn render_engine_latency(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let last_str = match app.last_autonomous_us {
        Some(us) => format!("{}us", us),
        None => "--".to_string(),
    };

    let avg_str = if app.autonomous_stats.avg_latency_us > 0.0 {
        format!("{:.0}us", app.autonomous_stats.avg_latency_us)
    } else {
        "--".to_string()
    };

    let line = Line::from(vec![
        Span::styled("  last: ", Style::default().fg(theme.muted)),
        Span::styled(last_str, Style::default().fg(theme.text)),
        Span::styled(" | avg: ", Style::default().fg(theme.muted)),
        Span::styled(avg_str, Style::default().fg(theme.text)),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

/// Render engine queries processed + JIT cache stats line.
///
/// Format: `  Q:123 | JIT: 5/2 (compiled/miss)`
fn render_engine_jit_stats(f: &mut Frame, area: Rect, app: &AppState, theme: &Theme) {
    let total_q = app.autonomous_stats.total_queries;

    // JIT stats: compiled = total unique plans, misses = cache misses
    // If we have an autonomous executor, read its stats; otherwise show defaults
    let (jit_compiled, jit_misses) = if let Some(ref executor) = app.autonomous_executor {
        let stats = executor.stats();
        (stats.jit_cache_misses, stats.jit_cache_misses)
    } else {
        (0, 0)
    };

    let line = Line::from(vec![
        Span::styled("  Q:", Style::default().fg(theme.muted)),
        Span::styled(format!("{}", total_q), Style::default().fg(theme.text)),
        Span::styled(" | JIT: ", Style::default().fg(theme.muted)),
        Span::styled(
            format!("{}/{}", jit_compiled, jit_misses),
            Style::default().fg(theme.text),
        ),
        Span::styled(" (plans/miss)", Style::default().fg(theme.muted)),
    ]);

    f.render_widget(Paragraph::new(line), area);
}

/// Render sparkline history of throughput over recent queries.
fn render_sparkline(f: &mut Frame, area: Rect, metrics: &GpuMetricsCollector, theme: &Theme) {
    let data = metrics.throughput_history_mbs();

    if data.is_empty() {
        let line = Line::from(Span::styled(
            "  (run queries to see history)",
            Style::default().fg(theme.muted),
        ));
        f.render_widget(Paragraph::new(line), area);
        return;
    }

    // Use the theme's glow gradient start color for the sparkline
    let spark_color = theme.glow.at(0.5);

    let sparkline = Sparkline::default()
        .data(&data)
        .style(Style::default().fg(spark_color));

    f.render_widget(sparkline, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::metrics::{GpuMetricsCollector, QueryMetrics};

    #[test]
    fn test_render_gpu_dashboard_no_panic_empty() {
        // Just ensure that creating the dashboard components doesn't panic
        let metrics = GpuMetricsCollector::new();
        let theme = Theme::thermal();

        // Verify metrics are in expected initial state
        assert_eq!(metrics.query_count, 0);
        assert!(metrics.throughput_history_mbs().is_empty());
        assert_eq!(metrics.utilization_estimate(), 0.0);
        assert_eq!(metrics.memory_utilization(), 0.0);

        // Verify theme gradients exist
        let _ = theme.thermal.at(0.5);
        let _ = theme.glow.at(0.5);
    }

    #[test]
    fn test_render_gpu_dashboard_no_panic_with_data() {
        let mut metrics = GpuMetricsCollector::new();
        metrics.record(QueryMetrics {
            gpu_time_ms: 5.0,
            memory_used_bytes: 1_000_000,
            scan_throughput_gbps: 50.0,
            rows_processed: 1_000_000,
            bytes_scanned: 100_000_000,
            is_warm: false,
        });

        let _theme = Theme::thermal();

        // Verify data is recorded for rendering
        assert_eq!(metrics.query_count, 1);
        assert_eq!(metrics.throughput_history_mbs().len(), 1);
        assert!(metrics.utilization_estimate() > 0.0);
        assert!(metrics.memory_utilization() > 0.0);

        // Verify sparkline data conversion
        let data = metrics.throughput_history_mbs();
        assert_eq!(data[0], 50000); // 50 GB/s = 50000 MB/s

        let time_data = metrics.time_history_us();
        assert_eq!(time_data[0], 5000); // 5ms = 5000us

        // Verify utilization estimate
        let util = metrics.utilization_estimate();
        assert!((util - 0.5).abs() < 0.01); // 50/100 = 0.5
    }

    #[test]
    fn test_all_themes_render_dashboard_colors() {
        let themes = [Theme::thermal(), Theme::glow(), Theme::ice(), Theme::mono()];

        for theme in &themes {
            // Each theme should provide valid gradient colors
            let gpu_0 = theme.thermal.at(0.0);
            let gpu_50 = theme.thermal.at(0.5);
            let gpu_100 = theme.thermal.at(1.0);

            // Colors should be different at different points
            assert_ne!(
                gpu_0, gpu_100,
                "theme {} gradient endpoints should differ",
                theme.name
            );
            let _ = gpu_50; // Just ensure no panic

            let glow_mid = theme.glow.at(0.5);
            let _ = glow_mid;
        }
    }

    #[test]
    fn test_sparkline_data_with_multiple_queries() {
        let mut metrics = GpuMetricsCollector::new();
        for i in 1..=5 {
            metrics.record(QueryMetrics {
                gpu_time_ms: i as f64 * 2.0,
                memory_used_bytes: i as u64 * 1_000_000,
                scan_throughput_gbps: i as f64 * 10.0,
                rows_processed: i as u64 * 100_000,
                bytes_scanned: i as u64 * 50_000_000,
                is_warm: false,
            });
        }

        let data = metrics.throughput_history_mbs();
        assert_eq!(data.len(), 5);
        assert_eq!(data[0], 10000); // 10 GB/s
        assert_eq!(data[4], 50000); // 50 GB/s

        let time_data = metrics.time_history_us();
        assert_eq!(time_data.len(), 5);
        assert_eq!(time_data[0], 2000); // 2ms
        assert_eq!(time_data[4], 10000); // 10ms
    }
}
