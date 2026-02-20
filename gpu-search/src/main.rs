//! gpu-search -- GPU-accelerated filesystem search tool for Apple Silicon.
//!
//! Launches a floating search panel powered by egui/eframe.

fn main() -> eframe::Result {
    gpu_search::ui::app::run_app()
}
