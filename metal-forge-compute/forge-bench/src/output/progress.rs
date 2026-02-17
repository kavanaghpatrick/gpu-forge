//! Progress bar output using indicatif.
//!
//! Provides a spinner-style progress indicator during benchmark measurement.

use indicatif::{ProgressBar, ProgressStyle};

/// A progress reporter that wraps an indicatif spinner.
pub struct BenchProgress {
    bar: ProgressBar,
}

impl BenchProgress {
    /// Create a new progress spinner.
    pub fn new() -> Self {
        let bar = ProgressBar::new_spinner();
        bar.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(100));
        Self { bar }
    }

    /// Update the progress message.
    pub fn update(&self, msg: &str) {
        self.bar.set_message(msg.to_string());
    }

    /// Finish and clear the progress spinner.
    pub fn finish(&self) {
        self.bar.finish_and_clear();
    }

    /// Return a callback closure for use with run_experiment.
    pub fn callback(&self) -> impl Fn(&str) + '_ {
        move |msg: &str| {
            self.bar.set_message(msg.to_string());
        }
    }
}
