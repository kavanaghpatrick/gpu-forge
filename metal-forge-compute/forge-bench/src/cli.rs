use clap::Parser;

/// GPU compute benchmark harness for Apple Silicon
#[derive(Parser, Debug)]
#[command(name = "forge-bench", version, about)]
pub struct ForgeArgs {
    /// Experiment names to run (e.g., reduce scan sort).
    /// Use "all" to run the full suite with crossover analysis.
    #[arg(value_name = "EXPERIMENTS")]
    pub experiments: Vec<String>,

    /// Element sizes to benchmark (e.g., 1M, 10M, 100K, 1000000)
    #[arg(long, value_delimiter = ',')]
    pub sizes: Option<Vec<String>>,

    /// Number of measured runs per size
    #[arg(long, default_value_t = 10)]
    pub runs: u32,

    /// Number of warmup runs before measurement
    #[arg(long, default_value_t = 3)]
    pub warmup: u32,

    /// Benchmark profile: quick (1M/3/1), standard (1M+10M/10/3), thorough (1M+10M+100M/30/3)
    #[arg(long)]
    pub profile: Option<String>,

    /// Write JSON results to file
    #[arg(long)]
    pub json_file: Option<String>,

    /// Write CSV results to file
    #[arg(long)]
    pub csv_file: Option<String>,
}

impl ForgeArgs {
    /// Returns true if this is an "all" suite run.
    pub fn is_all_suite(&self) -> bool {
        self.experiments.len() == 1 && self.experiments[0].eq_ignore_ascii_case("all")
            || self.experiments.is_empty()
    }
}
