use gpu_query::cli::args::CliArgs;
use clap::Parser;

fn main() {
    let args = CliArgs::parse();

    if args.is_non_interactive() {
        let code = gpu_query::cli::run_non_interactive(&args);
        std::process::exit(code);
    }

    // Default: launch interactive dashboard TUI
    // --dashboard flag is accepted but no longer required (dashboard is default)
    if let Err(e) = gpu_query::tui::run_dashboard(args.directory.clone(), &args.theme) {
        eprintln!("Dashboard error: {}", e);
        std::process::exit(1);
    }
}
