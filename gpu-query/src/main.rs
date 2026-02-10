use gpu_query::cli::args::CliArgs;
use clap::Parser;

fn main() {
    let args = CliArgs::parse();

    if args.is_non_interactive() {
        let code = gpu_query::cli::run_non_interactive(&args);
        std::process::exit(code);
    }

    // Interactive/dashboard mode placeholder -- will be implemented in Phase 3
    if args.dashboard {
        eprintln!("Dashboard mode not yet implemented. Use -e \"SQL\" for non-interactive mode.");
        std::process::exit(1);
    }

    // Default: show usage
    eprintln!("Usage: gpu-query <directory> -e \"<SQL>\" [--format csv|json|jsonl|table]");
    eprintln!("       gpu-query <directory> --dashboard");
    eprintln!("       echo \"SELECT ...\" | gpu-query <directory>");
    std::process::exit(1);
}
