use gpu_query::cli::args::CliArgs;
use clap::Parser;

fn main() {
    let args = CliArgs::parse();

    if args.is_non_interactive() {
        let code = gpu_query::cli::run_non_interactive(&args);
        std::process::exit(code);
    }

    // Interactive dashboard mode
    if args.dashboard {
        if let Err(e) = gpu_query::tui::run_dashboard(args.directory.clone(), &args.theme) {
            eprintln!("Dashboard error: {}", e);
            std::process::exit(1);
        }
        std::process::exit(0);
    }

    // Default: show usage
    eprintln!("Usage: gpu-query <directory> -e \"<SQL>\" [--format csv|json|jsonl|table]");
    eprintln!("       gpu-query <directory> --dashboard");
    eprintln!("       echo \"SELECT ...\" | gpu-query <directory>");
    std::process::exit(1);
}
