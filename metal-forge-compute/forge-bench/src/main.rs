mod cli;
mod config;
mod data_gen;
mod stats;

use clap::Parser;
use cli::ForgeArgs;
use config::{get_profile, parse_sizes};

fn main() {
    let args = ForgeArgs::parse();

    // Resolve sizes: CLI --sizes takes precedence, then --profile, then default
    let sizes = if let Some(ref raw_sizes) = args.sizes {
        match parse_sizes(raw_sizes) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error parsing sizes: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Some(ref profile_name) = args.profile {
        match get_profile(profile_name) {
            Some(p) => p.sizes,
            None => {
                eprintln!(
                    "Unknown profile '{}'. Valid: quick, standard, thorough",
                    profile_name
                );
                std::process::exit(1);
            }
        }
    } else {
        vec![1_000_000] // default: 1M
    };

    // Resolve runs/warmup from profile if not explicitly set
    let (runs, warmup) = if let Some(ref profile_name) = args.profile {
        if let Some(p) = get_profile(profile_name) {
            // CLI explicit values override profile defaults
            let r = if args.runs != 10 { args.runs } else { p.runs };
            let w = if args.warmup != 3 { args.warmup } else { p.warmup };
            (r, w)
        } else {
            (args.runs, args.warmup)
        }
    } else {
        (args.runs, args.warmup)
    };

    let experiments = if args.experiments.is_empty() {
        vec!["all".to_string()]
    } else {
        args.experiments.clone()
    };

    println!("forge-bench: GPU compute benchmark suite");
    println!("  Experiments: {:?}", experiments);
    println!("  Sizes: {:?}", sizes);
    println!("  Runs: {}, Warmup: {}", runs, warmup);
    if let Some(ref path) = args.json_file {
        println!("  JSON output: {}", path);
    }
    if let Some(ref path) = args.csv_file {
        println!("  CSV output: {}", path);
    }

    // TODO: dispatch to experiment harness
    println!("\nExperiment dispatch not yet implemented (stub).");
    println!("Available experiments: reduce, scan, compact, sort, histogram, filter, groupby, gemm, gemv, pipeline, duckdb, spreadsheet, timeseries, json_parse, hash_join");
}
