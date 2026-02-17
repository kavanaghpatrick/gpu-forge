mod cli;
mod config;
mod cpu_baselines;
mod data_gen;
mod experiments;
mod harness;
mod output;
mod stats;

use clap::Parser;
use cli::ForgeArgs;
use config::{get_profile, parse_sizes};
use forge_primitives::{HardwareInfo, MetalContext};
use harness::{run_experiment, BenchConfig, DataPoint};
use output::progress::BenchProgress;
use output::table::render_all_tables;

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

    let experiment_names = if args.experiments.is_empty() {
        vec!["all".to_string()]
    } else {
        args.experiments.clone()
    };

    // Create Metal context
    let ctx = MetalContext::new();
    let hardware = HardwareInfo::detect(&ctx.device);

    println!("forge-bench: GPU compute benchmark suite");
    println!("  Hardware: {} ({} GB/s)", hardware.chip_name, hardware.bandwidth_gbs);
    println!("  Experiments: {:?}", experiment_names);
    println!("  Sizes: {:?}", sizes);
    println!("  Runs: {}, Warmup: {}", runs, warmup);
    if let Some(ref path) = args.json_file {
        println!("  JSON output: {}", path);
    }
    if let Some(ref path) = args.csv_file {
        println!("  CSV output: {}", path);
    }
    println!();

    // Get all available experiments
    let mut all_exps = experiments::all_experiments();

    // Filter experiments based on CLI selection
    let selected: Vec<usize> = if experiment_names.contains(&"all".to_string()) {
        (0..all_exps.len()).collect()
    } else {
        let mut indices = Vec::new();
        for name in &experiment_names {
            if let Some(idx) = all_exps.iter().position(|e| e.name() == name.as_str()) {
                indices.push(idx);
            } else {
                eprintln!(
                    "Unknown experiment '{}'. Available: {}",
                    name,
                    all_exps
                        .iter()
                        .map(|e| e.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                std::process::exit(1);
            }
        }
        indices
    };

    let config = BenchConfig {
        sizes: sizes.clone(),
        runs,
        warmup,
    };

    // Run selected experiments
    let progress = BenchProgress::new();
    let mut all_results: Vec<DataPoint> = Vec::new();

    for &idx in &selected {
        let exp = &mut *all_exps[idx];
        let cb = progress.callback();
        let results = run_experiment(exp, &config, &ctx, &hardware, Some(&cb));
        all_results.extend(results);
    }

    progress.finish();

    // Render table output
    render_all_tables(&all_results);

    // Write JSON if requested
    if let Some(ref path) = args.json_file {
        if let Err(e) = output::json::write_json(path, &all_results, &hardware) {
            eprintln!("Error writing JSON: {}", e);
        }
    }
}
