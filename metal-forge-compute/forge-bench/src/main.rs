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
    let is_all = args.is_all_suite();

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
            let w = if args.warmup != 3 {
                args.warmup
            } else {
                p.warmup
            };
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

    let profile_label = args.profile.as_deref().unwrap_or("custom");

    println!("forge-bench: GPU compute benchmark suite");
    println!(
        "  Hardware: {} ({} GB/s)",
        hardware.chip_name, hardware.bandwidth_gbs
    );
    if is_all {
        println!("  Mode: full suite (all experiments)");
        println!("  Profile: {}", profile_label);
    } else {
        println!("  Experiments: {:?}", experiment_names);
    }
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
    let selected: Vec<usize> = if is_all {
        (0..all_exps.len()).collect()
    } else {
        let mut indices = Vec::new();
        for name in &experiment_names {
            if name.eq_ignore_ascii_case("all") {
                // Handle "all" mixed with other names -- just run everything
                run_all_and_exit(args, all_exps, sizes, runs, warmup, ctx, hardware);
            }
            if let Some(idx) = all_exps.iter().position(|e| e.name() == name.as_str()) {
                indices.push(idx);
            } else {
                eprintln!(
                    "Unknown experiment '{}'. Available: all, {}",
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

    // Print summary with verdicts
    output::summary::print_summary(&all_results);

    // Print crossover analysis and suite summary for "all" runs
    if is_all {
        output::summary::print_crossover_analysis(&all_results);
        output::summary::print_suite_summary(&all_results);
    }

    // Print ASCII roofline diagram
    output::roofline::print_roofline(&all_results, &hardware);

    // Write JSON if requested
    if let Some(ref path) = args.json_file {
        if let Err(e) = output::json::write_json(path, &all_results, &hardware) {
            eprintln!("Error writing JSON: {}", e);
        }
    }

    // Write CSV if requested
    if let Some(ref path) = args.csv_file {
        if let Err(e) = output::csv::write_csv(path, &all_results) {
            eprintln!("Error writing CSV: {}", e);
        }
    }
}

/// Helper: run all experiments when "all" is detected mid-argument list.
/// This avoids complex index tracking when "all" appears among other names.
fn run_all_and_exit(
    args: ForgeArgs,
    mut all_exps: Vec<Box<dyn experiments::Experiment>>,
    sizes: Vec<usize>,
    runs: u32,
    warmup: u32,
    ctx: MetalContext,
    hardware: HardwareInfo,
) -> ! {
    let config = BenchConfig {
        sizes,
        runs,
        warmup,
    };
    let progress = BenchProgress::new();
    let mut all_results: Vec<DataPoint> = Vec::new();

    for exp in all_exps.iter_mut() {
        let cb = progress.callback();
        let results = run_experiment(&mut **exp, &config, &ctx, &hardware, Some(&cb));
        all_results.extend(results);
    }

    progress.finish();

    render_all_tables(&all_results);
    output::summary::print_summary(&all_results);
    output::summary::print_crossover_analysis(&all_results);
    output::summary::print_suite_summary(&all_results);
    output::roofline::print_roofline(&all_results, &hardware);

    if let Some(ref path) = args.json_file {
        if let Err(e) = output::json::write_json(path, &all_results, &hardware) {
            eprintln!("Error writing JSON: {}", e);
        }
    }
    if let Some(ref path) = args.csv_file {
        if let Err(e) = output::csv::write_csv(path, &all_results) {
            eprintln!("Error writing CSV: {}", e);
        }
    }

    std::process::exit(0);
}
