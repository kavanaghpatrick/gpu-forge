use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[2] != "-e" {
        eprintln!("Usage: gpu-query <directory> -e \"<SQL>\"");
        eprintln!("Example: gpu-query ./test-data/ -e \"SELECT count(*) FROM sales WHERE amount > 100\"");
        std::process::exit(1);
    }

    let dir = PathBuf::from(&args[1]);
    let sql = &args[3];

    if !dir.is_dir() {
        eprintln!("Error: '{}' is not a directory", dir.display());
        std::process::exit(1);
    }

    // Scan directory for data files
    let catalog = gpu_query::io::catalog::scan_directory(&dir)
        .unwrap_or_else(|e| {
            eprintln!("Error scanning directory: {}", e);
            std::process::exit(1);
        });

    if catalog.is_empty() {
        eprintln!("No data files found in '{}'", dir.display());
        std::process::exit(1);
    }

    // Parse SQL
    let logical_plan = gpu_query::sql::parser::parse_query(sql)
        .unwrap_or_else(|e| {
            eprintln!("SQL parse error: {}", e);
            std::process::exit(1);
        });

    // Optimize logical plan (column pruning, predicate pushdown, constant folding)
    let logical_plan = gpu_query::sql::optimizer::optimize(logical_plan);

    // Convert to physical plan
    let physical_plan = gpu_query::sql::physical_plan::plan(&logical_plan)
        .unwrap_or_else(|e| {
            eprintln!("Plan error: {:?}", e);
            std::process::exit(1);
        });

    // Execute on GPU
    let mut executor = gpu_query::gpu::executor::QueryExecutor::new()
        .unwrap_or_else(|e| {
            eprintln!("GPU init error: {}", e);
            std::process::exit(1);
        });

    let result = executor.execute(&physical_plan, &catalog)
        .unwrap_or_else(|e| {
            eprintln!("Query execution error: {}", e);
            std::process::exit(1);
        });

    result.print();
}
