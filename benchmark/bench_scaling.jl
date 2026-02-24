# Scaling benchmarks - how performance varies with table size and load factor

function run_cuda_scaling_benchmarks()
    println("-" ^ 60)
    println("CUDA Scaling Benchmarks")
    println("-" ^ 60)

    # Scaling with table size
    println("\n1. Throughput vs Table Size (10M queries, load_factor=0.7)")
    println("-" ^ 40)

    n_queries = 10_000_000
    load_factor = 0.7

    @printf("%12s  %12s\n", "Table Size", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for n_entries in [100_000, 1_000_000, 10_000_000, 50_000_000]
        qps = benchmark_cuda_query(n_entries, n_queries, load_factor; n_iterations=5)
        size_str = n_entries >= 1_000_000 ? "$(n_entries ÷ 1_000_000)M" : "$(n_entries ÷ 1_000)K"
        @printf("%12s  %12.2f\n", size_str, qps / 1e6)
    end

    # Scaling with load factor
    println("\n2. Throughput vs Load Factor (10M entries, 10M queries)")
    println("-" ^ 40)

    n_entries = 10_000_000

    @printf("%12s  %12s\n", "Load Factor", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for load_factor in [0.5, 0.6, 0.7, 0.8, 0.9]
        qps = benchmark_cuda_query(n_entries, n_queries, load_factor; n_iterations=5)
        @printf("%12.1f  %12.2f\n", load_factor, qps / 1e6)
    end

    # Scaling with query batch size
    println("\n3. Throughput vs Query Batch Size (10M entries, load_factor=0.7)")
    println("-" ^ 40)

    n_entries = 10_000_000
    load_factor = 0.7

    @printf("%12s  %12s\n", "Batch Size", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for n_queries in [10_000, 100_000, 1_000_000, 10_000_000]
        # Adjust iterations to keep total work reasonable
        n_iter = max(1, 100_000_000 ÷ n_queries)
        qps = benchmark_cuda_query(n_entries, n_queries, load_factor; n_iterations=n_iter)
        size_str = n_queries >= 1_000_000 ? "$(n_queries ÷ 1_000_000)M" : "$(n_queries ÷ 1_000)K"
        @printf("%12s  %12.2f\n", size_str, qps / 1e6)
    end
end

function run_metal_scaling_benchmarks()
    println("-" ^ 60)
    println("Metal Scaling Benchmarks")
    println("-" ^ 60)

    # Scaling with table size
    println("\n1. Throughput vs Table Size (10M queries, load_factor=0.7)")
    println("-" ^ 40)

    n_queries = 10_000_000
    load_factor = 0.7

    @printf("%12s  %12s\n", "Table Size", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for n_entries in [100_000, 1_000_000, 10_000_000, 50_000_000]
        qps = benchmark_metal_query(n_entries, n_queries, load_factor; n_iterations=5)
        size_str = n_entries >= 1_000_000 ? "$(n_entries ÷ 1_000_000)M" : "$(n_entries ÷ 1_000)K"
        @printf("%12s  %12.2f\n", size_str, qps / 1e6)
    end

    # Scaling with load factor
    println("\n2. Throughput vs Load Factor (10M entries, 10M queries)")
    println("-" ^ 40)

    n_entries = 10_000_000

    @printf("%12s  %12s\n", "Load Factor", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for load_factor in [0.5, 0.6, 0.7, 0.8, 0.9]
        qps = benchmark_metal_query(n_entries, n_queries, load_factor; n_iterations=5)
        @printf("%12.1f  %12.2f\n", load_factor, qps / 1e6)
    end

    # Scaling with query batch size
    println("\n3. Throughput vs Query Batch Size (10M entries, load_factor=0.7)")
    println("-" ^ 40)

    n_entries = 10_000_000
    load_factor = 0.7

    @printf("%12s  %12s\n", "Batch Size", "M queries/s")
    @printf("%12s  %12s\n", "-" ^ 10, "-" ^ 10)

    for n_queries in [10_000, 100_000, 1_000_000, 10_000_000]
        # Adjust iterations to keep total work reasonable
        n_iter = max(1, 100_000_000 ÷ n_queries)
        qps = benchmark_metal_query(n_entries, n_queries, load_factor; n_iterations=n_iter)
        size_str = n_queries >= 1_000_000 ? "$(n_queries ÷ 1_000_000)M" : "$(n_queries ÷ 1_000)K"
        @printf("%12s  %12.2f\n", size_str, qps / 1e6)
    end
end
