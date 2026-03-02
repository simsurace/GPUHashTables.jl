# Scaling benchmarks - how performance varies with table size and load factor

function run_cuda_scaling_benchmarks()
    println("-" ^ 70)
    println("CUDA Scaling Benchmarks")
    println("-" ^ 70)

    # 1. Scaling with table size
    println("\n1. Throughput vs Table Size (10M queries, load_factor=0.7)")
    println("-" ^ 50)

    n_queries   = 10_000_000
    load_factor = 0.7

    @printf("%12s  %16s  %16s\n", "Table Size", "CuDoubleHT", "CuHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for n_entries in [100_000, 1_000_000, 10_000_000, 50_000_000]
        double_qps = benchmark_cuda_query(cu_double_builder, n_entries, n_queries, load_factor; n_iterations=5)
        hive_qps   = benchmark_cuda_query(cu_hive_builder,   n_entries, n_queries, load_factor; n_iterations=5)
        size_str = n_entries >= 1_000_000 ? "$(n_entries ÷ 1_000_000)M" : "$(n_entries ÷ 1_000)K"
        @printf("%12s  %13.2f M  %13.2f M\n", size_str, double_qps / 1e6, hive_qps / 1e6)
    end

    # 2. Scaling with load factor
    println("\n2. Throughput vs Load Factor (10M entries, 10M queries)")
    println("-" ^ 50)

    n_entries = 10_000_000

    @printf("%12s  %16s  %16s\n", "Load Factor", "CuDoubleHT", "CuHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for lf in [0.5, 0.6, 0.7, 0.8, 0.9]
        double_qps = benchmark_cuda_query(cu_double_builder, n_entries, n_queries, lf; n_iterations=5)
        hive_qps   = benchmark_cuda_query(cu_hive_builder,   n_entries, n_queries, lf; n_iterations=5)
        @printf("%12.1f  %13.2f M  %13.2f M\n", lf, double_qps / 1e6, hive_qps / 1e6)
    end

    # 3. Scaling with query batch size
    println("\n3. Throughput vs Query Batch Size (10M entries, load_factor=0.7)")
    println("-" ^ 50)

    n_entries   = 10_000_000
    load_factor = 0.7

    @printf("%12s  %16s  %16s\n", "Batch Size", "CuDoubleHT", "CuHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for n_q in [10_000, 100_000, 1_000_000, 10_000_000]
        n_iter = max(1, 100_000_000 ÷ n_q)
        double_qps = benchmark_cuda_query(cu_double_builder, n_entries, n_q, load_factor; n_iterations=n_iter)
        hive_qps   = benchmark_cuda_query(cu_hive_builder,   n_entries, n_q, load_factor; n_iterations=n_iter)
        size_str = n_q >= 1_000_000 ? "$(n_q ÷ 1_000_000)M" : "$(n_q ÷ 1_000)K"
        @printf("%12s  %13.2f M  %13.2f M\n", size_str, double_qps / 1e6, hive_qps / 1e6)
    end
end

function run_metal_scaling_benchmarks()
    println("-" ^ 70)
    println("Metal Scaling Benchmarks")
    println("-" ^ 70)

    # 1. Scaling with table size
    println("\n1. Throughput vs Table Size (10M queries, load_factor=0.7)")
    println("-" ^ 50)

    n_queries   = 10_000_000
    load_factor = 0.7

    @printf("%12s  %16s  %16s\n", "Table Size", "MtlDoubleHT", "MtlHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for n_entries in [100_000, 1_000_000, 10_000_000, 50_000_000]
        double_qps = benchmark_metal_query(mtl_double_builder, n_entries, n_queries, load_factor; n_iterations=5)
        hive_qps   = benchmark_metal_query(mtl_hive_builder,   n_entries, n_queries, load_factor; n_iterations=5)
        size_str = n_entries >= 1_000_000 ? "$(n_entries ÷ 1_000_000)M" : "$(n_entries ÷ 1_000)K"
        @printf("%12s  %13.2f M  %13.2f M\n", size_str, double_qps / 1e6, hive_qps / 1e6)
    end

    # 2. Scaling with load factor
    println("\n2. Throughput vs Load Factor (10M entries, 10M queries)")
    println("-" ^ 50)

    n_entries = 10_000_000

    @printf("%12s  %16s  %16s\n", "Load Factor", "MtlDoubleHT", "MtlHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for lf in [0.5, 0.6, 0.7, 0.8, 0.9]
        double_qps = benchmark_metal_query(mtl_double_builder, n_entries, n_queries, lf; n_iterations=5)
        hive_qps   = benchmark_metal_query(mtl_hive_builder,   n_entries, n_queries, lf; n_iterations=5)
        @printf("%12.1f  %13.2f M  %13.2f M\n", lf, double_qps / 1e6, hive_qps / 1e6)
    end

    # 3. Scaling with query batch size
    println("\n3. Throughput vs Query Batch Size (10M entries, load_factor=0.7)")
    println("-" ^ 50)

    n_entries   = 10_000_000
    load_factor = 0.7

    @printf("%12s  %16s  %16s\n", "Batch Size", "MtlDoubleHT", "MtlHiveHT")
    @printf("%12s  %16s  %16s\n", "-"^10, "-"^14, "-"^14)

    for n_q in [10_000, 100_000, 1_000_000, 10_000_000]
        n_iter = max(1, 100_000_000 ÷ n_q)
        double_qps = benchmark_metal_query(mtl_double_builder, n_entries, n_q, load_factor; n_iterations=n_iter)
        hive_qps   = benchmark_metal_query(mtl_hive_builder,   n_entries, n_q, load_factor; n_iterations=n_iter)
        size_str = n_q >= 1_000_000 ? "$(n_q ÷ 1_000_000)M" : "$(n_q ÷ 1_000)K"
        @printf("%12s  %13.2f M  %13.2f M\n", size_str, double_qps / 1e6, hive_qps / 1e6)
    end
end
