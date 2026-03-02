# Query throughput benchmarks

# =============================================================================
# Generic benchmark core
# =============================================================================

"""
    benchmark_cuda_query(build_fn, n_entries, n_queries, load_factor; n_iterations, positive_ratio)

Benchmark CUDA GPU query throughput for any table type.

`build_fn(keys, vals, load_factor)` must return a GPU table that supports `query!`.
Returns queries per second.
"""
function benchmark_cuda_query(
    build_fn,
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int = 10,
    positive_ratio::Float64 = 1.0
)
    Random.seed!(42)

    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)
    gpu_table = build_fn(keys, vals, load_factor)

    n_positive = round(Int, n_queries * positive_ratio)
    n_negative = n_queries - n_positive
    positive_keys = keys[rand(1:n_entries, n_positive)]
    negative_keys = rand(UInt32(2^31):(typemax(UInt32) - UInt32(1)), n_negative)
    query_keys = Vector{UInt32}(shuffle(vcat(positive_keys, negative_keys)))

    gpu_keys = CuVector(query_keys)
    results = CUDA.zeros(UInt32, n_queries)
    found = CUDA.zeros(Bool, n_queries)

    # Warmup
    for _ in 1:3
        query!(results, found, gpu_table, gpu_keys)
    end
    CUDA.synchronize()

    # Benchmark
    t_start = UInt64(0)
    t_end = UInt64(0)
    CUDA.@sync begin
        t_start = time_ns()
        for _ in 1:n_iterations
            query!(results, found, gpu_table, gpu_keys)
        end
        t_end = time_ns()
    end

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

"""
    benchmark_metal_query(build_fn, n_entries, n_queries, load_factor; n_iterations, positive_ratio)

Benchmark Metal GPU query throughput for any table type.

`build_fn(keys, vals, load_factor)` must return a GPU table that supports `query!`.
Returns queries per second.
"""
function benchmark_metal_query(
    build_fn,
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int = 10,
    positive_ratio::Float64 = 1.0
)
    Random.seed!(42)

    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)
    metal_table = build_fn(keys, vals, load_factor)

    n_positive = round(Int, n_queries * positive_ratio)
    n_negative = n_queries - n_positive
    positive_keys = keys[rand(1:n_entries, n_positive)]
    negative_keys = rand(UInt32(2^31):(typemax(UInt32) - UInt32(1)), n_negative)
    query_keys = Vector{UInt32}(shuffle(vcat(positive_keys, negative_keys)))

    metal_keys = MtlVector(query_keys)
    results = Metal.zeros(UInt32, n_queries)
    found = Metal.zeros(Bool, n_queries)

    # Warmup
    for _ in 1:3
        query!(results, found, metal_table, metal_keys)
    end
    Metal.synchronize()

    # Benchmark
    t_start = time_ns()
    for _ in 1:n_iterations
        query!(results, found, metal_table, metal_keys)
    end
    Metal.synchronize()
    t_end = time_ns()

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

"""
    benchmark_cpu_query(build_fn, n_entries, n_queries, load_factor; n_iterations)

Benchmark CPU query throughput for any table type.

`build_fn(keys, vals, load_factor)` must return a CPU table that supports `query!`.
Returns queries per second.
"""
function benchmark_cpu_query(
    build_fn,
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int = 10
)
    Random.seed!(42)

    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)
    cpu_table = build_fn(keys, vals, load_factor)

    query_keys = keys[rand(1:n_entries, n_queries)]
    results = Vector{UInt32}(undef, n_queries)
    found = Vector{Bool}(undef, n_queries)

    # Warmup
    query!(results, found, cpu_table, query_keys)

    # Benchmark
    t_start = time_ns()
    for _ in 1:n_iterations
        query!(results, found, cpu_table, query_keys)
    end
    t_end = time_ns()

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

# =============================================================================
# Table builders
# =============================================================================

cpu_double_builder(keys, vals, lf) = CPUDoubleHT(keys, vals; load_factor=lf)
cpu_hive_builder(keys, vals, lf)   = CPUHiveHT(keys, vals; load_factor=lf)

cu_double_builder(keys, vals, lf) = CuDoubleHT(CPUDoubleHT(keys, vals; load_factor=lf))
cu_hive_builder(keys, vals, lf)   = CuHiveHT(CPUHiveHT(keys, vals; load_factor=lf))

mtl_double_builder(keys, vals, lf) = MtlDoubleHT(CPUDoubleHT(keys, vals; load_factor=lf))
mtl_hive_builder(keys, vals, lf)   = MtlHiveHT(CPUHiveHT(keys, vals; load_factor=lf))

# =============================================================================
# Run functions
# =============================================================================

function run_cuda_query_benchmarks()
    println("-" ^ 70)
    println("CUDA Query Throughput Benchmarks")
    println("-" ^ 70)

    n_entries   = 10_000_000
    n_queries   = 10_000_000
    load_factor = 0.7

    println("\nTable size:  $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println("Load factor: $load_factor")
    println()

    @printf("%-28s  %14s  %14s\n", "Query Type", "CuDoubleHT", "CuHiveHT")
    @printf("%-28s  %14s  %14s\n", "-"^26, "-"^12, "-"^12)

    for (label, ratio) in [
        ("Positive (all keys exist)", 1.0),
        ("Negative (no keys exist)",  0.0),
        ("Mixed (50/50)",             0.5),
    ]
        double_qps = benchmark_cuda_query(cu_double_builder, n_entries, n_queries, load_factor; positive_ratio=ratio)
        hive_qps   = benchmark_cuda_query(cu_hive_builder,   n_entries, n_queries, load_factor; positive_ratio=ratio)
        @printf("%-28s  %11.2f M  %11.2f M\n", label, double_qps / 1e6, hive_qps / 1e6)
    end
end

function run_metal_query_benchmarks()
    println("-" ^ 70)
    println("Metal Query Throughput Benchmarks")
    println("-" ^ 70)

    n_entries   = 10_000_000
    n_queries   = 10_000_000
    load_factor = 0.7

    println("\nTable size:  $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println("Load factor: $load_factor")
    println()

    @printf("%-28s  %14s  %14s\n", "Query Type", "MtlDoubleHT", "MtlHiveHT")
    @printf("%-28s  %14s  %14s\n", "-"^26, "-"^12, "-"^12)

    for (label, ratio) in [
        ("Positive (all keys exist)", 1.0),
        ("Negative (no keys exist)",  0.0),
        ("Mixed (50/50)",             0.5),
    ]
        double_qps = benchmark_metal_query(mtl_double_builder, n_entries, n_queries, load_factor; positive_ratio=ratio)
        hive_qps   = benchmark_metal_query(mtl_hive_builder,   n_entries, n_queries, load_factor; positive_ratio=ratio)
        @printf("%-28s  %11.2f M  %11.2f M\n", label, double_qps / 1e6, hive_qps / 1e6)
    end
end

function run_comparison_benchmark(; use_cuda::Bool=false, use_metal::Bool=false)
    println("-" ^ 60)
    println("GPU vs CPU Comparison")
    println("-" ^ 60)

    n_entries   = 1_000_000
    n_queries   = 1_000_000
    load_factor = 0.7

    println("\nTable size:  $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println()

    cpu_double_qps = benchmark_cpu_query(cpu_double_builder, n_entries, n_queries, load_factor)
    cpu_hive_qps   = benchmark_cpu_query(cpu_hive_builder,   n_entries, n_queries, load_factor)
    @printf("CPU (DoubleHT):      %8.2f M queries/sec\n", cpu_double_qps / 1e6)
    @printf("CPU (HiveHT):        %8.2f M queries/sec\n", cpu_hive_qps / 1e6)

    if use_metal
        double_qps = benchmark_metal_query(mtl_double_builder, n_entries, n_queries, load_factor)
        hive_qps   = benchmark_metal_query(mtl_hive_builder,   n_entries, n_queries, load_factor)
        @printf("Metal MtlDoubleHT:   %8.2f M queries/sec  (%.1fx vs CPU)\n", double_qps / 1e6, double_qps / cpu_double_qps)
        @printf("Metal MtlHiveHT:     %8.2f M queries/sec  (%.1fx vs CPU)\n", hive_qps   / 1e6, hive_qps   / cpu_hive_qps)
    end

    if use_cuda
        double_qps = benchmark_cuda_query(cu_double_builder, n_entries, n_queries, load_factor)
        hive_qps   = benchmark_cuda_query(cu_hive_builder,   n_entries, n_queries, load_factor)
        @printf("CUDA CuDoubleHT:     %8.2f M queries/sec  (%.1fx vs CPU)\n", double_qps / 1e6, double_qps / cpu_double_qps)
        @printf("CUDA CuHiveHT:       %8.2f M queries/sec  (%.1fx vs CPU)\n", hive_qps   / 1e6, hive_qps   / cpu_hive_qps)
    end
end
