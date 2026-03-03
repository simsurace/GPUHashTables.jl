# Query throughput benchmarks
#
# All benchmarks use the allocating query(table, cpu_keys) interface,
# which includes CPU↔GPU transfer overhead for GPU tables.

# =============================================================================
# Generic benchmark core
# =============================================================================

"""
    benchmark_cuda_query(build_fn, n_entries, n_queries, load_factor; n_iterations, positive_ratio)

Benchmark CUDA GPU query throughput using `query(table, cpu_keys)`.
Includes CPU→GPU key transfer and GPU→CPU result transfer.
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

    # Warmup
    for _ in 1:3
        query(gpu_table, query_keys)
    end
    CUDA.synchronize()

    # Benchmark
    t_start = UInt64(0)
    t_end = UInt64(0)
    CUDA.@sync begin
        t_start = time_ns()
        for _ in 1:n_iterations
            query(gpu_table, query_keys)
        end
        t_end = time_ns()
    end

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

"""
    benchmark_metal_query(build_fn, n_entries, n_queries, load_factor; n_iterations, positive_ratio)

Benchmark Metal GPU query throughput using `query(table, cpu_keys)`.
Includes CPU→GPU key transfer and GPU→CPU result transfer.
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

    # Warmup
    for _ in 1:3
        query(metal_table, query_keys)
    end
    Metal.synchronize()

    # Benchmark
    t_start = time_ns()
    for _ in 1:n_iterations
        query(metal_table, query_keys)
    end
    Metal.synchronize()
    t_end = time_ns()

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

"""
    benchmark_cpu_query(build_fn, n_entries, n_queries, load_factor; n_iterations)

Benchmark CPU query throughput using `query(table, keys)`.
Returns queries per second.
"""
function benchmark_cpu_query(
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
    cpu_table = build_fn(keys, vals, load_factor)

    n_positive = round(Int, n_queries * positive_ratio)
    n_negative = n_queries - n_positive
    positive_keys = keys[rand(1:n_entries, n_positive)]
    negative_keys = rand(UInt32(2^31):(typemax(UInt32) - UInt32(1)), n_negative)
    query_keys = Vector{UInt32}(shuffle(vcat(positive_keys, negative_keys)))

    # Warmup
    query(cpu_table, query_keys)

    # Benchmark
    t_start = time_ns()
    for _ in 1:n_iterations
        query(cpu_table, query_keys)
    end
    t_end = time_ns()

    elapsed_seconds = (t_end - t_start) / 1e9
    return n_queries * n_iterations / elapsed_seconds
end

# =============================================================================
# Table builders
# =============================================================================

dict_builder(keys, vals, lf) = Dict(zip(keys, vals))

cpu_double_builder(keys, vals, lf) = CPUDoubleHT(keys, vals; load_factor=lf)
cpu_hive_builder(keys, vals, lf)   = CPUHiveHT(keys, vals; load_factor=lf)
cpu_simple_builder(keys, vals, lf) = CPUSimpleHT(keys, vals; load_factor=lf)

cu_double_builder(keys, vals, lf) = CuDoubleHT(CPUDoubleHT(keys, vals; load_factor=lf))
cu_hive_builder(keys, vals, lf)   = CuHiveHT(CPUHiveHT(keys, vals; load_factor=lf))
cu_simple_builder(keys, vals, lf) = CuSimpleHT(CPUSimpleHT(keys, vals; load_factor=lf))

mtl_double_builder(keys, vals, lf) = MtlDoubleHT(CPUDoubleHT(keys, vals; load_factor=lf))
mtl_hive_builder(keys, vals, lf)   = MtlHiveHT(CPUHiveHT(keys, vals; load_factor=lf))

# =============================================================================
# Run functions
# =============================================================================

function run_comparison_benchmark(n_entries, n_queries, load_factor; use_cuda::Bool=false, use_metal::Bool=false)
    println("-" ^ 70)
    println("Comparison Benchmark (1M entries, 1M queries, load factor 0.7)")
    println("-" ^ 70)

    println("\nTable size:  $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println()

    @printf("%-20s  %14s  %14s  %14s\n", "Hash Table", "Positive", "Negative", "Mixed")
    @printf("%-20s  %14s  %14s  %14s\n", "-"^18, "-"^12, "-"^12, "-"^12)

    function bench_row(label, build_fn, bench_fn)
        pos_qps = bench_fn(build_fn, n_entries, n_queries, load_factor; positive_ratio=1.0)
        neg_qps = bench_fn(build_fn, n_entries, n_queries, load_factor; positive_ratio=0.0)
        mix_qps = bench_fn(build_fn, n_entries, n_queries, load_factor; positive_ratio=0.5)
        @printf("%-20s  %11.2f M  %11.2f M  %11.2f M\n", label, pos_qps / 1e6, neg_qps / 1e6, mix_qps / 1e6)
    end

    bench_row("Base.Dict", dict_builder, benchmark_cpu_query)

    bench_row("CPUSimpleHT", cpu_simple_builder, benchmark_cpu_query)
    use_cuda && bench_row("CuSimpleHT",  cu_simple_builder, benchmark_cuda_query)
    # use_metal && bench_row("MtlSimpleHT", mtl_simple_builder, benchmark_metal_query)

    bench_row("CPUDoubleHT", cpu_double_builder, benchmark_cpu_query)
    use_cuda && bench_row("CuDoubleHT",  cu_double_builder, benchmark_cuda_query)
    use_metal && bench_row("MtlDoubleHT", mtl_double_builder, benchmark_metal_query)

    bench_row("CPUHiveHT", cpu_hive_builder, benchmark_cpu_query)
    use_cuda && bench_row("CuHiveHT", cu_hive_builder, benchmark_cuda_query)
    use_metal && bench_row("MtlHiveHT", mtl_hive_builder, benchmark_metal_query)
end
