# Query throughput benchmarks

"""
    benchmark_cuda_query(n_entries, n_queries, load_factor; n_iterations=10)

Benchmark CUDA GPU query throughput.

Returns queries per second.
"""
function benchmark_cuda_query(
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int=10,
    positive_ratio::Float64=1.0  # Fraction of queries for existing keys
)
    Random.seed!(42)

    # Build table
    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)

    cpu_table = CPUDoubleHT(keys, vals; load_factor=load_factor)
    gpu_table = CuDoubleHT(cpu_table)

    # Generate query keys
    n_positive = round(Int, n_queries * positive_ratio)
    n_negative = n_queries - n_positive

    positive_keys = keys[rand(1:n_entries, n_positive)]
    negative_keys = rand(UInt32(2^31):(typemax(UInt32) - UInt32(1)), n_negative)  # Unlikely to be in table
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
    CUDA.@sync begin
        t_start = time_ns()
        for _ in 1:n_iterations
            query!(results, found, gpu_table, gpu_keys)
        end
        t_end = time_ns()
    end

    elapsed_seconds = (t_end - t_start) / 1e9
    total_queries = n_queries * n_iterations
    queries_per_second = total_queries / elapsed_seconds

    return queries_per_second
end

"""
    benchmark_metal_query(n_entries, n_queries, load_factor; n_iterations=10)

Benchmark Metal GPU query throughput.

Returns queries per second.
"""
function benchmark_metal_query(
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int=10,
    positive_ratio::Float64=1.0  # Fraction of queries for existing keys
)
    Random.seed!(42)

    # Build table
    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)

    cpu_table = CPUDoubleHT(keys, vals; load_factor=load_factor)
    metal_table = MtlDoubleHT(cpu_table)

    # Generate query keys
    n_positive = round(Int, n_queries * positive_ratio)
    n_negative = n_queries - n_positive

    positive_keys = keys[rand(1:n_entries, n_positive)]
    negative_keys = rand(UInt32(2^31):(typemax(UInt32) - UInt32(1)), n_negative)  # Unlikely to be in table
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
    total_queries = n_queries * n_iterations
    queries_per_second = total_queries / elapsed_seconds

    return queries_per_second
end

"""
    benchmark_cpu_query(n_entries, n_queries, load_factor; n_iterations=10)

Benchmark CPU query throughput for comparison.
"""
function benchmark_cpu_query(
    n_entries::Int,
    n_queries::Int,
    load_factor::Float64;
    n_iterations::Int=10
)
    Random.seed!(42)

    keys = unique(rand(UInt32(1):UInt32(2^31 - 1), n_entries * 2))[1:n_entries]
    vals = rand(UInt32, n_entries)

    cpu_table = CPUDoubleHT(keys, vals; load_factor=load_factor)

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
    total_queries = n_queries * n_iterations
    queries_per_second = total_queries / elapsed_seconds

    return queries_per_second
end

function run_cuda_query_benchmarks()
    println("-" ^ 60)
    println("CUDA Query Throughput Benchmarks")
    println("-" ^ 60)

    n_entries = 10_000_000
    n_queries = 10_000_000
    load_factor = 0.7

    println("\nTable size: $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println("Load factor: $load_factor")
    println()

    # Positive queries (all keys exist)
    qps = benchmark_cuda_query(n_entries, n_queries, load_factor; positive_ratio=1.0)
    @printf("CUDA Positive queries:  %8.2f M queries/sec\n", qps / 1e6)

    # Negative queries (no keys exist)
    qps = benchmark_cuda_query(n_entries, n_queries, load_factor; positive_ratio=0.0)
    @printf("CUDA Negative queries:  %8.2f M queries/sec\n", qps / 1e6)

    # Mixed queries (50/50)
    qps = benchmark_cuda_query(n_entries, n_queries, load_factor; positive_ratio=0.5)
    @printf("CUDA Mixed queries:     %8.2f M queries/sec\n", qps / 1e6)
end

function run_metal_query_benchmarks()
    println("-" ^ 60)
    println("Metal Query Throughput Benchmarks")
    println("-" ^ 60)

    n_entries = 10_000_000
    n_queries = 10_000_000
    load_factor = 0.7

    println("\nTable size: $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println("Load factor: $load_factor")
    println()

    # Positive queries (all keys exist)
    qps = benchmark_metal_query(n_entries, n_queries, load_factor; positive_ratio=1.0)
    @printf("Metal Positive queries:  %8.2f M queries/sec\n", qps / 1e6)

    # Negative queries (no keys exist)
    qps = benchmark_metal_query(n_entries, n_queries, load_factor; positive_ratio=0.0)
    @printf("Metal Negative queries:  %8.2f M queries/sec\n", qps / 1e6)

    # Mixed queries (50/50)
    qps = benchmark_metal_query(n_entries, n_queries, load_factor; positive_ratio=0.5)
    @printf("Metal Mixed queries:     %8.2f M queries/sec\n", qps / 1e6)
end

function run_comparison_benchmark(; use_cuda::Bool=false, use_metal::Bool=false)
    println("-" ^ 60)
    println("GPU vs CPU Comparison")
    println("-" ^ 60)

    n_entries = 1_000_000
    n_queries = 1_000_000
    load_factor = 0.7

    println("\nTable size: $(n_entries ÷ 1_000_000)M entries")
    println("Query batch: $(n_queries ÷ 1_000_000)M queries")
    println()

    cpu_qps = benchmark_cpu_query(n_entries, n_queries, load_factor)
    @printf("CPU:   %8.2f M queries/sec\n", cpu_qps / 1e6)

    if use_cuda
        cuda_qps = benchmark_cuda_query(n_entries, n_queries, load_factor)
        @printf("CUDA:  %8.2f M queries/sec (%.1fx vs CPU)\n", cuda_qps / 1e6, cuda_qps / cpu_qps)
    end

    if use_metal
        metal_qps = benchmark_metal_query(n_entries, n_queries, load_factor)
        @printf("Metal: %8.2f M queries/sec (%.1fx vs CPU)\n", metal_qps / 1e6, metal_qps / cpu_qps)
    end
end
