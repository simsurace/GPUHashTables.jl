# Scaling benchmarks - how performance varies with table size, load factor, and batch size

safe_bench(f) = try f() catch _ NaN end
fmtqps(qps) = isnan(qps) ? "            OOM" : @sprintf("%13.2f M", qps / 1e6)

function run_scaling_benchmarks(
    n_queries::Int,
    n_entries::Int,
    load_factor::Float64;
    use_cuda::Bool = false,
    use_metal::Bool = false
)
    println("-" ^ 70)
    println("Scaling Benchmarks")
    println("-" ^ 70)

    fmt_size(n) = n ≥ 1_000_000_000 ? "1G" :
                  n ≥ 1_000_000     ? "$(n ÷ 1_000_000)M" :
                                       "$(n ÷ 1_000)K"

    function bench_all(ne, nq, lf; n_iterations=5)
        dict = safe_bench(() -> benchmark_cpu_query(dict_builder, ne, nq, lf; n_iterations))

        cpu_simple = safe_bench(() -> benchmark_cpu_query(cpu_simple_builder, ne, nq, lf; n_iterations))
        cu_simple = use_cuda ? safe_bench(() -> benchmark_cuda_query(cu_simple_builder, ne, nq, lf; n_iterations)) : NaN
        mtl_simple = use_metal ? safe_bench(() -> benchmark_metal_query(mtl_simple_builder, ne, nq, lf; n_iterations)) : NaN

        cpu_double = safe_bench(() -> benchmark_cpu_query(cpu_double_builder, ne, nq, lf; n_iterations))
        cu_double = use_cuda ? safe_bench(() -> benchmark_cuda_query(cu_double_builder, ne, nq, lf; n_iterations)) : NaN
        mtl_double = use_metal ? safe_bench(() -> benchmark_metal_query(mtl_double_builder, ne, nq, lf; n_iterations)) : NaN

        cpu_hive = safe_bench(() -> benchmark_cpu_query(cpu_hive_builder, ne, nq, lf; n_iterations))
        cu_hive = use_cuda ? safe_bench(() -> benchmark_cuda_query(cu_hive_builder, ne, nq, lf; n_iterations)) : NaN
        mtl_hive = use_metal ? safe_bench(() -> benchmark_metal_query(mtl_hive_builder, ne, nq, lf; n_iterations)) : NaN

        return (dict, cpu_simple, cu_simple, mtl_simple, cpu_double, cu_double, mtl_double, cpu_hive, cu_hive, mtl_hive)
    end

    col_names = ("Base.Dict", "CPUSimpleHT", "CuSimpleHT", "MtlSimpleHT", "CPUDoubleHT", "CuDoubleHT", "MtlDoubleHT", "CPUHiveHT", "CuHiveHT", "MtlHiveHT")

    function print_header(first_col)
        @printf("%12s", first_col)
        for name in col_names
            @printf("  %15s", name)
        end
        println()
        @printf("%12s", "-"^10)
        for _ in col_names
            @printf("  %15s", "-"^15)
        end
        println()
    end

    function print_row(label, values)
        @printf("%12s", label)
        for v in values
            print("  ", fmtqps(v))
        end
        println()
    end

    # 1. Scaling with table size
    println("\n1. Throughput vs Table Size ($(fmt_size(n_queries)) queries, load_factor=$load_factor)")
    println("-" ^ 50)
    print_header("Table Size")

    for ne in [100_000, 1_000_000, 10_000_000, 100_000_000]
        results = bench_all(ne, n_queries, load_factor)
        print_row(fmt_size(ne), results)
        if ne ≥ 10_000_000
            GC.gc(true)
            use_cuda && CUDA.reclaim()
        end
    end

    # 2. Scaling with load factor
    println("\n2. Throughput vs Load Factor ($(fmt_size(n_entries)) entries, $(fmt_size(n_queries)) queries)")
    println("-" ^ 50)
    print_header("Load Factor")

    for lf in [0.5, 0.6, 0.7, 0.8, 0.9]
        results = bench_all(n_entries, n_queries, lf)
        print_row(@sprintf("%.1f", lf), results)
    end

    # 3. Scaling with query batch size
    println("\n3. Throughput vs Query Batch Size ($(fmt_size(n_entries)) entries, load_factor=$load_factor)")
    println("-" ^ 50)
    print_header("Batch Size")

    for nq in [10_000, 100_000, 1_000_000, 10_000_000]
        n_iter = max(1, 100_000_000 ÷ nq)
        results = bench_all(n_entries, nq, load_factor; n_iterations=n_iter)
        print_row(fmt_size(nq), results)
    end
end
