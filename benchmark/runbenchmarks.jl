#!/usr/bin/env julia

using GPUHashTables
using CUDA
using Random
using Printf

include("bench_query.jl")
include("bench_scaling.jl")

function main()
    println("=" ^ 60)
    println("GPUHashTables Benchmark Suite")
    println("=" ^ 60)

    if !CUDA.functional()
        error("CUDA is not available - cannot run GPU benchmarks")
    end

    println("\nCUDA Device: ", CUDA.name(CUDA.device()))
    println()

    # Run benchmarks
    run_query_benchmarks()
    println()
    run_scaling_benchmarks()
    println()
    run_comparison_benchmark()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
