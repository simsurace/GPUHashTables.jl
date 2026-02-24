#!/usr/bin/env julia

using GPUHashTables
using CUDA
using Metal
using Random
using Printf

include("bench_query.jl")
include("bench_scaling.jl")

function main()
    println("=" ^ 60)
    println("GPUHashTables Benchmark Suite")
    println("=" ^ 60)

    # Detect available GPU backends
    has_cuda = CUDA.functional()
    has_metal = Metal.functional()

    if !has_cuda && !has_metal
        error("No GPU backend available - cannot run GPU benchmarks")
    end

    if has_cuda
        println("\nCUDA Device: ", CUDA.name(CUDA.device()))
    end
    if has_metal
        println("\nMetal Device: ", Metal.current_device().name)
    end
    println()

    # Run CUDA benchmarks
    if has_cuda
        run_cuda_query_benchmarks()
        println()
        run_cuda_scaling_benchmarks()
        println()
    end

    # Run Metal benchmarks
    if has_metal
        run_metal_query_benchmarks()
        println()
        run_metal_scaling_benchmarks()
        println()
    end

    # Run comparison benchmark
    run_comparison_benchmark(; use_cuda=has_cuda, use_metal=has_metal)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
