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

    # Run query throughput benchmarks
    run_comparison_benchmark(10^7, 10^7, 0.7; use_cuda=has_cuda, use_metal=has_metal)
    run_comparison_benchmark(10^6, 10^6, 0.7; use_cuda=has_cuda, use_metal=has_metal)

    # Run scaling benchmarks
    run_scaling_benchmarks(10^7, 10^7, 0.7; use_cuda=has_cuda, use_metal=has_metal)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
