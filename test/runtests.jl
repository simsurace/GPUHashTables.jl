using Test
using GPUHashTables
using Random

using CUDA
using Metal

@testset "GPUHashTables" begin
    include("test_hash.jl")
    include("test_double_cpu.jl")
    include("test_simple_cpu.jl")
    include("test_hive_cpu.jl")

    include("gpu_query_tests.jl")

    if CUDA.functional()
        @testset "CUDA" begin
            for GpuT in (CuDoubleHT, CuHiveHT, CuSimpleHT)
                @testset "$GpuT" begin
                    run_gpu_query_tests(GpuT)
                end
            end
        end
    else
        @warn "CUDA not available - skipping CUDA GPU tests"
    end

    if Metal.functional()
        using Metal: MtlVector
        @testset "Metal" begin
            for GpuT in (MtlDoubleHT, MtlHiveHT, MtlSimpleHT)
                @testset "$GpuT" begin
                    run_gpu_query_tests(GpuT)
                end
            end
            include("test_hive_metal.jl")
        end
    else
        @warn "Metal not available - skipping Metal GPU tests"
    end
end
