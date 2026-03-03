using Test
using GPUHashTables
using Random

using CUDA
using Metal

@testset "GPUHashTables" begin
    include("test_hash.jl")
    include("test_double_cpu.jl")
    include("test_simple_cpu.jl")

    if CUDA.functional()
        @testset "CUDA" begin
            include("test_double_cuda.jl")
            include("test_hive_cuda.jl")
            include("test_simple_cuda.jl")
        end
    else
        @warn "CUDA not available - skipping CUDA GPU tests"
    end

    if Metal.functional()
        using Metal: MtlVector
        @testset "Metal" begin
            include("test_double_metal.jl")
            include("test_hive_metal.jl")
            include("test_simple_metal.jl")
        end
    else
        @warn "Metal not available - skipping Metal GPU tests"
    end
end
