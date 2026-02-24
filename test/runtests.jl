using Test
using GPUHashTables
using Random

# Check GPU backend availability
using CUDA
using Metal
const HAS_CUDA = CUDA.functional()
const HAS_METAL = Metal.functional()

@testset "GPUHashTables" begin
    include("test_hash.jl")
    include("test_cpu.jl")

    if HAS_CUDA
        include("test_cuda.jl")
    else
        @warn "CUDA not available - skipping CUDA GPU tests"
    end

    if HAS_METAL
        using Metal: MtlVector
        include("test_metal.jl")
    else
        @warn "Metal not available - skipping Metal GPU tests"
    end
end
