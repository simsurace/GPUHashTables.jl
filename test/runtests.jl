using Test
using GPUHashTables
using Random

# Check if CUDA is available
using CUDA
const HAS_CUDA = CUDA.functional()

@testset "GPUHashTables" begin
    include("test_hash.jl")
    include("test_cpu.jl")

    if HAS_CUDA
        include("test_gpu.jl")
    else
        @warn "CUDA not available - skipping GPU tests"
    end
end
