module GPUHashTables

using CUDA
using Metal

# Constants
const BUCKET_SIZE = 8      # Slots per bucket (matches tile size)
const TILE_SIZE = 8        # Threads per cooperative tile
const MAX_PROBES = 80      # Maximum probe attempts
const DEFAULT_LOAD_FACTOR = 0.7
const MAX_LOAD_FACTOR = 0.9

# Sentinel values for UInt32
const EMPTY_KEY_U32 = typemax(UInt32)
const EMPTY_VAL_U32 = typemax(UInt32)

# Core types
include("types.jl")

# Hash functions
include("hash.jl")

# CPU implementation
include("cpu/table.jl")
include("cpu/operations.jl")

# CUDA GPU implementation
include("cuda/warp_utils.jl")
include("cuda/kernels.jl")
include("cuda/table.jl")

# Metal GPU implementation
include("metal/simd_utils.jl")
include("metal/kernels.jl")
include("metal/table.jl")

# Exports
export CPUDoubleHT, CuDoubleHT, MtlDoubleHT
export CuMutableDoubleHT, MtlMutableDoubleHT
export query, query!
export upsert!
export double_hash
export UPSERT_FAILED, UPSERT_INSERTED, UPSERT_UPDATED

# Runtime availability checks
"""
    has_metal() -> Bool

Check if Metal is available and functional at runtime.
"""
has_metal() = Metal.functional()

export has_cuda, has_metal

end # module GPUHashTables
