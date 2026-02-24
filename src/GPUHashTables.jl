module GPUHashTables

using CUDA

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

# GPU implementation
include("gpu/warp_utils.jl")
include("gpu/kernels.jl")
include("gpu/table.jl")

# Exports
export DoubleHashTable, GPUDoubleHashTable
export build_table, query, query!
export double_hash

end # module GPUHashTables
