module GPUHashTables

using CUDA
using Metal

# =============================================================================
# DoubleHT Constants
# =============================================================================
const BUCKET_SIZE = 8      # Slots per bucket (matches tile size)
const TILE_SIZE = 8        # Threads per cooperative tile
const MAX_PROBES = 80      # Maximum probe attempts
const DEFAULT_LOAD_FACTOR = 0.7
const MAX_LOAD_FACTOR = 0.9

# Sentinel values for UInt32
const EMPTY_KEY_U32 = typemax(UInt32)
const EMPTY_VAL_U32 = typemax(UInt32)

# Upsert result codes
const UPSERT_FAILED = UInt8(0)
const UPSERT_INSERTED = UInt8(1)
const UPSERT_UPDATED = UInt8(2)

# Base.Dict compatibility (for benchmarking)
include("dict-compat.jl")

# =============================================================================
# DoubleHT Implementation
# =============================================================================

# Core types
include("DoubleHT/types.jl")

# Hash functions
include("DoubleHT/hash.jl")

# CPU implementation
include("DoubleHT/cpu/table.jl")
include("DoubleHT/cpu/operations.jl")

# CUDA GPU implementation
include("DoubleHT/cuda/warp_utils.jl")
include("DoubleHT/cuda/kernels.jl")
include("DoubleHT/cuda/table.jl")

# Metal GPU implementation
include("DoubleHT/metal/simd_utils.jl")
include("DoubleHT/metal/kernels.jl")
include("DoubleHT/metal/table.jl")

# =============================================================================
# HiveHT Implementation
# =============================================================================

# HiveHT types and constants
include("HiveHT/types.jl")

# CPU HiveHT
include("HiveHT/cpu/table.jl")

# CUDA HiveHT
include("HiveHT/cuda/kernels.jl")
include("HiveHT/cuda/table.jl")

# Metal HiveHT
include("HiveHT/metal/kernels.jl")
include("HiveHT/metal/table.jl")

# =============================================================================
# SimpleHT Implementation
# =============================================================================

# CPU SimpleHT
include("SimpleHT/cpu/table.jl")
include("SimpleHT/cpu/operations.jl")

# CUDA SimpleHT
include("SimpleHT/cuda/kernels.jl")
include("SimpleHT/cuda/table.jl")

# Metal SimpleHT
include("SimpleHT/metal/kernels.jl")
include("SimpleHT/metal/table.jl")

# =============================================================================
# Exports
# =============================================================================

# DoubleHT types
export CPUDoubleHT, CuDoubleHT, MtlDoubleHT

# HiveHT types
export CPUHiveHT, CuHiveHT, MtlHiveHT
export HiveBucket

# SimpleHT types
export CPUSimpleHT, CuSimpleHT, MtlSimpleHT

# Operations (shared by all table types)
export query, query!
export upsert!

# Hash functions
export double_hash

# Status codes
export UPSERT_FAILED, UPSERT_INSERTED, UPSERT_UPDATED
export DELETE_FAILED, DELETE_SUCCESS

# HiveHT utilities
export pack_pair, unpack_key, unpack_val
export HIVE_EMPTY_PAIR, HIVE_TOMBSTONE

# Runtime availability checks
"""
    has_metal() -> Bool

Check if Metal is available and functional at runtime.
"""
has_metal() = Metal.functional()

export has_cuda, has_metal

end # module GPUHashTables
