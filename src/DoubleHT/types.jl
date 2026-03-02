# Core data types for DoubleHT hash table

# Lock state constants for bucket locking
const LOCK_FREE = UInt32(0)
const LOCK_HELD = UInt32(1)

"""
    Slot{K,V}

A single key-value slot in a hash table bucket.
"""
struct Slot{K,V}
    key::K
    val::V
end

"""
    Bucket{K,V,N}

A bucket containing N slots. Each bucket is accessed by one tile of threads.
For UInt32 keys/values with N=8: sizeof(Bucket) = 64 bytes.
"""
struct Bucket{K,V,N}
    slots::NTuple{N,Slot{K,V}}
end

# Default bucket type for UInt32 keys/values (CUDA uses 8-slot buckets)
const Bucket8{K,V} = Bucket{K,V,BUCKET_SIZE}

# Metal-specific bucket type (32-slot buckets to match simdgroup size)
const METAL_BUCKET_SIZE = 32
const METAL_TILE_SIZE = 32  # Full simdgroup as a tile
const Bucket32{K,V} = Bucket{K,V,METAL_BUCKET_SIZE}
