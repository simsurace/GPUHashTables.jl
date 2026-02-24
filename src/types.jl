# Core data types for DoubleHT hash table

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

# Default bucket type for UInt32 keys/values
const Bucket8{K,V} = Bucket{K,V,BUCKET_SIZE}

"""
    DoubleHashTable{K,V}

CPU-side double hashing hash table. Used for building the table and as a
reference implementation for testing.

# Fields
- `buckets`: Vector of buckets
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
mutable struct DoubleHashTable{K,V}
    buckets::Vector{Bucket8{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    GPUDoubleHashTable{K,V}

GPU-side double hashing hash table. Created by transferring a CPU table to GPU.

# Fields
- `buckets`: CuVector of buckets on GPU
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
struct GPUDoubleHashTable{K,V}
    buckets::CuVector{Bucket8{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end
