# GPU hash table wrapper and operations

# Upsert result codes
const UPSERT_FAILED = UInt8(0)
const UPSERT_INSERTED = UInt8(1)
const UPSERT_UPDATED = UInt8(2)

"""
    CuDoubleHT{K,V}

CUDA GPU-side double hashing hash table. Created by transferring a CPU table to GPU.
This is an immutable (query-only) view of the table.

# Fields
- `buckets`: CuVector of buckets on GPU
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
struct CuDoubleHT{K,V}
    buckets::CuVector{Bucket8{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    CuMutableDoubleHT{K,V}

CUDA GPU-side mutable double hashing hash table with support for concurrent upserts.
Includes per-bucket locks for thread-safe modifications.

# Fields
- `buckets`: CuVector of buckets on GPU
- `locks`: CuVector of per-bucket locks (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Approximate number of entries (not updated atomically)
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
mutable struct CuMutableDoubleHT{K,V}
    buckets::CuVector{Bucket8{K,V}}
    locks::CuVector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    CuDoubleHT(cpu_table::CPUDoubleHT{K,V}) -> CuDoubleHT{K,V}

Transfer a CPU hash table to the GPU.

# Example
```julia
cpu_table = CPUDoubleHT(keys, vals)
gpu_table = CuDoubleHT(cpu_table)
```
"""
function CuDoubleHT(cpu_table::CPUDoubleHT{K,V}) where {K,V}
    gpu_buckets = CuVector(cpu_table.buckets)
    return CuDoubleHT{K,V}(
        gpu_buckets,
        cpu_table.n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

"""
    query!(results::CuVector{V}, found::CuVector{Bool}, table::CuDoubleHT{K,V}, keys::CuVector{K})

Batch query keys on the GPU.

# Arguments
- `results`: Pre-allocated GPU vector for result values
- `found`: Pre-allocated GPU vector for found flags
- `table`: GPU hash table
- `keys`: GPU vector of keys to query
"""
function query!(
    results::CuVector{V},
    found::CuVector{Bool},
    table::CuDoubleHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    # Configure kernel launch
    # Each tile of TILE_SIZE threads handles one query
    threads_per_block = 256  # Must be multiple of TILE_SIZE
    tiles_per_block = threads_per_block ÷ TILE_SIZE
    n_blocks = cld(n_queries, tiles_per_block)

    @cuda threads=threads_per_block blocks=n_blocks query_kernel!(
        results,
        found,
        table.buckets,
        Int32(table.n_buckets),
        keys,
        Int32(n_queries),
        table.empty_key
    )

    # Synchronize to ensure results are ready
    CUDA.synchronize()

    return nothing
end

"""
    query(table::CuDoubleHT{K,V}, keys::CuVector{K}) -> (found::CuVector{Bool}, results::CuVector{V})

Batch query keys on the GPU, allocating result vectors.
"""
function query(table::CuDoubleHT{K,V}, keys::CuVector{K}) where {K,V}
    n = length(keys)
    results = CUDA.zeros(V, n)
    found = CUDA.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::CuDoubleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query GPU table with CPU keys, handling transfers automatically.
Results are returned as CPU vectors.
"""
function query(table::CuDoubleHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = CuVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end

# ============================================================================
# Mutable GPU Hash Table (supports upserts)
# ============================================================================

"""
    CuMutableDoubleHT(cpu_table::CPUDoubleHT{K,V}) -> CuMutableDoubleHT{K,V}

Transfer a CPU hash table to the GPU as a mutable table that supports upserts.

# Example
```julia
cpu_table = CPUDoubleHT(keys, vals)
gpu_table = CuMutableDoubleHT(cpu_table)
upsert!(gpu_table, new_keys, new_vals)
```
"""
function CuMutableDoubleHT(cpu_table::CPUDoubleHT{K,V}) where {K,V}
    gpu_buckets = CuVector(cpu_table.buckets)
    gpu_locks = CUDA.zeros(UInt32, cpu_table.n_buckets)
    return CuMutableDoubleHT{K,V}(
        gpu_buckets,
        gpu_locks,
        cpu_table.n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

"""
    CuMutableDoubleHT{K,V}(n_buckets::Int; load_factor=DEFAULT_LOAD_FACTOR) -> CuMutableDoubleHT{K,V}

Create an empty mutable GPU hash table with the specified number of buckets.

# Example
```julia
# Create empty table sized for ~1M entries at 70% load
n_entries = 1_000_000
n_buckets = cld(cld(n_entries, 0.7), BUCKET_SIZE)
table = CuMutableDoubleHT{UInt32,UInt32}(n_buckets)
```
"""
function CuMutableDoubleHT{K,V}(
    n_buckets::Int;
    empty_key::K = K === UInt32 ? EMPTY_KEY_U32 : typemax(K),
    empty_val::V = V === UInt32 ? EMPTY_VAL_U32 : typemax(V)
) where {K,V}
    # Initialize empty buckets
    empty_slot = Slot{K,V}(empty_key, empty_val)
    empty_bucket = Bucket8{K,V}(ntuple(_ -> empty_slot, BUCKET_SIZE))
    cpu_buckets = fill(empty_bucket, n_buckets)

    gpu_buckets = CuVector(cpu_buckets)
    gpu_locks = CUDA.zeros(UInt32, n_buckets)

    return CuMutableDoubleHT{K,V}(
        gpu_buckets,
        gpu_locks,
        n_buckets,
        0,
        empty_key,
        empty_val
    )
end

"""
    query!(results::CuVector{V}, found::CuVector{Bool}, table::CuMutableDoubleHT{K,V}, keys::CuVector{K})

Batch query keys on a mutable GPU table.
"""
function query!(
    results::CuVector{V},
    found::CuVector{Bool},
    table::CuMutableDoubleHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    threads_per_block = 256
    tiles_per_block = threads_per_block ÷ TILE_SIZE
    n_blocks = cld(n_queries, tiles_per_block)

    @cuda threads=threads_per_block blocks=n_blocks query_kernel!(
        results,
        found,
        table.buckets,
        Int32(table.n_buckets),
        keys,
        Int32(n_queries),
        table.empty_key
    )

    CUDA.synchronize()
    return nothing
end

"""
    query(table::CuMutableDoubleHT{K,V}, keys::CuVector{K}) -> (found::CuVector{Bool}, results::CuVector{V})

Batch query keys on a mutable GPU table, allocating result vectors.
"""
function query(table::CuMutableDoubleHT{K,V}, keys::CuVector{K}) where {K,V}
    n = length(keys)
    results = CUDA.zeros(V, n)
    found = CUDA.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::CuMutableDoubleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query mutable GPU table with CPU keys, handling transfers automatically.
"""
function query(table::CuMutableDoubleHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = CuVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end

"""
    upsert!(status::CuVector{UInt8}, table::CuMutableDoubleHT{K,V}, keys::CuVector{K}, vals::CuVector{V})

Batch upsert (insert or update) key-value pairs into the GPU table.

# Arguments
- `status`: Pre-allocated GPU vector for operation results
  - 0 = failed (table full or max probes exceeded)
  - 1 = inserted (new key)
  - 2 = updated (existing key)
- `table`: Mutable GPU hash table
- `keys`: GPU vector of keys to upsert
- `vals`: GPU vector of values to upsert
"""
function upsert!(
    status::CuVector{UInt8},
    table::CuMutableDoubleHT{K,V},
    keys::CuVector{K},
    vals::CuVector{V}
) where {K,V}
    n_ops = length(keys)
    @assert length(vals) == n_ops "Keys and values must have same length"
    @assert length(status) >= n_ops "Status vector too small"

    if n_ops == 0
        return nothing
    end

    threads_per_block = 256
    tiles_per_block = threads_per_block ÷ TILE_SIZE
    n_blocks = cld(n_ops, tiles_per_block)

    @cuda threads=threads_per_block blocks=n_blocks upsert_kernel!(
        status,
        table.buckets,
        table.locks,
        Int32(table.n_buckets),
        keys,
        vals,
        Int32(n_ops),
        table.empty_key
    )

    CUDA.synchronize()
    return nothing
end

"""
    upsert!(table::CuMutableDoubleHT{K,V}, keys::CuVector{K}, vals::CuVector{V}) -> CuVector{UInt8}

Batch upsert key-value pairs, allocating and returning status vector.
"""
function upsert!(
    table::CuMutableDoubleHT{K,V},
    keys::CuVector{K},
    vals::CuVector{V}
) where {K,V}
    n = length(keys)
    status = CUDA.zeros(UInt8, n)
    upsert!(status, table, keys, vals)
    return status
end

"""
    upsert!(table::CuMutableDoubleHT{K,V}, keys::Vector{K}, vals::Vector{V}) -> Vector{UInt8}

Upsert with CPU vectors, handling transfers automatically.
"""
function upsert!(
    table::CuMutableDoubleHT{K,V},
    keys::Vector{K},
    vals::Vector{V}
) where {K,V}
    gpu_keys = CuVector(keys)
    gpu_vals = CuVector(vals)
    status_gpu = upsert!(table, gpu_keys, gpu_vals)
    return Vector(status_gpu)
end
