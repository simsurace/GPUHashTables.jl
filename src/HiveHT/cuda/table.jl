# CUDA HiveHT hash table type and operations

"""
    CuHiveHT{K,V}

CUDA GPU HiveHT hash table with 64-bit packed key-value pairs.

Uses WCME protocol for lock-free queries and WABC protocol for upserts.
Supports deletion via tombstones.

# Fields
- `buckets`: CuVector of HiveBucket structs (32 packed pairs each)
- `freemasks`: CuVector of freemask bitmaps (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Approximate number of entries (not updated atomically)
- `empty_key`: Sentinel value for empty key slots

# Type Constraints
Currently only supports `K = V = UInt32` due to the 64-bit packed pair format.
"""
mutable struct CuHiveHT{K,V}
    buckets::CuVector{HiveBucket}
    freemasks::CuVector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
end

# =============================================================================
# Constructors
# =============================================================================

"""
    CuHiveHT{K,V}(n_buckets::Int; empty_key=typemax(K)) -> CuHiveHT{K,V}

Create an empty CUDA HiveHT hash table with the specified number of buckets.

Each bucket has 32 slots, so total capacity is `n_buckets * 32` entries.

# Example
```julia
# Create empty table sized for ~1M entries at 70% load
n_entries = 1_000_000
n_buckets = cld(cld(n_entries, 0.7), 32)
table = CuHiveHT{UInt32,UInt32}(n_buckets)
```
"""
function CuHiveHT{K,V}(
    n_buckets::Int;
    empty_key::K = K === UInt32 ? HIVE_EMPTY_KEY : typemax(K)
) where {K,V}
    @assert K === UInt32 && V === UInt32 "HiveHT currently only supports UInt32 keys and values"
    @assert n_buckets > 0 "Number of buckets must be positive"

    # Initialize all buckets with empty pairs
    cpu_buckets = fill(empty_hive_bucket(), n_buckets)
    gpu_buckets = CuVector(cpu_buckets)

    # Initialize all freemasks to all-ones (all slots free)
    gpu_freemasks = CUDA.fill(UInt32(0xFFFFFFFF), n_buckets)

    return CuHiveHT{K,V}(
        gpu_buckets,
        gpu_freemasks,
        n_buckets,
        0,
        empty_key
    )
end

"""
    CuHiveHT(cpu_table::CPUHiveHT{K,V}) -> CuHiveHT{K,V}

Transfer a CPU HiveHT table to the GPU without invoking the upsert kernel.
The bucket layout is identical, so the data is copied directly.

# Example
```julia
cpu_table = CPUHiveHT(keys, vals)
gpu_table = CuHiveHT(cpu_table)
found, results = query(gpu_table, keys)
```
"""
function CuHiveHT(cpu_table::CPUHiveHT{K,V}) where {K,V}
    gpu_buckets = CuVector(cpu_table.buckets)
    gpu_freemasks = CuVector(cpu_table.freemasks)
    return CuHiveHT{K,V}(
        gpu_buckets,
        gpu_freemasks,
        cpu_table.n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key
    )
end

"""
    CuHiveHT(keys::Vector{K}, vals::Vector{V}; load_factor=0.7) -> CuHiveHT{K,V}

Create a CUDA HiveHT table and populate it with the given key-value pairs.

# Arguments
- `keys`: Vector of keys to insert
- `vals`: Vector of values to insert
- `load_factor`: Target load factor (default 0.7)

# Example
```julia
keys = rand(UInt32(1):UInt32(2^31), 10000)
vals = rand(UInt32, 10000)
table = CuHiveHT(keys, vals)
```
"""
function CuHiveHT(
    keys::Vector{K},
    vals::Vector{V};
    load_factor::Float64 = DEFAULT_LOAD_FACTOR,
    empty_key::K = K === UInt32 ? HIVE_EMPTY_KEY : typemax(K)
) where {K,V}
    @assert K === UInt32 && V === UInt32 "HiveHT currently only supports UInt32 keys and values"
    @assert length(keys) == length(vals) "Keys and values must have same length"
    @assert 0.0 < load_factor <= MAX_LOAD_FACTOR "Load factor must be in (0, $MAX_LOAD_FACTOR]"

    n = length(keys)
    if n == 0
        return CuHiveHT{K,V}(1; empty_key=empty_key)
    end

    # Calculate number of buckets needed
    total_slots_needed = ceil(Int, n / load_factor)
    n_buckets = max(1, cld(total_slots_needed, HIVE_BUCKET_SIZE))

    # Create empty table
    table = CuHiveHT{K,V}(n_buckets; empty_key=empty_key)

    # Insert all key-value pairs
    status = upsert!(table, keys, vals)

    # Check that all inserts succeeded
    failed = count(s -> s == UPSERT_FAILED, status)
    if failed > 0
        @warn "HiveHT construction: $failed out of $n inserts failed"
    end

    table.n_entries = n - failed

    return table
end

# =============================================================================
# Query Operations
# =============================================================================

"""
    query!(results::CuVector{V}, found::CuVector{Bool}, table::CuHiveHT{K,V}, keys::CuVector{K})

Batch query keys on the GPU HiveHT table.

# Arguments
- `results`: Pre-allocated GPU vector for result values
- `found`: Pre-allocated GPU vector for found flags
- `table`: GPU HiveHT table
- `keys`: GPU vector of keys to query
"""
function query!(
    results::CuVector{V},
    found::CuVector{Bool},
    table::CuHiveHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    # Configure kernel launch
    # Each warp (32 threads) handles one query
    threads_per_block = 256  # Must be multiple of 32
    warps_per_block = threads_per_block ÷ 32
    n_blocks = cld(n_queries, warps_per_block)

    @cuda threads=threads_per_block blocks=n_blocks hive_query_kernel!(
        results,
        found,
        table.buckets,
        Int32(table.n_buckets),
        keys,
        Int32(n_queries)
    )

    CUDA.synchronize()
    return nothing
end

"""
    query(table::CuHiveHT{K,V}, keys::CuVector{K}) -> (found::CuVector{Bool}, results::CuVector{V})

Batch query keys on the GPU HiveHT table, allocating result vectors.
"""
function query(table::CuHiveHT{K,V}, keys::CuVector{K}) where {K,V}
    n = length(keys)
    results = CUDA.zeros(V, n)
    found = CUDA.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::CuHiveHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query GPU HiveHT table with CPU keys, handling transfers automatically.
Results are returned as CPU vectors.
"""
function query(table::CuHiveHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = CuVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end

# =============================================================================
# Upsert Operations
# =============================================================================

"""
    upsert!(status::CuVector{UInt8}, table::CuHiveHT{K,V}, keys::CuVector{K}, vals::CuVector{V})

Batch upsert (insert or update) key-value pairs into the GPU HiveHT table.

# Arguments
- `status`: Pre-allocated GPU vector for operation results
  - 0 = failed (table full, CAS conflict, or max probes exceeded)
  - 1 = inserted (new key)
  - 2 = updated (existing key)
- `table`: HiveHT GPU hash table
- `keys`: GPU vector of keys to upsert
- `vals`: GPU vector of values to upsert
"""
function upsert!(
    status::CuVector{UInt8},
    table::CuHiveHT{K,V},
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
    warps_per_block = threads_per_block ÷ 32
    n_blocks = cld(n_ops, warps_per_block)

    @cuda threads=threads_per_block blocks=n_blocks hive_upsert_kernel!(
        status,
        table.buckets,
        table.freemasks,
        Int32(table.n_buckets),
        keys,
        vals,
        Int32(n_ops)
    )

    CUDA.synchronize()
    return nothing
end

"""
    upsert!(table::CuHiveHT{K,V}, keys::CuVector{K}, vals::CuVector{V}) -> CuVector{UInt8}

Batch upsert key-value pairs, allocating and returning status vector.
"""
function upsert!(
    table::CuHiveHT{K,V},
    keys::CuVector{K},
    vals::CuVector{V}
) where {K,V}
    n = length(keys)
    status = CUDA.zeros(UInt8, n)
    upsert!(status, table, keys, vals)
    return status
end

"""
    upsert!(table::CuHiveHT{K,V}, keys::Vector{K}, vals::Vector{V}) -> Vector{UInt8}

Upsert with CPU vectors, handling transfers automatically.
"""
function upsert!(
    table::CuHiveHT{K,V},
    keys::Vector{K},
    vals::Vector{V}
) where {K,V}
    gpu_keys = CuVector(keys)
    gpu_vals = CuVector(vals)
    status_gpu = upsert!(table, gpu_keys, gpu_vals)
    return Vector(status_gpu)
end

# =============================================================================
# Delete Operations
# =============================================================================

"""
    delete!(status::CuVector{UInt8}, table::CuHiveHT{K,V}, keys::CuVector{K})

Batch delete keys from the GPU HiveHT table.

Deletion marks slots as tombstones, which can be reused for future inserts.

# Arguments
- `status`: Pre-allocated GPU vector for operation results
  - 0 = failed (key not found or CAS conflict)
  - 1 = success (key was deleted)
- `table`: HiveHT GPU hash table
- `keys`: GPU vector of keys to delete
"""
function Base.delete!(
    status::CuVector{UInt8},
    table::CuHiveHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n_ops = length(keys)
    @assert length(status) >= n_ops "Status vector too small"

    if n_ops == 0
        return nothing
    end

    threads_per_block = 256
    warps_per_block = threads_per_block ÷ 32
    n_blocks = cld(n_ops, warps_per_block)

    @cuda threads=threads_per_block blocks=n_blocks hive_delete_kernel!(
        status,
        table.buckets,
        table.freemasks,
        Int32(table.n_buckets),
        keys,
        Int32(n_ops)
    )

    CUDA.synchronize()
    return nothing
end

"""
    delete!(table::CuHiveHT{K,V}, keys::CuVector{K}) -> CuVector{UInt8}

Batch delete keys, allocating and returning status vector.
"""
function Base.delete!(
    table::CuHiveHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n = length(keys)
    status = CUDA.zeros(UInt8, n)
    delete!(status, table, keys)
    return status
end

"""
    delete!(table::CuHiveHT{K,V}, keys::Vector{K}) -> Vector{UInt8}

Delete with CPU vectors, handling transfers automatically.
"""
function Base.delete!(
    table::CuHiveHT{K,V},
    keys::Vector{K}
) where {K,V}
    gpu_keys = CuVector(keys)
    status_gpu = delete!(table, gpu_keys)
    return Vector(status_gpu)
end
