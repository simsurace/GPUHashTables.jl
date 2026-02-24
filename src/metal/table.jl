# Metal GPU hash table wrapper and operations

using Metal: MtlVector, MtlDeviceVector, @metal, Metal

"""
    MtlDoubleHT{K,V}

Metal GPU-side double hashing hash table. Created by transferring a CPU table to GPU.

# Fields
- `buckets`: MtlVector of buckets on GPU
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
struct MtlDoubleHT{K,V}
    buckets::MtlVector{Bucket8{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    MtlDoubleHT(cpu_table::CPUDoubleHT{K,V}) -> MtlDoubleHT{K,V}

Transfer a CPU hash table to Metal GPU.

# Example
```julia
cpu_table = CPUDoubleHT(keys, vals)
metal_table = MtlDoubleHT(cpu_table)
```
"""
function MtlDoubleHT(cpu_table::CPUDoubleHT{K,V}) where {K,V}
    metal_buckets = MtlVector(cpu_table.buckets)
    return MtlDoubleHT{K,V}(
        metal_buckets,
        cpu_table.n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

"""
    query!(results::MtlVector{V}, found::MtlVector{Bool}, table::MtlDoubleHT{K,V}, keys::MtlVector{K})

Batch query keys on the Metal GPU.

# Arguments
- `results`: Pre-allocated GPU vector for result values
- `found`: Pre-allocated GPU vector for found flags
- `table`: Metal GPU hash table
- `keys`: GPU vector of keys to query
"""
function query!(
    results::MtlVector{V},
    found::MtlVector{Bool},
    table::MtlDoubleHT{K,V},
    keys::MtlVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    # Configure kernel launch
    # Each tile of TILE_SIZE threads handles one query
    threads_per_threadgroup = 256  # Must be multiple of TILE_SIZE
    tiles_per_threadgroup = threads_per_threadgroup ÷ TILE_SIZE
    n_threadgroups = cld(n_queries, tiles_per_threadgroup)

    @metal threads=threads_per_threadgroup groups=n_threadgroups metal_query_kernel!(
        results,
        found,
        table.buckets,
        Int32(table.n_buckets),
        keys,
        Int32(n_queries),
        table.empty_key
    )

    # Synchronize to ensure results are ready
    Metal.synchronize()

    return nothing
end

"""
    query(table::MtlDoubleHT{K,V}, keys::MtlVector{K}) -> (found::MtlVector{Bool}, results::MtlVector{V})

Batch query keys on the Metal GPU, allocating result vectors.
"""
function query(table::MtlDoubleHT{K,V}, keys::MtlVector{K}) where {K,V}
    n = length(keys)
    results = Metal.zeros(V, n)
    found = Metal.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::MtlDoubleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query Metal GPU table with CPU keys, handling transfers automatically.
Results are returned as CPU vectors.
"""
function query(table::MtlDoubleHT{K,V}, keys::Vector{K}) where {K,V}
    metal_keys = MtlVector(keys)
    found_gpu, results_gpu = query(table, metal_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end
