# GPU hash table wrapper and operations

"""
    GPUDoubleHashTable(cpu_table::DoubleHashTable{K,V}) -> GPUDoubleHashTable{K,V}

Transfer a CPU hash table to the GPU.

# Example
```julia
cpu_table = DoubleHashTable(keys, vals)
gpu_table = GPUDoubleHashTable(cpu_table)
```
"""
function GPUDoubleHashTable(cpu_table::DoubleHashTable{K,V}) where {K,V}
    gpu_buckets = CuVector(cpu_table.buckets)
    return GPUDoubleHashTable{K,V}(
        gpu_buckets,
        cpu_table.n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

"""
    query!(results::CuVector{V}, found::CuVector{Bool}, table::GPUDoubleHashTable{K,V}, keys::CuVector{K})

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
    table::GPUDoubleHashTable{K,V},
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
    query(table::GPUDoubleHashTable{K,V}, keys::CuVector{K}) -> (found::CuVector{Bool}, results::CuVector{V})

Batch query keys on the GPU, allocating result vectors.
"""
function query(table::GPUDoubleHashTable{K,V}, keys::CuVector{K}) where {K,V}
    n = length(keys)
    results = CUDA.zeros(V, n)
    found = CUDA.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::GPUDoubleHashTable{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query GPU table with CPU keys, handling transfers automatically.
Results are returned as CPU vectors.
"""
function query(table::GPUDoubleHashTable{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = CuVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end
