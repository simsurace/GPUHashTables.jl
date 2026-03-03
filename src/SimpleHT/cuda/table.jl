# CUDA SimpleHT hash table type and query operations

"""
    CuSimpleHT{K,V}

CUDA GPU linear-probing hash table. Ported from the SimpleGPUHashTable design
(https://github.com/nosferalatu/SimpleGPUHashTable).

Stores a flat `CuVector{Slot{K,V}}` of `n_slots` slots (power of two). Each
query is handled by a single thread — no warp cooperation. This minimises
overhead for workloads with short probe sequences (low-to-moderate load).

Deletion is handled via tombstones on the CPU side (see `CPUSimpleHT`). GPU
upsert/delete kernels are not yet implemented.

# Fields
- `slots`: Flat `CuVector` of `Slot{K,V}` (length `n_slots`)
- `n_slots`: Table capacity (power of two)
- `n_entries`: Number of live entries at construction time
- `empty_key`: Sentinel for an unused slot
- `empty_val`: Sentinel indicating a deleted entry
"""
mutable struct CuSimpleHT{K,V}
    slots::CuVector{Slot{K,V}}
    n_slots::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    CuSimpleHT(cpu_table::CPUSimpleHT{K,V}) -> CuSimpleHT{K,V}

Upload a CPU-built `CPUSimpleHT` to the GPU.

# Example
```julia
cpu = CPUSimpleHT(keys, vals; load_factor=0.7)
gpu = CuSimpleHT(cpu)
found, results = query(gpu, query_keys)
```
"""
function CuSimpleHT(cpu_table::CPUSimpleHT{K,V}) where {K,V}
    gpu_slots = CuVector(cpu_table.slots)
    return CuSimpleHT{K,V}(
        gpu_slots,
        cpu_table.n_slots,
        cpu_table.n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

# ---------------------------------------------------------------------------
# Query operations
# ---------------------------------------------------------------------------

"""
    query!(results::CuVector{V}, found::CuVector{Bool}, table::CuSimpleHT{K,V}, keys::CuVector{K})

In-place batch query on the GPU.
"""
function query!(
    results::CuVector{V},
    found::CuVector{Bool},
    table::CuSimpleHT{K,V},
    keys::CuVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found)   >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    threads_per_block = 256
    n_blocks = cld(n_queries, threads_per_block)

    @cuda threads=threads_per_block blocks=n_blocks simple_query_kernel!(
        results,
        found,
        table.slots,
        Int32(table.n_slots),
        keys,
        Int32(n_queries),
        table.empty_key,
        table.empty_val
    )

    CUDA.synchronize()
    return nothing
end

"""
    query(table::CuSimpleHT{K,V}, keys::CuVector{K}) -> (found::CuVector{Bool}, results::CuVector{V})

Allocating batch query with GPU input keys.
"""
function query(table::CuSimpleHT{K,V}, keys::CuVector{K}) where {K,V}
    n       = length(keys)
    results = CUDA.zeros(V, n)
    found   = CUDA.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::CuSimpleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Allocating batch query with CPU input keys; handles transfers automatically.
"""
function query(table::CuSimpleHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = CuVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end
