# Metal SimpleHT hash table type and query operations

using Metal: thread_position_in_grid

"""
    MtlSimpleHT{K,V}

Metal GPU linear-probing hash table. Ported from the SimpleGPUHashTable design
(https://github.com/nosferalatu/SimpleGPUHashTable).

Stores flat `MtlVector`s of keys and values with `n_slots` entries (power of
two). Each query is handled by a single thread — no simdgroup cooperation. This
minimises overhead for workloads with short probe sequences (low-to-moderate
load).

Deletion is handled via tombstones on the CPU side (see `CPUSimpleHT`). GPU
upsert/delete kernels are not yet implemented.

# Fields
- `slots_keys`: Flat `MtlVector{K}` of keys (length `n_slots`)
- `slots_vals`: Flat `MtlVector{V}` of values (length `n_slots`)
- `n_slots`: Table capacity (power of two)
- `n_entries`: Number of live entries at construction time
- `empty_key`: Sentinel for an unused slot
- `empty_val`: Sentinel indicating a deleted entry
"""
mutable struct MtlSimpleHT{K,V}
    slots_keys::MtlVector{K}
    slots_vals::MtlVector{V}
    n_slots::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    MtlSimpleHT(cpu_table::CPUSimpleHT{K,V}) -> MtlSimpleHT{K,V}

Upload a CPU-built `CPUSimpleHT` to the Metal GPU.

# Example
```julia
cpu = CPUSimpleHT(keys, vals; load_factor=0.7)
gpu = MtlSimpleHT(cpu)
found, results = query(gpu, query_keys)
```
"""
function MtlSimpleHT(cpu_table::CPUSimpleHT{K,V}) where {K,V}
    cpu_keys = Vector{K}(undef, cpu_table.n_slots)
    cpu_vals = Vector{V}(undef, cpu_table.n_slots)
    for i in 1:cpu_table.n_slots
        cpu_keys[i] = cpu_table.slots[i].key
        cpu_vals[i] = cpu_table.slots[i].val
    end
    return MtlSimpleHT{K,V}(
        MtlVector(cpu_keys),
        MtlVector(cpu_vals),
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
    query!(results::MtlVector{V}, found::MtlVector{Bool}, table::MtlSimpleHT{K,V}, keys::MtlVector{K})

In-place batch query on the Metal GPU.
"""
function query!(
    results::MtlVector{V},
    found::MtlVector{Bool},
    table::MtlSimpleHT{K,V},
    keys::MtlVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found)   >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    threads_per_group = 256
    n_groups = cld(n_queries, threads_per_group)

    @metal threads=threads_per_group groups=n_groups metal_simple_query_kernel!(
        results,
        found,
        table.slots_keys,
        table.slots_vals,
        Int32(table.n_slots),
        keys,
        Int32(n_queries),
        table.empty_key,
        table.empty_val
    )

    Metal.synchronize()
    return nothing
end

"""
    query(table::MtlSimpleHT{K,V}, keys::MtlVector{K}) -> (found::MtlVector{Bool}, results::MtlVector{V})

Allocating batch query with Metal GPU input keys.
"""
function query(table::MtlSimpleHT{K,V}, keys::MtlVector{K}) where {K,V}
    n       = length(keys)
    results = MtlVector(zeros(V, n))
    found   = MtlVector(zeros(Bool, n))
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::MtlSimpleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Allocating batch query with CPU input keys; handles transfers automatically.
"""
function query(table::MtlSimpleHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = MtlVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end
