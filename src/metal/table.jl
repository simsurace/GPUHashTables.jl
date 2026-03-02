# Metal GPU hash table wrapper and operations
# Metal uses 32-slot buckets with full simdgroup (32 threads) per tile

using Metal: MtlVector, MtlDeviceVector, @metal, Metal

"""
    MtlDoubleHT{K,V}

Metal GPU-side double hashing hash table. Created by transferring a CPU table to GPU.

Uses 32-slot buckets for optimal performance with Metal's simdgroup (32 threads).

# Fields
- `buckets`: MtlVector of Bucket32 on GPU (32 slots per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
struct MtlDoubleHT{K,V}
    buckets::MtlVector{Bucket32{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    MtlDoubleHT(cpu_table::CPUDoubleHT{K,V}) -> MtlDoubleHT{K,V}

Transfer a CPU hash table to Metal GPU.

Note: CPU tables use 8-slot buckets; this constructor converts to 32-slot buckets
by repacking entries. The number of Metal buckets may differ from CPU buckets.

# Example
```julia
cpu_table = CPUDoubleHT(keys, vals)
metal_table = MtlDoubleHT(cpu_table)
```
"""
function MtlDoubleHT(cpu_table::CPUDoubleHT{K,V}) where {K,V}
    # Extract all entries from CPU table and repack into 32-slot buckets
    entries = Tuple{K,V}[]
    for bucket in cpu_table.buckets
        for slot in bucket.slots
            if slot.key != cpu_table.empty_key
                push!(entries, (slot.key, slot.val))
            end
        end
    end

    # Calculate number of Metal buckets needed (32 slots each)
    n_entries = length(entries)
    n_buckets = max(1, cld(Int(ceil(n_entries / DEFAULT_LOAD_FACTOR)), METAL_BUCKET_SIZE))

    # Initialize empty buckets
    empty_slot = Slot{K,V}(cpu_table.empty_key, cpu_table.empty_val)
    empty_bucket = Bucket32{K,V}(ntuple(_ -> empty_slot, METAL_BUCKET_SIZE))
    metal_buckets_cpu = fill(empty_bucket, n_buckets)

    # Insert entries using double hashing
    for (key, val) in entries
        h1, h2 = double_hash(key)
        step = n_buckets == 1 ? UInt32(1) : h2 % UInt32(n_buckets - 1) + UInt32(1)

        inserted = false
        for probe in 0:(MAX_PROBES - 1)
            bucket_idx = Int((h1 + step * UInt32(probe)) % UInt32(n_buckets)) + 1
            bucket = metal_buckets_cpu[bucket_idx]

            # Find empty slot in bucket
            for slot_idx in 1:METAL_BUCKET_SIZE
                if bucket.slots[slot_idx].key == cpu_table.empty_key
                    # Insert at this slot
                    new_slots = ntuple(METAL_BUCKET_SIZE) do i
                        i == slot_idx ? Slot{K,V}(key, val) : bucket.slots[i]
                    end
                    metal_buckets_cpu[bucket_idx] = Bucket32{K,V}(new_slots)
                    inserted = true
                    break
                end
            end
            if inserted
                break
            end
        end
    end

    metal_buckets = MtlVector(metal_buckets_cpu)
    return MtlDoubleHT{K,V}(
        metal_buckets,
        n_buckets,
        n_entries,
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
    # Each tile (simdgroup of 32 threads) handles one query
    threads_per_threadgroup = 256  # Must be multiple of METAL_TILE_SIZE (32)
    tiles_per_threadgroup = threads_per_threadgroup ÷ METAL_TILE_SIZE
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

# ============================================================================
# Mutable Metal GPU Hash Table (supports upserts)
# ============================================================================

"""
    MtlMutableDoubleHT{K,V}

Metal GPU-side mutable double hashing hash table with support for concurrent upserts.
Includes per-bucket locks for thread-safe modifications.

Uses 32-slot buckets for optimal performance with Metal's simdgroup (32 threads).

# Fields
- `buckets`: MtlVector of Bucket32 on GPU (32 slots per bucket)
- `locks`: MtlVector of per-bucket locks (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Approximate number of entries (not updated atomically)
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
mutable struct MtlMutableDoubleHT{K,V}
    buckets::MtlVector{Bucket32{K,V}}
    locks::MtlVector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    MtlMutableDoubleHT(cpu_table::CPUDoubleHT{K,V}) -> MtlMutableDoubleHT{K,V}

Transfer a CPU hash table to Metal GPU as a mutable table that supports upserts.

Note: CPU tables use 8-slot buckets; this constructor converts to 32-slot buckets
by repacking entries. The number of Metal buckets may differ from CPU buckets.

# Example
```julia
cpu_table = CPUDoubleHT(keys, vals)
metal_table = MtlMutableDoubleHT(cpu_table)
upsert!(metal_table, new_keys, new_vals)
```
"""
function MtlMutableDoubleHT(cpu_table::CPUDoubleHT{K,V}) where {K,V}
    # Extract all entries from CPU table and repack into 32-slot buckets
    entries = Tuple{K,V}[]
    for bucket in cpu_table.buckets
        for slot in bucket.slots
            if slot.key != cpu_table.empty_key
                push!(entries, (slot.key, slot.val))
            end
        end
    end

    # Calculate number of Metal buckets needed (32 slots each)
    n_entries = length(entries)
    n_buckets = max(1, cld(Int(ceil(n_entries / DEFAULT_LOAD_FACTOR)), METAL_BUCKET_SIZE))

    # Initialize empty buckets
    empty_slot = Slot{K,V}(cpu_table.empty_key, cpu_table.empty_val)
    empty_bucket = Bucket32{K,V}(ntuple(_ -> empty_slot, METAL_BUCKET_SIZE))
    metal_buckets_cpu = fill(empty_bucket, n_buckets)

    # Insert entries using double hashing
    for (key, val) in entries
        h1, h2 = double_hash(key)
        step = n_buckets == 1 ? UInt32(1) : h2 % UInt32(n_buckets - 1) + UInt32(1)

        inserted = false
        for probe in 0:(MAX_PROBES - 1)
            bucket_idx = Int((h1 + step * UInt32(probe)) % UInt32(n_buckets)) + 1
            bucket = metal_buckets_cpu[bucket_idx]

            # Find empty slot in bucket
            for slot_idx in 1:METAL_BUCKET_SIZE
                if bucket.slots[slot_idx].key == cpu_table.empty_key
                    # Insert at this slot
                    new_slots = ntuple(METAL_BUCKET_SIZE) do i
                        i == slot_idx ? Slot{K,V}(key, val) : bucket.slots[i]
                    end
                    metal_buckets_cpu[bucket_idx] = Bucket32{K,V}(new_slots)
                    inserted = true
                    break
                end
            end
            if inserted
                break
            end
        end
    end

    metal_buckets = MtlVector(metal_buckets_cpu)
    metal_locks = Metal.zeros(UInt32, n_buckets)

    return MtlMutableDoubleHT{K,V}(
        metal_buckets,
        metal_locks,
        n_buckets,
        n_entries,
        cpu_table.empty_key,
        cpu_table.empty_val
    )
end

"""
    MtlMutableDoubleHT{K,V}(n_buckets::Int) -> MtlMutableDoubleHT{K,V}

Create an empty mutable Metal GPU hash table with the specified number of buckets.

Each bucket has 32 slots to match Metal's simdgroup size.

# Example
```julia
# Create empty table sized for ~1M entries at 70% load
n_entries = 1_000_000
n_buckets = cld(cld(n_entries, 0.7), METAL_BUCKET_SIZE)  # Use 32 slots per bucket
table = MtlMutableDoubleHT{UInt32,UInt32}(n_buckets)
```
"""
function MtlMutableDoubleHT{K,V}(
    n_buckets::Int;
    empty_key::K = K === UInt32 ? EMPTY_KEY_U32 : typemax(K),
    empty_val::V = V === UInt32 ? EMPTY_VAL_U32 : typemax(V)
) where {K,V}
    # Initialize empty 32-slot buckets
    empty_slot = Slot{K,V}(empty_key, empty_val)
    empty_bucket = Bucket32{K,V}(ntuple(_ -> empty_slot, METAL_BUCKET_SIZE))
    cpu_buckets = fill(empty_bucket, n_buckets)

    metal_buckets = MtlVector(cpu_buckets)
    metal_locks = Metal.zeros(UInt32, n_buckets)

    return MtlMutableDoubleHT{K,V}(
        metal_buckets,
        metal_locks,
        n_buckets,
        0,
        empty_key,
        empty_val
    )
end

"""
    query!(results::MtlVector{V}, found::MtlVector{Bool}, table::MtlMutableDoubleHT{K,V}, keys::MtlVector{K})

Batch query keys on a mutable Metal GPU table.
"""
function query!(
    results::MtlVector{V},
    found::MtlVector{Bool},
    table::MtlMutableDoubleHT{K,V},
    keys::MtlVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    threads_per_threadgroup = 256
    tiles_per_threadgroup = threads_per_threadgroup ÷ METAL_TILE_SIZE
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

    Metal.synchronize()
    return nothing
end

"""
    query(table::MtlMutableDoubleHT{K,V}, keys::MtlVector{K}) -> (found::MtlVector{Bool}, results::MtlVector{V})

Batch query keys on a mutable Metal GPU table, allocating result vectors.
"""
function query(table::MtlMutableDoubleHT{K,V}, keys::MtlVector{K}) where {K,V}
    n = length(keys)
    results = Metal.zeros(V, n)
    found = Metal.zeros(Bool, n)
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::MtlMutableDoubleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query mutable Metal GPU table with CPU keys, handling transfers automatically.
"""
function query(table::MtlMutableDoubleHT{K,V}, keys::Vector{K}) where {K,V}
    metal_keys = MtlVector(keys)
    found_gpu, results_gpu = query(table, metal_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end

"""
    upsert!(status::MtlVector{UInt8}, table::MtlMutableDoubleHT{K,V}, keys::MtlVector{K}, vals::MtlVector{V})

Batch upsert (insert or update) key-value pairs into the Metal GPU table.

# Arguments
- `status`: Pre-allocated GPU vector for operation results
  - 0 = failed (table full or max probes exceeded)
  - 1 = inserted (new key)
  - 2 = updated (existing key)
- `table`: Mutable Metal GPU hash table
- `keys`: GPU vector of keys to upsert
- `vals`: GPU vector of values to upsert

# Notes
Uses host-side retry for operations that fail due to lock contention.

**Important**: Due to Metal's relaxed memory ordering (only `memory_order_relaxed` is supported),
large batches (>128 keys) may have some writes that aren't visible to subsequent queries even
after the operation completes. For reliable results, use batch sizes of 64-128 keys and call
`upsert!` multiple times rather than one large batch. Each call synchronizes with the GPU,
ensuring previous writes are globally visible.
"""
function upsert!(
    status::MtlVector{UInt8},
    table::MtlMutableDoubleHT{K,V},
    keys::MtlVector{K},
    vals::MtlVector{V};
    max_retries::Int = 1000
) where {K,V}
    n_ops = length(keys)
    @assert length(vals) == n_ops "Keys and values must have same length"
    @assert length(status) >= n_ops "Status vector too small"

    if n_ops == 0
        return nothing
    end

    threads_per_threadgroup = 256
    tiles_per_threadgroup = threads_per_threadgroup ÷ METAL_TILE_SIZE

    # Initial attempt
    n_threadgroups = cld(n_ops, tiles_per_threadgroup)
    @metal threads=threads_per_threadgroup groups=n_threadgroups metal_upsert_kernel!(
        status,
        table.buckets,
        table.locks,
        Int32(table.n_buckets),
        keys,
        vals,
        Int32(n_ops),
        table.empty_key
    )
    Metal.synchronize()

    # Host-side retry loop for failed operations
    status_cpu = Vector(status)
    for retry in 1:max_retries
        # Find failed operations
        failed_indices = findall(x -> x == UInt8(0), status_cpu)
        if isempty(failed_indices)
            break
        end

        # Extract failed keys and vals
        keys_cpu = Vector(keys)
        vals_cpu = Vector(vals)
        retry_keys = MtlVector(keys_cpu[failed_indices])
        retry_vals = MtlVector(vals_cpu[failed_indices])
        retry_status = Metal.zeros(UInt8, length(failed_indices))

        # Retry failed operations
        n_retry = length(failed_indices)
        n_threadgroups_retry = cld(n_retry, tiles_per_threadgroup)
        @metal threads=threads_per_threadgroup groups=n_threadgroups_retry metal_upsert_kernel!(
            retry_status,
            table.buckets,
            table.locks,
            Int32(table.n_buckets),
            retry_keys,
            retry_vals,
            Int32(n_retry),
            table.empty_key
        )
        Metal.synchronize()

        # Update status for retried operations
        retry_status_cpu = Vector(retry_status)
        for (i, orig_idx) in enumerate(failed_indices)
            status_cpu[orig_idx] = retry_status_cpu[i]
        end
    end

    # Copy final status back to GPU
    copyto!(status, status_cpu)
    return nothing
end

"""
    upsert!(table::MtlMutableDoubleHT{K,V}, keys::MtlVector{K}, vals::MtlVector{V}) -> MtlVector{UInt8}

Batch upsert key-value pairs, allocating and returning status vector.
"""
function upsert!(
    table::MtlMutableDoubleHT{K,V},
    keys::MtlVector{K},
    vals::MtlVector{V}
) where {K,V}
    n = length(keys)
    status = Metal.zeros(UInt8, n)
    upsert!(status, table, keys, vals)
    return status
end

"""
    upsert!(table::MtlMutableDoubleHT{K,V}, keys::Vector{K}, vals::Vector{V}) -> Vector{UInt8}

Upsert with CPU vectors, handling transfers automatically.
"""
function upsert!(
    table::MtlMutableDoubleHT{K,V},
    keys::Vector{K},
    vals::Vector{V}
) where {K,V}
    metal_keys = MtlVector(keys)
    metal_vals = MtlVector(vals)
    status_gpu = upsert!(table, metal_keys, metal_vals)
    return Vector(status_gpu)
end
