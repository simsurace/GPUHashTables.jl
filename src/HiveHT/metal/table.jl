# Metal HiveHT hash table type and operations
#
# Metal implementation uses a different data layout than CUDA because
# Metal doesn't support 64-bit atomic CAS. Instead we use:
# - Flat MtlVector{UInt64} for pairs (n_buckets * 32 elements)
# - MtlVector{UInt32} per-slot locks for atomic updates
# - Lock-free queries, per-slot locking for upserts/deletes

"""
    MtlHiveHT{K,V}

Metal GPU HiveHT hash table with 64-bit packed key-value pairs.

Uses lock-free queries and per-slot locking for upserts/deletes
(because Metal doesn't support 64-bit atomic CAS).

# Fields
- `pairs`: MtlVector{UInt64} flat array of packed pairs (n_buckets * 32 elements)
- `locks`: MtlVector{UInt32} per-slot locks (one per slot)
- `freemasks`: MtlVector{UInt32} freemask bitmaps (one per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Approximate number of entries (not updated atomically)
- `empty_key`: Sentinel value for empty key slots

# Type Constraints
Currently only supports `K = V = UInt32` due to the 64-bit packed pair format.

# Notes
Due to Metal's lack of 64-bit atomics, we use per-slot locking instead of
64-bit CAS. Failed operations are automatically retried on the host side.
"""
mutable struct MtlHiveHT{K,V}
    pairs::MtlVector{UInt64}
    locks::MtlVector{UInt32}
    freemasks::MtlVector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
end

# =============================================================================
# Constructors
# =============================================================================

"""
    MtlHiveHT{K,V}(n_buckets::Int; empty_key=typemax(K)) -> MtlHiveHT{K,V}

Create an empty Metal HiveHT hash table with the specified number of buckets.

Each bucket has 32 slots, so total capacity is `n_buckets * 32` entries.
"""
function MtlHiveHT{K,V}(
    n_buckets::Int;
    empty_key::K = K === UInt32 ? HIVE_EMPTY_KEY : typemax(K)
) where {K,V}
    @assert K === UInt32 && V === UInt32 "HiveHT currently only supports UInt32 keys and values"
    @assert n_buckets > 0 "Number of buckets must be positive"

    total_slots = n_buckets * HIVE_BUCKET_SIZE

    # Initialize all pairs as empty
    cpu_pairs = fill(HIVE_EMPTY_PAIR, total_slots)
    gpu_pairs = MtlVector(cpu_pairs)

    # Initialize all locks as free (0)
    gpu_locks = MtlVector(zeros(UInt32, total_slots))

    # Initialize all freemasks to all-ones (all slots free)
    gpu_freemasks = MtlVector(fill(UInt32(0xFFFFFFFF), n_buckets))

    return MtlHiveHT{K,V}(
        gpu_pairs,
        gpu_locks,
        gpu_freemasks,
        n_buckets,
        0,
        empty_key
    )
end

"""
    MtlHiveHT(cpu_table::CPUHiveHT{K,V}) -> MtlHiveHT{K,V}

Transfer a CPU HiveHT table to Metal GPU without invoking the upsert kernel.
The HiveBucket array is flattened into the flat-pair layout used by MtlHiveHT.

# Example
```julia
cpu_table = CPUHiveHT(keys, vals)
gpu_table = MtlHiveHT(cpu_table)
found, results = query(gpu_table, keys)
```
"""
function MtlHiveHT(cpu_table::CPUHiveHT{K,V}) where {K,V}
    n_buckets = cpu_table.n_buckets
    total_slots = n_buckets * HIVE_BUCKET_SIZE

    # Flatten HiveBucket array into the 1-D pairs layout used by MtlHiveHT
    cpu_pairs = Vector{UInt64}(undef, total_slots)
    for b in 1:n_buckets
        bucket = cpu_table.buckets[b]
        base = (b - 1) * HIVE_BUCKET_SIZE
        for s in 1:HIVE_BUCKET_SIZE
            cpu_pairs[base + s] = bucket.pairs[s]
        end
    end

    gpu_pairs = MtlVector(cpu_pairs)
    gpu_locks = MtlVector(zeros(UInt32, total_slots))
    gpu_freemasks = MtlVector(cpu_table.freemasks)

    return MtlHiveHT{K,V}(
        gpu_pairs,
        gpu_locks,
        gpu_freemasks,
        n_buckets,
        cpu_table.n_entries,
        cpu_table.empty_key
    )
end

"""
    MtlHiveHT(keys::Vector{K}, vals::Vector{V}; load_factor=0.7) -> MtlHiveHT{K,V}

Create a Metal HiveHT table and populate it with the given key-value pairs.
"""
function MtlHiveHT(
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
        return MtlHiveHT{K,V}(1; empty_key=empty_key)
    end

    # Calculate number of buckets needed
    total_slots_needed = ceil(Int, n / load_factor)
    n_buckets = max(1, cld(total_slots_needed, HIVE_BUCKET_SIZE))

    # Create empty table
    table = MtlHiveHT{K,V}(n_buckets; empty_key=empty_key)

    # Insert all key-value pairs (with automatic retry for failures)
    status = upsert!(table, keys, vals)

    # Check that all inserts succeeded
    failed = count(s -> s == UPSERT_FAILED, status)
    if failed > 0
        @warn "MtlHiveHT construction: $failed out of $n inserts failed after retries"
    end

    table.n_entries = n - failed

    return table
end

# =============================================================================
# Query Operations
# =============================================================================

"""
    query!(results::MtlVector{V}, found::MtlVector{Bool}, table::MtlHiveHT{K,V}, keys::MtlVector{K})

Batch query keys on the Metal GPU HiveHT table.
"""
function query!(
    results::MtlVector{V},
    found::MtlVector{Bool},
    table::MtlHiveHT{K,V},
    keys::MtlVector{K}
) where {K,V}
    n_queries = length(keys)
    @assert length(results) >= n_queries "Results vector too small"
    @assert length(found) >= n_queries "Found vector too small"

    if n_queries == 0
        return nothing
    end

    # Queries are completely lock-free (read-only), so no sub-batching needed.
    # Unlike upsert!, there are no write visibility concerns.
    threads_per_group = 32  # 1 simdgroup per group, each group handles one query
    n_groups = n_queries

    @metal threads=threads_per_group groups=n_groups metal_hive_query_kernel!(
        results,
        found,
        table.pairs,
        Int32(table.n_buckets),
        keys,
        Int32(n_queries)
    )

    Metal.synchronize()
    return nothing
end

"""
    query(table::MtlHiveHT{K,V}, keys::MtlVector{K}) -> (found::MtlVector{Bool}, results::MtlVector{V})

Batch query keys on the Metal GPU HiveHT table, allocating result vectors.
"""
function query(table::MtlHiveHT{K,V}, keys::MtlVector{K}) where {K,V}
    n = length(keys)
    results = MtlVector(zeros(V, n))
    found = MtlVector(zeros(Bool, n))
    query!(results, found, table, keys)
    return (found, results)
end

"""
    query(table::MtlHiveHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Query Metal GPU HiveHT table with CPU keys, handling transfers automatically.
"""
function query(table::MtlHiveHT{K,V}, keys::Vector{K}) where {K,V}
    gpu_keys = MtlVector(keys)
    found_gpu, results_gpu = query(table, gpu_keys)
    return (Vector(found_gpu), Vector(results_gpu))
end

# =============================================================================
# Upsert Operations (with host-side retry)
# =============================================================================

"""
    upsert!(status::MtlVector{UInt8}, table::MtlHiveHT{K,V}, keys::MtlVector{K}, vals::MtlVector{V}; max_retries=100)

Batch upsert (insert or update) key-value pairs into the Metal GPU HiveHT table.

Failed operations are automatically retried up to `max_retries` times.

Note: Operations are processed in sub-batches of up to 128 concurrent threadgroups
to avoid Metal GPU write visibility issues with large numbers of concurrent groups.
"""
function upsert!(
    status::MtlVector{UInt8},
    table::MtlHiveHT{K,V},
    keys::MtlVector{K},
    vals::MtlVector{V};
    max_retries::Int = 100
) where {K,V}
    n_ops = length(keys)
    @assert length(vals) == n_ops "Keys and values must have same length"
    @assert length(status) >= n_ops "Status vector too small"

    if n_ops == 0
        return nothing
    end

    # Use 1 simdgroup (32 threads) per group
    threads_per_group = 32

    # IMPORTANT: Limit batch size to avoid Metal GPU write visibility issues
    # Testing showed that >250 concurrent groups can cause writes to be lost
    max_groups_per_launch = 128

    # First attempt - process in sub-batches
    cpu_keys = Vector(keys)
    cpu_vals = Vector(vals)
    cpu_status = Vector(status)

    for batch_start in 1:max_groups_per_launch:n_ops
        batch_end = min(batch_start + max_groups_per_launch - 1, n_ops)
        batch_n = batch_end - batch_start + 1

        batch_keys = MtlVector(cpu_keys[batch_start:batch_end])
        batch_vals = MtlVector(cpu_vals[batch_start:batch_end])
        batch_status = MtlVector(zeros(UInt8, batch_n))

        @metal threads=threads_per_group groups=batch_n metal_hive_upsert_kernel!(
            batch_status,
            table.pairs,
            table.locks,
            table.freemasks,
            Int32(table.n_buckets),
            batch_keys,
            batch_vals,
            Int32(batch_n)
        )
        Metal.synchronize()

        cpu_status[batch_start:batch_end] = Vector(batch_status)
    end

    copyto!(status, MtlVector(cpu_status))

    # Retry failed operations
    failed_indices = findall(s -> s == UPSERT_FAILED, cpu_status)

    retry = 0
    while !isempty(failed_indices) && retry < max_retries
        retry += 1

        # Extract failed keys/vals
        n_failed = length(failed_indices)
        cpu_failed_keys = cpu_keys[failed_indices]
        cpu_failed_vals = cpu_vals[failed_indices]
        cpu_failed_status = zeros(UInt8, n_failed)

        # Process retry in sub-batches too
        for batch_start in 1:max_groups_per_launch:n_failed
            batch_end = min(batch_start + max_groups_per_launch - 1, n_failed)
            batch_n = batch_end - batch_start + 1

            batch_keys = MtlVector(cpu_failed_keys[batch_start:batch_end])
            batch_vals = MtlVector(cpu_failed_vals[batch_start:batch_end])
            batch_status = MtlVector(zeros(UInt8, batch_n))

            @metal threads=threads_per_group groups=batch_n metal_hive_upsert_kernel!(
                batch_status,
                table.pairs,
                table.locks,
                table.freemasks,
                Int32(table.n_buckets),
                batch_keys,
                batch_vals,
                Int32(batch_n)
            )
            Metal.synchronize()

            cpu_failed_status[batch_start:batch_end] = Vector(batch_status)
        end

        # Update status for succeeded operations
        for (i, orig_idx) in enumerate(failed_indices)
            if cpu_failed_status[i] != UPSERT_FAILED
                cpu_status[orig_idx] = cpu_failed_status[i]
            end
        end

        # Find remaining failures
        failed_indices = findall(s -> s == UPSERT_FAILED, cpu_status)
    end

    # Post-insert verification: read raw table data on CPU and check which
    # keys are actually present. Re-insert any that are missing.
    # This catches GPU write visibility issues that affect both writes and queries.
    cpu_pairs = Vector(table.pairs)

    # Build a set of all keys in the table (CPU-side scan)
    present_keys = Set{K}()
    for pair in cpu_pairs
        if pair != HIVE_EMPTY_PAIR && pair != HIVE_TOMBSTONE
            present_keys = push!(present_keys, unpack_key(pair))
        end
    end

    # Find keys that reported success but aren't in the table
    missing_mask = [cpu_status[i] != UPSERT_FAILED && !(cpu_keys[i] in present_keys)
                    for i in 1:n_ops]
    missing_indices = findall(missing_mask)

    verify_retry = 0
    while !isempty(missing_indices) && verify_retry < max_retries
        verify_retry += 1

        n_missing = length(missing_indices)
        missing_keys = cpu_keys[missing_indices]
        missing_vals = cpu_vals[missing_indices]

        for batch_start in 1:max_groups_per_launch:n_missing
            batch_end = min(batch_start + max_groups_per_launch - 1, n_missing)
            batch_n = batch_end - batch_start + 1

            batch_keys = MtlVector(missing_keys[batch_start:batch_end])
            batch_vals = MtlVector(missing_vals[batch_start:batch_end])
            batch_status = MtlVector(zeros(UInt8, batch_n))

            @metal threads=threads_per_group groups=batch_n metal_hive_upsert_kernel!(
                batch_status,
                table.pairs,
                table.locks,
                table.freemasks,
                Int32(table.n_buckets),
                batch_keys,
                batch_vals,
                Int32(batch_n)
            )
            Metal.synchronize()
        end

        # Re-verify on CPU
        cpu_pairs = Vector(table.pairs)
        present_keys = Set{K}()
        for pair in cpu_pairs
            if pair != HIVE_EMPTY_PAIR && pair != HIVE_TOMBSTONE
                present_keys = push!(present_keys, unpack_key(pair))
            end
        end

        missing_mask = [cpu_status[i] != UPSERT_FAILED && !(cpu_keys[i] in present_keys)
                        for i in 1:n_ops]
        missing_indices = findall(missing_mask)
    end

    # Copy final status back to GPU
    copyto!(status, MtlVector(cpu_status))

    return nothing
end

"""
    upsert!(table::MtlHiveHT{K,V}, keys::MtlVector{K}, vals::MtlVector{V}; max_retries=100) -> MtlVector{UInt8}

Batch upsert key-value pairs, allocating and returning status vector.
"""
function upsert!(
    table::MtlHiveHT{K,V},
    keys::MtlVector{K},
    vals::MtlVector{V};
    max_retries::Int = 100
) where {K,V}
    n = length(keys)
    status = MtlVector(zeros(UInt8, n))
    upsert!(status, table, keys, vals; max_retries=max_retries)
    return status
end

"""
    upsert!(table::MtlHiveHT{K,V}, keys::Vector{K}, vals::Vector{V}; max_retries=100) -> Vector{UInt8}

Upsert with CPU vectors, handling transfers automatically.
"""
function upsert!(
    table::MtlHiveHT{K,V},
    keys::Vector{K},
    vals::Vector{V};
    max_retries::Int = 100
) where {K,V}
    gpu_keys = MtlVector(keys)
    gpu_vals = MtlVector(vals)
    status_gpu = upsert!(table, gpu_keys, gpu_vals; max_retries=max_retries)
    return Vector(status_gpu)
end

# =============================================================================
# Delete Operations (with host-side retry)
# =============================================================================

"""
    delete!(status::MtlVector{UInt8}, table::MtlHiveHT{K,V}, keys::MtlVector{K}; max_retries=100)

Batch delete keys from the Metal GPU HiveHT table.

Deletion marks slots as tombstones, which can be reused for future inserts.
Failed operations are automatically retried up to `max_retries` times.

Note: Operations are processed in sub-batches of up to 128 concurrent threadgroups
to avoid Metal GPU write visibility issues with large numbers of concurrent groups.
"""
function Base.delete!(
    status::MtlVector{UInt8},
    table::MtlHiveHT{K,V},
    keys::MtlVector{K};
    max_retries::Int = 100
) where {K,V}
    n_ops = length(keys)
    @assert length(status) >= n_ops "Status vector too small"

    if n_ops == 0
        return nothing
    end

    # Use 1 simdgroup (32 threads) per group
    threads_per_group = 32

    # IMPORTANT: Limit batch size to avoid Metal GPU write visibility issues
    max_groups_per_launch = 128

    cpu_keys = Vector(keys)
    cpu_status = Vector(status)

    for batch_start in 1:max_groups_per_launch:n_ops
        batch_end = min(batch_start + max_groups_per_launch - 1, n_ops)
        batch_n = batch_end - batch_start + 1

        batch_keys = MtlVector(cpu_keys[batch_start:batch_end])
        batch_status = MtlVector(zeros(UInt8, batch_n))

        @metal threads=threads_per_group groups=batch_n metal_hive_delete_kernel!(
            batch_status,
            table.pairs,
            table.locks,
            table.freemasks,
            Int32(table.n_buckets),
            batch_keys,
            Int32(batch_n)
        )
        Metal.synchronize()

        cpu_status[batch_start:batch_end] = Vector(batch_status)
    end

    copyto!(status, MtlVector(cpu_status))

    # Retry failed operations
    failed_indices = findall(s -> s == DELETE_FAILED, cpu_status)

    retry = 0
    while !isempty(failed_indices) && retry < max_retries
        retry += 1

        n_failed = length(failed_indices)
        cpu_failed_keys = cpu_keys[failed_indices]
        cpu_failed_status = zeros(UInt8, n_failed)

        for batch_start in 1:max_groups_per_launch:n_failed
            batch_end = min(batch_start + max_groups_per_launch - 1, n_failed)
            batch_n = batch_end - batch_start + 1

            batch_keys = MtlVector(cpu_failed_keys[batch_start:batch_end])
            batch_status = MtlVector(zeros(UInt8, batch_n))

            @metal threads=threads_per_group groups=batch_n metal_hive_delete_kernel!(
                batch_status,
                table.pairs,
                table.locks,
                table.freemasks,
                Int32(table.n_buckets),
                batch_keys,
                Int32(batch_n)
            )
            Metal.synchronize()

            cpu_failed_status[batch_start:batch_end] = Vector(batch_status)
        end

        for (i, orig_idx) in enumerate(failed_indices)
            if cpu_failed_status[i] != DELETE_FAILED
                cpu_status[orig_idx] = cpu_failed_status[i]
            end
        end

        failed_indices = findall(s -> s == DELETE_FAILED, cpu_status)
    end

    # Copy final status back to GPU
    copyto!(status, MtlVector(cpu_status))

    return nothing
end

"""
    delete!(table::MtlHiveHT{K,V}, keys::MtlVector{K}) -> MtlVector{UInt8}

Batch delete keys, allocating and returning status vector.
"""
function Base.delete!(
    table::MtlHiveHT{K,V},
    keys::MtlVector{K}
) where {K,V}
    n = length(keys)
    status = MtlVector(zeros(UInt8, n))
    delete!(status, table, keys)
    return status
end

"""
    delete!(table::MtlHiveHT{K,V}, keys::Vector{K}) -> Vector{UInt8}

Delete with CPU vectors, handling transfers automatically.
"""
function Base.delete!(
    table::MtlHiveHT{K,V},
    keys::Vector{K}
) where {K,V}
    gpu_keys = MtlVector(keys)
    status_gpu = delete!(table, gpu_keys)
    return Vector(status_gpu)
end
