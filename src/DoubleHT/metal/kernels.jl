# Metal kernels for GPU hash table operations
# Metal uses 32-slot buckets with full simdgroup (32 threads) per tile

# Helper to update a single slot in a 32-slot bucket at a given lane index
# Uses @generated to create compile-time dispatch for all 32 lanes
@generated function update_bucket_slot32(bucket::Bucket32{K,V}, lane::Int, new_slot::Slot{K,V}) where {K,V}
    # Generate code that checks lane and builds new bucket
    cases = [
        :(if lane == $(i-1)
            return Bucket32{K,V}(($([:( $(j == i) ? new_slot : bucket.slots[$j]) for j in 1:32]...),))
        end)
        for i in 1:32
    ]
    quote
        $(cases...)
        return bucket  # Fallback (shouldn't happen)
    end
end

"""
    metal_query_kernel!(results, found, buckets, n_buckets, keys, n_queries, empty_key)

Metal kernel for batch querying the hash table.

Each tile (full simdgroup of 32 threads) cooperatively handles one query:
1. Compute hash and probe position
2. Each thread loads one slot from the 32-slot bucket
3. Ballot vote to find matches or empty slots
4. First matching thread writes result, or continue probing

# Arguments
- `results`: Output array for values (length >= n_queries)
- `found`: Output array for found flags (length >= n_queries)
- `buckets`: Array of Bucket32 structs (32 slots each)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to query
- `n_queries`: Number of queries
- `empty_key`: Sentinel value for empty slots
"""
function metal_query_kernel!(
    results::MtlDeviceVector{V},
    found::MtlDeviceVector{Bool},
    buckets::MtlDeviceVector{Bucket32{K,V}},
    n_buckets::Int32,
    keys::MtlDeviceVector{K},
    n_queries::Int32,
    empty_key::K
) where {K,V}
    # Which query does this tile handle?
    query_idx = metal_tile_id()

    if query_idx > n_queries
        return nothing
    end

    lane = metal_tile_lane()  # 0 to 31

    key = keys[query_idx]
    h1, h2 = double_hash_gpu(key)

    # Odd step is always coprime with a power-of-two n_buckets.
    step = h2 | UInt32(1)

    # Probe sequence
    for probe in Int32(0):Int32(MAX_PROBES - 1)
        bucket_idx = ((h1 + step * UInt32(probe)) & UInt32(n_buckets - 1)) + UInt32(1)

        # Load bucket - all 32 threads load simultaneously
        bucket = buckets[bucket_idx]

        # Each thread checks one slot (lane corresponds to slot index)
        slot = bucket.slots[lane + 1]

        is_match = slot.key == key
        is_empty = slot.key == empty_key

        # SIMD ballot across full simdgroup (all 32 threads)
        match_ballot = metal_tile_ballot(is_match)
        empty_ballot = metal_tile_ballot(is_empty)

        # Check for match
        if match_ballot != UInt32(0)
            # Found! First matching lane writes the result
            winner = metal_first_set_lane(match_ballot)
            if lane == winner
                results[query_idx] = slot.val
                found[query_idx] = true
            end
            return nothing
        end

        # Check for empty - key not in table
        if empty_ballot != UInt32(0)
            # Not found - first lane writes result
            if lane == 0
                found[query_idx] = false
            end
            return nothing
        end

        # Bucket full, no match - continue to next probe position
    end

    # Exceeded max probes - key not found
    if lane == 0
        found[query_idx] = false
    end

    return nothing
end

"""
    metal_try_acquire_lock(locks::MtlDeviceVector{UInt32}, bucket_idx::Integer) -> Bool

Attempt to acquire the lock for a bucket using atomic CAS.
Returns true if lock was acquired, false if already held.
Only lane 0 should call this.
"""
@inline function metal_try_acquire_lock(locks::MtlDeviceVector{UInt32}, bucket_idx::Integer)
    old = Metal.atomic_compare_exchange_weak_explicit(
        pointer(locks, bucket_idx),
        LOCK_FREE,
        LOCK_HELD
    )
    return old == LOCK_FREE
end

"""
    metal_release_lock(locks::MtlDeviceVector{UInt32}, bucket_idx::Integer)

Release the lock for a bucket.
Only lane 0 should call this.
"""
@inline function metal_release_lock(locks::MtlDeviceVector{UInt32}, bucket_idx::Integer)
    Metal.atomic_store_explicit(pointer(locks, bucket_idx), LOCK_FREE)
    return nothing
end

"""
    metal_upsert_kernel!(results, buckets, locks, n_buckets, keys, vals, n_ops, empty_key)

Metal kernel for batch upsert (insert or update) into the hash table.

Each tile (full simdgroup of 32 threads) cooperatively handles one upsert:
1. Compute hash and probe position
2. Try to acquire bucket lock (lane 0 only, single attempt)
3. If lock not acquired, return FAILED (host will retry)
4. All 32 threads search the 32-slot bucket for key or empty slot
5. Insert/update, sync, and release lock

Using full simdgroup as tile eliminates intra-simdgroup deadlock since
there's only one tile per simdgroup competing for locks.

# Arguments
- `results`: Output array for operation status (0=failed, 1=inserted, 2=updated)
- `buckets`: Array of Bucket32 structs (modified in place, 32 slots each)
- `locks`: Array of bucket locks (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to upsert
- `vals`: Input array of values to upsert
- `n_ops`: Number of upsert operations
- `empty_key`: Sentinel value for empty slots
"""
function metal_upsert_kernel!(
    results::MtlDeviceVector{UInt8},
    buckets::MtlDeviceVector{Bucket32{K,V}},
    locks::MtlDeviceVector{UInt32},
    n_buckets::Int32,
    keys::MtlDeviceVector{K},
    vals::MtlDeviceVector{V},
    n_ops::Int32,
    empty_key::K
) where {K,V}
    # Which operation does this tile (simdgroup) handle?
    op_idx = metal_tile_id()

    if op_idx > n_ops
        return nothing
    end

    lane = metal_tile_lane()  # 0 to 31

    key = keys[op_idx]
    val = vals[op_idx]
    h1, h2 = double_hash_gpu(key)

    # Odd step is always coprime with a power-of-two n_buckets.
    step = h2 | UInt32(1)

    # Probe sequence with locking
    # When lock is contended, we return FAILED and let host retry
    # No intra-simdgroup deadlock since tile == simdgroup
    for probe in Int32(0):Int32(MAX_PROBES - 1)
        bucket_idx = ((h1 + step * UInt32(probe)) & UInt32(n_buckets - 1)) + UInt32(1)

        # Lane 0 tries to acquire lock (single attempt, no spinning)
        lock_acquired = false
        if lane == 0
            lock_acquired = metal_try_acquire_lock(locks, bucket_idx)
        end

        # Broadcast lock status to all lanes in tile (simdgroup)
        lock_ballot = metal_tile_ballot(lock_acquired)
        if lock_ballot == UInt32(0)
            # Lock not acquired - mark as failed for host retry
            # Do NOT continue to next probe (would break hash table invariant)
            if lane == 0
                results[op_idx] = UInt8(0)  # Failed - needs retry
            end
            return nothing
        end

        # Lock acquired! All 32 threads search the 32-slot bucket
        bucket = buckets[bucket_idx]
        slot = bucket.slots[lane + 1]

        is_match = slot.key == key
        is_empty = slot.key == empty_key

        match_ballot = metal_tile_ballot(is_match)
        empty_ballot = metal_tile_ballot(is_empty)

        # Check for existing key - update
        if match_ballot != UInt32(0)
            winner = metal_first_set_lane(match_ballot)
            if lane == winner
                new_slot = Slot{K,V}(key, val)
                buckets[bucket_idx] = update_bucket_slot32(bucket, lane, new_slot)
                results[op_idx] = UInt8(2)  # Updated
            end
            # Sync simdgroup before releasing lock
            metal_tile_sync()
            if lane == 0
                metal_release_lock(locks, bucket_idx)
            end
            return nothing
        end

        # Check for empty slot - insert
        if empty_ballot != UInt32(0)
            winner = metal_first_set_lane(empty_ballot)
            if lane == winner
                new_slot = Slot{K,V}(key, val)
                buckets[bucket_idx] = update_bucket_slot32(bucket, lane, new_slot)
                results[op_idx] = UInt8(1)  # Inserted
            end
            # Sync simdgroup before releasing lock
            metal_tile_sync()
            if lane == 0
                metal_release_lock(locks, bucket_idx)
            end
            return nothing
        end

        # Bucket full, no match - release lock and continue probing
        if lane == 0
            metal_release_lock(locks, bucket_idx)
        end
    end

    # Exceeded max probes - operation failed (table full)
    if lane == 0
        results[op_idx] = UInt8(0)  # Failed
    end

    return nothing
end
