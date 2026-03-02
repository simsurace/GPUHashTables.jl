# CUDA kernels for GPU hash table operations

"""
    query_kernel!(results, found, buckets, n_buckets, keys, n_queries, empty_key)

CUDA kernel for batch querying the hash table.

Each tile of TILE_SIZE threads cooperatively handles one query:
1. Compute hash and probe position
2. Each thread loads one slot from the bucket
3. Ballot vote to find matches or empty slots
4. First matching thread writes result, or continue probing

# Arguments
- `results`: Output array for values (length >= n_queries)
- `found`: Output array for found flags (length >= n_queries)
- `buckets`: Array of Bucket structs
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to query
- `n_queries`: Number of queries
- `empty_key`: Sentinel value for empty slots
"""
function query_kernel!(
    results::CuDeviceVector{V},
    found::CuDeviceVector{Bool},
    buckets::CuDeviceVector{Bucket8{K,V}},
    n_buckets::Int32,
    keys::CuDeviceVector{K},
    n_queries::Int32,
    empty_key::K
) where {K,V}
    # Which query does this tile handle?
    query_idx = tile_id()

    if query_idx > n_queries
        return nothing
    end

    lane = tile_lane()  # 0 to TILE_SIZE-1
    mask = tile_mask()

    key = keys[query_idx]
    h1, h2 = double_hash_gpu(key)

    # Ensure step is non-zero
    # Handle edge case where n_buckets == 1
    if n_buckets == Int32(1)
        step = UInt32(1)
    else
        step = h2 % UInt32(n_buckets - 1) + UInt32(1)
    end

    # Probe sequence
    for probe in Int32(0):Int32(MAX_PROBES - 1)
        bucket_idx = (h1 + step * UInt32(probe)) % UInt32(n_buckets) + UInt32(1)

        # Load bucket - all threads load simultaneously
        bucket = buckets[bucket_idx]

        # Each thread checks one slot (lane corresponds to slot index)
        slot = bucket.slots[lane + 1]

        is_match = slot.key == key
        is_empty = slot.key == empty_key

        # Warp ballot within tile
        match_ballot = tile_ballot(mask, is_match)
        empty_ballot = tile_ballot(mask, is_empty)

        # Check for match
        if match_ballot != UInt32(0)
            # Found! First matching lane writes the result
            winner = first_set_lane(match_ballot)
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
    try_acquire_lock(locks::CuDeviceVector{UInt32}, bucket_idx::Integer) -> Bool

Attempt to acquire the lock for a bucket using atomic CAS.
Returns true if lock was acquired, false if already held.
Only lane 0 should call this.
"""
@inline function try_acquire_lock(locks::CuDeviceVector{UInt32}, bucket_idx::Integer)
    old = CUDA.atomic_cas!(pointer(locks, bucket_idx), LOCK_FREE, LOCK_HELD)
    return old == LOCK_FREE
end

"""
    release_lock(locks::CuDeviceVector{UInt32}, bucket_idx::Integer)

Release the lock for a bucket.
Only lane 0 should call this.
"""
@inline function release_lock(locks::CuDeviceVector{UInt32}, bucket_idx::Integer)
    CUDA.atomic_xchg!(pointer(locks, bucket_idx), LOCK_FREE)
    return nothing
end

"""
    upsert_kernel!(results, buckets, locks, n_buckets, keys, vals, n_ops, empty_key)

CUDA kernel for batch upsert (insert or update) into the hash table.

Each tile of TILE_SIZE threads cooperatively handles one upsert:
1. Compute hash and probe position
2. Try to acquire bucket lock (tile leader only)
3. If locked by another, skip to next probe position
4. Search bucket for key or empty slot
5. Insert/update and release lock

# Arguments
- `results`: Output array for operation status (0=failed, 1=inserted, 2=updated)
- `buckets`: Array of Bucket structs (modified in place)
- `locks`: Array of bucket locks (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to upsert
- `vals`: Input array of values to upsert
- `n_ops`: Number of upsert operations
- `empty_key`: Sentinel value for empty slots
"""
function upsert_kernel!(
    results::CuDeviceVector{UInt8},
    buckets::CuDeviceVector{Bucket8{K,V}},
    locks::CuDeviceVector{UInt32},
    n_buckets::Int32,
    keys::CuDeviceVector{K},
    vals::CuDeviceVector{V},
    n_ops::Int32,
    empty_key::K
) where {K,V}
    # Which operation does this tile handle?
    op_idx = tile_id()

    if op_idx > n_ops
        return nothing
    end

    lane = tile_lane()  # 0 to TILE_SIZE-1
    mask = tile_mask()

    key = keys[op_idx]
    val = vals[op_idx]
    h1, h2 = double_hash_gpu(key)

    # Ensure step is non-zero
    if n_buckets == Int32(1)
        step = UInt32(1)
    else
        step = h2 % UInt32(n_buckets - 1) + UInt32(1)
    end

    # Probe sequence with locking
    for probe in Int32(0):Int32(MAX_PROBES - 1)
        bucket_idx = (h1 + step * UInt32(probe)) % UInt32(n_buckets) + UInt32(1)

        # Lane 0 tries to acquire lock
        lock_acquired = false
        if lane == 0
            lock_acquired = try_acquire_lock(locks, bucket_idx)
        end

        # Broadcast lock status to all lanes in tile
        lock_ballot = tile_ballot(mask, lock_acquired)
        if lock_ballot == UInt32(0)
            # Lock not acquired, try next probe position
            continue
        end

        # Lock acquired! Search bucket
        bucket = buckets[bucket_idx]
        slot = bucket.slots[lane + 1]

        is_match = slot.key == key
        is_empty = slot.key == empty_key

        match_ballot = tile_ballot(mask, is_match)
        empty_ballot = tile_ballot(mask, is_empty)

        # Check for existing key - update
        if match_ballot != UInt32(0)
            winner = first_set_lane(match_ballot)
            if lane == winner
                # Update value using atomic store to the slot
                # We need to write to the bucket's slot
                new_slot = Slot{K,V}(key, val)
                new_slots = ntuple(i -> i == lane + 1 ? new_slot : bucket.slots[i], Val(BUCKET_SIZE))
                buckets[bucket_idx] = Bucket8{K,V}(new_slots)
                results[op_idx] = UInt8(2)  # Updated
            end
            # Release lock
            if lane == 0
                release_lock(locks, bucket_idx)
            end
            return nothing
        end

        # Check for empty slot - insert
        if empty_ballot != UInt32(0)
            winner = first_set_lane(empty_ballot)
            if lane == winner
                new_slot = Slot{K,V}(key, val)
                new_slots = ntuple(i -> i == lane + 1 ? new_slot : bucket.slots[i], Val(BUCKET_SIZE))
                buckets[bucket_idx] = Bucket8{K,V}(new_slots)
                results[op_idx] = UInt8(1)  # Inserted
            end
            # Release lock
            if lane == 0
                release_lock(locks, bucket_idx)
            end
            return nothing
        end

        # Bucket full, no match - release lock and continue probing
        if lane == 0
            release_lock(locks, bucket_idx)
        end
    end

    # Exceeded max probes - operation failed
    if lane == 0
        results[op_idx] = UInt8(0)  # Failed
    end

    return nothing
end
