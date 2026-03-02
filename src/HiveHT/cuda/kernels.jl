# CUDA kernels for HiveHT hash table operations
# Implements WCME (query) and WABC (upsert) protocols from the Hive paper

using CUDA: threadIdx, blockIdx, blockDim, vote_ballot_sync, shfl_sync

# =============================================================================
# Warp utilities for HiveHT (32-thread warps)
# =============================================================================

"""
    hive_warp_id() -> Int

Get the warp ID for the current thread (1-indexed).
Each warp handles one operation in HiveHT.
"""
@inline function hive_warp_id()
    thread_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    return (thread_id - 1) ÷ 32 + 1
end

"""
    hive_lane_id() -> Int

Get the lane (position) within the warp for the current thread (0-indexed, 0 to 31).
"""
@inline function hive_lane_id()
    return (threadIdx().x - 1) % 32
end

"""
    hive_ballot(predicate::Bool) -> UInt32

Perform ballot vote across the full warp (32 threads).
Returns a bitmask where bit i is set if thread i's predicate is true.
"""
@inline function hive_ballot(predicate::Bool)
    return vote_ballot_sync(0xFFFFFFFF, predicate)
end

"""
    hive_shfl(val, src_lane::Int) -> typeof(val)

Broadcast a value from src_lane to all threads in the warp.
"""
@inline function hive_shfl(val::UInt32, src_lane::Int32)
    return shfl_sync(0xFFFFFFFF, val, src_lane)
end

"""
    hive_first_set_lane(ballot::UInt32) -> Int

Find the first set bit in a ballot result (0-indexed lane number).
Returns -1 if no bits are set.
"""
@inline function hive_first_set_lane(ballot::UInt32)
    if ballot == UInt32(0)
        return -1
    end
    return trailing_zeros(ballot)
end

# =============================================================================
# Query Kernel (WCME Protocol - Lock-Free)
# =============================================================================

"""
    hive_query_kernel!(results, found, buckets, n_buckets, keys, n_queries)

CUDA kernel for batch querying the HiveHT hash table using WCME protocol.

Each warp (32 threads) cooperatively handles one query:
1. All 32 threads load their slot's pair in parallel (coalesced)
2. Each thread extracts the key and compares to the query key
3. Ballot vote to find matching lane
4. Matching thread writes result, or continue probing

This is completely lock-free - no synchronization needed.

# Arguments
- `results`: Output array for values (length >= n_queries)
- `found`: Output array for found flags (length >= n_queries)
- `buckets`: Array of HiveBucket structs (32 packed pairs each)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to query
- `n_queries`: Number of queries
"""
function hive_query_kernel!(
    results::CuDeviceVector{UInt32},
    found::CuDeviceVector{Bool},
    buckets::CuDeviceVector{HiveBucket},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    n_queries::Int32
)
    # Which query does this warp handle?
    query_idx = hive_warp_id()

    if query_idx > n_queries
        return nothing
    end

    lane = hive_lane_id()  # 0 to 31

    key = keys[query_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets) + UInt32(1)

        # Load bucket - all 32 threads load their pair simultaneously (coalesced)
        bucket = buckets[bucket_idx]
        pair = bucket.pairs[lane + 1]

        # Extract key from packed pair
        pair_key = unpack_key_gpu(pair)

        # Check for match (but not if it's a tombstone)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        # Warp ballot
        match_ballot = hive_ballot(is_match)
        empty_ballot = hive_ballot(is_empty)

        # Check for match
        if match_ballot != UInt32(0)
            # Found! First matching lane writes the result
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                results[query_idx] = unpack_val_gpu(pair)
                found[query_idx] = true
            end
            return nothing
        end

        # Check for empty slot - key not in table
        # (We only stop at empty, not tombstones - continue probing past tombstones)
        if empty_ballot != UInt32(0)
            # Not found - lane 0 writes result
            if lane == 0
                found[query_idx] = false
            end
            return nothing
        end

        # Bucket has no empty slots and no match - continue to next probe position
    end

    # Exceeded max probes - key not found
    if lane == 0
        found[query_idx] = false
    end

    return nothing
end

# =============================================================================
# Upsert Kernel (WABC Protocol)
# =============================================================================

"""
    hive_upsert_kernel!(status, buckets, freemasks, n_buckets, keys, vals, n_ops)

CUDA kernel for batch upsert (insert or update) into the HiveHT hash table.

Uses the WABC (Warp-Aggregated-Bitmask-Claim) protocol:
1. Lane 0 loads freemask, broadcasts to all lanes
2. All lanes load their pair and check for match or free slot
3. Elect winner (first match for update, first free for insert)
4. Winner does 64-bit atomicCAS
5. On insert success, atomically update freemask

# Arguments
- `status`: Output array for operation status (0=failed, 1=inserted, 2=updated)
- `buckets`: Array of HiveBucket structs (modified in place)
- `freemasks`: Array of freemask bitmaps (modified in place)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to upsert
- `vals`: Input array of values to upsert
- `n_ops`: Number of upsert operations
"""
function hive_upsert_kernel!(
    status::CuDeviceVector{UInt8},
    buckets::CuDeviceVector{HiveBucket},
    freemasks::CuDeviceVector{UInt32},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    vals::CuDeviceVector{UInt32},
    n_ops::Int32
)
    # Which operation does this warp handle?
    op_idx = hive_warp_id()

    if op_idx > n_ops
        return nothing
    end

    lane = hive_lane_id()  # 0 to 31

    key = keys[op_idx]
    val = vals[op_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets) + UInt32(1)

        # All threads load freemask directly (same address = cache broadcast)
        freemask = freemasks[bucket_idx]

        # All 32 threads load their pair
        bucket = buckets[bucket_idx]
        pair = bucket.pairs[lane + 1]
        pair_key = unpack_key_gpu(pair)

        # Check for existing key (update case)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        match_ballot = hive_ballot(is_match)

        if match_ballot != UInt32(0)
            # Found existing key - update it
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                new_pair = pack_pair_gpu(key, val)
                # Atomic CAS to update the pair
                # We need to get pointer to the specific slot
                bucket_ptr = pointer(buckets, bucket_idx)
                # The pairs are at offset 0 in the struct, each is 8 bytes
                pair_ptr = reinterpret(Core.LLVMPtr{UInt64, 1}, bucket_ptr) + lane
                old = CUDA.atomic_cas!(pair_ptr, pair, new_pair)
                if old == pair
                    status[op_idx] = UPSERT_UPDATED
                else
                    # CAS failed - another thread modified it
                    status[op_idx] = UPSERT_FAILED
                end
            end
            return nothing
        end

        # Check for free or tombstone slot (insert case)
        is_free = ((freemask >> lane) & UInt32(1)) == UInt32(1)
        is_tombstone = (pair == HIVE_TOMBSTONE)
        is_insertable = is_free || is_tombstone

        insertable_ballot = hive_ballot(is_insertable)

        if insertable_ballot != UInt32(0)
            # Found insertable slot
            winner = hive_first_set_lane(insertable_ballot)
            if lane == winner
                new_pair = pack_pair_gpu(key, val)
                # Determine expected value for CAS
                expected = is_tombstone ? HIVE_TOMBSTONE : HIVE_EMPTY_PAIR

                bucket_ptr = pointer(buckets, bucket_idx)
                pair_ptr = reinterpret(Core.LLVMPtr{UInt64, 1}, bucket_ptr) + lane
                old = CUDA.atomic_cas!(pair_ptr, expected, new_pair)

                if old == expected
                    # Successfully inserted - update freemask
                    # Clear the bit for this lane
                    CUDA.atomic_and!(pointer(freemasks, bucket_idx), ~(UInt32(1) << lane))
                    status[op_idx] = UPSERT_INSERTED
                else
                    # CAS failed - slot was taken by another thread
                    status[op_idx] = UPSERT_FAILED
                end
            end
            return nothing
        end

        # Bucket is full with no matches - continue to next probe position
    end

    # Exceeded max probes - operation failed
    if lane == 0
        status[op_idx] = UPSERT_FAILED
    end

    return nothing
end

# =============================================================================
# Delete Kernel
# =============================================================================

"""
    hive_delete_kernel!(status, buckets, freemasks, n_buckets, keys, n_ops)

CUDA kernel for batch deletion from the HiveHT hash table.

Deletion replaces the pair with a TOMBSTONE sentinel value and marks
the slot as free in the freemask (so it can be reused for inserts).

# Arguments
- `status`: Output array for operation status (0=not found, 1=deleted)
- `buckets`: Array of HiveBucket structs (modified in place)
- `freemasks`: Array of freemask bitmaps (modified in place)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to delete
- `n_ops`: Number of delete operations
"""
function hive_delete_kernel!(
    status::CuDeviceVector{UInt8},
    buckets::CuDeviceVector{HiveBucket},
    freemasks::CuDeviceVector{UInt32},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    n_ops::Int32
)
    # Which operation does this warp handle?
    op_idx = hive_warp_id()

    if op_idx > n_ops
        return nothing
    end

    lane = hive_lane_id()  # 0 to 31

    key = keys[op_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets) + UInt32(1)

        # All 32 threads load their pair
        bucket = buckets[bucket_idx]
        pair = bucket.pairs[lane + 1]
        pair_key = unpack_key_gpu(pair)

        # Check for match (not tombstone)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        match_ballot = hive_ballot(is_match)
        empty_ballot = hive_ballot(is_empty)

        if match_ballot != UInt32(0)
            # Found the key - delete it
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                bucket_ptr = pointer(buckets, bucket_idx)
                pair_ptr = reinterpret(Core.LLVMPtr{UInt64, 1}, bucket_ptr) + lane
                old = CUDA.atomic_cas!(pair_ptr, pair, HIVE_TOMBSTONE)

                if old == pair
                    # Successfully deleted - mark slot as free
                    CUDA.atomic_or!(pointer(freemasks, bucket_idx), UInt32(1) << lane)
                    status[op_idx] = DELETE_SUCCESS
                else
                    # CAS failed - key was modified by another thread
                    status[op_idx] = DELETE_FAILED
                end
            end
            return nothing
        end

        # If we found an empty slot, key doesn't exist
        if empty_ballot != UInt32(0)
            if lane == 0
                status[op_idx] = DELETE_FAILED
            end
            return nothing
        end

        # Continue probing
    end

    # Exceeded max probes - key not found
    if lane == 0
        status[op_idx] = DELETE_FAILED
    end

    return nothing
end
