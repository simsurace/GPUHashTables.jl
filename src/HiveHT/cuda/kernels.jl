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
    hive_lane_id() -> Int32

Get the lane (position) within the warp for the current thread (0-indexed, 0 to 31).
"""
@inline function hive_lane_id()
    return (threadIdx().x - Int32(1)) % Int32(32)
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
    hive_shfl(val, src_lane::Int32) -> typeof(val)

Broadcast a value from src_lane to all threads in the warp.
"""
@inline function hive_shfl(val::UInt32, src_lane::Int32)
    return shfl_sync(0xFFFFFFFF, val, src_lane)
end

"""
    hive_first_set_lane(ballot::UInt32) -> Int32

Find the first set bit in a ballot result (0-indexed lane number).
Returns -1 if no bits are set.
"""
@inline function hive_first_set_lane(ballot::UInt32)
    if ballot == UInt32(0)
        return Int32(-1)
    end
    return Int32(trailing_zeros(ballot))
end

# =============================================================================
# Query Kernel (WCME Protocol - Lock-Free)
# =============================================================================

"""
    hive_query_kernel!(results, found, pairs, n_buckets, keys, n_queries)

CUDA kernel for batch querying the HiveHT hash table using WCME protocol.

Each warp (32 threads) cooperatively handles one query:
1. All 32 threads load their slot's pair in parallel (coalesced)
2. Each thread extracts the key and compares to the query key
3. Ballot vote to find matching lane
4. Matching thread writes result, or continue probing

The pairs array is flat (n_buckets × 32 elements). Thread with lane `l` in the
warp probing bucket `b` (0-indexed) loads element `b*32 + l + 1` (1-indexed).
All 32 loads in the warp are to consecutive addresses, giving a perfectly
coalesced 256-byte read per probe.

# Arguments
- `results`: Output array for values (length >= n_queries)
- `found`: Output array for found flags (length >= n_queries)
- `pairs`: Flat array of UInt64 pairs (n_buckets × 32 elements)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to query
- `n_queries`: Number of queries
"""
function hive_query_kernel!(
    results::CuDeviceVector{UInt32},
    found::CuDeviceVector{Bool},
    pairs::CuDeviceVector{UInt64},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    n_queries::Int32
)
    query_idx = hive_warp_id()

    if query_idx > n_queries
        return nothing
    end

    lane = hive_lane_id()  # Int32, 0 to 31

    key = keys[query_idx]
    h   = hive_hash_gpu(key)

    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        # 0-indexed bucket
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)

        # Coalesced load: thread `lane` reads slot `lane` of this bucket.
        # All 32 threads access consecutive 8-byte words — one 256-byte transaction.
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)
        pair     = pairs[slot_idx]

        pair_key = unpack_key_gpu(pair)

        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        match_ballot = hive_ballot(is_match)
        empty_ballot = hive_ballot(is_empty)

        if match_ballot != UInt32(0)
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                results[query_idx] = unpack_val_gpu(pair)
                found[query_idx]   = true
            end
            return nothing
        end

        if empty_ballot != UInt32(0)
            if lane == Int32(0)
                found[query_idx] = false
            end
            return nothing
        end
    end

    if lane == Int32(0)
        found[query_idx] = false
    end

    return nothing
end

# =============================================================================
# Upsert Kernel (WABC Protocol)
# =============================================================================

"""
    hive_upsert_kernel!(status, pairs, freemasks, n_buckets, keys, vals, n_ops)

CUDA kernel for batch upsert (insert or update) into the HiveHT hash table.

Uses the WABC (Warp-Aggregated-Bitmask-Claim) protocol:
1. All lanes load freemask (same address → cache broadcast)
2. All lanes load their pair via coalesced flat-array indexing
3. Elect winner (first match for update, first free for insert)
4. Winner does 64-bit atomicCAS directly on the pairs array
5. On insert success, atomically update freemask

# Arguments
- `status`: Output array for operation status (0=failed, 1=inserted, 2=updated)
- `pairs`: Flat array of UInt64 pairs (modified in place)
- `freemasks`: Array of freemask bitmaps (modified in place)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to upsert
- `vals`: Input array of values to upsert
- `n_ops`: Number of upsert operations
"""
function hive_upsert_kernel!(
    status::CuDeviceVector{UInt8},
    pairs::CuDeviceVector{UInt64},
    freemasks::CuDeviceVector{UInt32},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    vals::CuDeviceVector{UInt32},
    n_ops::Int32
)
    op_idx = hive_warp_id()

    if op_idx > n_ops
        return nothing
    end

    lane = hive_lane_id()  # Int32, 0 to 31

    key = keys[op_idx]
    val = vals[op_idx]
    h   = hive_hash_gpu(key)

    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)  # 0-indexed

        # All threads load freemask from the same address (cache broadcast)
        freemask = freemasks[bucket_idx + UInt32(1)]

        # Coalesced load: each thread reads its own 8-byte slot
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)
        pair     = pairs[slot_idx]
        pair_key = unpack_key_gpu(pair)

        # Update case: key already present
        is_match     = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        match_ballot = hive_ballot(is_match)

        if match_ballot != UInt32(0)
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                new_pair = pack_pair_gpu(key, val)
                old      = CUDA.atomic_cas!(pointer(pairs, slot_idx), pair, new_pair)
                if old == pair
                    status[op_idx] = UPSERT_UPDATED
                else
                    status[op_idx] = UPSERT_FAILED
                end
            end
            return nothing
        end

        # Insert case: free or tombstone slot
        is_free       = ((freemask >> UInt32(lane)) & UInt32(1)) == UInt32(1)
        is_tombstone  = (pair == HIVE_TOMBSTONE)
        is_insertable = is_free || is_tombstone

        insertable_ballot = hive_ballot(is_insertable)

        if insertable_ballot != UInt32(0)
            winner = hive_first_set_lane(insertable_ballot)
            if lane == winner
                new_pair = pack_pair_gpu(key, val)
                expected = is_tombstone ? HIVE_TOMBSTONE : HIVE_EMPTY_PAIR
                old      = CUDA.atomic_cas!(pointer(pairs, slot_idx), expected, new_pair)

                if old == expected
                    # Clear the freemask bit for this lane
                    CUDA.atomic_and!(
                        pointer(freemasks, bucket_idx + UInt32(1)),
                        ~(UInt32(1) << UInt32(lane))
                    )
                    status[op_idx] = UPSERT_INSERTED
                else
                    status[op_idx] = UPSERT_FAILED
                end
            end
            return nothing
        end
    end

    if lane == Int32(0)
        status[op_idx] = UPSERT_FAILED
    end

    return nothing
end

# =============================================================================
# Delete Kernel
# =============================================================================

"""
    hive_delete_kernel!(status, pairs, freemasks, n_buckets, keys, n_ops)

CUDA kernel for batch deletion from the HiveHT hash table.

Deletion replaces the pair with a TOMBSTONE sentinel value and marks
the slot as free in the freemask (so it can be reused for inserts).

# Arguments
- `status`: Output array for operation status (0=not found, 1=deleted)
- `pairs`: Flat array of UInt64 pairs (modified in place)
- `freemasks`: Array of freemask bitmaps (modified in place)
- `n_buckets`: Number of buckets
- `keys`: Input array of keys to delete
- `n_ops`: Number of delete operations
"""
function hive_delete_kernel!(
    status::CuDeviceVector{UInt8},
    pairs::CuDeviceVector{UInt64},
    freemasks::CuDeviceVector{UInt32},
    n_buckets::Int32,
    keys::CuDeviceVector{UInt32},
    n_ops::Int32
)
    op_idx = hive_warp_id()

    if op_idx > n_ops
        return nothing
    end

    lane = hive_lane_id()  # Int32, 0 to 31

    key = keys[op_idx]
    h   = hive_hash_gpu(key)

    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)  # 0-indexed

        # Coalesced load
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)
        pair     = pairs[slot_idx]
        pair_key = unpack_key_gpu(pair)

        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        match_ballot = hive_ballot(is_match)
        empty_ballot = hive_ballot(is_empty)

        if match_ballot != UInt32(0)
            winner = hive_first_set_lane(match_ballot)
            if lane == winner
                old = CUDA.atomic_cas!(pointer(pairs, slot_idx), pair, HIVE_TOMBSTONE)

                if old == pair
                    # Mark slot as free in freemask
                    CUDA.atomic_or!(
                        pointer(freemasks, bucket_idx + UInt32(1)),
                        UInt32(1) << UInt32(lane)
                    )
                    status[op_idx] = DELETE_SUCCESS
                else
                    status[op_idx] = DELETE_FAILED
                end
            end
            return nothing
        end

        if empty_ballot != UInt32(0)
            if lane == Int32(0)
                status[op_idx] = DELETE_FAILED
            end
            return nothing
        end
    end

    if lane == Int32(0)
        status[op_idx] = DELETE_FAILED
    end

    return nothing
end
