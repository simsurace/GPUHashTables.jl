# Metal kernels for HiveHT hash table operations
#
# IMPORTANT: Metal only supports 32-bit atomics, not 64-bit CAS.
# We use a different strategy than CUDA:
# - Store pairs as a flat MtlVector{UInt64} (not HiveBucket structs)
# - Use per-slot locks (MtlVector{UInt32}) for atomic updates
# - Lock-free queries (just read pairs)
# - Per-slot locking for upserts and deletes

using Metal: thread_position_in_threadgroup, threadgroup_position_in_grid,
             threads_per_threadgroup, simd_ballot, simd_shuffle,
             threadgroup_barrier, simdgroup_barrier, MemoryFlagDevice

# =============================================================================
# Simdgroup utilities for HiveHT (32-thread simdgroups)
# =============================================================================

"""
    metal_hive_simdgroup_id() -> Int

Get the simdgroup ID for the current thread (1-indexed).
Each simdgroup handles one operation in HiveHT.
"""
@inline function metal_hive_simdgroup_id()
    thread_id = thread_position_in_threadgroup().x +
                (threadgroup_position_in_grid().x - 1) * threads_per_threadgroup().x
    return (thread_id - 1) ÷ 32 + 1
end

"""
    metal_hive_lane_id() -> Int

Get the lane (position) within the simdgroup for the current thread (0-indexed, 0 to 31).
"""
@inline function metal_hive_lane_id()
    return (thread_position_in_threadgroup().x - 1) % 32
end

"""
    metal_hive_ballot(predicate::Bool) -> UInt32

Perform ballot vote across the full simdgroup (32 threads).
Returns a bitmask where bit i is set if thread i's predicate is true.
"""
@inline function metal_hive_ballot(predicate::Bool)
    full_ballot = simd_ballot(predicate)
    return UInt32(full_ballot & 0xFFFFFFFF)
end

"""
    metal_hive_shfl(val::UInt32, src_lane::Int) -> UInt32

Broadcast a value from src_lane (0-indexed) to all threads in the simdgroup.

NOTE: Metal's simd_shuffle uses 1-based lane indexing, so we add 1.
"""
@inline function metal_hive_shfl(val::UInt32, src_lane::Int)
    # Metal simd_shuffle uses 1-based lane indices
    return simd_shuffle(val, UInt16(src_lane + 1))
end

"""
    metal_hive_first_set_lane(ballot::UInt32) -> Int

Find the first set bit in a ballot result (0-indexed lane number).
Returns -1 if no bits are set.
"""
@inline function metal_hive_first_set_lane(ballot::UInt32)
    if ballot == UInt32(0)
        return -1
    end
    return trailing_zeros(ballot)
end

"""
    metal_hive_sync()

Synchronize all threads in the threadgroup with device memory fence.
Use this after writes to ensure visibility.
"""
@inline function metal_hive_sync()
    threadgroup_barrier(MemoryFlagDevice)
    return nothing
end

# =============================================================================
# Lock utilities for per-slot locking
# =============================================================================

const METAL_HIVE_LOCK_FREE = UInt32(0)
const METAL_HIVE_LOCK_HELD = UInt32(1)

"""
Try to acquire the lock for a specific slot.
Returns true if lock was acquired, false if already held.
"""
@inline function metal_hive_try_lock(locks::MtlDeviceVector{UInt32}, slot_idx::Int)
    old = Metal.atomic_compare_exchange_weak_explicit(
        pointer(locks, slot_idx),
        METAL_HIVE_LOCK_FREE,
        METAL_HIVE_LOCK_HELD
    )
    return old == METAL_HIVE_LOCK_FREE
end

"""
Release the lock for a specific slot.
"""
@inline function metal_hive_unlock(locks::MtlDeviceVector{UInt32}, slot_idx::Int)
    Metal.atomic_store_explicit(pointer(locks, slot_idx), METAL_HIVE_LOCK_FREE)
    return nothing
end

# =============================================================================
# Query Kernel (Lock-Free)
# =============================================================================

"""
    metal_hive_query_kernel!(results, found, pairs, n_buckets, keys, n_queries)

Metal kernel for batch querying the HiveHT hash table.

Each simdgroup (32 threads) cooperatively handles one query:
1. All 32 threads load their slot's pair in parallel (coalesced)
2. Each thread extracts the key and compares to the query key
3. Ballot vote to find matching lane
4. Matching thread writes result, or continue probing

This is completely lock-free.

# Arguments
- `pairs`: Flat array of UInt64 pairs (n_buckets * 32 elements)
- Each bucket is 32 consecutive elements
"""
function metal_hive_query_kernel!(
    results::MtlDeviceVector{UInt32},
    found::MtlDeviceVector{Bool},
    pairs::MtlDeviceVector{UInt64},
    n_buckets::Int32,
    keys::MtlDeviceVector{UInt32},
    n_queries::Int32
)
    # Which query does this simdgroup handle?
    query_idx = metal_hive_simdgroup_id()

    if query_idx > n_queries
        return nothing
    end

    lane = metal_hive_lane_id()  # 0 to 31

    key = keys[query_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)

        # Compute flat index for this slot: bucket * 32 + lane + 1 (1-indexed)
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)

        # All 32 threads load their pair simultaneously (coalesced)
        pair = pairs[slot_idx]

        # Extract key from packed pair
        pair_key = unpack_key_gpu(pair)

        # Check for match (but not if it's a tombstone)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        # Simdgroup ballot
        match_ballot = metal_hive_ballot(is_match)
        empty_ballot = metal_hive_ballot(is_empty)

        # Check for match
        if match_ballot != UInt32(0)
            # Found! First matching lane writes the result
            winner = metal_hive_first_set_lane(match_ballot)
            if lane == winner
                results[query_idx] = unpack_val_gpu(pair)
                found[query_idx] = true
            end
            return nothing
        end

        # Check for empty slot - key not in table
        # If there's any empty slot and no match, key can't exist
        # (it would have been inserted in the empty slot)
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
# Upsert Kernel (Per-Slot Locking)
# =============================================================================

"""
    metal_hive_upsert_kernel!(status, pairs, locks, freemasks, n_buckets, keys, vals, n_ops)

Metal kernel for batch upsert (insert or update) into the HiveHT hash table.

Since Metal doesn't support 64-bit CAS, we use per-slot locks:
1. Each thread in simdgroup checks one slot
2. Ballot to find match or free slot
3. Winner tries to acquire slot lock
4. If lock acquired, update pair and freemask, then release
5. If lock not acquired, return FAILED for host retry
"""
function metal_hive_upsert_kernel!(
    status::MtlDeviceVector{UInt8},
    pairs::MtlDeviceVector{UInt64},
    locks::MtlDeviceVector{UInt32},
    freemasks::MtlDeviceVector{UInt32},
    n_buckets::Int32,
    keys::MtlDeviceVector{UInt32},
    vals::MtlDeviceVector{UInt32},
    n_ops::Int32
)
    # Which operation does this simdgroup handle?
    op_idx = metal_hive_simdgroup_id()

    if op_idx > n_ops
        return nothing
    end

    lane = metal_hive_lane_id()  # 0 to 31

    key = keys[op_idx]
    val = vals[op_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)

        # Lane 0 loads freemask and broadcasts to all lanes
        local_freemask = UInt32(0)
        if lane == 0
            local_freemask = freemasks[bucket_idx + 1]
        end
        freemask = metal_hive_shfl(local_freemask, 0)

        # Compute flat index for this slot
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)

        # All 32 threads load their pair
        pair = pairs[slot_idx]
        pair_key = unpack_key_gpu(pair)

        # Check for existing key (update case)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        match_ballot = metal_hive_ballot(is_match)

        if match_ballot != UInt32(0)
            # Found existing key - update it
            winner = metal_hive_first_set_lane(match_ballot)
            if lane == winner
                # Try to acquire slot lock
                if metal_hive_try_lock(locks, Int(slot_idx))
                    # Re-read pair to verify it wasn't changed
                    current_pair = pairs[slot_idx]
                    current_key = unpack_key_gpu(current_pair)

                    if current_key == key && current_pair != HIVE_TOMBSTONE
                        # Still matches, update it
                        new_pair = pack_pair_gpu(key, val)
                        pairs[slot_idx] = new_pair
                        status[op_idx] = UPSERT_UPDATED
                    else
                        # Key changed while we waited - treat as failed
                        status[op_idx] = UPSERT_FAILED
                    end
                    metal_hive_unlock(locks, Int(slot_idx))
                else
                    # Lock not acquired - return FAILED for host retry
                    status[op_idx] = UPSERT_FAILED
                end
            end
            return nothing
        end

        # Check for free or tombstone slot (insert case)
        is_free = ((freemask >> lane) & UInt32(1)) == UInt32(1)
        is_tombstone = (pair == HIVE_TOMBSTONE)
        is_insertable = is_free || is_tombstone

        insertable_ballot = metal_hive_ballot(is_insertable)

        if insertable_ballot != UInt32(0)
            # Found insertable slot
            winner = metal_hive_first_set_lane(insertable_ballot)
            if lane == winner
                # Try to acquire slot lock
                if metal_hive_try_lock(locks, Int(slot_idx))
                    # Re-read pair to verify slot is still available
                    current_pair = pairs[slot_idx]
                    still_available = (current_pair == HIVE_EMPTY_PAIR) ||
                                     (current_pair == HIVE_TOMBSTONE)

                    if still_available
                        # Insert the new pair
                        new_pair = pack_pair_gpu(key, val)
                        pairs[slot_idx] = new_pair

                        # Update freemask - clear the bit for this lane
                        Metal.atomic_fetch_and_explicit(
                            pointer(freemasks, Int(bucket_idx + 1)),
                            ~(UInt32(1) << lane)
                        )
                        status[op_idx] = UPSERT_INSERTED
                    else
                        # Slot was taken - return FAILED for host retry
                        status[op_idx] = UPSERT_FAILED
                    end
                    metal_hive_unlock(locks, Int(slot_idx))
                else
                    # Lock not acquired - return FAILED for host retry
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
# Delete Kernel (Per-Slot Locking)
# =============================================================================

"""
    metal_hive_delete_kernel!(status, pairs, locks, freemasks, n_buckets, keys, n_ops)

Metal kernel for batch deletion from the HiveHT hash table.

Deletion replaces the pair with a TOMBSTONE sentinel value and marks
the slot as free in the freemask (so it can be reused for inserts).
"""
function metal_hive_delete_kernel!(
    status::MtlDeviceVector{UInt8},
    pairs::MtlDeviceVector{UInt64},
    locks::MtlDeviceVector{UInt32},
    freemasks::MtlDeviceVector{UInt32},
    n_buckets::Int32,
    keys::MtlDeviceVector{UInt32},
    n_ops::Int32
)
    # Which operation does this simdgroup handle?
    op_idx = metal_hive_simdgroup_id()

    if op_idx > n_ops
        return nothing
    end

    lane = metal_hive_lane_id()  # 0 to 31

    key = keys[op_idx]
    h = hive_hash_gpu(key)

    # Linear probe sequence
    for probe in Int32(0):Int32(HIVE_MAX_PROBES - 1)
        bucket_idx = (h + UInt32(probe)) % UInt32(n_buckets)

        # Compute flat index for this slot
        slot_idx = bucket_idx * UInt32(32) + UInt32(lane) + UInt32(1)

        # All 32 threads load their pair
        pair = pairs[slot_idx]
        pair_key = unpack_key_gpu(pair)

        # Check for match (not tombstone)
        is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
        is_empty = (pair == HIVE_EMPTY_PAIR)

        match_ballot = metal_hive_ballot(is_match)
        empty_ballot = metal_hive_ballot(is_empty)

        if match_ballot != UInt32(0)
            # Found the key - delete it
            winner = metal_hive_first_set_lane(match_ballot)
            if lane == winner
                # Try to acquire slot lock
                if metal_hive_try_lock(locks, Int(slot_idx))
                    # Re-read pair to verify it's still the same key
                    current_pair = pairs[slot_idx]
                    current_key = unpack_key_gpu(current_pair)

                    if current_key == key && current_pair != HIVE_TOMBSTONE
                        # Write tombstone
                        pairs[slot_idx] = HIVE_TOMBSTONE

                        # Mark slot as free
                        Metal.atomic_fetch_or_explicit(
                            pointer(freemasks, Int(bucket_idx + 1)),
                            UInt32(1) << lane
                        )
                        status[op_idx] = DELETE_SUCCESS
                    else
                        # Key was changed/deleted - treat as not found
                        status[op_idx] = DELETE_FAILED
                    end
                    metal_hive_unlock(locks, Int(slot_idx))
                else
                    # Lock not acquired - return FAILED
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
