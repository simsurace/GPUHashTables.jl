# SIMD-group utilities for Metal kernels
# Metal uses simdgroups (32 threads) similar to CUDA warps
#
# Key design choice for Metal:
# - TILE_SIZE = SIMDGROUP_SIZE = 32 (full simdgroup per tile)
# - BUCKET_SIZE = 32 (all threads check slots)
# This avoids intra-simdgroup deadlock that occurs when smaller tiles compete

using Metal: thread_position_in_threadgroup, threadgroup_position_in_grid,
             threads_per_threadgroup, simd_ballot, simdgroup_barrier, threadgroup_barrier, MemoryFlagDevice

# Metal simdgroup size - this is also our tile size
const SIMDGROUP_SIZE = 32

# =============================================================================
# Primary tile utilities for Metal (tile == simdgroup, 32 threads)
# All kernels (query and upsert) use these functions
# =============================================================================

"""
    metal_tile_id() -> Int

Get the tile ID for the current thread (1-indexed).
Each tile is a full simdgroup (32 threads) and handles one operation.
"""
@inline function metal_tile_id()
    thread_id = thread_position_in_threadgroup().x +
                (threadgroup_position_in_grid().x - 1) * threads_per_threadgroup().x
    return (thread_id - 1) ÷ SIMDGROUP_SIZE + 1
end

"""
    metal_tile_lane() -> Int

Get the lane (position) within the tile for the current thread (0-indexed, 0 to 31).
Each lane loads one slot from the 32-slot bucket.
"""
@inline function metal_tile_lane()
    return (thread_position_in_threadgroup().x - 1) % SIMDGROUP_SIZE
end

"""
    metal_tile_ballot(predicate::Bool) -> UInt32

Perform ballot vote across the full simdgroup (tile).
Returns lower 32 bits of the ballot - one bit per thread/lane.
"""
@inline function metal_tile_ballot(predicate::Bool)
    full_ballot = simd_ballot(predicate)
    return UInt32(full_ballot & 0xFFFFFFFF)
end

"""
    metal_first_set_lane(ballot::UInt32) -> Int

Find the first set bit in a ballot result (0-indexed lane number).
Returns -1 if no bits are set.
"""
@inline function metal_first_set_lane(ballot::UInt32)
    if ballot == 0
        return -1
    end
    return trailing_zeros(ballot)
end

"""
    metal_tile_sync()

Synchronize all threads in the tile (simdgroup) with device memory fence.
Use this after writes to ensure visibility before releasing locks.

Note: We use threadgroup_barrier instead of simdgroup_barrier for stronger
memory ordering guarantees. This ensures writes to device memory are visible
across the entire threadgroup, not just within a simdgroup.
"""
@inline function metal_tile_sync()
    threadgroup_barrier(MemoryFlagDevice)
    return nothing
end
