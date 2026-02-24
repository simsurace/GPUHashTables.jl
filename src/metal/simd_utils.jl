# SIMD-group utilities for Metal kernels
# Metal uses simdgroups (32 threads) similar to CUDA warps

using Metal: thread_position_in_threadgroup, threadgroup_position_in_grid,
             threads_per_threadgroup, simd_ballot

"""
    metal_tile_id() -> Int

Get the tile ID for the current thread (1-indexed).
Each tile handles one query.
"""
@inline function metal_tile_id()
    thread_id = thread_position_in_threadgroup().x +
                (threadgroup_position_in_grid().x - 1) * threads_per_threadgroup().x
    return (thread_id - 1) ÷ TILE_SIZE + 1
end

"""
    metal_tile_lane() -> Int

Get the lane (position) within the tile for the current thread (0-indexed, 0 to TILE_SIZE-1).
Each lane loads one slot from the bucket.
"""
@inline function metal_tile_lane()
    return (thread_position_in_threadgroup().x - 1) % TILE_SIZE
end

"""
    metal_tile_ballot(predicate::Bool) -> UInt32

Perform ballot vote within the tile, returning only tile-relevant bits.
Result is shifted so bit 0 corresponds to lane 0 of the tile.

Uses Metal's simd_ballot which returns UInt64 (we take lower 32 bits and mask to tile).
"""
@inline function metal_tile_ballot(predicate::Bool)
    # simd_ballot returns UInt64 with bit i set if thread i's predicate is true
    full_ballot = simd_ballot(predicate)
    # Calculate which tile within the simdgroup (32 threads)
    simd_lane = (thread_position_in_threadgroup().x - 1) % 32
    tile_in_simd = simd_lane ÷ TILE_SIZE
    # Shift and mask to get just this tile's 8 bits
    return UInt32((full_ballot >> (tile_in_simd * TILE_SIZE)) & 0xFF)
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
