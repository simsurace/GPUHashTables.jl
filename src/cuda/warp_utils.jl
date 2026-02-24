# Warp-level utilities for GPU kernels

# Tile configuration
# Each tile of TILE_SIZE threads cooperatively handles one query

"""
    tile_id() -> Int

Get the tile ID for the current thread (1-indexed).
Each tile handles one query.
"""
@inline function tile_id()
    thread_id = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    return (thread_id - 1) ÷ TILE_SIZE + 1
end

"""
    tile_lane() -> Int

Get the lane (position) within the tile for the current thread (0-indexed, 0 to TILE_SIZE-1).
Each lane loads one slot from the bucket.
"""
@inline function tile_lane()
    return (threadIdx().x - 1) % TILE_SIZE
end

"""
    tile_mask() -> UInt32

Get the warp ballot mask for threads in the current tile.
Used for warp-level voting operations.
"""
@inline function tile_mask()
    # Calculate which 8-thread tile within the warp
    warp_lane = (threadIdx().x - 1) % 32
    tile_in_warp = warp_lane ÷ TILE_SIZE
    # Create mask for just this tile's threads
    return UInt32(0xFF) << (tile_in_warp * TILE_SIZE)
end

"""
    tile_ballot(mask::UInt32, predicate::Bool) -> UInt32

Perform ballot vote within the tile, returning only tile-relevant bits.
Result is shifted so bit 0 corresponds to lane 0 of the tile.
"""
@inline function tile_ballot(mask::UInt32, predicate::Bool)
    full_ballot = CUDA.vote_ballot_sync(mask, predicate)
    warp_lane = (threadIdx().x - 1) % 32
    tile_in_warp = warp_lane ÷ TILE_SIZE
    # Shift and mask to get just this tile's 8 bits
    return (full_ballot >> (tile_in_warp * TILE_SIZE)) & UInt32(0xFF)
end

"""
    first_set_lane(ballot::UInt32) -> Int

Find the first set bit in a ballot result (0-indexed lane number).
Returns -1 if no bits are set.
"""
@inline function first_set_lane(ballot::UInt32)
    if ballot == 0
        return -1
    end
    return trailing_zeros(ballot)
end
