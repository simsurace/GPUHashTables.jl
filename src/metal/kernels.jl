# Metal kernels for GPU hash table operations

"""
    metal_query_kernel!(results, found, buckets, n_buckets, keys, n_queries, empty_key)

Metal kernel for batch querying the hash table.

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
function metal_query_kernel!(
    results::MtlDeviceVector{V},
    found::MtlDeviceVector{Bool},
    buckets::MtlDeviceVector{Bucket8{K,V}},
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

    lane = metal_tile_lane()  # 0 to TILE_SIZE-1

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

        # SIMD ballot within tile
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
