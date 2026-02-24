# CPU hash table query operations

"""
    query(table::DoubleHashTable{K,V}, key::K) -> (found::Bool, value::V)

Query a single key in the CPU hash table.

Returns a tuple of (found, value) where:
- `found`: true if key was found
- `value`: the associated value if found, or empty_val if not found
"""
function query(table::DoubleHashTable{K,V}, key::K)::Tuple{Bool,V} where {K,V}
    h1, h2 = double_hash(key)

    # Handle edge case where n_buckets == 1
    if table.n_buckets == 1
        step = UInt32(1)
    else
        step = h2 % UInt32(table.n_buckets - 1) + UInt32(1)
    end

    for probe in 0:MAX_PROBES-1
        bucket_idx = (h1 + step * UInt32(probe)) % UInt32(table.n_buckets) + UInt32(1)
        bucket = table.buckets[bucket_idx]

        found_empty = false
        for slot_idx in 1:BUCKET_SIZE
            slot = bucket.slots[slot_idx]

            if slot.key == key
                return (true, slot.val)
            elseif slot.key == table.empty_key
                found_empty = true
            end
        end

        # If we found an empty slot in this bucket, key doesn't exist
        # (it would have been inserted here or earlier)
        if found_empty
            return (false, table.empty_val)
        end
        # Bucket was full but no match - continue probing
    end

    # Max probes exceeded - key not found
    return (false, table.empty_val)
end

"""
    query!(results::Vector{V}, found::Vector{Bool}, table::DoubleHashTable{K,V}, keys::Vector{K})

Batch query multiple keys in the CPU hash table.

# Arguments
- `results`: Pre-allocated vector to store result values
- `found`: Pre-allocated vector to store found flags
- `table`: The hash table to query
- `keys`: Vector of keys to look up
"""
function query!(
    results::Vector{V},
    found::Vector{Bool},
    table::DoubleHashTable{K,V},
    keys::Vector{K}
) where {K,V}
    n = length(keys)
    @assert length(results) >= n "Results vector too small"
    @assert length(found) >= n "Found vector too small"

    for i in 1:n
        found[i], results[i] = query(table, keys[i])
    end

    return nothing
end

"""
    query(table::DoubleHashTable{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Batch query multiple keys, allocating result vectors.
"""
function query(table::DoubleHashTable{K,V}, keys::Vector{K})::Tuple{Vector{Bool},Vector{V}} where {K,V}
    n = length(keys)
    results = Vector{V}(undef, n)
    found = Vector{Bool}(undef, n)
    query!(results, found, table, keys)
    return (found, results)
end
