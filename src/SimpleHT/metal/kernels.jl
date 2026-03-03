# Metal kernel for SimpleHT linear-probing query

using Metal: thread_position_in_threadgroup, threadgroup_position_in_grid,
             threads_per_threadgroup

"""
    metal_simple_query_kernel!(results, found, slots_keys, slots_vals, n_slots, keys, n_queries, empty_key, empty_val)

Metal kernel for batch querying a linear-probing hash table.

One thread handles one query. Each thread:
1. Hashes its key to an initial slot
2. Scans forward linearly until it finds the key, an empty slot, or exhausts all slots
3. Writes to `results` and `found`

# Arguments
- `results`: Output array for values
- `found`: Output array for found flags
- `slots_keys`: Flat array of keys (length `n_slots`)
- `slots_vals`: Flat array of values (length `n_slots`)
- `n_slots`: Table capacity (power of two)
- `keys`: Query keys
- `n_queries`: Number of queries
- `empty_key`: Sentinel for an unused slot
- `empty_val`: Sentinel value indicating a deleted (tombstoned) entry
"""
function metal_simple_query_kernel!(
    results::MtlDeviceVector{V},
    found::MtlDeviceVector{Bool},
    slots_keys::MtlDeviceVector{K},
    slots_vals::MtlDeviceVector{V},
    n_slots::Int32,
    keys::MtlDeviceVector{K},
    n_queries::Int32,
    empty_key::K,
    empty_val::V
) where {K,V}
    tid = Int32(thread_position_in_threadgroup().x) +
          (Int32(threadgroup_position_in_grid().x) - Int32(1)) * Int32(threads_per_threadgroup().x)

    if tid > n_queries
        return nothing
    end

    key    = keys[tid]
    h1, _  = double_hash_gpu(key)
    slot_0 = h1 & UInt32(n_slots - Int32(1))  # 0-indexed starting position

    for _ in Int32(1):n_slots
        slot_key = slots_keys[slot_0 + UInt32(1)]  # convert to 1-indexed access

        if slot_key == key
            slot_val = slots_vals[slot_0 + UInt32(1)]
            if slot_val != empty_val
                results[tid] = slot_val
                found[tid]   = true
            else
                found[tid] = false             # tombstone: key deleted
            end
            return nothing
        end

        if slot_key == empty_key
            found[tid] = false                 # empty slot: key never inserted
            return nothing
        end

        slot_0 = (slot_0 + UInt32(1)) & UInt32(n_slots - Int32(1))
    end

    # Exhausted all slots (table full with no match or empty) — shouldn't
    # happen at reasonable load factors.
    found[tid] = false
    return nothing
end
