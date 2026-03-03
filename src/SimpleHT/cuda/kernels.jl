# CUDA kernel for SimpleHT linear-probing query

"""
    simple_query_kernel!(results, found, slots, n_slots, keys, n_queries, empty_key, empty_val)

CUDA kernel for batch querying a linear-probing hash table.

One thread handles one query. Each thread:
1. Hashes its key to an initial slot
2. Scans forward linearly until it finds the key, an empty slot, or exhausts all slots
3. Writes to `results` and `found`

This is the simplest possible GPU hash table kernel — no warp cooperation, no
buckets, no shared memory. Performance is limited by memory latency, but at
low-to-moderate load factors the average probe length is short (≈ 1–2 slots).

# Arguments
- `results`: Output array for values
- `found`: Output array for found flags
- `slots`: Flat array of `Slot{K,V}` (length `n_slots`)
- `n_slots`: Table capacity (power of two)
- `keys`: Query keys
- `n_queries`: Number of queries
- `empty_key`: Sentinel for an unused slot
- `empty_val`: Sentinel value indicating a deleted (tombstoned) entry
"""
function simple_query_kernel!(
    results::CuDeviceVector{V},
    found::CuDeviceVector{Bool},
    slots::CuDeviceVector{Slot{K,V}},
    n_slots::Int32,
    keys::CuDeviceVector{K},
    n_queries::Int32,
    empty_key::K,
    empty_val::V
) where {K,V}
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if tid > n_queries
        return nothing
    end

    key    = keys[tid]
    h1, _  = double_hash_gpu(key)
    slot_0 = h1 & UInt32(n_slots - Int32(1))  # 0-indexed starting position

    for _ in Int32(1):n_slots
        slot = slots[slot_0 + UInt32(1)]       # convert to 1-indexed access

        if slot.key == key
            if slot.val != empty_val
                results[tid] = slot.val
                found[tid]   = true
            else
                found[tid] = false             # tombstone: key deleted
            end
            return nothing
        end

        if slot.key == empty_key
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
