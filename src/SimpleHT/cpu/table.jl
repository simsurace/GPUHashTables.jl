# CPU linear-probing hash table (SimpleHT)

"""
    CPUSimpleHT{K,V}

CPU-side linear-probing hash table.

Uses a flat array of `Slot{K,V}` with a power-of-two capacity. Each key maps
to an initial slot via a single hash; collisions are resolved by scanning
forward linearly (with wrap-around).

Deletion leaves the key in place and sets the value to `empty_val` (tombstone).
This preserves probe chains for other keys. Tombstoned slots can only be
reclaimed by re-inserting the same key.

# Fields
- `slots`: Flat array of all key-value slots
- `n_slots`: Total number of slots (power of two)
- `n_entries`: Number of live (non-deleted) entries
- `empty_key`: Sentinel for an empty (never-used) slot
- `empty_val`: Sentinel used as a tombstone value after deletion
"""
struct CPUSimpleHT{K,V}
    slots::Vector{Slot{K,V}}
    n_slots::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    CPUSimpleHT(keys, vals; load_factor=0.5, empty_key=typemax(K), empty_val=typemax(V))

Build a CPU linear-probing hash table from `keys` and `vals`.

`load_factor` controls the ratio of entries to slots. Linear probing
degrades sharply above ~0.7; the default is 0.5 to match the original
SimpleGPUHashTable design.
"""
function CPUSimpleHT(
    keys::Vector{K},
    vals::Vector{V};
    load_factor::Float64 = 0.5,
    empty_key::K = K === UInt32 ? EMPTY_KEY_U32 : typemax(K),
    empty_val::V = V === UInt32 ? EMPTY_VAL_U32 : typemax(V)
) where {K,V}
    n = length(keys)
    @assert length(vals) == n "Keys and values must have the same length"
    @assert 0.0 < load_factor < 1.0 "Load factor must be in (0, 1)"

    if any(==(empty_key), keys)
        error("CPUSimpleHT: cannot insert sentinel (empty) key $empty_key")
    end

    n_slots = n == 0 ? 1 : nextpow(2, ceil(Int, n / load_factor))

    empty_slot = Slot{K,V}(empty_key, empty_val)
    slots = fill(empty_slot, n_slots)

    n_inserted = 0
    for i in 1:n
        if _insert_slot!(slots, n_slots, keys[i], vals[i], empty_key)
            n_inserted += 1
        end
    end

    return CPUSimpleHT{K,V}(slots, n_slots, n_inserted, empty_key, empty_val)
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _slot_index(key, n_slots, empty_key) -> initial 1-indexed slot

Compute the 1-indexed starting slot for `key` in a table of `n_slots` slots.
"""
@inline function _initial_slot(key::K, n_slots::Int) where {K}
    h1, _ = double_hash(key)
    return Int(h1 & UInt32(n_slots - 1)) + 1
end

"""
    _next_slot(slot_idx, n_slots) -> next 1-indexed slot (linear step with wrap)
"""
@inline function _next_slot(slot_idx::Int, n_slots::Int)
    return slot_idx % n_slots + 1
end

"""
    _insert_slot!(slots, n_slots, key, val, empty_key) -> inserted::Bool

Insert `key => val` into `slots`. Returns `true` if a new slot was claimed,
`false` if an existing key was updated. Errors if the table is full.
"""
function _insert_slot!(
    slots::Vector{Slot{K,V}},
    n_slots::Int,
    key::K,
    val::V,
    empty_key::K
) where {K,V}
    slot_idx = _initial_slot(key, n_slots)
    for _ in 1:n_slots
        slot = slots[slot_idx]
        if slot.key == empty_key
            slots[slot_idx] = Slot{K,V}(key, val)
            return true   # new insertion
        elseif slot.key == key
            slots[slot_idx] = Slot{K,V}(key, val)
            return false  # update
        end
        slot_idx = _next_slot(slot_idx, n_slots)
    end
    error("CPUSimpleHT: table is full (load factor too high or duplicate keys not unique)")
end

# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

"""
    Base.delete!(table::CPUSimpleHT{K,V}, key::K) -> Bool

Delete `key` from the table by setting its value to `empty_val` (tombstone).
The key slot remains occupied to preserve probe chains.
Returns `true` if the key was found and deleted, `false` if not present.
"""
function Base.delete!(table::CPUSimpleHT{K,V}, key::K) where {K,V}
    slot_idx = _initial_slot(key, table.n_slots)
    for _ in 1:table.n_slots
        slot = table.slots[slot_idx]
        if slot.key == key
            table.slots[slot_idx] = Slot{K,V}(key, table.empty_val)
            return true
        elseif slot.key == table.empty_key
            return false
        end
        slot_idx = _next_slot(slot_idx, table.n_slots)
    end
    return false
end
