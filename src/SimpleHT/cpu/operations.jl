# CPU query operations for SimpleHT

"""
    query(table::CPUSimpleHT{K,V}, key::K) -> (found::Bool, value::V)

Query a single key. Returns `(true, value)` if found, `(false, empty_val)` if not.
Deleted entries (tombstones) are treated as not found.
"""
function query(table::CPUSimpleHT{K,V}, key::K)::Tuple{Bool,V} where {K,V}
    slot_idx = _initial_slot(key, table.n_slots)
    for _ in 1:table.n_slots
        slot = table.slots[slot_idx]
        if slot.key == key
            if slot.val != table.empty_val
                return (true, slot.val)
            else
                return (false, table.empty_val)  # deleted (tombstone)
            end
        elseif slot.key == table.empty_key
            return (false, table.empty_val)
        end
        slot_idx = _next_slot(slot_idx, table.n_slots)
    end
    return (false, table.empty_val)
end

"""
    query!(results, found, table::CPUSimpleHT{K,V}, keys)

Batch query, writing into pre-allocated `results` and `found` vectors.
"""
function query!(
    results::Vector{V},
    found::Vector{Bool},
    table::CPUSimpleHT{K,V},
    keys::Vector{K}
) where {K,V}
    n = length(keys)
    @assert length(results) >= n
    @assert length(found)   >= n
    for i in 1:n
        found[i], results[i] = query(table, keys[i])
    end
    return nothing
end

"""
    query(table::CPUSimpleHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Batch query, allocating result vectors.
"""
function query(table::CPUSimpleHT{K,V}, keys::Vector{K}) where {K,V}
    n = length(keys)
    results = Vector{V}(undef, n)
    found   = Vector{Bool}(undef, n)
    query!(results, found, table, keys)
    return (found, results)
end
