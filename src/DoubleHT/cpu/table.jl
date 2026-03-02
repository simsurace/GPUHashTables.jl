"""
    CPUDoubleHT{K,V}

CPU-side double hashing hash table. Used for building the table and as a
reference implementation for testing.

# Fields
- `buckets`: Vector of buckets
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
- `empty_val`: Sentinel value for empty value slots
"""
mutable struct CPUDoubleHT{K,V}
    buckets::Vector{Bucket8{K,V}}
    n_buckets::Int
    n_entries::Int
    empty_key::K
    empty_val::V
end

"""
    CPUDoubleHT(keys::Vector{K}, values::Vector{V}; load_factor=DEFAULT_LOAD_FACTOR) where {K,V}

Build a CPU double hashing hash table from key-value pairs.

# Arguments
- `keys`: Vector of keys (must not contain the empty sentinel value)
- `values`: Vector of values
- `load_factor`: Target load factor (default: 0.7, max: 0.9)

# Returns
A `CPUDoubleHT` ready for queries or transfer to GPU.

# Example
```julia
keys = rand(UInt32(1):UInt32(2^31), 1_000_000)
vals = rand(UInt32, 1_000_000)
table = CPUDoubleHT(keys, vals)
```
"""
function CPUDoubleHT(
    keys::Vector{K},
    values::Vector{V};
    load_factor::Float64=DEFAULT_LOAD_FACTOR,
    empty_key::K=K === UInt32 ? EMPTY_KEY_U32 : typemax(K),
    empty_val::V=V === UInt32 ? EMPTY_VAL_U32 : typemax(V)
) where {K,V}
    n_entries = length(keys)
    @assert length(values) == n_entries "Keys and values must have same length"
    @assert 0.0 < load_factor <= MAX_LOAD_FACTOR "Load factor must be in (0, $MAX_LOAD_FACTOR]"

    # Check for sentinel in keys
    for key in keys
        if key == empty_key
            error("Keys cannot contain the empty sentinel value ($empty_key)")
        end
    end

    # Calculate number of buckets needed
    # Total slots = n_buckets * BUCKET_SIZE
    # load_factor = n_entries / total_slots
    total_slots_needed = ceil(Int, n_entries / load_factor)
    n_buckets = ceil(Int, total_slots_needed / BUCKET_SIZE)
    n_buckets = max(n_buckets, 1)  # At least one bucket

    # Initialize empty buckets
    empty_slot = Slot{K,V}(empty_key, empty_val)
    empty_bucket = Bucket8{K,V}(ntuple(_ -> empty_slot, BUCKET_SIZE))
    buckets = fill(empty_bucket, n_buckets)

    # Insert all key-value pairs
    table = CPUDoubleHT{K,V}(buckets, n_buckets, 0, empty_key, empty_val)

    for i in 1:n_entries
        success = insert_cpu!(table, keys[i], values[i])
        if !success
            error("Failed to insert key at index $i - table may be too full")
        end
    end

    return table
end

"""
    insert_cpu!(table::CPUDoubleHT{K,V}, key::K, val::V) -> Bool

Insert a key-value pair into the CPU table using double hashing.
Returns true on success, false if max probes exceeded.
"""
function insert_cpu!(table::CPUDoubleHT{K,V}, key::K, val::V)::Bool where {K,V}
    h1, h2 = double_hash(key)

    # Ensure step is non-zero and coprime-ish to n_buckets
    # Handle edge case where n_buckets == 1
    if table.n_buckets == 1
        step = UInt32(1)
    else
        step = h2 % UInt32(table.n_buckets - 1) + UInt32(1)
    end

    for probe in 0:MAX_PROBES-1
        bucket_idx = (h1 + step * UInt32(probe)) % UInt32(table.n_buckets) + UInt32(1)
        bucket = table.buckets[bucket_idx]

        # Search for empty slot or existing key in this bucket
        for slot_idx in 1:BUCKET_SIZE
            slot = bucket.slots[slot_idx]

            if slot.key == table.empty_key
                # Found empty slot - insert here
                new_slot = Slot{K,V}(key, val)
                new_slots = ntuple(i -> i == slot_idx ? new_slot : bucket.slots[i], BUCKET_SIZE)
                table.buckets[bucket_idx] = Bucket8{K,V}(new_slots)
                table.n_entries += 1
                return true
            elseif slot.key == key
                # Key already exists - update value
                new_slot = Slot{K,V}(key, val)
                new_slots = ntuple(i -> i == slot_idx ? new_slot : bucket.slots[i], BUCKET_SIZE)
                table.buckets[bucket_idx] = Bucket8{K,V}(new_slots)
                return true
            end
        end
        # Bucket full, continue probing
    end

    return false  # Max probes exceeded
end
