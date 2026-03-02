# CPU HiveHT hash table type and operations
#
# Provides a pure-CPU implementation of HiveHT, mirroring the GPU layout so
# the table can be transferred to CuHiveHT or MtlHiveHT without re-running
# the GPU upsert kernel.

"""
    CPUHiveHT{K,V}

CPU-side HiveHT hash table. Used for building the table on CPU and as a
reference implementation for testing. Shares the same HiveBucket layout as
CuHiveHT so it can be transferred to the GPU without invoking the upsert kernel.

# Fields
- `buckets`: Vector of HiveBucket structs (32 packed pairs each)
- `freemasks`: Vector of freemask bitmaps (one UInt32 per bucket)
- `n_buckets`: Number of buckets
- `n_entries`: Number of key-value pairs stored
- `empty_key`: Sentinel value for empty key slots
"""
mutable struct CPUHiveHT{K,V}
    buckets::Vector{HiveBucket}
    freemasks::Vector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
end

"""
    CPUHiveHT(keys::Vector{K}, vals::Vector{V}; load_factor=0.7) -> CPUHiveHT{K,V}

Build a CPU HiveHT hash table from key-value pairs using linear probing.

# Arguments
- `keys`: Vector of keys to insert
- `vals`: Vector of values to insert
- `load_factor`: Target load factor (default 0.7, max 0.9)

# Example
```julia
keys = rand(UInt32(1):UInt32(2^31), 1_000_000)
vals = rand(UInt32, 1_000_000)
table = CPUHiveHT(keys, vals)
```
"""
function CPUHiveHT(
    keys::Vector{K},
    vals::Vector{V};
    load_factor::Float64 = DEFAULT_LOAD_FACTOR,
    empty_key::K = K === UInt32 ? HIVE_EMPTY_KEY : typemax(K)
) where {K,V}
    @assert K === UInt32 && V === UInt32 "HiveHT currently only supports UInt32 keys and values"
    @assert length(keys) == length(vals) "Keys and values must have same length"
    @assert 0.0 < load_factor <= MAX_LOAD_FACTOR "Load factor must be in (0, $MAX_LOAD_FACTOR]"

    n = length(keys)
    if n == 0
        return CPUHiveHT{K,V}(
            [empty_hive_bucket()],
            [UInt32(0xFFFFFFFF)],
            1, 0, empty_key
        )
    end

    total_slots_needed = ceil(Int, n / load_factor)
    n_buckets = max(1, cld(total_slots_needed, HIVE_BUCKET_SIZE))

    buckets = fill(empty_hive_bucket(), n_buckets)
    freemasks = fill(UInt32(0xFFFFFFFF), n_buckets)

    table = CPUHiveHT{K,V}(buckets, freemasks, n_buckets, 0, empty_key)

    for i in 1:n
        success = _hive_insert_cpu!(table, keys[i], vals[i])
        if !success
            @warn "CPUHiveHT construction: failed to insert key at index $i"
        end
    end

    return table
end

# =============================================================================
# Internal insert
# =============================================================================

"""
    _hive_insert_cpu!(table::CPUHiveHT{K,V}, key::K, val::V) -> Bool

Insert or update a key-value pair using linear probing. Mirrors the WABC
protocol used by the GPU upsert kernel. Returns true on success.
"""
function _hive_insert_cpu!(table::CPUHiveHT{K,V}, key::K, val::V)::Bool where {K,V}
    h = hive_hash(key)
    new_pair = pack_pair(key, val)

    for probe in 0:HIVE_MAX_PROBES-1
        bucket_idx = Int((h + UInt32(probe)) % UInt32(table.n_buckets)) + 1
        bucket = table.buckets[bucket_idx]

        first_insertable = 0  # 1-indexed slot index, 0 = none found yet

        for slot in 1:HIVE_BUCKET_SIZE
            pair = bucket.pairs[slot]

            if pair != HIVE_TOMBSTONE && unpack_key(pair) == key
                # Key already exists — update value in place
                new_pairs = ntuple(i -> i == slot ? new_pair : bucket.pairs[i], HIVE_BUCKET_SIZE)
                table.buckets[bucket_idx] = HiveBucket(new_pairs)
                return true
            elseif (pair == HIVE_EMPTY_PAIR || pair == HIVE_TOMBSTONE) && first_insertable == 0
                first_insertable = slot
            end
        end

        if first_insertable != 0
            # Insert into the first free/tombstone slot found
            new_pairs = ntuple(i -> i == first_insertable ? new_pair : bucket.pairs[i], HIVE_BUCKET_SIZE)
            table.buckets[bucket_idx] = HiveBucket(new_pairs)
            # Clear the freemask bit for this slot (bit position = slot - 1)
            table.freemasks[bucket_idx] &= ~(UInt32(1) << (first_insertable - 1))
            table.n_entries += 1
            return true
        end

        # Bucket is completely full — continue to next probe position
    end

    return false
end

# =============================================================================
# Query operations
# =============================================================================

"""
    query(table::CPUHiveHT{K,V}, key::K) -> (found::Bool, value::V)

Query a single key in the CPU HiveHT. Mirrors the WCME protocol used by the
GPU query kernel: probes whole buckets, stops at an empty slot.
"""
function query(table::CPUHiveHT{K,V}, key::K)::Tuple{Bool,V} where {K,V}
    h = hive_hash(key)

    for probe in 0:HIVE_MAX_PROBES-1
        bucket_idx = Int((h + UInt32(probe)) % UInt32(table.n_buckets)) + 1
        bucket = table.buckets[bucket_idx]

        found_empty = false

        for slot in 1:HIVE_BUCKET_SIZE
            pair = bucket.pairs[slot]

            if pair != HIVE_TOMBSTONE && unpack_key(pair) == key
                return (true, unpack_val(pair))
            elseif pair == HIVE_EMPTY_PAIR
                found_empty = true
            end
        end

        # If the bucket contained an empty slot (and no match), the key was
        # never inserted — stop probing.
        if found_empty
            return (false, typemax(V))
        end

        # Bucket is fully occupied with non-matching entries or tombstones —
        # continue linear probing.
    end

    return (false, typemax(V))
end

"""
    query!(results::Vector{V}, found::Vector{Bool}, table::CPUHiveHT{K,V}, keys::Vector{K})

Batch query multiple keys in the CPU HiveHT (in-place).
"""
function query!(
    results::Vector{V},
    found::Vector{Bool},
    table::CPUHiveHT{K,V},
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
    query(table::CPUHiveHT{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Batch query multiple keys, allocating result vectors.
"""
function query(table::CPUHiveHT{K,V}, keys::Vector{K})::Tuple{Vector{Bool},Vector{V}} where {K,V}
    n = length(keys)
    results = Vector{V}(undef, n)
    found = Vector{Bool}(undef, n)
    query!(results, found, table, keys)
    return (found, results)
end
