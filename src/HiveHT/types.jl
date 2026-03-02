# Core types for HiveHT hash table
# Based on the Hive Hash Table paper (https://arxiv.org/abs/2510.15095)

# =============================================================================
# Constants
# =============================================================================

"""
Hive bucket size - 32 slots per bucket, matching warp/simdgroup size.
Each slot is a 64-bit packed key-value pair.
"""
const HIVE_BUCKET_SIZE = 32

"""
Maximum probe attempts before giving up on an operation.
Linear probing with 32-slot buckets typically needs fewer probes than double hashing.
"""
const HIVE_MAX_PROBES = 300

"""
Empty pair sentinel value.
Key = 0xFFFFFFFF (typemax(UInt32)), Value = 0xFFFFFFFF
This pair indicates an unoccupied slot.
"""
const HIVE_EMPTY_PAIR = 0xFFFFFFFF_FFFFFFFF

"""
Tombstone sentinel value for deleted slots.
Key = 0xFFFFFFFE, Value = 0xFFFFFFFF
Tombstones are skipped during queries but can be reused for insertions.
"""
const HIVE_TOMBSTONE = 0xFFFFFFFE_FFFFFFFF

"""
Empty key sentinel - cannot be used as an actual key.
"""
const HIVE_EMPTY_KEY = typemax(UInt32)

"""
Tombstone key sentinel - cannot be used as an actual key.
"""
const HIVE_TOMBSTONE_KEY = UInt32(0xFFFFFFFE)

# =============================================================================
# Delete operation result codes
# =============================================================================

const DELETE_FAILED = UInt8(0)    # Key not found or operation failed
const DELETE_SUCCESS = UInt8(1)   # Key was deleted

# =============================================================================
# Pack/Unpack Utilities
# =============================================================================

"""
    pack_pair(key::UInt32, val::UInt32) -> UInt64

Pack a key-value pair into a single 64-bit word.
Format: (value << 32) | key

This enables atomic updates of both key and value with a single 64-bit CAS.
"""
@inline function pack_pair(key::UInt32, val::UInt32)::UInt64
    return (UInt64(val) << 32) | UInt64(key)
end

"""
    unpack_key(pair::UInt64) -> UInt32

Extract the key (lower 32 bits) from a packed pair.
"""
@inline function unpack_key(pair::UInt64)::UInt32
    return UInt32(pair & 0xFFFFFFFF)
end

"""
    unpack_val(pair::UInt64) -> UInt32

Extract the value (upper 32 bits) from a packed pair.
"""
@inline function unpack_val(pair::UInt64)::UInt32
    return UInt32(pair >> 32)
end

# =============================================================================
# GPU-compatible versions (inlined for kernel use)
# =============================================================================

"""
GPU kernel-compatible pack function.
"""
@inline function pack_pair_gpu(key::UInt32, val::UInt32)::UInt64
    return (UInt64(val) << 32) | UInt64(key)
end

"""
GPU kernel-compatible unpack key function.
"""
@inline function unpack_key_gpu(pair::UInt64)::UInt32
    return UInt32(pair & 0xFFFFFFFF)
end

"""
GPU kernel-compatible unpack value function.
"""
@inline function unpack_val_gpu(pair::UInt64)::UInt32
    return UInt32(pair >> 32)
end

# =============================================================================
# Data Structures
# =============================================================================

"""
    HiveBucket

A bucket containing 32 packed key-value pairs (64-bit each).
Total size: 32 × 8 = 256 bytes (cache-aligned).

Each slot stores both key and value packed into a single UInt64,
enabling atomic CAS updates without separate locking.
"""
struct HiveBucket
    pairs::NTuple{32, UInt64}
end

"""
Create an empty HiveBucket with all slots initialized to HIVE_EMPTY_PAIR.
"""
function empty_hive_bucket()::HiveBucket
    return HiveBucket(ntuple(_ -> HIVE_EMPTY_PAIR, 32))
end

# =============================================================================
# Hash Function for Linear Probing
# =============================================================================

"""
    hive_hash(key::UInt32) -> UInt32

Compute hash for linear probing in HiveHT.
Uses MurmurHash64A and takes lower 32 bits.
"""
@inline function hive_hash(key::UInt32)::UInt32
    h = murmur_hash_64a(key)
    return UInt32(h & 0xFFFFFFFF)
end

"""
GPU kernel-compatible hash function for HiveHT.
Inlined MurmurHash64A returning lower 32 bits.
"""
@inline function hive_hash_gpu(key::UInt32)::UInt32
    m = 0xc6a4a7935bd1e995
    r = 47

    h = UInt64(4) * m  # seed = 0
    k = UInt64(key)
    k *= m
    k = k ⊻ (k >> r)
    k *= m
    h = h ⊻ k
    h *= m
    h = h ⊻ (h >> r)
    h *= m
    h = h ⊻ (h >> r)

    return UInt32(h & 0xFFFFFFFF)
end
