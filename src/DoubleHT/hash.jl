# MurmurHash64A implementation and double hashing utilities
# Reference: https://github.com/aappleby/smhasher

"""
    murmur_hash_64a(key::UInt32, seed::UInt64=0x0) -> UInt64

MurmurHash64A hash function for 32-bit keys.
Returns a 64-bit hash value suitable for double hashing.

This implementation matches warpSpeed's hash function for compatibility.
"""
@inline function murmur_hash_64a(key::UInt32, seed::UInt64=0x0000000000000000)::UInt64
    m = 0xc6a4a7935bd1e995
    r = 47

    # Initialize hash with seed XORed with length * m
    # We treat the key as 4 bytes
    h = seed ⊻ (UInt64(4) * m)

    # Mix in the 4-byte key (extend to 64-bit, process as one block)
    k = UInt64(key)
    k *= m
    k ⊻= k >> r
    k *= m

    h ⊻= k
    h *= m

    # Finalization mix
    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    return h
end

"""
    murmur_hash_64a(key::UInt64, seed::UInt64=0x0) -> UInt64

MurmurHash64A hash function for 64-bit keys.
"""
@inline function murmur_hash_64a(key::UInt64, seed::UInt64=0x0000000000000000)::UInt64
    m = 0xc6a4a7935bd1e995
    r = 47

    # Initialize hash with seed XORed with length * m
    # We treat the key as 8 bytes
    h = seed ⊻ (UInt64(8) * m)

    # Mix in the 8-byte key
    k = key
    k *= m
    k ⊻= k >> r
    k *= m

    h ⊻= k
    h *= m

    # Finalization mix
    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    return h
end

"""
    double_hash(key::UInt32) -> (h1::UInt32, h2::UInt32)

Compute double hash values for a key.
- h1: Primary hash for bucket index
- h2: Secondary hash for probe step size

Usage:
    h1, h2 = double_hash(key)
    bucket_idx = h1 % n_buckets + 1
    step = h2 % (n_buckets - 1) + 1  # Ensure non-zero step
"""
@inline function double_hash(key::UInt32)::Tuple{UInt32,UInt32}
    h = murmur_hash_64a(key)
    h1 = UInt32(h & 0xFFFFFFFF)
    h2 = UInt32(h >> 32)
    return (h1, h2)
end

"""
    double_hash(key::UInt64) -> (h1::UInt32, h2::UInt32)

Compute double hash values for a 64-bit key.
"""
@inline function double_hash(key::UInt64)::Tuple{UInt32,UInt32}
    h = murmur_hash_64a(key)
    h1 = UInt32(h & 0xFFFFFFFF)
    h2 = UInt32(h >> 32)
    return (h1, h2)
end

# GPU-compatible versions using CUDA.@inline
"""
    double_hash_gpu(key::UInt32) -> (h1::UInt32, h2::UInt32)

GPU kernel-compatible version of double_hash.
"""
@inline function double_hash_gpu(key::UInt32)::Tuple{UInt32,UInt32}
    m = 0xc6a4a7935bd1e995
    r = 47

    h = UInt64(4) * m  # seed = 0
    k = UInt64(key)
    k *= m
    k ⊻= k >> r
    k *= m
    h ⊻= k
    h *= m
    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    h1 = UInt32(h & 0xFFFFFFFF)
    h2 = UInt32(h >> 32)
    return (h1, h2)
end
