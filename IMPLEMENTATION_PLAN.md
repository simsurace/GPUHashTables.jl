# GPUHashTables Implementation Plan

A Julia package implementing high-performance GPU hash tables for CUDA and Metal backends.

## Package Overview

**Goal**: Implement multiple GPU hash table designs with unified APIs, targeting >1 billion operations/second.

**Hash Table Implementations**:
1. **DoubleHT** - Double hashing with warp/simdgroup-cooperative parallelism (implemented)
2. **HiveHT** - Hive-style with 64-bit packed KV pairs and lock-free queries (implemented)

**Supported Backends**: CUDA, Metal

**Key Types**: `UInt32` keys, `UInt32` values

---

## File Structure

```
GPUHashTables/
├── src/
│   ├── GPUHashTables.jl          # Module definition, exports
│   ├── DoubleHT/
│   │   ├── types.jl              # Slot, Bucket, CPUDoubleHT types
│   │   ├── hash.jl               # MurmurHash, double_hash
│   │   ├── cpu/
│   │   │   ├── table.jl          # CPUDoubleHT constructor
│   │   │   └── operations.jl     # CPU query operations
│   │   ├── cuda/
│   │   │   ├── warp_utils.jl     # tile_id, tile_lane, tile_ballot
│   │   │   ├── kernels.jl        # query_kernel!, upsert_kernel!
│   │   │   └── table.jl          # CuDoubleHT, CuMutableDoubleHT
│   │   └── metal/
│   │       ├── simd_utils.jl     # metal_tile_*, simd_ballot wrappers
│   │       ├── kernels.jl        # metal_query_kernel!, metal_upsert_kernel!
│   │       └── table.jl          # MtlDoubleHT, MtlMutableDoubleHT
│   └── HiveHT/
│       ├── types.jl              # HiveBucket, pack/unpack, constants
│       ├── cuda/
│       │   ├── kernels.jl        # CUDA WABC/WCME kernels
│       │   └── table.jl          # CuHiveHT type and API
│       └── metal/
│           ├── kernels.jl        # Metal WABC/WCME kernels
│           └── table.jl          # MtlHiveHT type and API
├── test/
│   ├── runtests.jl
│   ├── test_hash.jl
│   ├── test_double_cpu.jl
│   ├── test_double_cuda.jl
│   ├── test_double_metal.jl
│   ├── test_hive_cuda.jl
│   └── test_hive_metal.jl
└── benchmark/
    ├── runbenchmarks.jl
    ├── bench_query.jl
    └── bench_scaling.jl
```

---

# DoubleHT (Implemented)

Double hashing hash table with warp/simdgroup-cooperative parallelism. Port of warpSpeed's DoubleHT.

## Status: Complete

| Phase | Status |
|-------|--------|
| Foundation (types, hash) | ✅ |
| CPU reference implementation | ✅ |
| CUDA query | ✅ |
| CUDA upsert | ✅ |
| Metal query | ✅ |
| Metal upsert | ✅ |
| Benchmarking | ✅ |

## Design

- **Bucket size**: 8 slots (CUDA), 32 slots (Metal)
- **Tile size**: 8 threads (CUDA), 32 threads (Metal - full simdgroup)
- **Probe sequence**: Double hashing `(h1 + h2 * i) % n_buckets`
- **Locking**: Per-bucket locks for upsert operations
- **Memory**: Separate key/value fields in Slot struct

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ CPUDoubleHT{K,V}                                        │
├─────────────────────────────────────────────────────────┤
│ buckets::Vector{Bucket{K,V}}   # CPU bucket storage     │
│ n_buckets::Int                 # Number of buckets      │
│ n_entries::Int                 # Number of entries      │
│ empty_key::K                   # Sentinel for empty     │
│ empty_val::V                   # Sentinel for empty     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Bucket{K,V} (64/256 bytes)                              │
├─────────────────────────────────────────────────────────┤
│ slots::NTuple{N, Slot{K,V}}                             │
│   where N = 8 (CUDA) or 32 (Metal)                      │
│   each Slot = (key::K, val::V)                          │
└─────────────────────────────────────────────────────────┘
```

## Types

```julia
# CUDA
struct CuDoubleHT{K,V}        # Immutable, query-only
struct CuMutableDoubleHT{K,V} # Mutable, supports upsert

# Metal
struct MtlDoubleHT{K,V}        # Immutable, query-only
struct MtlMutableDoubleHT{K,V} # Mutable, supports upsert
```

## API

```julia
# Construction
CPUDoubleHT(keys, vals; load_factor=0.7)
CuDoubleHT(cpu_table)
CuMutableDoubleHT{K,V}(n_buckets)

# Operations
query(table, keys) -> (found::Vector{Bool}, results::Vector{V})
query!(results, found, table, keys)
upsert!(table, keys, vals) -> status::Vector{UInt8}
```

## Known Limitations

- **Metal memory ordering**: Relaxed memory model requires sub-batching for large upserts (64-128 keys per batch recommended)
- **Sentinel values**: Cannot store `typemax(K)` as key (reserved for empty)
- **No deletion**: Tombstones not implemented

## Future Work for DoubleHT

- [ ] Delete operation with tombstones
- [ ] GPU-native constructor (build directly on GPU)
- [ ] Dynamic resizing
- [ ] Extended type support (UInt64, strings)

---

# HiveHT (Implemented)

Hive-style hash table with 64-bit packed KV pairs, lock-free queries, and warp-cooperative WABC/WCME protocols.

Based on [Hive Hash Table paper](https://arxiv.org/abs/2510.15095).

## Status

| Phase | Status |
|-------|--------|
| Core types (hive/types.jl) | ✅ |
| CUDA kernels | ✅ |
| CUDA table API | ✅ |
| CUDA tests | ✅ |
| Metal kernels | ✅ |
| Metal table API | ✅ |
| Metal tests | ✅ |
| Integration | ✅ |
| Benchmarking | ⬜ |

## Design Advantages over DoubleHT

| Feature | DoubleHT | HiveHT |
|---------|----------|--------|
| Query locking | None | None (same) |
| Upsert mechanism | Per-bucket lock | 64-bit CAS (lock-free fast path) |
| Memory per slot | 8 bytes (separate K,V) | 8 bytes (packed) |
| Freemask | None | 32-bit bitmap per bucket |
| Deletion | Not supported | Native support |
| Key stability | N/A | Keys never move after insert |
| Metal suitability | Requires sub-batching | Better (single CAS, stable keys) |

## Key Design Elements

### 64-bit Packed Key-Value Pairs

```julia
# Pack: (value << 32) | key
pack_pair(key::UInt32, val::UInt32) = (UInt64(val) << 32) | UInt64(key)
unpack_key(pair::UInt64) = UInt32(pair & 0xFFFFFFFF)
unpack_val(pair::UInt64) = UInt32(pair >> 32)

# Enables single 64-bit CAS for atomic key-value updates
```

### WABC Protocol (Warp-Aggregated-Bitmask-Claim) for Insert

1. Lane 0 loads `freemask`, broadcasts via shuffle
2. All lanes ballot to find free slots
3. Elect winner (first free slot)
4. Winner does 64-bit `atomicCAS(pairs[lane], EMPTY, packed_kv)`
5. On success, lane 0 atomically clears freemask bit

### WCME Protocol (Warp-Cooperative Match-and-Elect) for Query

1. All 32 lanes load `pairs[lane]` in parallel (coalesced)
2. Each lane extracts key, compares to query key
3. Ballot to find matching lane
4. Matching lane returns value (or first lane reports not found)

**Completely lock-free** - critical for Metal's relaxed memory model.

## Types

```julia
# Hive-specific constants
const HIVE_BUCKET_SIZE = 32
const HIVE_EMPTY_PAIR = 0xFFFFFFFF_FFFFFFFF
const HIVE_TOMBSTONE = 0xFFFFFFFE_FFFFFFFF

struct HiveBucket
    pairs::NTuple{32, UInt64}  # 32 packed key-value pairs
end

mutable struct CuHiveHT{K,V}
    buckets::CuVector{HiveBucket}
    freemasks::CuVector{UInt32}  # Bitmap per bucket
    n_buckets::Int
    n_entries::Int
    empty_key::K
end

mutable struct MtlHiveHT{K,V}
    buckets::MtlVector{HiveBucket}
    freemasks::MtlVector{UInt32}
    n_buckets::Int
    n_entries::Int
    empty_key::K
end
```

## API (Matching DoubleHT)

```julia
# Construction
CuHiveHT{K,V}(n_buckets::Int; empty_key=typemax(K))
CuHiveHT(keys::Vector{K}, vals::Vector{V}; load_factor=0.7)
MtlHiveHT{K,V}(n_buckets::Int; empty_key=typemax(K))

# Operations (same as DoubleHT)
query(table, keys) -> (found::Vector{Bool}, results::Vector{V})
query!(results, found, table, keys)
upsert!(table, keys, vals) -> status::Vector{UInt8}

# NEW: Delete operation
delete!(table, keys) -> status::Vector{UInt8}
```

## Implementation Phases

### Phase 1: Core Types ✅

- [x] `src/hive/types.jl`
  - HiveBucket struct
  - pack_pair, unpack_key, unpack_val
  - Constants (EMPTY_PAIR, TOMBSTONE)
  - hive_hash, hive_hash_gpu

### Phase 2: CUDA Implementation

- [x] `src/hive/cuda/kernels.jl`
  - `hive_query_kernel!` (WCME protocol)
  - `hive_upsert_kernel!` (WABC protocol)
  - `hive_delete_kernel!`

- [x] `src/hive/cuda/table.jl`
  - CuHiveHT type
  - Constructors
  - query, query!, upsert!, delete! functions

- [x] `test/test_hive_cuda.jl`
  - Basic insert/query tests
  - Update tests
  - Delete tests
  - Slot reuse after delete

### Phase 3: Metal Implementation

- [x] `src/hive/metal/kernels.jl`
  - Same protocols using Metal primitives
  - `simd_ballot`, `simd_shuffle`
  - Per-slot locking (Metal lacks 64-bit CAS)

- [x] `src/hive/metal/table.jl`
  - MtlHiveHT type
  - Host-side retry for failed upserts

- [x] `test/test_hive_metal.jl`
  - Same test structure as CUDA

### Phase 4: Integration

- [x] Update `src/GPUHashTables.jl`
  - Add includes for hive modules
  - Export CuHiveHT, MtlHiveHT, delete!
  - Add DELETE_SUCCESS, DELETE_FAILED constants

- [x] Update `test/runtests.jl`
  - Include hive tests

### Phase 5: Benchmarking

- [ ] Compare HiveHT vs DoubleHT
  - Query throughput
  - Upsert throughput
  - Mixed workloads
  - Metal reliability (need for sub-batching?)

## Key Implementation Details

### Freemask Initialization

```julia
# All slots start free: all 32 bits set
freemasks = CUDA.fill(UInt32(0xFFFFFFFF), n_buckets)

# All pairs start empty
empty_bucket = HiveBucket(ntuple(_ -> HIVE_EMPTY_PAIR, 32))
buckets = CUDA.fill(empty_bucket, n_buckets)
```

### WABC Upsert Kernel (Pseudocode)

```julia
function hive_upsert_kernel!(status, buckets, freemasks, n_buckets, keys, vals, n_ops)
    op_idx = warp_id()
    lane = lane_id()  # 0-31

    key, val = keys[op_idx], vals[op_idx]
    h = hive_hash(key)

    for probe in 0:HIVE_MAX_PROBES-1
        bucket_idx = (h + probe) % n_buckets + 1

        # Lane 0 loads freemask, broadcasts
        mask = shfl_sync(freemasks[bucket_idx], 0)
        pair = buckets[bucket_idx].pairs[lane + 1]
        pair_key = unpack_key(pair)

        # Check for existing key (update)
        is_match = (pair_key == key)
        match_ballot = ballot_sync(is_match)

        if match_ballot != 0
            winner = trailing_zeros(match_ballot)
            if lane == winner
                new_pair = pack_pair(key, val)
                old = atomicCAS(ptr, pair, new_pair)
                status[op_idx] = (old == pair) ? UPDATED : FAILED
            end
            return
        end

        # Check for free slot (insert)
        is_free = ((mask >> lane) & 1) == 1
        free_ballot = ballot_sync(is_free)

        if free_ballot != 0
            winner = trailing_zeros(free_ballot)
            if lane == winner
                new_pair = pack_pair(key, val)
                old = atomicCAS(ptr, HIVE_EMPTY_PAIR, new_pair)
                if old == HIVE_EMPTY_PAIR
                    atomicAnd(freemasks_ptr, ~(1 << lane))
                    status[op_idx] = INSERTED
                else
                    status[op_idx] = FAILED
                end
            end
            return
        end

        # Bucket full, continue probing
    end

    status[op_idx] = FAILED
end
```

### Tombstone Handling

```julia
# In query: skip tombstones but continue probing
is_match = (pair_key == key) && (pair != HIVE_TOMBSTONE)
is_empty = (pair == HIVE_EMPTY_PAIR)  # Stop only at empty

# In delete: replace pair with tombstone
old = atomicCAS(ptr, pair, HIVE_TOMBSTONE)
if old == pair
    atomicOr(freemasks_ptr, 1 << lane)  # Mark slot reusable
end

# In insert: can use tombstone slots
is_insertable = is_free || (pair == HIVE_TOMBSTONE)
```

## Verification

```julia
using GPUHashTables, CUDA

# Create table
ht = CuHiveHT{UInt32,UInt32}(1000)

# Insert
keys = UInt32.(1:100)
vals = UInt32.(101:200)
status = upsert!(ht, keys, vals)
@assert all(status .== UPSERT_INSERTED)

# Query
found, results = query(ht, keys)
@assert all(found)
@assert results == vals

# Delete
status = delete!(ht, keys[1:10])
@assert all(status .== DELETE_SUCCESS)

# Verify deleted
found, _ = query(ht, keys[1:10])
@assert !any(found)

# Reinsert (slot reuse)
status = upsert!(ht, keys[1:10], vals[1:10])
@assert all(status .== UPSERT_INSERTED)
```

---

## References

- [warpSpeed GitHub](https://github.com/saltsystemslab/warpSpeed)
- [warpSpeed Paper](https://arxiv.org/pdf/2509.16407)
- [Hive Hash Table Paper](https://arxiv.org/abs/2510.15095)
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/)
- [Metal.jl Documentation](https://metal.juliagpu.org/stable/)
