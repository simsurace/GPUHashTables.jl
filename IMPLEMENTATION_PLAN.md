# GPUHashTables Implementation Plan

Port of warpSpeed's DoubleHT dynamic hash table to Julia using CUDA.jl.

## Overview

**Goal**: Implement a high-performance GPU hash table using double hashing with warp-cooperative parallelism.

**Scope (Phase 1)**:
- Key type: `UInt32`
- Value type: `UInt32`
- Operations: Build on CPU, query on GPU
- CPU reference implementation for correctness testing

**Target Performance**: Competitive with warpSpeed's ~2 billion queries/second on modern GPUs.

---

## Architecture

### Data Structure

```
┌─────────────────────────────────────────────────────────┐
│ DoubleHashTable{K,V}                                    │
├─────────────────────────────────────────────────────────┤
│ buckets::CuVector{Bucket{K,V}}   # GPU bucket storage   │
│ n_buckets::Int                   # Number of buckets    │
│ n_entries::Int                   # Number of entries    │
│ empty_key::K                     # Sentinel for empty   │
│ empty_val::V                     # Sentinel for empty   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Bucket{K,V} (128 bytes, cache-line aligned)             │
├─────────────────────────────────────────────────────────┤
│ slots::NTuple{BUCKET_SIZE, Slot{K,V}}                   │
│   where BUCKET_SIZE = 8 for UInt32/UInt32               │
│   each Slot = (key::K, val::V)                          │
└─────────────────────────────────────────────────────────┘
```

### Double Hashing Probe Sequence

```
h1(key) = murmur_hash(key) & 0xFFFFFFFF        # Primary bucket
h2(key) = murmur_hash(key) >> 32               # Step size

probe(i) = (h1 + h2 * i) % n_buckets           # i ∈ [0, MAX_PROBES)
```

### Warp-Cooperative Query

Each query uses 8 threads (a "tile") working together:
1. All 8 threads load one slot each from the bucket
2. Ballot vote identifies matching/empty slots
3. If match found → return value
4. If empty found → key not present
5. Otherwise → advance to next probe position

---

## File Structure

```
GPUHashTables/
├── Project.toml
├── IMPLEMENTATION_PLAN.md
├── src/
│   ├── GPUHashTables.jl          # Module definition, exports
│   ├── types.jl                  # Bucket, Slot, Table structs
│   ├── hash.jl                   # MurmurHash implementation
│   ├── cpu/
│   │   ├── table.jl              # CPU hash table implementation
│   │   └── operations.jl         # CPU build, query operations
│   └── gpu/
│       ├── table.jl              # GPU hash table wrapper
│       ├── kernels.jl            # CUDA kernels for query
│       └── warp_utils.jl         # Ballot, shuffle helpers
├── test/
│   ├── runtests.jl
│   ├── test_hash.jl              # Hash function tests
│   ├── test_cpu.jl               # CPU implementation tests
│   └── test_gpu.jl               # GPU implementation tests
└── benchmark/
    ├── runbenchmarks.jl
    ├── bench_query.jl            # Query throughput benchmarks
    └── bench_scaling.jl          # Scaling with table size
```

---

## Implementation Phases

### Phase 1: Foundation ✅

#### 1.1 Project Setup
- [x] Add dependencies to Project.toml: `CUDA`, `Random`, `Test`, `BenchmarkTools`
- [x] Set up module structure with includes
- [x] Define exports

#### 1.2 Hash Function
- [x] Implement MurmurHash64A (matches warpSpeed for compatibility)
- [x] Create `double_hash(key) -> (h1::UInt32, h2::UInt32)` helper
- [x] Test against known test vectors

```julia
# Target API
h1, h2 = double_hash(key)
bucket_idx = h1 % n_buckets
step = h2 % (n_buckets - 1) + 1  # Ensure non-zero step
```

#### 1.3 Data Types
- [x] Define `Slot{K,V}` as immutable struct (key-value pair)
- [x] Define `Bucket{K,V}` with `BUCKET_SIZE` slots
- [x] Define `DoubleHashTable{K,V}` container
- [x] Ensure proper memory alignment (128-byte buckets)

```julia
struct Slot{K,V}
    key::K
    val::V
end

struct Bucket{K,V}
    slots::NTuple{8, Slot{K,V}}
end

# Verify: sizeof(Bucket{UInt32,UInt32}) == 64 bytes
# (8 slots × 8 bytes each)
```

### Phase 2: CPU Reference Implementation ✅

#### 2.1 Table Construction
- [x] `DoubleHashTable(keys, values; load_factor=0.7)` constructor
- [x] Compute appropriate bucket count from load factor
- [x] Insert all key-value pairs using double hashing

```julia
# Target API
keys = rand(UInt32, 1_000_000)
vals = rand(UInt32, 1_000_000)
table = DoubleHashTable(keys, vals)
```

#### 2.2 CPU Query
- [x] `query(table, key) -> (found::Bool, value::V)`
- [x] `query!(results, found, table, keys)` batch query (mutates results array)
- [x] Implement probe sequence with MAX_PROBES limit (80)

#### 2.3 CPU Tests
- [x] Test empty table queries (all return not found)
- [x] Test single element insert/query
- [x] Test bulk insert with random keys
- [x] Test query for non-existent keys
- [x] Test at various load factors (50%, 70%, 90%)
- [x] Test probe sequence doesn't infinite loop

### Phase 3: GPU Implementation ✅

#### 3.1 GPU Table Transfer
- [x] `GPUDoubleHashTable(cpu_table)` - transfer to GPU
- [x] Store buckets in `CuVector{Bucket{K,V}}`
- [x] Keep metadata (n_buckets, sentinels) accessible to kernels

```julia
# Target API
cpu_table = DoubleHashTable(keys, vals)
gpu_table = GPUDoubleHashTable(cpu_table)
```

#### 3.2 Warp Utilities
- [x] Define `TILE_SIZE = 8` (threads per query)
- [x] Helper: `tile_id()` - which tile this thread belongs to
- [x] Helper: `tile_lane()` - position within tile (0-7)
- [x] Helper: `tile_ballot(mask, predicate)` - vote within tile

```julia
# Warp utility functions for kernels
@inline function tile_lane()
    return (threadIdx().x - 1) % TILE_SIZE
end

@inline function tile_ballot(mask, predicate)
    # Use vote_ballot_sync with appropriate mask
    full_ballot = vote_ballot_sync(0xFFFFFFFF, predicate)
    tile_offset = ((threadIdx().x - 1) ÷ TILE_SIZE) * TILE_SIZE
    return (full_ballot >> tile_offset) & 0xFF
end
```

#### 3.3 Query Kernel
- [x] Implement `query_kernel!(results, found, buckets, n_buckets, keys, n_queries, empty_key)`
- [x] Each tile of 8 threads handles one query
- [x] Cooperative bucket loading (each thread loads one slot)
- [x] Ballot-based match/empty detection
- [x] Probe sequence iteration

```julia
function query_kernel!(results, found, buckets, n_buckets, keys,
                       empty_key, empty_val)
    # Calculate which query this tile handles
    tile_idx = (threadIdx().x - 1) ÷ TILE_SIZE +
               (blockIdx().x - 1) * (blockDim().x ÷ TILE_SIZE)
    lane = tile_lane()

    if tile_idx > length(keys)
        return
    end

    key = keys[tile_idx]
    h1, h2 = double_hash(key)

    for probe in 0:MAX_PROBES-1
        bucket_idx = (h1 + h2 * probe) % n_buckets + 1
        bucket = buckets[bucket_idx]

        # Each thread checks one slot
        slot = bucket.slots[lane + 1]
        is_match = slot.key == key
        is_empty = slot.key == empty_key

        match_ballot = tile_ballot(is_match)
        empty_ballot = tile_ballot(is_empty)

        if match_ballot != 0
            # Found - first matching thread writes result
            if lane == trailing_zeros(match_ballot)
                results[tile_idx] = slot.val
                found[tile_idx] = true
            end
            return
        end

        if empty_ballot != 0
            # Empty slot found - key not in table
            if lane == 0
                found[tile_idx] = false
            end
            return
        end
    end

    # Exceeded max probes
    if lane == 0
        found[tile_idx] = false
    end
end
```

#### 3.4 Query API
- [x] `query!(results, found, gpu_table, keys)` - batch GPU query
- [x] Handle kernel launch configuration (threads, blocks)
- [x] Synchronize and return

```julia
function query!(results::CuVector, found::CuVector{Bool},
                table::GPUDoubleHashTable, keys::CuVector)
    n_queries = length(keys)
    threads_per_block = 256  # Must be multiple of TILE_SIZE
    tiles_per_block = threads_per_block ÷ TILE_SIZE
    n_blocks = cld(n_queries, tiles_per_block)

    @cuda threads=threads_per_block blocks=n_blocks query_kernel!(
        results, found, table.buckets, table.n_buckets,
        keys, table.empty_key, table.empty_val
    )

    synchronize()
end
```

#### 3.5 GPU Tests
- [x] Compare GPU results against CPU reference for same inputs
- [x] Test with various query batch sizes (1, 7, 32, 100, 1K, 10K)
- [x] Test positive queries (key exists)
- [x] Test negative queries (key doesn't exist)
- [x] Test mixed positive/negative queries

### Phase 4: Benchmarking ✅

#### 4.1 Query Throughput Benchmark
- [x] Measure queries/second at various table sizes (100K, 1M, 10M, 50M entries)
- [x] Measure at various load factors (50%, 60%, 70%, 80%, 90%)
- [x] Compare GPU vs CPU throughput
- [ ] Report bandwidth utilization

```julia
# Benchmark structure
function benchmark_query_throughput(n_entries, n_queries, load_factor)
    # Setup
    keys = rand(UInt32, n_entries)
    vals = rand(UInt32, n_entries)
    cpu_table = CPUDoubleHashTable(keys, vals; load_factor)
    gpu_table = GPUDoubleHashTable(cpu_table)

    query_keys = keys[rand(1:n_entries, n_queries)]  # Positive queries
    gpu_keys = CuVector(query_keys)
    results = CUDA.zeros(UInt32, n_queries)
    found = CUDA.zeros(Bool, n_queries)

    # Warmup
    query!(results, found, gpu_table, gpu_keys)

    # Benchmark
    CUDA.@sync begin
        t = @elapsed for _ in 1:10
            query!(results, found, gpu_table, gpu_keys)
        end
    end

    queries_per_second = (n_queries * 10) / t
    return queries_per_second
end
```

#### 4.2 Scaling Benchmark
- [x] Throughput vs table size (fixed query count)
- [x] Throughput vs query count (fixed table size)
- [x] Throughput vs load factor

#### 4.3 Comparison Benchmark
- [x] Compare against CPU DoubleHashTable
- [ ] Compare against simple linear probing GPU hash table
- [ ] Document speedup factors

### Phase 5: Future Extensions (Not in Initial Scope)

These are documented for future reference but not implemented in Phase 1:

#### 5.1 GPU Insert Operations
- [ ] Bitmap-based locking mechanism
- [ ] `insert!(gpu_table, keys, values)` kernel
- [ ] Handle concurrent insert conflicts

#### 5.2 GPU Delete Operations
- [ ] Tombstone sentinel value
- [ ] `delete!(gpu_table, keys)` kernel
- [ ] Tombstone-aware queries

#### 5.3 Dynamic Resizing
- [ ] Rehash when load factor exceeded
- [ ] Incremental vs full rehash strategies

#### 5.4 Extended Type Support
- [ ] `UInt64` keys and values
- [ ] Generic parametric types with size constraints
- [ ] String keys via hashing

---

## Key Implementation Details

### MurmurHash64A

```julia
function murmur_hash_64a(key::UInt32, seed::UInt64 = 0x0)::UInt64
    const m = 0xc6a4a7935bd1e995
    const r = 47

    h = seed ⊻ (8 * m)  # len = 8 bytes conceptually

    # Mix in the key (treating as 8 bytes, upper 4 are zero)
    k = UInt64(key)
    k *= m
    k ⊻= k >> r
    k *= m

    h ⊻= k
    h *= m

    # Finalization
    h ⊻= h >> r
    h *= m
    h ⊻= h >> r

    return h
end

function double_hash(key::UInt32)::Tuple{UInt32, UInt32}
    h = murmur_hash_64a(key)
    h1 = UInt32(h & 0xFFFFFFFF)
    h2 = UInt32(h >> 32)
    return (h1, h2)
end
```

### Memory Layout Considerations

For `UInt32` keys and values:
- `Slot{UInt32, UInt32}` = 8 bytes
- `Bucket` with 8 slots = 64 bytes
- This fits in one cache line on GPU (128 bytes with padding)

For optimal memory coalescing:
- Bucket array should be aligned to 128 bytes
- Each warp tile accesses consecutive slots within one bucket

### Sentinel Values

```julia
const EMPTY_KEY = typemax(UInt32)      # 0xFFFFFFFF
const EMPTY_VAL = typemax(UInt32)      # 0xFFFFFFFF
```

Users cannot store `EMPTY_KEY` as an actual key. Document this limitation.

### Error Handling

- Table construction fails if keys contain `EMPTY_KEY`
- Query returns `(false, EMPTY_VAL)` for not-found
- Probe limit exceeded treated as not-found (table too full)

---

## Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_murmur_hash` | Known test vectors, distribution quality |
| `test_double_hash` | Both components are non-zero, good distribution |
| `test_cpu_empty_table` | Queries on empty table return not-found |
| `test_cpu_single_entry` | Insert one, query one |
| `test_cpu_bulk` | 1M random entries, all found |
| `test_cpu_negative` | Query for keys not inserted |
| `test_cpu_load_factors` | Works at 50%, 70%, 90% load |
| `test_gpu_matches_cpu` | GPU and CPU return identical results |
| `test_gpu_batch_sizes` | Various batch sizes work correctly |

### Property-Based Tests

- For any set of unique keys, all inserted keys are found
- For any set of unique keys, random other keys are not found
- GPU results exactly match CPU results for same inputs

### Stress Tests

- 100M entry table with 100M queries
- Near-maximum load factor (95%)
- Adversarial key patterns (sequential, clustered hashes)

---

## Benchmark Targets

Based on warpSpeed paper results on RTX 3090:

| Metric | Target | Notes |
|--------|--------|-------|
| Query throughput | >1B queries/sec | Positive queries, 70% load |
| Negative query | >800M queries/sec | Key not in table |
| Mixed workload | >900M queries/sec | 50% positive, 50% negative |
| CPU speedup | >100x | vs single-threaded Julia Dict |

---

## Dependencies

```toml
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[extras]
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[targets]
test = ["Test", "BenchmarkTools", "Random"]
```

---

## Design Decisions (Following warpSpeed)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Bucket size** | 8 slots | Matches tile size (8 threads), each thread loads one slot |
| **Tile size** | 8 threads | Optimal for ballot operations within a warp |
| **Default load factor** | 70% | Good balance of space efficiency and probe length |
| **Max load factor** | 90% | DoubleHT handles high loads well due to reduced clustering |
| **Max probes** | 80 | Empirically sufficient for worst-case at high load factors |
| **Sentinel handling** | User responsibility | No runtime check; document that `EMPTY_KEY` cannot be stored |

---

## References

- [warpSpeed GitHub](https://github.com/saltsystemslab/warpSpeed)
- [warpSpeed Paper](https://arxiv.org/pdf/2509.16407)
- [CUDA.jl Documentation](https://cuda.juliagpu.org/stable/)
- [MurmurHash Reference](https://github.com/aappleby/smhasher)
