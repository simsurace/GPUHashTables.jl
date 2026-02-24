# GPUHashTables.jl

A high-performance GPU hash table implementation for Julia, supporting both NVIDIA GPUs (via CUDA.jl) and Apple Silicon GPUs (via Metal.jl).

Based on the DoubleHT algorithm from [warpSpeed](https://github.com/saltsystemslab/warpSpeed), this package implements a static hash table optimized for massively parallel lookups. The design uses double hashing with warp/simdgroup-cooperative probing, where groups of 8 threads work together to scan buckets in parallel using ballot voting primitives.

## Features

- **Multi-backend support**: CUDA for NVIDIA GPUs, Metal for Apple Silicon
- **High throughput**: Optimized for batch queries with millions of lookups per kernel launch
- **Warp-cooperative design**: 8 threads per tile collaborate using ballot voting for efficient parallel probing
- **Configurable load factor**: Trade off memory usage vs. probe depth (default: 0.7)
- **Simple API**: Build on CPU, transfer to GPU, query in batches

## Installation

```julia
using Pkg
Pkg.add("GPUHashTables")
```

## Usage

### Building a Hash Table

Tables are built on the CPU and then transferred to the GPU:

```julia
using GPUHashTables
using Random

# Generate random key-value pairs
n = 10_000_000
keys = unique(rand(UInt32(1):UInt32(2^31-1), n * 2))[1:n]
vals = rand(UInt32, n)

# Build CPU table (also usable for queries)
cpu_table = CPUDoubleHT(keys, vals; load_factor=0.7)
```

### CPU Queries

```julia
# Query some keys
query_keys = keys[1:1000]
found, results = query(cpu_table, query_keys)

# found[i] is true if query_keys[i] was found
# results[i] contains the value if found
```

### CUDA GPU Queries

```julia
using CUDA

# Check if CUDA is available
if has_cuda()
    # Transfer table to GPU
    gpu_table = CuDoubleHT(cpu_table)

    # Query with CPU vectors (automatic transfer)
    found, results = query(gpu_table, query_keys)

    # Or use GPU vectors directly for best performance
    gpu_keys = CuVector(query_keys)
    gpu_found, gpu_results = query(gpu_table, gpu_keys)
end
```

### Metal GPU Queries (Apple Silicon)

```julia
using Metal

# Check if Metal is available
if has_metal()
    # Transfer table to GPU
    metal_table = MtlDoubleHT(cpu_table)

    # Query with CPU vectors (automatic transfer)
    found, results = query(metal_table, query_keys)

    # Or use GPU vectors directly for best performance
    metal_keys = MtlVector(query_keys)
    metal_found, metal_results = query(metal_table, metal_keys)
end
```

### In-place Queries

For repeated queries, pre-allocate result buffers to avoid allocations:

```julia
# CUDA
results = CUDA.zeros(UInt32, length(query_keys))
found = CUDA.zeros(Bool, length(query_keys))
query!(results, found, gpu_table, gpu_keys)

# Metal
results = Metal.zeros(UInt32, length(query_keys))
found = Metal.zeros(Bool, length(query_keys))
query!(results, found, metal_table, metal_keys)
```

## API Reference

### Types

- `CPUDoubleHT{K,V}` - CPU hash table for building and reference queries
- `CuDoubleHT{K,V}` - CUDA GPU hash table
- `MtlDoubleHT{K,V}` - Metal GPU hash table

### Functions

- `CPUDoubleHT(keys, vals; load_factor=0.7)` - Build a hash table from key-value vectors
- `CuDoubleHT(cpu_table)` - Transfer CPU table to CUDA GPU
- `MtlDoubleHT(cpu_table)` - Transfer CPU table to Metal GPU
- `query(table, keys)` - Batch query, returns `(found, results)` vectors
- `query!(results, found, table, keys)` - In-place batch query
- `has_cuda()` - Check CUDA availability at runtime
- `has_metal()` - Check Metal availability at runtime

## Benchmarks

### Metal (Apple M3 Pro)

Tested with 10M entries table, 10M query batch, load factor 0.7:

| Metric | Performance |
|--------|-------------|
| Positive queries (all keys exist) | 390 M queries/sec |
| Negative queries (no keys exist) | 376 M queries/sec |
| Mixed queries (50/50) | 375 M queries/sec |

**Scaling with table size** (10M queries, load factor 0.7):

| Table Size | Throughput |
|------------|------------|
| 100K | 390 M queries/sec |
| 1M | 391 M queries/sec |
| 10M | 390 M queries/sec |
| 50M | 389 M queries/sec |

**Scaling with load factor** (10M entries, 10M queries):

| Load Factor | Throughput |
|-------------|------------|
| 0.5 | 394 M queries/sec |
| 0.6 | 393 M queries/sec |
| 0.7 | 391 M queries/sec |
| 0.8 | 386 M queries/sec |
| 0.9 | 376 M queries/sec |

**GPU vs CPU comparison** (1M entries, 1M queries):

| Backend | Throughput | Speedup |
|---------|------------|---------|
| CPU | 40 M queries/sec | 1.0x |
| Metal (M3 Pro) | 388 M queries/sec | 9.6x |

### Running Benchmarks

```bash
julia --project=. benchmark/runbenchmarks.jl
```

## How It Works

The hash table uses **double hashing** for collision resolution:
1. A 64-bit MurmurHash is computed for each key
2. The hash is split into `h1` (bucket index) and `h2` (probe step)
3. Buckets are probed in sequence: `bucket[i] = (h1 + i * h2) % n_buckets`

Each bucket contains 8 slots. On the GPU, **8 threads form a cooperative tile** that searches a bucket in parallel:
1. Each thread checks one slot in the bucket
2. Threads use ballot voting to communicate findings across the tile
3. If the key is found or an empty slot is reached, the search terminates
4. Otherwise, the tile moves to the next bucket in the probe sequence

This design achieves high throughput by:
- Minimizing warp divergence (all 8 threads in a tile follow the same control flow)
- Using fast ballot intrinsics instead of shared memory
- Coalescing memory accesses within each bucket

## Limitations

- **Static tables**: Tables are built once and cannot be modified on the GPU
- **UInt32 keys/values**: Currently optimized for 32-bit unsigned integers
- **Sentinel values**: `typemax(UInt32)` is reserved as the empty sentinel and cannot be used as a key

## Acknowledgments

This implementation is based on the DoubleHT algorithm from the [warpSpeed](https://github.com/saltsystemslab/warpSpeed) library by the Salt Systems Lab.

## License

MIT
