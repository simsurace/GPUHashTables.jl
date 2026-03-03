# GPUHashTables.jl

A library of hash table implementations in pure Julia, supporting both NVIDIA GPUs (via CUDA.jl) and Apple Silicon GPUs (via Metal.jl).

>[!WARNING]
>The implementations in this package are to be considered highly experimental.
>In particular, since most research on GPU hash tables is being conducted in CUDA, porting them to Metal is a best-effort endeavour with unclear odds of being successful.

## Installation

```julia
using Pkg
Pkg.add("GPUHashTables")
```

You need either a CUDA or Metal-capable GPU to use this package meaningfully. You can query
the presence of either with `has_cuda()` and `has_metal()` after loading the package.

## Implemented hash table designs

- DoubleHT from the [warpSpeed library](), see [arXiv paper](https://arxiv.org/pdf/2509.16407): `CPUDoubleHT`, `CuDoubleHT`, `MtlDoubleHT`.
- Hive Hash Table, c.f. [arXiv paper](https://arxiv.org/pdf/2510.15095), `CPUHiveHT`, `CuHiveHT`, `MtlHiveHT`.

## Usage

### Building a Hash Table

Tables can be built on the CPU and then transferred to the GPU, e.g.

```julia
using GPUHashTables
using Random

n = 10_000_000
keys = unique(rand(UInt32(1):UInt32(2^31-1), n * 2))[1:n]
vals = rand(UInt32, n)
cpu_table = CPUDoubleHT(keys, vals; load_factor=0.7)
gpu_table = CuDoubleHT(cpu_table)
```

### Querying

```julia
found, results = query(gpu_table, query_keys)
```

## Benchmarks

Metal benchmarks are with Apple M3 Pro, CUDA benchmarks are with an RTX 2070.
All benchmarks use the allocating `query(table, cpu_keys)` interface, which includes
CPU↔GPU transfer overhead for GPU tables.

### Query Throughput

Tested with 10M entries table, 10M query batch, load factor 0.7:

| Metric                            | `Base.Dict` | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|-----------------------------------|-------------|---------------|--------------|-------------|------------|
| Positive queries (all keys exist) | 50.2 M      | 75.5 M        | 114.0 M      | 70.3 M      | 27.6 M     |
| Negative queries (no keys exist)  | 36.7 M      | 77.7 M        | 219.1 M      | 78.1 M      | 27.6 M     |
| Mixed queries (50/50)             | 21.2 M      | 76.4 M        | 266.3 M      | 78.6 M      | 27.7 M     |

### Query Scaling

**Scaling with table size** (10M queries, load factor 0.7):

| Table Size | `Base.Dict` | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|------------|-------------|---------------|--------------|-------------|------------|
| 100K       | 179 M       | 82.7 M        | 224.7 M      | 85.5 M      | 29.8 M     |
| 1M         | 113 M       | 74.0 M        | 232.1 M      | 85.0 M      | 28.5 M     |
| 10M        | 49.4 M      | 70.6 M        | 259.2 M      | 81.3 M      | 28.4 M     |
| 50M        | 51.4 M      | 78.8 M        | 104.5 M      | 50.4 M      | 25.7 M     |

**Scaling with load factor** (10M entries, 10M queries):

| Load Factor | `Base.Dict` | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|-------------|-------------|---------------|--------------|-------------|------------|
| 0.5         | 51.0 M      | 70.8 M        | 235.8 M      | 82.7 M      | 29.2 M     |
| 0.6         | 49.0 M      | 69.8 M        | 257.6 M      | 79.2 M      | 29.2 M     |
| 0.7         | 49.0 M      | 70.7 M        | 251.3 M      | 83.1 M      | 29.4 M     |
| 0.8         | 48.0 M      | 69.4 M        | 240.5 M      | 79.1 M      | 29.3 M     |
| 0.9         | 46.6 M      | 59.4 M        | 237.4 M      | 82.0 M      | 28.4 M     |

**Scaling with query batch size** (10M entries, load_factor=0.7):

| Batch Size | `Base.Dict` | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|------------|-------------|---------------|--------------|-------------|------------|
| 10K        | 150 M       | 14.3 M        | 133.6 M      | 11.6 M      | 25.6 M     |
| 100K       | 55.2 M      | 57.3 M        | 307.2 M      | 49.5 M      | 27.7 M     |
| 1M         | 39.9 M      | 86.2 M        | 190.8 M      | 81.4 M      | 28.3 M     |
| 10M        | 51.1 M      | 82.0 M        | 158.0 M      | 75.1 M      | 27.7 M     |

**GPU vs CPU comparison** (1M entries, 1M queries):

| Hash Table              | Queries/Sec | Speedup |
|-------------------------|-------------|---------|
| `Base.Dict` (M3 Pro)    | 116 M       | 1.0x    |
| `CPUDoubleHT` (M3 Pro)  | 40.6 M      | 0.4x    |
| `CPUHiveHT` (M3 Pro)    | 29.2 M      | 0.3x    |
| `MtlDoubleHT` (M3 Pro)  | 87.9 M      | 0.8x    |
| `MtlHiveHT` (M3 Pro)    | 84.8 M      | 0.7x    |
| `CuDoubleHT` (RTX 2070) | 415.6 M     | 13.7x   |
| `CuHiveHT` (RTX 2070)   | 31.3 M      | 1.0x    |

### Running Benchmarks

```bash
julia --project=. benchmark/runbenchmarks.jl
```

## TODOs:

- Power-of-two n_buckets + bitwise AND to replace expensive modulo arithmetic in bucket
  indexing. Requires changing the constructor to round up n_buckets to the next power of
  two, and replacing % UInt32(n_buckets) with & UInt32(n_buckets - 1) in all kernels. The 
  step calculation changes from h2 % (n_buckets - 1) + 1 to h2 | UInt32(1) (ensures odd
  step, coprime with power-of-two size).
