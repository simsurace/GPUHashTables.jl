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

### Query Throughput

Tested with 10M entries table, 10M query batch, load factor 0.7:

| Metric                            | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|-----------------------------------|---------------|--------------|-------------|------------|
| Positive queries (all keys exist) | 97.6 M        | 651 M        | 107 M       | 29.6 M     |
| Negative queries (no keys exist)  | 98.5 M        | 636 M        | 109 M       | 31.7 M     |
| Mixed queries (50/50)             | 98.3 M        | 660 M        | 108 M       | 32.0 M     |

### Query Scaling

**Scaling with table size** (10M queries, load factor 0.7):

| Table Size | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|------------|---------------|--------------|-------------|------------|
| 100K       | 98.0 M        | 1110 M       | 108 M       | 33.0 M     |
| 1M         | 98.1 M        | 820 M        | 107 M       | 30.8 M     |
| 10M        | 97.8 M        | 804 M        | 108 M       | 31.4 M     |
| 50M        | 97.5 M        | 800 M        | 107 M       | 32.5 M     |

**Scaling with load factor** (10M entries, 10M queries):

| Load Factor | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|-------------|---------------|--------------|-------------|------------|
| 0.5         | 97.0 M        | 855 M        | 107 M       | 31.7 M     |
| 0.6         | 98.0 M        | 837 M        | 108 M       | 32.4 M     |
| 0.7         | 97.7 M        | 644 M        | 108 M       | 32.5 M     |
| 0.8         | 98.0 M        | 748 M        | 108 M       | 31.3 M     |
| 0.9         | 98.0 M        | 658 M        | 107 M       | 30.9 M     |

**Scaling with query batch size** (10M entries, load_factor=0.7):

| Batch Size | `MtlDoubleHT` | `CuDoubleHT` | `MtlHiveHT` | `CuHiveHT` |
|------------|---------------|--------------|-------------|------------|
| 10K        | 34.3 M        | 464 M        | 29.5 M      | 30.6 M     |
| 100K       | 99.7 M        | 742 M        | 109 M       | 31.7 M     |
| 1M         | 107 M         | 606 M        | 119 M       | 29.8 M     |
| 10M        | 98.1 M        | 773 M        | 108 M       | 29.4 M     |

**GPU vs CPU comparison** (1M entries, 1M queries):

| Backend                 | Queries/Sec | Speedup |
|-------------------------|-------------|---------|
| `CPUDoubleHT` (M3 Pro)  | 41.0 M      | 1.0x    |
| `MtlDoubleHT` (M3 Pro)  | 107 M       | 2.6x    |
| `CuDoubleHT` (RTX 2070) | 878 M       | 22x     |
| `CPUHiveHT` (M3 Pro)    | 29.4 M      | 1.0x    |
| `MtlHiveHT` (M3 Pro)    | 119 M       | 4.1x    |
| `CuHiveHT` (RTX 2070)   | 26.7 M      | 0.9x    |

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
