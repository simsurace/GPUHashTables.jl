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

## Implementation status

### Hash table designs

- A linear-probing design ported from [SimpleGPUHashTable](https://github.com/nosferalatu/SimpleGPUHashTable): `CPUSimpleHT`, `CuSimpleHT`.
- DoubleHT from the [warpSpeed library](https://github.com/saltsystemslab/warpSpeed), see [arXiv paper](https://arxiv.org/pdf/2509.16407): `CPUDoubleHT`, `CuDoubleHT`, `MtlDoubleHT`.
- Hive Hash Table, c.f. [arXiv paper](https://arxiv.org/pdf/2510.15095): `CPUHiveHT`, `CuHiveHT`, `MtlHiveHT`.

### Features

- Batch querying: `query`

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
CPU↔GPU transfer overhead for GPU tables. Speedups are relative to `Base.Dict` on M3 Pro.

### Query Throughput

**10M entries, 10M query batch, load factor 0.7:**

| Hash Table              | Positive       | Negative       | Mixed           |
|-------------------------|----------------|----------------|-----------------|
| `Base.Dict` (M3 Pro)    | 50.2 M (1.0x)  | 36.7 M (1.0x)  | 21.2 M (1.0x)   |
| `CPUSimpleHT` (M3 Pro)  | TODO           | TODO           | TODO            |
| `CuSimpleHT` (RTX 2070) | 176 M (3.5x)   | 167 M (4.6x)   | 177 M (8.3x)    |
| `CPUDoubleHT` (M3 Pro)  | TODO           | TODO           | TODO            |
| `CuDoubleHT` (RTX 2070) | 112 M (2.2x)   | 227 M (6.2x)   | 278 M (13.1x)   |
| `MtlDoubleHT` (M3 Pro)  | 79.9 M (1.6x)  | 81.9 M (2.2x)  | 84.3 M (4.0x)   |
| `CPUHiveHT` (M3 Pro)    | TODO           | TODO           | TODO            |
| `CuHiveHT` (RTX 2070)   | 161 M (3.2x)   | 231 M (6.3x)   | 239 M (11.3x)   |
| `MtlHiveHT` (M3 Pro)    | 70.3 M (1.4x)  | 78.1 M (2.1x)  | 78.6 M (3.7x)   |

**1M entries, 1M query batch, load factor 0.7:**

| Hash Table              | Positive        | Negative | Mixed |
|-------------------------|-----------------|----------|-------|
| `Base.Dict` (M3 Pro)    | 116 M (1.0x)    | TODO     | TODO  |
| `CPUSimpleHT` (M3 Pro)  | TODO            | TODO     | TODO  |
| `CuSimpleHT` (RTX 2070) | 634 M (5.5x)    | TODO     | TODO  |
| `CPUDoubleHT` (M3 Pro)  | 40.6 M (0.4x)   | TODO     | TODO  |
| `MtlDoubleHT` (M3 Pro)  | 94.3 M (0.8x)   | TODO     | TODO  |
| `CuDoubleHT` (RTX 2070) | 405 M (3.5x)    | TODO     | TODO  |
| `CPUHiveHT` (M3 Pro)    | 29.2 M (0.3x)   | TODO     | TODO  |
| `MtlHiveHT` (M3 Pro)    | 84.8 M (0.7x)   | TODO     | TODO  |
| `CuHiveHT` (RTX 2070)   | 391 M (3.4x)    | TODO     | TODO  |

### Query Scaling

**Scaling with table size** (10M queries, load factor 0.7, positive):

| Hash Table              | 100K            | 1M              | 10M             | 50M             |
|-------------------------|-----------------|-----------------|-----------------|-----------------|
| `Base.Dict` (M3 Pro)    | 179 M (1.0x)    | 113 M (1.0x)    | 49.4 M (1.0x)   | 51.4 M (1.0x)   |
| `CuSimpleHT` (RTX 2070) | 152 M (0.8x)    | 373 M (3.3x)    | 245 M (5.0x)    | 299 M (5.8x)    |
| `CuDoubleHT` (RTX 2070) | 135 M (0.8x)    | 140 M (1.2x)    | 247 M (5.0x)    | 212 M (4.1x)    |
| `MtlDoubleHT` (M3 Pro)  | 78.6 M (0.4x)   | 79.5 M (0.7x)   | 75.1 M (1.5x)   | 82.7 M (1.6x)   |
| `CuHiveHT` (RTX 2070)   | 230 M (1.3x)    | 335 M (3.0x)    | 258 M (5.2x)    | 106 M (2.1x)    |
| `MtlHiveHT` (M3 Pro)    | 85.5 M (0.5x)   | 85.0 M (0.8x)   | 81.3 M (1.6x)   | 50.4 M (1.0x)   |

**Scaling with load factor** (10M entries, 10M queries, positive):

| Hash Table              | 0.5             | 0.6             | 0.7             | 0.8             | 0.9             |
|-------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| `Base.Dict` (M3 Pro)    | 51.0 M (1.0x)   | 49.0 M (1.0x)   | 49.0 M (1.0x)   | 48.0 M (1.0x)   | 46.6 M (1.0x)   |
| `CuSimpleHT` (RTX 2070) | 242 M (4.7x)    | 237 M (4.8x)    | 235 M (4.8x)    | 234 M (4.9x)    | 233 M (5.0x)    |
| `CuDoubleHT` (RTX 2070) | 214 M (4.2x)    | 260 M (5.3x)    | 256 M (5.2x)    | 263 M (5.5x)    | 263 M (5.6x)    |
| `MtlDoubleHT` (M3 Pro)  | 89.1 M (1.7x)   | 76.0 M (1.6x)   | 74.4 M (1.5x)   | 76.3 M (1.6x)   | 75.0 M (1.6x)   |
| `CuHiveHT` (RTX 2070)   | 212 M (4.2x)    | 251 M (5.1x)    | 248 M (5.1x)    | 250 M (5.2x)    | 238 M (5.1x)    |
| `MtlHiveHT` (M3 Pro)    | 82.7 M (1.6x)   | 79.2 M (1.6x)   | 83.1 M (1.7x)   | 79.1 M (1.6x)   | 82.0 M (1.8x)   |

**Scaling with query batch size** (10M entries, load factor 0.7, positive):

| Hash Table              | 10K             | 100K            | 1M              | 10M             |
|-------------------------|-----------------|-----------------|-----------------|-----------------|
| `Base.Dict` (M3 Pro)    | 150 M (1.0x)    | 55.2 M (1.0x)   | 39.9 M (1.0x)   | 51.1 M (1.0x)   |
| `CuSimpleHT` (RTX 2070) | 111 M (0.7x)    | 199 M (3.6x)    | 210 M (5.3x)    | 323 M (6.3x)    |
| `CuDoubleHT` (RTX 2070) | 145 M (1.0x)    | 319 M (5.8x)    | 200 M (5.0x)    | 175 M (3.4x)    |
| `MtlDoubleHT` (M3 Pro)  | 14.6 M (0.1x)   | 56.6 M (1.0x)   | 87.9 M (2.2x)   | 89.9 M (1.8x)   |
| `CuHiveHT` (RTX 2070)   | 146 M (1.0x)    | 321 M (5.8x)    | 204 M (5.1x)    | 182 M (3.6x)    |
| `MtlHiveHT` (M3 Pro)    | 11.6 M (0.1x)   | 49.5 M (0.9x)   | 81.4 M (2.0x)   | 75.1 M (1.5x)   |

### Running Benchmarks

```bash
julia --project=. benchmark/runbenchmarks.jl
```

