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

- A linear-probing design ported from [SimpleGPUHashTable](https://github.com/nosferalatu/SimpleGPUHashTable): `CPUSimpleHT`, `CuSimpleHT`, `MtlSimpleHT`.
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

| Hash Table              | Positive         | Negative         | Mixed            |
|-------------------------|------------------|------------------|------------------|
| `Base.Dict` (M3 Pro)    | 50.2 M (1.0x)    | 36.7 M (1.0x)    | 21.2 M (1.0x)    |
| `CPUSimpleHT` (M3 Pro)  | 53.6 M (1.1x)    | 11.8 M (0.3x)    | 10.7 M (0.5x)    |
| `CuSimpleHT` (RTX 2070) | 174 M (3.5x)     | 180 M (4.9x)     | 128 M (6.1x)     |
| `MtlSimpleHT` (M3 Pro)  | **184 M (3.7x)** | **235 M (6.4x)** | 194 M (9.2x)     |
| `CPUDoubleHT` (M3 Pro)  | 24.3 M (0.5x)    | 28.4 M (0.8x)    | 27.3 M (1.3x)    |
| `CuDoubleHT` (RTX 2070) | 109 M (2.2x)     | 219 M (6.0x)     | 220 M (10.4x)    |
| `MtlDoubleHT` (M3 Pro)  | 79.9 M (1.6x)    | 81.9 M (2.2x)    | 84.3 M (4.0x)    |
| `CPUHiveHT` (M3 Pro)    | 7.1 M (0.1x)     | 6.5 M (0.2x)     | 6.7 M (0.3x)     |
| `CuHiveHT` (RTX 2070)   | 159 M (3.2x)     | 220 M (6.0x)     | **224 M (10.6x)**|
| `MtlHiveHT` (M3 Pro)    | 70.3 M (1.4x)    | 78.1 M (2.1x)    | 78.6 M (3.7x)    |

**1M entries, 1M query batch, load factor 0.7:**

| Hash Table              | Positive         | Negative         | Mixed            |
|-------------------------|------------------|------------------|------------------|
| `Base.Dict` (M3 Pro)    | 116 M (1.0x)     | 77.8 M (1.0x)    | 65.7 M (1.0x)    |
| `CPUSimpleHT` (M3 Pro)  | 109 M (0.9x)     | 50.6 M (0.7x)    | 41.2 M (0.6x)    |
| `CuSimpleHT` (RTX 2070) | 355 M (3.1x)     | **568 M (7.3x)** | 45.1 M (0.7x)    |
| `MtlSimpleHT` (M3 Pro)  | 59.6 M (0.5x)    | 171 M (2.2x)     | 207 M (3.1x)     |
| `CPUDoubleHT` (M3 Pro)  | 40.6 M (0.4x)    | 98.5 M (1.3x)    | 59.1 M (0.9x)    |
| `MtlDoubleHT` (M3 Pro)  | 94.3 M (0.8x)    | 90.8 M (1.2x)    | 93.9 M (1.4x)    |
| `CuDoubleHT` (RTX 2070) | **436 M (3.8x)** | 323 M (4.2x)     | **427 M (6.5x)** |
| `CPUHiveHT` (M3 Pro)    | 29.2 M (0.3x)    | 22.8 M (0.3x)    | 28.2 M (0.4x)    |
| `MtlHiveHT` (M3 Pro)    | 84.8 M (0.7x)    | 54.9 M (0.7x)    | 49.9 M (0.8x)    |
| `CuHiveHT` (RTX 2070)   | 389 M (3.4x)     | 360 M (4.6x)     | 38.6 M (0.6x)    |

### Query Scaling

**Scaling with table size** (10M queries, load factor 0.7, positive):

| Hash Table              | 100K             | 1M               | 10M              | 100M             |
|-------------------------|------------------|------------------|------------------|------------------|
| `Base.Dict` (M3 Pro)    | 179 M (1.0x)     | 113 M (1.0x)     | 49.4 M (1.0x)    | 51.2 M (1.0x)    |
| `CPUSimpleHT` (M3 Pro)  | 123 M (0.7x)     | 87.5 M (0.8x)    | 56.8 M (1.1x)    | 75.1 M (1.5x)    |
| `CuSimpleHT` (RTX 2070) | **295 M (1.6x)** | **300 M (2.7x)** | 139 M (2.8x)     | 126 M (2.5x)     |
| `MtlSimpleHT` (M3 Pro)  | 151 M (0.8x)     | 250 M (2.2x)     | 156 M (3.2x)     | 214 M (4.2x)     |
| `CPUDoubleHT` (M3 Pro)  | 55.8 M (0.3x)    | 38.9 M (0.3x)    | 24.9 M (0.5x)    | 28.1 M (0.5x)    |
| `CuDoubleHT` (RTX 2070) | 225 M (1.3x)     | 265 M (2.3x)     | 219 M (4.4x)     | **278 M (5.4x)** |
| `MtlDoubleHT` (M3 Pro)  | 78.6 M (0.4x)    | 79.5 M (0.7x)    | 75.1 M (1.5x)    | 83.9 M (1.6x)    |
| `CPUHiveHT` (M3 Pro)    | 30.5 M (0.2x)    | 27.3 M (0.2x)    | 7.4 M (0.1x)     | 6.4 M (0.1x)     |
| `CuHiveHT` (RTX 2070)   | 120 M (0.7x)     | 242 M (2.1x)     | **286 M (5.8x)** | 266 M (5.2x)     |
| `MtlHiveHT` (M3 Pro)    | 85.5 M (0.5x)    | 85.0 M (0.8x)    | 81.3 M (1.6x)    | 82.5 M (1.6x)    |

**Scaling with load factor** (10M entries, 10M queries, positive):

| Hash Table              | 0.5              | 0.6              | 0.7              | 0.8              | 0.9              |
|-------------------------|------------------|------------------|------------------|------------------|------------------|
| `Base.Dict` (M3 Pro)    | 51.0 M (1.0x)    | 49.0 M (1.0x)    | 49.0 M (1.0x)    | 48.0 M (1.0x)    | 46.6 M (1.0x)    |
| `CPUSimpleHT` (M3 Pro)  | 86.7 M (1.7x)    | 50.3 M (1.0x)    | 53.3 M (1.1x)    | 57.2 M (1.2x)    | 50.6 M (1.1x)    |
| `CuSimpleHT` (RTX 2070) | 131 M (2.6x)     | 142 M (2.9x)     | 142 M (2.9x)     | 141 M (2.9x)     | 141 M (3.0x)     |
| `MtlSimpleHT` (M3 Pro)  | 138 M (2.7x)     | 159 M (3.2x)     | 129 M (2.6x)     | 150 M (3.1x)     | 156 M (3.4x)     |
| `CPUDoubleHT` (M3 Pro)  | 29.9 M (0.6x)    | 25.1 M (0.5x)    | 25.1 M (0.5x)    | 25.2 M (0.5x)    | 24.2 M (0.5x)    |
| `CuDoubleHT` (RTX 2070) | 209 M (4.1x)     | 264 M (5.4x)     | 265 M (5.4x)     | 266 M (5.5x)     | 263 M (5.6x)     |
| `MtlDoubleHT` (M3 Pro)  | 89.1 M (1.7x)    | 76.0 M (1.6x)    | 74.4 M (1.5x)    | 76.3 M (1.6x)    | 75.0 M (1.6x)    |
| `CPUHiveHT` (M3 Pro)    | 7.2 M (0.1x)     | 7.3 M (0.1x)     | 7.2 M (0.1x)     | 7.4 M (0.2x)     | 7.1 M (0.2x)     |
| `CuHiveHT` (RTX 2070)   | **261 M (5.1x)** | **297 M (6.1x)** | **296 M (6.0x)** | **297 M (6.2x)** | **294 M (6.3x)** |
| `MtlHiveHT` (M3 Pro)    | 82.7 M (1.6x)    | 79.2 M (1.6x)    | 83.1 M (1.7x)    | 79.1 M (1.6x)    | 82.0 M (1.8x)    |

**Scaling with query batch size** (10M entries, load factor 0.7, positive):

| Hash Table              | 10K              | 100K             | 1M               | 10M              |
|-------------------------|------------------|------------------|------------------|------------------|
| `Base.Dict` (M3 Pro)    | 150 M (1.0x)     | 55.2 M (1.0x)    | 39.9 M (1.0x)    | 51.1 M (1.0x)    |
| `CPUSimpleHT` (M3 Pro)  | **204 M (1.4x)** | 84.9 M (1.5x)    | 55.4 M (1.4x)    | 54.5 M (1.1x)    |
| `CuSimpleHT` (RTX 2070) | 116 M (0.8x)     | 195 M (3.5x)     | **202 M (5.1x)** | 124 M (2.4x)     |
| `MtlSimpleHT` (M3 Pro)  | 11.6 M (0.1x)    | 64.7 M (1.2x)    | 173 M (4.3x)     | **199 M (3.9x)** |
| `CPUDoubleHT` (M3 Pro)  | 131 M (0.9x)     | 47.9 M (0.9x)    | 24.4 M (0.6x)    | 18.8 M (0.4x)    |
| `CuDoubleHT` (RTX 2070) | 144 M (1.0x)     | 307 M (5.6x)     | 196 M (4.9x)     | 117 M (2.3x)     |
| `MtlDoubleHT` (M3 Pro)  | 14.6 M (0.1x)    | 56.6 M (1.0x)    | 87.9 M (2.2x)    | 89.9 M (1.8x)    |
| `CPUHiveHT` (M3 Pro)    | 38.6 M (0.3x)    | 8.0 M (0.1x)     | 7.2 M (0.2x)     | 7.2 M (0.1x)     |
| `CuHiveHT` (RTX 2070)   | 157 M (1.0x)     | **332 M (6.0x)** | 198 M (5.0x)     | 162 M (3.2x)     |
| `MtlHiveHT` (M3 Pro)    | 11.6 M (0.1x)    | 49.5 M (0.9x)    | 81.4 M (2.0x)    | 75.1 M (1.5x)    |

### Running Benchmarks

```bash
julia --project=. benchmark/runbenchmarks.jl
```

