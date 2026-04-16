# LayerNorm Kernel Development Log

## 1. Goal

This round of work focused on turning a LayerNorm demo kernel into a usable project operator inside `cuda-oplib`, while keeping a running record of:

- kernel version evolution
- benchmark methodology
- correctness and performance issues discovered along the way
- fixes applied during development
- final project integration steps

The target path eventually became:

- experimental kernels in `src/pydemo/`
- formal operator implementation in `src/kernel/layernorm_half2.cu`
- public API in `include/cuda_oplib/layernorm.h`
- build integration via CMake
- correctness test, benchmark, and example programs

## 2. Initial State

At the beginning, the repository only had a fully wired `vector_add` operator. `layernorm` existed only as a Python demo and a standalone CUDA demo file under `src/pydemo/`.

Initial reference points:

- `src/pydemo/layernorm.py`
- `src/pydemo/layernorm.cu`

The Python demo compared:

- `torch.nn.LayerNorm`
- a manual PyTorch tensor implementation:
  - mean
  - variance
  - normalize
  - affine

This established the first correctness baseline before moving into custom CUDA kernels.

## 3. Benchmark Methodology

The benchmark approach evolved into a fixed-shape latency comparison with:

- identical input `x`
- identical `gamma` and `beta`
- identical `eps`
- warmup iterations before timing
- average latency over multiple iterations
- CUDA timing using `cudaEvent` or `torch.cuda.Event`

The main fixed comparison shape used during experiments was:

- `B = 512`
- `D = 768`
- `dtype = float32`
- `device = cuda`

This shape was used repeatedly to compare the effects of different kernel designs.

## 4. Version 0: Pure PyTorch Reference

The first working baseline was:

- official `torch.nn.LayerNorm`
- a manual PyTorch tensor expression version

Observed result from an early run:

- official `torch.nn.LayerNorm`: `0.084 ms / iter`
- python tensor implementation: `0.358 ms / iter`

Interpretation:

- the Python implementation was not "real Python scalar math"
- it was still backed by optimized PyTorch kernels
- therefore it was slower than official fused LayerNorm, but much faster than an unoptimized custom CUDA kernel

## 5. Version 1: Naive CUDA Kernel

The first custom CUDA version was `layernorm_naive_kernel`.

Design:

- one block per row
- one thread per column
- thread `0` computed full-row mean and variance serially
- all threads waited
- each thread normalized one output element

Main issue:

- almost all threads sat idle while thread `0` computed row statistics
- this defeated GPU parallelism

Measured result from an early run:

- official `torch.nn.LayerNorm`: `0.084 ms / iter`
- python tensor implementation: `0.358 ms / iter`
- current cuda naive kernel: `1.289 ms / iter`

Interpretation:

- custom CUDA did not automatically mean faster
- this version was much slower than both official LayerNorm and PyTorch tensor composition

## 6. Version 2: Shared-Memory Reduction Kernel

The next step was `layernorm_reduction_kernel`.

Design:

- still one block per row
- use shared memory buffer
- parallel reduction for mean
- parallel reduction for variance

Improvement:

- this version removed the "thread 0 computes everything" bottleneck
- row statistics were now reduced cooperatively

Important flaw discovered during review:

- the reduction logic assumed a power-of-two reduction pattern
- for `N = 768`, correctness was not guaranteed
- this version was therefore not considered a safe general implementation

Later measured result:

- official: `0.026 ms / iter`
- reduction kernel: `0.037 ms / iter`

Interpretation:

- even when it ran, synchronization overhead and shared-memory reduction cost were high
- it was slower than official LayerNorm

## 7. Version 3: Stride Kernel

The next kernel was `layernorm_stride_kernel`.

Design:

- fixed block size, typically `256`
- each thread handled multiple elements with a stride loop
- local sums and local variances were reduced cooperatively
- output writeback also used strided processing

Why this was better:

- threads no longer required `blockDim.x == hidden`
- the kernel became usable for more hidden sizes
- work was more evenly distributed

This version was an important transition from "toy reduction kernel" to something structurally closer to a usable LayerNorm kernel.

## 8. Version 4: Warp-Reduction Kernel

The next design replaced much of the shared-memory reduction logic with warp-level primitives.

Relevant helper structure:

- `warp_reduce_sum`
- `block_reduce_sum`
- `layernorm_wrap_kernel`

Design intent:

- reduce synchronization
- use warp shuffle intrinsics
- improve performance on row-wise reduction

This version performed well on the fixed benchmark shape.

One key measured result:

- official `torch.nn.LayerNorm`: `0.014 ms / iter`
- wrap kernel: `0.011 ms / iter`

Later multi-kernel comparison:

- official: `0.026 ms / iter`
- warp kernel: `0.014 ms / iter`

Interpretation:

- on the tested shape, warp reduction significantly improved performance
- this was the first custom version that outperformed the official implementation in the fixed benchmark setup

Important note:

- output strings initially used wording like `0.546x slower`
- this wording was incorrect when the ratio was below `1.0`
- the real meaning was that the custom kernel used about `54.6%` of the official runtime

## 9. Version 5: Welford Kernel

The next kernel introduced numerically stable variance computation:

- `WelfordData`
- `welford_update`
- `welford_combine`
- `welford_warp_reduce`
- `welford_block_reduce`
- `layernorm_welford_kernel`

Design goal:

- preserve good reduction performance
- improve numerical robustness for mean/variance computation

### 9.1 First Critical Bug

During review, a race condition was identified:

- after block reduction, all threads wrote shared `mean` and `var`
- but only one thread actually held the correct reduced `local_state`

This was fixed by restricting the final shared writes to `tid == 0`.

### 9.2 Compile-Time Bug

The first version of Welford helper functions compiled as host functions because they were missing device qualifiers.

Observed compilation errors included:

- calling `__shfl_down_sync` from a host function
- using `__syncthreads()` from a host function
- declaring `static __shared__` inside a host function body

Fix applied:

- mark helper routines as `__device__ __forceinline__`
- pass block shared memory explicitly into `welford_block_reduce`
- remove invalid host-side use of device intrinsics

### 9.3 Measured Result

From the multi-kernel comparison:

- official: `0.026 ms / iter`
- welford kernel: `0.015 ms / iter`

Interpretation:

- Welford remained close to warp-reduction performance
- it traded a small amount of speed for a more principled variance calculation

## 10. Version 6: Half Kernel

After the float kernels, work moved to half precision:

- `layernorm_half_kernel`

Design:

- inputs, `gamma`, `beta`, and output stored as `half`
- mean/variance accumulation done in `float`
- final output converted back to `half`

Assessment during review:

- no fatal correctness issue under normal assumptions
- this was a stable "half storage + float accumulate" version

Limitations:

- not vectorized
- still scalar half loads/stores
- not the final performance target

## 11. Version 7: Half2 Kernel

The next major step was:

- `layernorm_half2_kernel`

Goal:

- use `half2` vectorized loads/stores where possible
- preserve Welford-style accumulation in float

### 11.1 First Review Findings

Two critical issues were identified:

1. Odd `N` was not handled.
   - the kernel processed only `N / 2` `half2` elements
   - if `N` was odd, the final scalar element was ignored

2. `half2` alignment was assumed unsafely.
   - `reinterpret_cast<const half2*>(row_x)` requires 4-byte alignment
   - row pointers are not automatically safe for `half2` if the row stride produces misaligned addresses

### 11.2 Minimal Fix Applied

The kernel was minimally patched to:

- check alignment of `row_x` and `row_y`
- use vectorized `half2` path only when aligned
- compute `vecN = N / 2` only for the aligned path
- process any remainder from `tail_start` to `N` using scalar half logic

This ensured correctness for:

- odd hidden sizes
- rows whose starting addresses are not safe for `half2`

### 11.3 Why This Fix Was Chosen

The goal was to keep the kernel structure intact rather than redesign it.

The chosen strategy was:

- vectorize when safe
- gracefully fall back to scalar processing for unsafe or leftover elements

This was the smallest change that fixed the real correctness issues.

## 12. Transition from Demo to Formal Operator

The project then moved from experimental demo code to formal operator integration.

New formal implementation file:

- `src/kernel/layernorm_half2.cu`

Public API:

- `include/cuda_oplib/layernorm.h`

Exported function:

- `cuda_oplib::layernorm_half(...)`

This established the correct call chain:

- external caller includes `layernorm.h`
- caller links against `cuda_oplib`
- caller calls `cuda_oplib::layernorm_half(...)`
- host launcher computes launch config
- launcher invokes `layernorm_half2_kernel<<<...>>>`

## 13. CMake Integration

The build system originally only compiled `vector_add`.

CMake was updated so that:

- `src/CMakeLists.txt` builds `kernel/vector_add.cu`
- if present, it also builds `kernel/layernorm_half2.cu`

Additional optional CMake integration was added for:

- `tests/cpp/test_layernorm.cu`
- `benchmarks/bench_layernorm.cu`
- `examples/cpp/layernorm_example.cu`

These targets are only added when the files exist, keeping the build robust during staged development.

## 14. Formal Operator Files Added

The following files were added or completed as part of formal integration:

- `src/kernel/layernorm_half2.cu`
- `include/cuda_oplib/layernorm.h`
- `tests/cpp/test_layernorm.cu`
- `benchmarks/bench_layernorm.cu`
- `examples/cpp/layernorm_example.cu`

Updated build integration:

- `src/CMakeLists.txt`
- `tests/CMakeLists.txt`
- `benchmarks/CMakeLists.txt`
- `examples/CMakeLists.txt`

## 15. Formal Test Added

`tests/cpp/test_layernorm.cu` was added as a minimal correctness check.

It verifies:

- project-level call path through `cuda_oplib::layernorm_half(...)`
- odd hidden size handling using `kHidden = 7`
- output correctness against a CPU reference implementation

This test was chosen specifically to exercise the edge case that used to break the half2 kernel.

## 16. Formal Benchmark Added

`benchmarks/bench_layernorm.cu` was added to provide a project-native benchmark.

This benchmark:

- allocates device buffers
- warms up the kernel
- times repeated calls to `cuda_oplib::layernorm_half(...)`
- reports average latency and approximate bandwidth

This moves benchmarking from ad hoc Python-only experiments to the actual project build.

## 17. Example Program Added

`examples/cpp/layernorm_example.cu` was added so that:

- users can see the minimal host-side usage of the operator
- the correct library call path is demonstrated

This file acts as a very small integration example and onboarding reference.

## 18. Interface Cleanup

A mismatch was found between the public header and the implementation.

Header:

- used `std::size_t`
- exposed a `cudaStream_t stream = 0` parameter

Implementation:

- still used `int`
- ignored `stream`

Fix applied:

- implementation updated to match the public header contract
- zero-size and null-pointer checks added
- oversized dimensions guarded before casting to `int`
- stream now passed into kernel launch

This made the library API internally consistent and ready for downstream use.

## 19. Build Validation

After integration, the project was configured and built successfully:

- `cmake -S /home/gs_cs/LLM/cuda-oplib -B /home/gs_cs/LLM/cuda-oplib/build`
- `cmake --build /home/gs_cs/LLM/cuda-oplib/build -j4`

Build outputs included:

- `libcuda_oplib.so`
- `test_layernorm`
- `bench_layernorm`
- `layernorm_example`

This confirmed that:

- the operator is formally connected into the project
- CMake wiring is valid
- the new API compiles and links
- tests/benchmarks/examples now recognize the operator

## 20. Git History

A formal integration commit was created:

- `59ca22f Add layernorm half2 operator pipeline`

Push to GitHub was attempted, but the remote push was blocked by local network/proxy/credential issues in the execution environment. The commit exists locally even if remote synchronization was not completed inside the session.

## 21. Summary of Kernel Evolution

The development sequence for this round was:

1. PyTorch reference and manual tensor implementation
2. naive CUDA kernel
3. shared-memory reduction kernel
4. stride kernel
5. warp-reduction kernel
6. Welford kernel
7. half kernel
8. half2 kernel with alignment and tail handling
9. formal operator integration into `cuda-oplib`

## 22. Main Lessons Learned

- A CUDA kernel can easily be slower than PyTorch if parallel reduction is poorly designed.
- Shared-memory reduction helps, but synchronization can still dominate.
- Warp-level primitives can materially improve LayerNorm latency.
- Welford improves numerical robustness but introduces its own implementation complexity.
- `half2` vectorization is only safe when alignment and odd-tail cases are handled carefully.
- A kernel is not really part of the project until it has:
  - public API
  - build integration
  - correctness test
  - benchmark
  - example or binding path

## 23. Current State After This Round

At the end of this round, the repository contains:

- formal `layernorm_half` operator API
- half2-based project kernel implementation
- build system integration
- a project-native test
- a project-native benchmark
- a minimal example

The operator has moved from experimental demo status to a usable project component.
