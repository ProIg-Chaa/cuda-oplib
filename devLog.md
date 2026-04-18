# CUDA Operator Development Log

## 1. Goal

This round of work focused on turning prototype CUDA normalization kernels into usable project operators inside `cuda-oplib`, while keeping a running record of:

- kernel version evolution
- benchmark methodology
- correctness and performance issues discovered along the way
- fixes applied during development
- final project integration steps
- later development-grade experiment framework work
- RMSNorm prototype evolution and formal integration

The target path eventually became:

- experimental kernels in `src/pydemo/`
- formal operator implementation in `src/kernel/layernorm_half2.cu`
- public API in `include/cuda_oplib/layernorm.h`
- build integration via CMake
- correctness test, benchmark, and example programs

Later, the project expanded into a broader two-layer structure:

- development-grade experiment modules in `python/experiments/`
- engineering-grade formal operators in `include/`, `src/kernel/`, `tests/`, `benchmarks/`, and `examples/`

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

## 20. Development-Grade Experiment Framework

After the formal `layernorm_half` operator path was integrated into the project,
the next structural improvement was to separate:

- development-grade experimentation
- engineering-grade formal integration

The motivation was straightforward:

- ad hoc benchmark scripts were becoming harder to extend
- kernel comparison conditions were not always centralized
- adding a new prototype kernel required repeated manual edits
- the project needed a clearer path from "experimental version" to "formal operator"

### 20.1 New LayerNorm Experiment Layer

A development-grade experiment layer was added under:

- `python/experiments/layernorm/`

This layer is intentionally different from the formal library path in
`include/`, `src/kernel/`, `tests/`, `benchmarks/`, and `examples/`.

Its role is to support:

- fast prototype comparison
- correctness checks against references
- benchmark consistency across kernel variants
- teaching-oriented recording of version evolution

### 20.2 Files Added to the Experiment Layer

The LayerNorm experiment framework was split into small focused modules:

- `compare.py`
  - unified experiment entrypoint
- `registry.py`
  - implementation registration table
- `cases.py`
  - shared benchmark cases such as `main_fp32` and `main_fp16`
- `report.py`
  - terminal table output and markdown table output
- `refs.py`
  - reference implementations such as `torch.nn.LayerNorm`
- `layernorm_wrap.cu`
  - CUDA extension wrapper used to expose experiment kernels to Python

This replaced the earlier "single script does everything" style with a clearer
division of responsibilities.

### 20.3 Registration Model

The experiment framework introduced an explicit registration model through
`LayerNormImpl` in `registry.py`.

Each registered implementation now carries:

- `name`
- `stage`
- `supported_dtypes`
- `notes`
- `builder`

This made it possible to treat implementations uniformly while still preserving
important metadata about maturity and applicability.

The current registered LayerNorm implementations are:

- `torch_official`
- `torch_python`
- `warp`
- `reduction`
- `welford`
- `half2`

The `stage` field was introduced to distinguish roles such as:

- `baseline`
- `candidate`
- `draft`

This was especially useful for keeping obviously flawed kernels, such as the
old reduction variant, visible as part of the learning path without treating
them as recommended implementations.

### 20.4 Unified Call Path in the Experiment Layer

The experiment layer now follows a consistent call chain:

- `compare.py` selects a case from `cases.py`
- inputs are generated once for the selected case
- `registry.py` provides the list of registered implementations
- `torch_official` is used as the main correctness and performance reference
- each implementation is run under the same conditions
- `report.py` formats the result table for terminal or markdown use

For CUDA-backed experiment kernels, the path is:

- `compare.py`
- `registry.py`
- `layernorm_wrap.cu`
- `layernorm.cu`
- GPU kernel execution

This gave the project a real development pipeline instead of a set of isolated
benchmark scripts.

### 20.5 Unified Benchmark Cases

Benchmark cases were extracted into `cases.py` so that every implementation
would be measured under shared conditions.

Examples include:

- `debug_fp32`
- `main_fp32`
- `main_fp16`
- `large_fp16`

This was an important step because it removed case definitions from
individual scripts and made benchmark runs easier to reproduce.

### 20.6 Unified Correctness and Benchmark Flow

The experiment driver now runs in two explicit phases:

1. correctness check against the official reference
2. benchmark timing only if correctness passes

This was an intentional design choice.

One of the key lessons from earlier iterations was:

- fast but incorrect kernels are not useful performance results

So the framework now refuses to treat a failing implementation as a valid
benchmark candidate.

### 20.7 Representative Output From the New Framework

A representative `main_fp32` run produced results like:

- `torch_official`: correct, baseline
- `torch_python`: correct, slower than official
- `warp`: correct, very close to official
- `reduction`: incorrect on this case
- `welford`: correct, slower than warp
- `half2`: skipped for float32 because of dtype filtering

This was the first point where the project could automatically express not only
"how fast is a kernel?" but also:

- is it correct?
- what stage is it in?
- is it applicable to the current dtype?

That made the experiment layer much more useful both for engineering and for
teaching.

### 20.8 Separation Between Development and Engineering Layers

After this change, the project structure became more intentional:

- development-grade layer:
  - `python/experiments/`
- engineering-grade layer:
  - `include/`
  - `src/kernel/`
  - `tests/`
  - `benchmarks/`
  - `examples/`

The intended lifecycle for future operators is now:

- prototype or iterate in the experiment layer
- compare against references and older variants
- decide which implementation is stable enough
- promote the chosen path into the formal project layer

This distinction is one of the most important structural upgrades in the
repository so far.

## 21. RMSNorm Experiment Layer

After the LayerNorm experiment framework stabilized, the same structure was
extended to RMSNorm.

The new development-grade RMSNorm module was added under:

- `python/experiments/RMSnorm/`

Its structure mirrors the LayerNorm experiment layer:

- `compare.py`
- `registry.py`
- `cases.py`
- `report.py`
- `refs.py`
- `rmsnorm_wrap.cu`
- `rmsnorm.cu`

The current registered RMSNorm implementations are:

- `torch_official`
- `torch_python`
- `f32_warp`
- `half2_warp`

This mattered for two reasons:

- it proved that the experiment-layer design was reusable across operators
- it established the beginning of a project-wide workflow rather than a
  LayerNorm-only workflow

One representative `main_fp16` run produced results like:

- `torch_official`: correct, fastest baseline on that case
- `torch_python`: failed current correctness tolerance in float16
- `f32_warp`: skipped because dtype filtering marked it unsupported
- `half2_warp`: correct and very close to official RMSNorm

This made RMSNorm the second operator in the repository to gain a structured
development-grade comparison path.

## 22. RMSNorm Formal Integration

Once the `half2_warp` RMSNorm path looked stable in the experiment layer, it
was promoted into the engineering-grade project layer.

The formal integration chain now includes:

- public API:
  - `include/cuda_oplib/rmsnorm.h`
- formal project kernel:
  - `src/kernel/rmsnorm_half2.cu`
- correctness test:
  - `tests/cpp/test_rmsnorm.cu`
- project benchmark:
  - `benchmarks/bench_rmsnorm.cu`
- minimal example:
  - `examples/cpp/rmsnorm_example.cu`
- build registration:
  - `src/CMakeLists.txt`
  - `tests/CMakeLists.txt`
  - `benchmarks/CMakeLists.txt`
  - `examples/CMakeLists.txt`

The final formal API exposed for this path is:

- `cuda_oplib::rmsnorm_half(...)`

The runtime call chain for the engineering-grade RMSNorm path is now:

- example / test / benchmark
- `cuda_oplib::rmsnorm_half(...)`
- `rmsnorm_half2_kernel<<<...>>>`

This is the exact point where RMSNorm stopped being "just an experiment" and
became an actual project operator.

The build and runtime path were validated during integration:

- CMake configure succeeded
- full project build succeeded
- `test_rmsnorm` passed
- `rmsnorm_example` ran successfully

### 22.1 Engineering Benchmark Results

After formal integration, both operator-level engineering benchmarks were run
through the project-native benchmark binaries.

Measured results:

- `bench_layernorm 4096 768 200`
  - `avg_ms = 0.208`
  - `approx_throughput_GBps = 90.735`
- `bench_rmsnorm 4096 768 200`
  - `avg_ms = 0.114`
  - `approx_throughput_GBps = 165.056`

Interpretation:

- the formal `rmsnorm_half` path is substantially faster than the formal
  `layernorm_half` path on this benchmark shape
- this is consistent with the operator structure, because RMSNorm avoids the
  mean-centering work and the extra `beta` affine path present in LayerNorm
- these measurements are especially useful because they come from the
  engineering-grade benchmark binaries rather than prototype scripts

## 23. Git History

A formal integration commit was created:

- `59ca22f Add layernorm half2 operator pipeline`

Push to GitHub was attempted, but the remote push was blocked by local network/proxy/credential issues in the execution environment. The commit exists locally even if remote synchronization was not completed inside the session.

## 24. Summary of Kernel Evolution

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
10. development-grade experiment framework for unified comparison
11. RMSNorm experiment-layer replication
12. RMSNorm half2 formal operator integration

## 25. Main Lessons Learned

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
- Once multiple variants exist, a unified experiment layer becomes necessary.
- Registration, shared cases, and shared reporting are as valuable for learning
  as they are for engineering.
- A reusable experiment-layer template makes it much easier to onboard the next
  operator instead of rebuilding ad hoc scripts from scratch.

## 26. Current State After This Round

At the end of this round, the repository contains:

- formal `layernorm_half` operator API
- half2-based project kernel implementation
- build system integration
- a project-native test
- a project-native benchmark
- a minimal example
- a development-grade experiment framework for comparing LayerNorm variants
- a development-grade experiment framework for comparing RMSNorm variants
- formal `rmsnorm_half` operator API
- half2-based RMSNorm project kernel implementation
- RMSNorm test, benchmark, and example targets
- recorded engineering benchmark results for both formal normalization operators

The repository has now moved beyond a single-operator demo and into a small but
coherent operator-learning project with both development-grade and
engineering-grade layers.
