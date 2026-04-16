# cuda-oplib

`cuda-oplib` is a CUDA operator library scaffold for building, benchmarking,
testing, and later open-sourcing custom GPU kernels.

This repository is intentionally organized around long-term growth:

- `include/`: public C++ headers
- `src/kernel/`: CUDA operator implementations
- `tests/`: correctness tests
- `benchmarks/`: performance measurement entry points
- `examples/`: minimal usage examples
- `bindings/`: framework integrations such as PyTorch
- `docs/`: architecture, operator conventions, and roadmap
- `scripts/`: local developer workflows

## Current status

The repository currently includes two integrated operators:

- `vector_add`: a simple float32 elementwise add kernel used to validate the
  end-to-end project layout
- `layernorm_half`: a half-precision LayerNorm operator backed by a `half2`
  kernel with float accumulation, odd-width tail handling, and project-level
  test / benchmark / example coverage

The current `layernorm_half` path is designed as a practical operator-integration
exercise:

- public API in `include/cuda_oplib/layernorm.h`
- implementation in `src/kernel/layernorm_half2.cu`
- correctness test in `tests/cpp/test_layernorm.cu`
- benchmark in `benchmarks/bench_layernorm.cu`
- example in `examples/cpp/layernorm_example.cu`

At a high level, the operator:

- computes per-row mean and variance using Welford-style reduction
- accumulates statistics in float for numerical stability
- uses `half2` vectorized load/store when row pointers are aligned
- falls back to scalar half processing for odd tails or unaligned cases

This makes it a useful stepping stone toward a broader operator library while
still remaining small enough to iterate on quickly.

## Build

Requirements:

- CUDA Toolkit 12.x or newer
- CMake 3.24 or newer
- A C++17-capable host compiler

```bash
./scripts/build.sh
```

Run tests:

```bash
./scripts/run_tests.sh
```

## Suggested growth path

1. Add one directory pair per operator interface and implementation.
2. Keep correctness tests and benchmarks together with every new operator.
3. Add framework bindings only after the C++/CUDA core API stabilizes.
4. Publish reproducible benchmark configs before the first open-source release.

## Planned first-wave operators

- layernorm
- rmsnorm
- softmax
- gemm epilogue fusion
- quantize / dequantize
- rope / rotary-related kernels
- cache and attention utility kernels

## License

Apache-2.0 for now. Change it before publishing if your release strategy
requires a different license.
