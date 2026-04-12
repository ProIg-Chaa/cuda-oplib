# cuda-oplib

`cuda-oplib` is a CUDA operator library scaffold for building, benchmarking,
testing, and later open-sourcing custom GPU kernels.

This repository is intentionally organized around long-term growth:

- `include/`: public C++ headers
- `src/operators/`: CUDA operator implementations
- `tests/`: correctness tests
- `benchmarks/`: performance measurement entry points
- `examples/`: minimal usage examples
- `bindings/`: framework integrations such as PyTorch
- `docs/`: architecture, operator conventions, and roadmap
- `scripts/`: local developer workflows

## Current status

The repository includes one minimal reference operator:

- `vector_add`: a simple float32 elementwise add kernel

Its purpose is to validate the end-to-end build layout, not to represent the
final operator set.

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

