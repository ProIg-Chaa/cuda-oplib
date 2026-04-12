# Architecture

## Design principles

- Keep public operator APIs small and stable.
- Keep CUDA kernel details out of public headers.
- Require every operator to ship with a correctness test and a benchmark.
- Add framework bindings only after the operator-level API is clear.

## Directory responsibilities

- `include/cuda_oplib/`: public headers that downstream users include
- `src/operators/`: CUDA kernels and launch wrappers
- `tests/cpp/`: native correctness tests
- `benchmarks/`: performance binaries with explicit input sizes
- `bindings/pytorch/`: future integration layer for custom ops

## Recommended operator pattern

For each new operator, keep this split:

1. Public API in `include/cuda_oplib/<op_name>.h`
2. Implementation in `src/operators/<op_name>.cu`
3. Test in `tests/cpp/test_<op_name>.cu`
4. Benchmark in `benchmarks/bench_<op_name>.cu`
5. Optional framework wrapper in `bindings/`

## Future expansion

When the repository reaches 5 to 10 real operators, consider splitting into:

- `src/common/`: launch helpers, dtype dispatch, layout utilities
- `src/attention/`: attention-specific kernels
- `src/quantization/`: quantization and packing kernels
- `src/normalization/`: layernorm and rmsnorm kernels

