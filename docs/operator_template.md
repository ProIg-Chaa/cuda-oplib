# Operator Template

Use this checklist when adding a new operator.

## Files

- `include/cuda_oplib/<op_name>.h`
- `src/operators/<op_name>.cu`
- `tests/cpp/test_<op_name>.cu`
- `benchmarks/bench_<op_name>.cu`

## API checklist

- The public signature should be explicit about dtype, layout, and stream.
- Validate null pointers and zero-sized inputs early.
- Return `cudaError_t` from the launch wrapper unless there is a stronger
  reason to wrap errors differently.

## Benchmark checklist

- Warm up before timing.
- Print the exact tensor size or problem size.
- Report throughput or effective bandwidth, not just raw latency.

## Open-source checklist

- Add a usage example if the operator is user-facing.
- Describe assumptions and limitations in `README.md` or `docs/`.
- Record supported architectures before tagging a release.

