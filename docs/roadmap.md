# Roadmap

## Phase 1: repository stabilization

- Keep the build system minimal and deterministic.
- Add CI checks that do not depend on GPU availability.
- Decide the first 3 to 5 operators before exposing a public API promise.

## Phase 2: kernel growth

- Implement normalization and quantization primitives first.
- Add fused kernels only after baseline unfused versions are correct.
- Track performance per architecture with reproducible benchmark scripts.

## Phase 3: open-source release

- Add English-first documentation and contribution guidelines.
- Publish benchmark methodology and environment details.
- Tag a `v0.1.0` release only after tests, examples, and install flow work cleanly.

