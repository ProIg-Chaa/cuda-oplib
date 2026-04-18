# cuda-oplib

> Personal CUDA operator learning lab with teaching-oriented kernels, lightweight engineering, and public dev notes.

![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
[![CI](https://img.shields.io/badge/CI-metadata%20check-2ea44f)](./.github/workflows/ci.yml)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

<a id="top"></a>
<a id="zh-cn"></a>

## Quick Links

- [中文](#zh-cn)
- [English](#english)
- [开发日志 / Dev Log](./devLog.md)
- [LayerNorm API](./include/cuda_oplib/layernorm.h)
- [LayerNorm Kernel](./src/kernel/layernorm_half2.cu)
- [LayerNorm Test](./tests/cpp/test_layernorm.cu)
- [LayerNorm Benchmark](./benchmarks/bench_layernorm.cu)
- [LayerNorm Example](./examples/cpp/layernorm_example.cu)
- [LayerNorm Experiment Driver](./python/experiments/layernorm/compare.py)
- [CI Workflow](./.github/workflows/ci.yml)

<details open>
<summary><strong>中文</strong></summary>

## 项目定位

`cuda-oplib` 是我的个人 CUDA 算子开发学习项目。

它的核心目标不是只做出几个能跑的 kernel，而是把 CUDA 算子从 `demo -> benchmark -> 修 bug -> 正式接入` 的全过程尽量保留下来，逐步整理成一个同时具备学习价值和轻度工程化结构的个人项目。

这个仓库会持续朝四个方向推进：

- `学习化`：系统练习 CUDA 算子开发
- `笔记化`：记录版本演进、实验结果和设计取舍
- `工程化`：把算子逐步接入统一 API、实现、测试、benchmark、example
- `输出化`：作为我公开分享 CUDA 学习过程和实践结果的载体

> [!NOTE]
> 这不是一个只追求最终性能数字的黑盒仓库。我更在意把开发过程、思考路径、踩坑记录和教学级代码一起留下来。

## 当前内容

### Operator Snapshot

| Operator | Status | What It Is | Entry |
|---|---|---|---|
| `vector_add` | Integrated | 最小 float32 元素加法算子，用来验证项目骨架 | [`src/kernel/vector_add.cu`](./src/kernel/vector_add.cu) |
| `layernorm_half` | Integrated | 基于 `half2` 路径的 half 精度 LayerNorm，统计使用 float 累加 | [`src/kernel/layernorm_half2.cu`](./src/kernel/layernorm_half2.cu) |

### LayerNorm Overview

`layernorm_half` 目前已经具备完整的正式接入链路：

- API: [`include/cuda_oplib/layernorm.h`](./include/cuda_oplib/layernorm.h)
- Kernel: [`src/kernel/layernorm_half2.cu`](./src/kernel/layernorm_half2.cu)
- Test: [`tests/cpp/test_layernorm.cu`](./tests/cpp/test_layernorm.cu)
- Benchmark: [`benchmarks/bench_layernorm.cu`](./benchmarks/bench_layernorm.cu)
- Example: [`examples/cpp/layernorm_example.cu`](./examples/cpp/layernorm_example.cu)
- Dev Log: [`devLog.md`](./devLog.md)

这个算子目前重点体现的是：

- 行级 LayerNorm 的 CUDA 实现
- Welford 风格的均值/方差统计
- half 存储、float 累加
- 对齐时走 `half2` 向量化路径
- odd tail 或未对齐时自动退回标量处理

除了正式接入链路，LayerNorm 现在还有一套开发级实验层，用来统一做原型验证、版本对比和教学记录：

- Experiment Driver: [`python/experiments/layernorm/compare.py`](./python/experiments/layernorm/compare.py)
- Registry: [`python/experiments/layernorm/registry.py`](./python/experiments/layernorm/registry.py)
- Cases: [`python/experiments/layernorm/cases.py`](./python/experiments/layernorm/cases.py)
- Reports: [`python/experiments/layernorm/report.py`](./python/experiments/layernorm/report.py)
- References: [`python/experiments/layernorm/refs.py`](./python/experiments/layernorm/refs.py)

这套实验层当前支持把 `torch_official`、`torch_python`、`warp`、`reduction`、`welford`、`half2` 注册进统一对比流程，在同一组 case 下跑 correctness 和 benchmark。

## LayerNorm Benchmark Snapshot

下表是当前阶段具有代表性的实验结果，用来展示内核演进方向，而不是作为最终性能结论。

| Variant | Shape | Dtype | Avg Latency |
|---|---:|---|---:|
| `torch.nn.LayerNorm` | `512 x 768` | `float32` | `0.026 ms` |
| `warp kernel` | `512 x 768` | `float32` | `0.014 ms` |
| `reduction kernel` | `512 x 768` | `float32` | `0.037 ms` |
| `welford kernel` | `512 x 768` | `float32` | `0.015 ms` |

更完整的开发过程、版本差异和实验背景记录在 [`devLog.md`](./devLog.md)。

## 仓库结构

```text
cuda-oplib
├── include/                public APIs
├── src/kernel/             formal CUDA operator implementations
├── src/pydemo/             older experiments and prototype-stage scripts
├── python/experiments/     development-grade experiment modules and compare drivers
├── tests/                  correctness tests
├── benchmarks/             performance benchmarks
├── examples/               minimal usage examples
├── bindings/               future framework bindings
└── docs/                   notes, architecture, and planning
```

这也是这个仓库想强调的一个方向：先在开发级实验层里快速验证，再把稳定版本推进到工程级正式层。

```text
development-grade experiments
-> registry / compare / report
-> stable kernel selection
-> formal operator integration
-> test / benchmark / example
```

其中：

- `python/experiments/`
  负责开发级实验、统一 case、统一注册和统一报告输出
- `src/kernel/ + include/ + tests/ + benchmarks/ + examples/`
  负责工程级正式算子能力

## 开发风格

这个项目尽量保持“教学可读 + 工程可落地”的平衡。

我的推进方式通常是：

1. 先实现一个容易解释的版本
2. 再逐步做性能优化
3. 在每一轮优化中记录为什么这么改
4. 最后把稳定路径接进正式项目结构

因此你会在仓库里看到：

- baseline 和优化版本并存
- 开发级统一实验入口
- bug 修复的上下文
- 从实验原型走向正式接入的完整痕迹

## Build

要求：

- CUDA Toolkit 12.x 或更新
- CMake 3.24 或更新
- 支持 C++17 的主机编译器

```bash
./scripts/build.sh
```

运行测试：

```bash
./scripts/run_tests.sh
```

或者直接使用 CMake：

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

开发级实验层当前的统一入口示例：

```bash
python3 python/experiments/layernorm/compare.py --case main_fp32
python3 python/experiments/layernorm/compare.py --case main_fp16 --markdown
```

## Roadmap

- [x] `vector_add` scaffold
- [x] `layernorm_half` half2 operator integration
- [x] LayerNorm correctness test
- [x] LayerNorm benchmark
- [x] LayerNorm example
- [x] LayerNorm development-grade experiment framework
- [ ] float LayerNorm path
- [ ] PyTorch binding
- [ ] RMSNorm
- [ ] Softmax
- [ ] More public dev notes and teaching-grade kernels

## License

Apache-2.0

</details>

<a id="english"></a>

<details>
<summary><strong>English</strong></summary>

## Project Purpose

`cuda-oplib` is my personal CUDA operator learning project.

The goal is not just to produce a few working kernels. I want this repository to preserve the full path from `demo -> benchmark -> bug fixing -> formal integration`, and gradually shape that process into a project that is both educational and lightly engineered.

This repository is intentionally built around four parallel goals:

- `Learning-oriented`: systematic CUDA operator practice
- `Notebook-oriented`: version history, experiment results, and design tradeoffs
- `Lightly engineered`: operators wired into API, implementation, tests, benchmarks, and examples
- `Public-facing`: a place to share what I am learning and building

> [!NOTE]
> This is not meant to be a black-box repository that only shows final performance numbers. The development process, reasoning, bugs, and teaching-oriented code are part of the product.

## Current Status

### Operator Snapshot

| Operator | Status | What It Is | Entry |
|---|---|---|---|
| `vector_add` | Integrated | Minimal float32 add operator used to validate the project scaffold | [`src/kernel/vector_add.cu`](./src/kernel/vector_add.cu) |
| `layernorm_half` | Integrated | Half-precision LayerNorm centered around a `half2` execution path with float accumulation | [`src/kernel/layernorm_half2.cu`](./src/kernel/layernorm_half2.cu) |

### LayerNorm Overview

The current `layernorm_half` path already includes:

- API: [`include/cuda_oplib/layernorm.h`](./include/cuda_oplib/layernorm.h)
- Kernel: [`src/kernel/layernorm_half2.cu`](./src/kernel/layernorm_half2.cu)
- Test: [`tests/cpp/test_layernorm.cu`](./tests/cpp/test_layernorm.cu)
- Benchmark: [`benchmarks/bench_layernorm.cu`](./benchmarks/bench_layernorm.cu)
- Example: [`examples/cpp/layernorm_example.cu`](./examples/cpp/layernorm_example.cu)
- Dev Log: [`devLog.md`](./devLog.md)

At a high level, it currently emphasizes:

- row-wise LayerNorm in CUDA
- Welford-style mean/variance reduction
- half storage with float accumulation
- `half2` vectorized execution when alignment permits
- scalar fallback for odd tails or unaligned cases

Besides the formal operator path, LayerNorm now also has a development-grade experiment layer used for prototype comparison, version tracking, and teaching-oriented benchmarking:

- Experiment Driver: [`python/experiments/layernorm/compare.py`](./python/experiments/layernorm/compare.py)
- Registry: [`python/experiments/layernorm/registry.py`](./python/experiments/layernorm/registry.py)
- Cases: [`python/experiments/layernorm/cases.py`](./python/experiments/layernorm/cases.py)
- Reports: [`python/experiments/layernorm/report.py`](./python/experiments/layernorm/report.py)
- References: [`python/experiments/layernorm/refs.py`](./python/experiments/layernorm/refs.py)

This experiment layer currently supports registering `torch_official`, `torch_python`, `warp`, `reduction`, `welford`, and `half2`, then comparing them under unified cases for correctness and latency.

## LayerNorm Benchmark Snapshot

The table below shows representative prototype-stage results. It is intended as a development snapshot, not a final performance claim.

| Variant | Shape | Dtype | Avg Latency |
|---|---:|---|---:|
| `torch.nn.LayerNorm` | `512 x 768` | `float32` | `0.026 ms` |
| `warp kernel` | `512 x 768` | `float32` | `0.014 ms` |
| `reduction kernel` | `512 x 768` | `float32` | `0.037 ms` |
| `welford kernel` | `512 x 768` | `float32` | `0.015 ms` |

For the full iteration history, notes, and debugging context, see [`devLog.md`](./devLog.md).

## Repository Layout

```text
cuda-oplib
├── include/                public APIs
├── src/kernel/             formal CUDA operator implementations
├── src/pydemo/             older experiments and prototype-stage scripts
├── python/experiments/     development-grade experiment modules and compare drivers
├── tests/                  correctness tests
├── benchmarks/             performance benchmarks
├── examples/               minimal usage examples
├── bindings/               future framework bindings
└── docs/                   notes, architecture, and planning
```

One of the main ideas behind this repository is to keep the path visible: validate quickly in the development-grade experiment layer, then promote stable kernels into the formal project layer.

```text
development-grade experiments
-> registry / compare / report
-> stable kernel selection
-> formal operator integration
-> test / benchmark / example
```

In practice:

- `python/experiments/`
  handles fast comparison, shared cases, and unified experiment reports
- `src/kernel/ + include/ + tests/ + benchmarks/ + examples/`
  handle formal project integration

## Development Style

This project tries to balance teaching readability with practical engineering.

My usual workflow is:

1. build a version that is easy to explain
2. optimize it step by step
3. document why each optimization exists
4. promote the stable path into the formal project structure

That is why the repository intentionally keeps:

- baseline and optimized versions
- a development-grade unified experiment entrypoint
- bug-fix context
- traces of how a prototype evolves into an operator

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

Or directly with CMake:

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build
```

Current development-grade experiment entrypoints include:

```bash
python3 python/experiments/layernorm/compare.py --case main_fp32
python3 python/experiments/layernorm/compare.py --case main_fp16 --markdown
```

## Roadmap

- [x] `vector_add` scaffold
- [x] `layernorm_half` half2 operator integration
- [x] LayerNorm correctness test
- [x] LayerNorm benchmark
- [x] LayerNorm example
- [x] LayerNorm development-grade experiment framework
- [ ] float LayerNorm path
- [ ] PyTorch binding
- [ ] RMSNorm
- [ ] Softmax
- [ ] More public dev notes and teaching-grade kernels

## License

Apache-2.0

</details>
