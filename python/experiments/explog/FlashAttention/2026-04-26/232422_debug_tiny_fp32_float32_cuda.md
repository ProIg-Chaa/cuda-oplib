# FlashAttention Experiment Log

- Date: `2026-04-26T23:24:22`
- Case: `debug_tiny_fp32`
- Sq: `1`
- Sk: `8`
- Hidden: `8`
- Dtype: `float32`
- Device: `cuda`
- Warmup: `1`
- Iters: `1`
- Seed: `0`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.191 | 1.011 | torch scaled_dot_product_attention reference |
| torch_python | baseline | float32 | yes | 2.459e-07 | 0.265 | 0.726 | manual attention reference |
| online_warp | candidate | float32 | yes | 2.384e-07 | 0.172 | 1.120 | fsattention online warp CUDA kernel |
