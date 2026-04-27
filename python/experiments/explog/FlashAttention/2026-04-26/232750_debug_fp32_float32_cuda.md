# FlashAttention Experiment Log

- Date: `2026-04-26T23:27:50`
- Case: `debug_fp32`
- Sq: `8`
- Sk: `16`
- Hidden: `32`
- Dtype: `float32`
- Device: `cuda`
- Warmup: `3`
- Iters: `10`
- Seed: `0`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.092 | 0.719 | torch scaled_dot_product_attention reference |
| torch_python | baseline | float32 | yes | 7.153e-07 | 0.200 | 0.332 | manual attention reference |
| online_warp | candidate | float32 | yes | 5.364e-07 | 0.109 | 0.607 | fsattention online warp CUDA kernel |
