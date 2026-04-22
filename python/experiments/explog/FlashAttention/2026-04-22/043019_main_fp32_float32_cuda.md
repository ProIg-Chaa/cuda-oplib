# FlashAttention Experiment Log

- Date: `2026-04-22T04:30:19`
- Case: `main_fp32`
- Sq: `64`
- Sk: `128`
- Hidden: `64`
- Dtype: `float32`
- Device: `cuda`
- Warmup: `20`
- Iters: `100`
- Seed: `0`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.024 | 1.118 | torch scaled_dot_product_attention reference |
| torch_python | baseline | float32 | yes | 4.172e-07 | 0.071 | 0.383 | manual attention reference |
| online_warp | candidate | float32 | no | nan | -- | -- | fsattention online warp CUDA kernel |
