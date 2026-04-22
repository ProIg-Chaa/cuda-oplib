# FlashAttention Experiment Log

- Date: `2026-04-22T04:54:57`
- Case: `debug_fp32`
- Sq: `16`
- Sk: `32`
- Hidden: `64`
- Dtype: `float32`
- Device: `cuda`
- Warmup: `5`
- Iters: `20`
- Seed: `0`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.094 | 1.986 | torch scaled_dot_product_attention reference |
| torch_python | baseline | float32 | yes | 5.960e-07 | 0.098 | 1.908 | manual attention reference |
| online_warp | candidate | float32 | yes | 7.153e-07 | 0.181 | 1.031 | fsattention online warp CUDA kernel |
