# FlashAttention Experiment Log

- Date: `2026-04-22T05:57:44`
- Case: `large_fp32`
- Sq: `128`
- Sk: `256`
- Hidden: `128`
- Dtype: `float32`
- Device: `cuda`
- Warmup: `20`
- Iters: `100`
- Seed: `0`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.604 | 0.999 | torch scaled_dot_product_attention reference |
| torch_python | baseline | float32 | yes | 6.557e-07 | 0.136 | 4.423 | manual attention reference |
| online_warp | candidate | float32 | yes | 9.239e-07 | 0.771 | 0.782 | fsattention online warp CUDA kernel |
