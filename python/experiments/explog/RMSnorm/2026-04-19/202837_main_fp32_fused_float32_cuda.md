# RMSNorm Experiment Log

- Date: `2026-04-19T20:28:37`
- Case: `main_fp32_fused`
- Rows: `512`
- Hidden: `768`
- Dtype: `float32`
- Device: `cuda`
- Eps: `1e-05`
- Warmup: `20`
- Iters: `100`
- Seed: `0`
- Residual Path: `enabled`

## Results

| name | stage | dtype | correct | max_abs_diff | avg_ms | speedup_vs_ref | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_official | baseline | float32 | yes | 0.000e+00 | 0.045 | 1.042 | torch.nn.RMSNorm reference if available |
| torch_python | baseline | float32 | yes | 1.907e-06 | 0.100 | 0.468 | manual PyTorch tensor reference |
| f32_warp | candidate | float32 | yes | 1.431e-06 | 0.019 | 2.449 | f32 warp-reduction CUDA kernel |
| half2_warp | candidate | float32 | skip | -- | -- | -- | unsupported for float32 |
| fused_add_f32 | candidate | float32 | yes | 1.431e-06 | 0.017 | 2.775 | fused add + f32 warp-reduction CUDA kernel |
| fused_add_half2 | candidate | float32 | skip | -- | -- | -- | unsupported for float32 |
