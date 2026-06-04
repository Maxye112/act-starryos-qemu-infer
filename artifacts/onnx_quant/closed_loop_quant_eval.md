# Closed-loop Quantization Evaluation

- State mode: feedback predicted state within each episode; reset to GT state at episode boundary.
- GT straight frames are ignored for turn accuracy.
- `coverage@eps`: among GT turn frames, fraction where prediction is also non-straight.
- `sign_no_straight@eps`: direction accuracy after ignoring prediction=straight on GT turn frames.

## Main Metrics

| model | diff MAE | diff RMSE | diff<=0.01 | turn@0.005 | coverage@0.005 | sign no straight@0.005 | turn@1e-6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.022137 | 0.060239 | 0.8934 | 0.3214 | 0.3929 | 0.8182 | 0.7143 |
| fp32_action_head_fp16 | 0.022156 | 0.060246 | 0.8934 | 0.3214 | 0.3929 | 0.8182 | 0.7143 |
| dyn_conv_matmul_head_fp16 | 0.021231 | 0.058142 | 0.8604 | 0.2500 | 0.5357 | 0.4667 | 0.6786 |
| static_conv_matmul_head_fp16 | 0.019926 | 0.057714 | 0.9084 | 0.2143 | 0.2321 | 0.9231 | 0.6786 |
| balanced_static_conv_matmul_head_fp16 | 0.019771 | 0.057689 | 0.8979 | 0.2143 | 0.2321 | 0.9231 | 0.6607 |
| repr_static_conv_matmul_head_fp16 | 0.020248 | 0.058155 | 0.8829 | 0.1964 | 0.3750 | 0.5238 | 0.5357 |
| balanced_static_attn_ffn_head_fp16 | 0.019881 | 0.057607 | 0.9114 | 0.1429 | 0.2143 | 0.6667 | 0.6964 |
| dyn_attn_ffn_head_fp16 | 0.021252 | 0.057947 | 0.8408 | 0.3393 | 0.5893 | 0.5758 | 0.6786 |
| pre_static_conv_matmul_head_fp16 | 0.019926 | 0.057714 | 0.9084 | 0.2143 | 0.2321 | 0.9231 | 0.6786 |
| pre_dyn_attn_ffn_head_fp16 | 0.021252 | 0.057947 | 0.8408 | 0.3393 | 0.5893 | 0.5758 | 0.6786 |

## Turn Threshold eps=0.005000

| model | turn acc | coverage | sign no straight | pred straight | opposite |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.3214 | 0.3929 | 0.8182 | 34 | 4 |
| fp32_action_head_fp16 | 0.3214 | 0.3929 | 0.8182 | 34 | 4 |
| dyn_conv_matmul_head_fp16 | 0.2500 | 0.5357 | 0.4667 | 26 | 16 |
| static_conv_matmul_head_fp16 | 0.2143 | 0.2321 | 0.9231 | 43 | 1 |
| balanced_static_conv_matmul_head_fp16 | 0.2143 | 0.2321 | 0.9231 | 43 | 1 |
| repr_static_conv_matmul_head_fp16 | 0.1964 | 0.3750 | 0.5238 | 35 | 10 |
| balanced_static_attn_ffn_head_fp16 | 0.1429 | 0.2143 | 0.6667 | 44 | 4 |
| dyn_attn_ffn_head_fp16 | 0.3393 | 0.5893 | 0.5758 | 23 | 14 |
| pre_static_conv_matmul_head_fp16 | 0.2143 | 0.2321 | 0.9231 | 43 | 1 |
| pre_dyn_attn_ffn_head_fp16 | 0.3393 | 0.5893 | 0.5758 | 23 | 14 |

## Turn Threshold eps=0.010000

| model | turn acc | coverage | sign no straight | pred straight | opposite |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.1250 | 0.1250 | 1.0000 | 49 | 0 |
| fp32_action_head_fp16 | 0.1250 | 0.1250 | 1.0000 | 49 | 0 |
| dyn_conv_matmul_head_fp16 | 0.0893 | 0.2500 | 0.3571 | 42 | 9 |
| static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| balanced_static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| repr_static_conv_matmul_head_fp16 | 0.0000 | 0.0536 | 0.0000 | 53 | 3 |
| balanced_static_attn_ffn_head_fp16 | 0.0179 | 0.0179 | 1.0000 | 55 | 0 |
| dyn_attn_ffn_head_fp16 | 0.0893 | 0.1964 | 0.4545 | 45 | 6 |
| pre_static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| pre_dyn_attn_ffn_head_fp16 | 0.0893 | 0.1964 | 0.4545 | 45 | 6 |

## Turn Threshold eps=0.020000

| model | turn acc | coverage | sign no straight | pred straight | opposite |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| fp32_action_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| dyn_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| balanced_static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| repr_static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| balanced_static_attn_ffn_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| dyn_attn_ffn_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| pre_static_conv_matmul_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |
| pre_dyn_attn_ffn_head_fp16 | 0.0000 | 0.0000 | 0.0000 | 56 | 0 |

## Turn Threshold eps=0.000001

| model | turn acc | coverage | sign no straight | pred straight | opposite |
|---|---:|---:|---:|---:|---:|
| fp32 | 0.7143 | 1.0000 | 0.7143 | 0 | 16 |
| fp32_action_head_fp16 | 0.7143 | 1.0000 | 0.7143 | 0 | 16 |
| dyn_conv_matmul_head_fp16 | 0.6786 | 1.0000 | 0.6786 | 0 | 18 |
| static_conv_matmul_head_fp16 | 0.6786 | 1.0000 | 0.6786 | 0 | 18 |
| balanced_static_conv_matmul_head_fp16 | 0.6607 | 1.0000 | 0.6607 | 0 | 19 |
| repr_static_conv_matmul_head_fp16 | 0.5357 | 1.0000 | 0.5357 | 0 | 26 |
| balanced_static_attn_ffn_head_fp16 | 0.6964 | 1.0000 | 0.6964 | 0 | 17 |
| dyn_attn_ffn_head_fp16 | 0.6786 | 1.0000 | 0.6786 | 0 | 18 |
| pre_static_conv_matmul_head_fp16 | 0.6786 | 1.0000 | 0.6786 | 0 | 18 |
| pre_dyn_attn_ffn_head_fp16 | 0.6786 | 1.0000 | 0.6786 | 0 | 18 |
