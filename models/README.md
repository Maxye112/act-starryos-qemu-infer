# Models

This repository includes the selected deployment model:

```text
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

It is the 50 MB balanced-calibration static QDQ Conv/MatMul quantized ACT model
with the action head kept in FP16.

The FP32 model and intermediate quantization variants are not included here to
keep the repository compact. Their evaluation results are included under:

```text
artifacts/onnx_quant/closed_loop_quant_eval.md
```
