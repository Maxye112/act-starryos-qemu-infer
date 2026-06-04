# ACT StarryOS QEMU CPU Inference

This repository contains a reproducible user-space ACT inference pipeline for
StarryOS running under QEMU on CPU.

The delivered pipeline is:

```text
RGB image -> resize/normalize -> ONNX Runtime CPU inference -> action denormalization -> left/right/straight decision
```

Main deliverable:

```text
deploy/cpp_onnxruntime/DELIVERABLE.md
```

Caveats are separated from the main report:

```text
deploy/cpp_onnxruntime/LIMITATIONS.md
```

## Contents

| path | purpose |
| --- | --- |
| `deploy/cpp_onnxruntime/src/act_ort_infer.cpp` | C++ user-space inference/evaluation program |
| `deploy/cpp_onnxruntime/CMakeLists.txt` | RISC-V build target |
| `deploy/cpp_onnxruntime/deploy_to_starryos_rootfs.sh` | copy executable/model/libs/data into StarryOS rootfs |
| `deploy/cpp_onnxruntime/run_starryos_benchmark.sh` | target-side benchmark entry |
| `deploy/cpp_onnxruntime/config/act_params.json` | quantile normalization and latent parameters |
| `deploy/cpp_onnxruntime/data/eval_manifest.csv` | 666-frame closed-loop evaluation manifest |
| `tools/make_cpp_eval_manifest.py` | manifest generator from LeRobot parquet |
| `models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx` | selected deployment ONNX model |
| `bin/riscv64/act_ort_infer` | prebuilt RISC-V executable |
| `artifacts/onnx_quant/closed_loop_quant_eval.md` | quantization comparison report |
| `starryos_patches/` | proc/stat files used for StarryOS memory reporting reference |

## Verified Summary

Selected model:

```text
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

Size:

```text
RISC-V executable: 172 KB
selected model:    50 MB
FP32 reference:    194 MB
```

StarryOS/QEMU CPU single-frame observation:

```text
VmHWM:             ~76.8 MB
inference average: ~5660 ms
```

Closed-loop validation for the selected model:

```text
samples: 666
diff_mae: 0.019771
diff_acc_abs_le_0.010: 0.897898
sign accuracy ignoring prediction=straight at eps=0.005: 0.923077
turn_pred_opposite at eps=0.005: 1
```

## Quick Build

```bash
cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/home/sakura/Deploy-ACT/third_party/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/home/sakura/Deploy-ACT/onnxruntime-riscv64-musl.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

## Quick Closed-loop Evaluation

```bash
deploy/cpp_onnxruntime/build/act_ort_infer \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --params deploy/cpp_onnxruntime/config/act_params.json \
  --eval-manifest deploy/cpp_onnxruntime/data/eval_manifest.csv \
  --dataset-root output/dataset \
  --threads 1 \
  --eval-turn-eps 0.005
```

For full StarryOS/QEMU commands, see:

```text
deploy/cpp_onnxruntime/DELIVERABLE.md
```
