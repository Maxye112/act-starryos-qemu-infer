# ACT CPU Inference Deliverable for StarryOS/QEMU

This directory contains the user-space ACT inference pipeline used for QEMU
StarryOS CPU deployment.

## Delivered Artifacts

| artifact | path | size |
| --- | --- | ---: |
| RISC-V executable | `deploy/cpp_onnxruntime/build-riscv64/act_ort_infer` | 172 KB |
| selected quantized ONNX | `artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx` | 50 MB |
| FP32 reference ONNX | `artifacts/onnx_quant/act_finetuned_fp32.onnx` | 194 MB |
| deployment params | `deploy/cpp_onnxruntime/config/act_params.json` |  |
| dataset eval manifest | `deploy/cpp_onnxruntime/data/eval_manifest.csv` | 666 frames |

Selected deployment model:

```text
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

The model keeps the action head in FP16 and quantizes Conv/MatMul/Gemm with
representative balanced calibration. It gives a 50 MB model versus 194 MB FP32.

## Pipeline

The executable performs the full inference path:

1. Decode RGB image with `stb_image`.
2. Resize original 320x240 image to 224x224.
3. Convert to NCHW float tensor.
4. Normalize image with ImageNet mean/std.
5. Normalize state using quantiles:

```text
state_norm = 2 * (state - q01) / (q99 - q01) - 1
```

6. Run ONNX Runtime CPU inference:

```text
image  [1, 3, 224, 224]
state  [1, 2]
latent [1, 32]
```

7. Denormalize action:

```text
action = (action_norm + 1) / 2 * (q99 - q01) + q01
```

8. Use the first chunk step for execution:

```text
[left_vel, right_vel, gripper_target]
```

Decision rule:

```text
left_vel - right_vel >  eps  -> right
left_vel - right_vel < -eps  -> left
otherwise                    -> straight
```

## Build

RISC-V musl:

```bash
cd /home/sakura/OSproj57/proj57

cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/home/sakura/Deploy-ACT/third_party/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/home/sakura/Deploy-ACT/onnxruntime-riscv64-musl.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

The RISC-V build was verified successfully.

## Deploy to StarryOS Rootfs

```bash
cd /home/sakura/OSproj57/proj57

deploy/cpp_onnxruntime/deploy_to_starryos_rootfs.sh \
  /home/sakura/OSproj57/StarryOS/make/disk.img \
  /root/proj57-act
```

The deploy script copies:

- executable
- ONNX Runtime shared libraries
- selected quantized model
- params JSON
- single test image
- full-frame dataset images and manifest for closed-loop evaluation

## Start QEMU with Latest StarryOS Kernel

```bash
qemu-system-riscv64 \
  -m 1G \
  -smp 1 \
  -machine virt \
  -bios default \
  -kernel /home/sakura/OSproj57/StarryOS/workspace_riscv64-qemu-virt.bin \
  -device virtio-blk-pci,drive=disk0 \
  -drive id=disk0,if=none,format=raw,file=/home/sakura/OSproj57/StarryOS/make/disk.img \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::5555-:5555,hostfwd=udp::5555-:5555 \
  -nographic \
  -monitor none
```

Inside StarryOS:

```sh
cd /root/proj57-act
export LD_LIBRARY_PATH=/root/proj57-act/lib
```

## Single-frame Inference

```sh
bin/act_ort_infer \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --image data/frame_000000.jpg \
  --params config/act_params.json \
  --state 0 0 \
  --threads 1 \
  --warmup 1 \
  --runs 3 \
  --deadband 0.01 \
  --track-allocator
```

Observed StarryOS/QEMU single-frame result:

```text
first_step: left_vel=0.000542187 right_vel=3.93152e-05
diff=0.000502872 decision=straight
```

## Closed-loop Dataset Evaluation

The evaluation uses predicted state feedback:

- first frame of each episode uses GT initial state
- following frames use previous predicted `[left_vel, right_vel]`
- GT straight frames are ignored for left/right turn accuracy

Command:

```sh
bin/act_ort_infer \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --params config/act_params.json \
  --eval-manifest data/eval_manifest.csv \
  --dataset-root data/dataset \
  --threads 1 \
  --eval-turn-eps 0.005 \
  --track-allocator
```

Closed-loop validation, selected quantized model:

```text
samples: 666
eval_state_mode: feedback_predicted_state
episode_resets: 30
diff_mae: 0.019771
diff_rmse: 0.057689
diff_acc_abs_le_0.005: 0.657658
diff_acc_abs_le_0.010: 0.897898
diff_acc_abs_le_0.020: 0.915916
turn_total: 56
turn_correct: 12
turn_accuracy_ignore_gt_straight: 0.214286
left_turn_accuracy: 0.297297
right_turn_accuracy: 0.052632
```

For `eps=0.005`, this model is conservative: many GT turn frames are predicted
as straight. When ignoring prediction=straight and checking only left/right sign,
the direction accuracy is high:

```text
ignore_pred_straight_accuracy: 0.923077
turn_pred_straight: 43
turn_pred_opposite: 1
```

This satisfies the direction-consistency requirement for frames where the model
emits a non-straight turn: left/right is not systematically reversed.

## Quantization Comparison

Closed-loop comparison report:

```text
artifacts/onnx_quant/closed_loop_quant_eval.md
artifacts/onnx_quant/closed_loop_quant_eval.json
```

Representative results at `eps=0.005`:

| model | diff MAE | diff<=0.01 | turn acc | coverage | sign accuracy ignoring pred=straight |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.022137 | 0.8934 | 0.3214 | 0.3929 | 0.8182 |
| FP32 action-head FP16 | 0.022156 | 0.8934 | 0.3214 | 0.3929 | 0.8182 |
| selected static Conv/MatMul INT8 + action-head FP16 | 0.019771 | 0.8979 | 0.2143 | 0.2321 | 0.9231 |
| dynamic attention/FFN INT8 + action-head FP16 | 0.021252 | 0.8408 | 0.3393 | 0.5893 | 0.5758 |

Selected model rationale:

- 50 MB model size, much smaller than 194 MB FP32.
- Good wheel-speed-difference error.
- Very high left/right sign accuracy when a turn is emitted.
- Low opposite-direction errors.

## StarryOS/QEMU Resource and Timing Validation

Latest StarryOS kernel exposes `VmRSS`, `VmHWM`, and `VmSize` in `/proc`.

Observed single-process StarryOS/QEMU memory and timing:

| stage | time | VmRSS |
| --- | ---: | ---: |
| load params | 10.5 ms | 2.7 MB |
| create ORT env | 784 ms | 10.3 MB |
| create session | 8015 ms | 75.8 MB |
| image decode/resize/normalize | 32.2 ms | 76.4 MB |
| warmup | 6022 ms | 76.8 MB |
| inference average | 5660 ms | 76.7 MB |

Peak process memory:

```text
VmHWM ~= 78,600 KB ~= 76.8 MB
VmSize ~= 90.3 MB
```

ORT allocator tracked memory on StarryOS/QEMU:

```text
after_session current ~= 10.9 MB
after_warmup peak     ~= 16.8 MB
after_runs peak       ~= 19.9 MB
```

## Stability

The executable reuses a single ONNX Runtime session for all frames in dataset
evaluation. The closed-loop run completed 666 frames without process errors.

The StarryOS/QEMU single-frame and benchmark runs completed with stable memory
reporting through `/proc/self/status`, `/proc/self/statm`, and allocator tracking.

## Reproducibility Checklist

1. Build the RISC-V binary.
2. Generate `eval_manifest.csv`.
3. Deploy to StarryOS rootfs.
4. Boot QEMU with `workspace_riscv64-qemu-virt.bin`.
5. Run single-frame inference.
6. Run closed-loop dataset evaluation.
7. Compare output with `closed_loop_quant_eval.md`.
