# ACT ONNX Runtime C++ CPU Inference

For the StarryOS/QEMU CPU deliverable and verified results, see
`DELIVERABLE.md`.

This directory contains a small user-space C++ inference program for ACT ONNX models.
It is written to keep dependencies light for StarryOS-style deployment:

- ONNX Runtime C/C++ API
- `stb_image.h` for JPEG/PNG decoding
- hand-written bilinear resize, normalization, state/action quantile scaling

## Inputs

The program feeds the exported ACT ONNX graph with:

| name | shape | source |
| --- | --- | --- |
| `image` | `[1, 3, 224, 224]` | decoded RGB image, resized and normalized |
| `state` | `[1, 2]` | raw `[left_vel, right_vel]`, quantile-normalized |
| `latent` | `[1, 32]` | fixed CVAE latent mean from `final_model.pt` |

Parameters are stored in:

```text
deploy/cpp_onnxruntime/config/act_params.json
```

## Run on StarryOS/QEMU

```bash
bin/act_ort_infer \
  --model artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --image output/dataset/videos/observation.images.fpv/chunk-000/frame_000000.jpg \
  --params deploy/cpp_onnxruntime/config/act_params.json \
  --state 0 0 \
  --threads 4 \
  --warmup 3 \
  --runs 20 \
  --deadband 0.01
```

Add `--print-chunk` to print all 8 predicted action steps.

## Dataset Evaluation

Create a lightweight CSV manifest from the LeRobot parquet files:

```bash
/home/sakura/miniforge3/envs/lerobot/bin/python tools/make_cpp_eval_manifest.py \
  --data-dir output/dataset \
  --output deploy/cpp_onnxruntime/data/eval_manifest.csv
```

Run every frame and compare the first predicted action step with GT:

```bash
bin/act_ort_infer \
  --model artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --params deploy/cpp_onnxruntime/config/act_params.json \
  --eval-manifest deploy/cpp_onnxruntime/data/eval_manifest.csv \
  --dataset-root output/dataset \
  --threads 1 \
  --eval-turn-eps 0.005 \
  --track-allocator
```

The report includes wheel-speed-difference MAE/RMSE and left/right turn accuracy.
By default dataset evaluation uses closed-loop state feedback: the first frame of
each episode uses the manifest GT state, and later frames use the previous
prediction's `[left_vel, right_vel]` as the next `state` input. Use
`--eval-open-loop-state` only for debugging the older GT-state-per-frame
evaluation. GT straight frames are ignored for turn accuracy.

## StarryOS / RISC-V build sketch

Use an ONNX Runtime package built for the StarryOS user-space ABI. A riscv64-musl
package is available locally in this workspace:

```bash
cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/home/sakura/Deploy-ACT/third_party/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/starryos-riscv64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

Bundle these files on the target:

```text
act_ort_infer
libonnxruntime.so*
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
act_params.json
input image(s)
```

Set `LD_LIBRARY_PATH` if ONNX Runtime is not in the system library path:

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

## CPU performance knobs

The program exposes ONNX Runtime CPU options:

- `--threads N`: sets intra-op threads to `N`, inter-op to `max(1, N/2)`.
- `--spin`: enables ONNX Runtime thread spinning. This can reduce latency but burns CPU.
- `--no-arena`: disables ORT CPU memory arena. Usually slower, but useful for debugging memory pressure.
- `--no-mem-pattern`: disables memory pattern optimization. Usually keep enabled for fixed-shape inference.
- `--warmup N --runs N`: use warmup and averaged timed runs for stable latency numbers.

For fixed-shape ACT inference, the default memory arena and memory pattern should stay enabled.
Start with `--threads` equal to available CPU cores, then benchmark lower values to avoid oversubscription.

## Output

The model returns `action [1, 8, 3]`. The program denormalizes it using:

```text
(action + 1) / 2 * (q99 - q01) + q01
```

It prints the first step:

```text
first_step: left_vel=... right_vel=... gripper_target=... diff=... decision=...
```

`decision` uses:

```text
abs(left_vel - right_vel) < deadband -> straight
left_vel - right_vel > deadband     -> right
left_vel - right_vel < -deadband    -> left
```
