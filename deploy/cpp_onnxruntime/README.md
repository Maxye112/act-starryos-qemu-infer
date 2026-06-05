# ACT ONNX Runtime C++ CPU 推理

StarryOS/QEMU CPU 交付物与已验证结果见 `DELIVERABLE.md`。

本目录包含用于 ACT ONNX 模型的轻量用户态 C++ 推理程序，依赖尽量少，便于 StarryOS 类环境部署：

- ONNX Runtime C/C++ API
- `stb_image.h` 解码 JPEG/PNG
- 手写双线性缩放、归一化、状态/动作分位数缩放

## 输入

程序向导出的 ACT ONNX 图提供：

| 名称 | 形状 | 来源 |
| --- | --- | --- |
| `image` | `[1, 3, 224, 224]` | 解码后的 RGB 图像，缩放并归一化 |
| `state` | `[1, 2]` | 原始 `[left_vel, right_vel]`，经分位数归一化 |
| `latent` | `[1, 32]` | 来自 `final_model.pt` 的固定 CVAE latent 均值 |

参数文件：

```text
deploy/cpp_onnxruntime/config/act_params.json
```

## 在 StarryOS/QEMU 上运行

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

加 `--print-chunk` 可打印全部 8 步预测动作。

## 数据集评测

从 LeRobot parquet 生成轻量 CSV 清单：

```bash
python tools/make_cpp_eval_manifest.py \
  --data-dir output/dataset \
  --output deploy/cpp_onnxruntime/data/eval_manifest.csv
```

逐帧运行并与 GT 比较第一步预测：

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

报告包含轮速差 MAE/RMSE 与左/右转准确率。默认采用闭环状态反馈：每个 episode 首帧使用清单中的 GT 状态，后续帧使用上一帧预测的 `[left_vel, right_vel]` 作为下一帧 `state` 输入。仅调试旧版「每帧 GT 状态」评测时使用 `--eval-open-loop-state`。计算转向准确率时忽略 GT 为直行的帧。

## StarryOS / RISC-V 构建要点

使用为 StarryOS 用户态 ABI 构建的 ONNX Runtime 包（本工作区可用 riscv64-musl 包）：

```bash
cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/starryos-riscv64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

目标机上需打包：

```text
act_ort_infer
libonnxruntime.so*
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
act_params.json
输入图像
```

若 ONNX Runtime 不在系统库路径，设置：

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

## CPU 性能相关参数

程序暴露的 ONNX Runtime CPU 选项：

- `--threads N`：算子内线程数为 `N`，算子间为 `max(1, N/2)`。
- `--spin`：启用 ORT 线程自旋，可能降低延迟但占用更多 CPU。
- `--no-arena`：关闭 ORT CPU 内存 arena，通常更慢，便于排查内存压力。
- `--no-mem-pattern`：关闭内存模式优化；固定形状推理一般保持默认开启。
- `--warmup N --runs N`：预热与多次计时取平均，便于稳定测延迟。

固定形状 ACT 推理建议保持默认 memory arena 与 memory pattern。`--threads` 可先设为可用 CPU 核数，再逐步降低以避免过度订阅。

## 输出

模型返回 `action [1, 8, 3]`，反归一化公式：

```text
(action + 1) / 2 * (q99 - q01) + q01
```

打印第一步：

```text
first_step: left_vel=... right_vel=... gripper_target=... diff=... decision=...
```

`decision` 判定规则：

```text
|left_vel - right_vel| < deadband → 直行 (straight)
left_vel - right_vel > deadband     → 右转 (right)
left_vel - right_vel < -deadband    → 左转 (left)
```
