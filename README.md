# ACT 在 StarryOS / QEMU 上的 CPU 推理

本仓库提供可在 **StarryOS（QEMU 模拟 RISC-V）** 用户态复现的 **ACT 模型 CPU 推理** 流水线，对应赛题 **Proj57 任务三**（QEMU + CPU + C/C++）。

## 推理流水线

```text
RGB 图像 → 缩放/归一化 → ONNX Runtime CPU 推理 → 动作反归一化 → 左/右/直行决策
```

完整技术说明与 StarryOS 操作命令见 `[deploy/cpp_onnxruntime/DELIVERABLE.md](deploy/cpp_onnxruntime/DELIVERABLE.md)`。

## 目录结构


| 路径                                                                       | 用途                               |
| ------------------------------------------------------------------------ | -------------------------------- |
| `deploy/cpp_onnxruntime/src/act_ort_infer.cpp`                           | C++ 用户态推理与评测程序                   |
| `deploy/cpp_onnxruntime/CMakeLists.txt`                                  | riscv64 构建配置                     |
| `deploy/cpp_onnxruntime/deploy_to_starryos_rootfs.sh`                    | 将可执行文件、模型、库、数据写入 StarryOS rootfs |
| `deploy/cpp_onnxruntime/run_starryos_benchmark.sh`                       | 在目标系统上运行基准测试                     |
| `deploy/cpp_onnxruntime/config/act_params.json`                          | 分位数归一化与 latent 参数                |
| `deploy/cpp_onnxruntime/data/eval_manifest.csv`                          | 666 帧闭环评测清单                      |
| `tools/make_cpp_eval_manifest.py`                                        | 从 LeRobot parquet 生成评测清单         |
| `models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx` | 选定的部署用 ONNX 模型                   |
| `bin/riscv64/act_ort_infer`                                              | 预编译 RISC-V 可执行文件                 |
| `bin/x64/act_ort_infer`                                                  | 预编译 x64 可执行文件                    |
| `artifacts/onnx_quant/closed_loop_quant_eval.md`                         | 量化方案对比报告                         |
| `starryos_patches/`                                                      | StarryOS `/proc` 内存统计相关补丁说明      |
| `results/`                                                               | StarryOS/QEMU 单帧推理结果样例           |


## 选定模型与体积

**部署模型：**

```text
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

对 Conv/MatMul/Gemm 做静态 QDQ 量化（balanced 校准），**动作头保持 FP16**，体积约 **50 MB**（FP32 参考约 **194 MB**）。


| 产物           | 大小（约）  |
| ------------ | ------ |
| RISC-V 可执行文件 | 172 KB |
| x64 可执行文件    | 252 KB |
| 选定 ONNX 模型   | 50 MB  |
| FP32 参考模型    | 194 MB |


## 已验证结果摘要

### StarryOS / QEMU 单帧观测

```text
进程峰值内存 VmHWM：约 76.8 MB（通过对StarryOS内核打补丁查看）
推理平均耗时：      约 5660 ms（QEMU TCG 模拟）
```

单帧输出示例（`frame_000000.jpg`，初态 `[0, 0]`）：

```text
first_step: left_vel=0.000542187 right_vel=3.93152e-05
diff=0.000502872 decision=straight
```

### RISC-V 闭环评测（选定量化模型）

在开发机上对 **666 帧** 做闭环评测（下一帧状态使用上一帧预测轮速）：

```text
样本数：     666
diff MAE：   0.019771
|误差|≤0.01： 89.79%
转向符号准确率（忽略预测为直行，eps=0.005）： 92.31%
反向预测次数（eps=0.005）：                   1
```

转向判定规则（`eps` 为左右轮速差阈值）：

```text
left_vel - right_vel >  eps  → 右转
left_vel - right_vel < -eps  → 左转
否则                         → 直行
```

更完整的量化对比、内存分阶段统计见 `[DELIVERABLE.md](deploy/cpp_onnxruntime/DELIVERABLE.md)`。

## 快速构建

请将 `ONNXRUNTIME_ROOT`、工具链路径改为你本机实际位置（示例来自 Linux/WSL 环境）。

```bash
cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/onnxruntime-riscv64-musl.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

也可直接使用仓库内预编译二进制：`bin/riscv64/act_ort_infer`、`bin/x64/act_ort_infer`。

## 部署到 StarryOS / QEMU

1. 构建或选用 `riscv64` 版 `act_ort_infer`
2. 执行 `deploy/cpp_onnxruntime/deploy_to_starryos_rootfs.sh`，将产物写入 StarryOS 的 `disk.img`
3. 使用 StarryOS 内核与 QEMU 启动（命令见 `[DELIVERABLE.md](deploy/cpp_onnxruntime/DELIVERABLE.md)`）
4. 在 StarryOS shell 中：

```sh
cd /root/proj57-act
export LD_LIBRARY_PATH=/root/proj57-act/lib

bin/act_ort_infer \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --image data/frame_000000.jpg \
  --params config/act_params.json \
  --state 0 0 \
  --threads 1 \
  --warmup 1 \
  --runs 3
```

## 进度


| 赛题任务                          | 本仓库                    |
| ----------------------------- | ---------------------- |
| **任务三**：QEMU + StarryOS + CPU | ✅ 已实现（本文档范围）           |
| **任务二**：RK3588 NPU            | ❌ 需 RKNN 等厂商工具链        |
| **任务一**：SG2002 TPU            | ❌ 需 Sophon SDK 与极强内存优化 |


## 相关资源

- 赛题仓库：[chenlongos/proj57](https://github.com/chenlongos/proj57)
- StarryOS：[Starry-OS/StarryOS](https://github.com/Starry-OS/StarryOS)
- 模型说明：`[models/README.md](models/README.md)`
- 单帧 QEMU 结果样例：`[results/frame_000007_starryos_single_infer.md](results/frame_000007_starryos_single_infer.md)`

