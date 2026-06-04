# StarryOS / QEMU 上的 ACT CPU 推理交付说明

本目录包含用于 **QEMU 中 StarryOS** 用户态 **CPU 推理** 的 ACT 流水线实现与验证记录。

## 交付物清单


| 交付物          | 路径                                                                                     | 大小     |
| ------------ | -------------------------------------------------------------------------------------- | ------ |
| RISC-V 可执行文件 | `deploy/cpp_onnxruntime/build-riscv64/act_ort_infer`                                   | 172 KB |
| 选定量化 ONNX    | `artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx` | 50 MB  |
| FP32 参考 ONNX | `artifacts/onnx_quant/act_finetuned_fp32.onnx`                                         | 194 MB |
| 部署参数         | `deploy/cpp_onnxruntime/config/act_params.json`                                        | —      |
| 数据集评测清单      | `deploy/cpp_onnxruntime/data/eval_manifest.csv`                                        | 666 帧  |


**选定部署模型：**

```text
balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

动作头保持 **FP16**；对 Conv / MatMul / Gemm 使用 representative balanced 校准做静态量化。模型约 **50 MB**，相对 FP32（约 **194 MB**）显著减小。

## 推理流水线

可执行文件 `act_ort_infer` 完成端到端路径：

1. 使用 `stb_image` 解码 RGB 图像。
2. 将原始 **320×240** 图像缩放为 **224×224**。
3. 转为 NCHW `float32` 张量。
4. 按 ImageNet mean/std 归一化图像。
5. 按分位数归一化状态：

```text
state_norm = 2 * (state - q01) / (q99 - q01) - 1
```

1. ONNX Runtime CPU 推理，输入为：

```text
image  [1, 3, 224, 224]
state  [1, 2]
latent [1, 32]
```

1. 动作反归一化：

```text
action = (action_norm + 1) / 2 * (q99 - q01) + q01
```

1. 取 action chunk 的**第一步**作为执行输出：

```text
[left_vel, right_vel, gripper_target]
```

**转向判定规则**（`eps` 为左右轮速差阈值）：

```text
left_vel - right_vel >  eps  → 右转 (right)
left_vel - right_vel < -eps  → 左转 (left)
否则                         → 直行 (straight)
```

## 构建

### RISC-V musl

在仓库根目录执行（请将路径替换为本机实际位置）：

```bash
cd /path/to/act-starryos-qemu-infer

cmake -S deploy/cpp_onnxruntime \
  -B deploy/cpp_onnxruntime/build-riscv64 \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-riscv64-musl \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/onnxruntime-riscv64-musl.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release

cmake --build deploy/cpp_onnxruntime/build-riscv64 -j$(nproc)
```

RISC-V 版本已成功编译验证。

## 部署到 StarryOS rootfs

```bash
cd /path/to/act-starryos-qemu-infer

deploy/cpp_onnxruntime/deploy_to_starryos_rootfs.sh \
  /path/to/StarryOS/make/disk.img \
  /root/proj57-act
```

部署脚本会拷贝：

- 可执行文件 `act_ort_infer`
- ONNX Runtime 动态库
- 选定的量化 ONNX 模型
- `act_params.json`
- 单帧测试图片
- 闭环评测所需的完整帧图像与 `eval_manifest.csv`

## 使用最新 StarryOS 内核启动 QEMU

```bash
qemu-system-riscv64 \
  -m 1G \
  -smp 1 \
  -machine virt \
  -bios default \
  -kernel /path/to/StarryOS/workspace_riscv64-qemu-virt.bin \
  -device virtio-blk-pci,drive=disk0 \
  -drive id=disk0,if=none,format=raw,file=/path/to/StarryOS/make/disk.img \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::5555-:5555,hostfwd=udp::5555-:5555 \
  -nographic \
  -monitor none
```

进入 StarryOS 后：

```sh
cd /root/proj57-act
export LD_LIBRARY_PATH=/root/proj57-act/lib
```

也可使用 StarryOS 官方 `make ARCH=riscv64 run`（若内核与 `disk.img` 路径与上述一致）。

## 单帧推理

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

**StarryOS / QEMU 实测单帧输出：**

```text
first_step: left_vel=0.000542187 right_vel=3.93152e-05
diff=0.000502872 decision=straight
```

更完整的单帧对照（含 GT、左转样本）见仓库根目录 `results/frame_000007_starryos_single_infer.md`。

## 闭环数据集评测

评测采用**预测状态反馈**：

- 每个 episode 的**首帧**使用 GT 初始状态；
- 后续帧使用上一帧预测的 `[left_vel, right_vel]` 作为当前 `state`；
- 计算左/右转准确率时，**忽略 GT 为直行的帧**。

命令：

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

**选定量化模型 — 闭环验证结果：**

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

在 `eps=0.005` 下，该模型偏**保守**：许多 GT 为转向的帧被预测为直行。若**忽略预测为直行**的样本，仅检查左/右符号，方向一致性很高：

```text
ignore_pred_straight_accuracy: 0.923077
turn_pred_straight: 43
turn_pred_opposite: 1
```

对模型输出**非直行**转向的帧，左右关系**未出现系统性反转**，满足赛题对「决策方向与参考一致」的要求。

## 量化方案对比

完整报告：

```text
artifacts/onnx_quant/closed_loop_quant_eval.md
artifacts/onnx_quant/closed_loop_quant_eval.json
```

`eps=0.005` 时代表性结果：


| 模型                                    | diff MAE     | diff≤0.01  | 转向准确率  | 覆盖率    | 忽略预测直行后的符号准确率 |
| ------------------------------------- | ------------ | ---------- | ------ | ------ | ------------- |
| FP32                                  | 0.022137     | 0.8934     | 0.3214 | 0.3929 | 0.8182        |
| FP32 动作头 FP16                         | 0.022156     | 0.8934     | 0.3214 | 0.3929 | 0.8182        |
| **选定：静态 Conv/MatMul INT8 + 动作头 FP16** | **0.019771** | **0.8979** | 0.2143 | 0.2321 | **0.9231**    |
| 动态 attention/FFN INT8 + 动作头 FP16      | 0.021252     | 0.8408     | 0.3393 | 0.5893 | 0.5758        |


**选定该模型的理由：**

- 体积约 50 MB，远小于 194 MB FP32。
- 轮速差（wheel-speed diff）误差表现良好。
- 在输出转向时，左/右符号准确率高。
- 反向预测（opposite）次数极少（本表为 1 次）。

## StarryOS / QEMU 资源与耗时验证

当前使用的 StarryOS 内核在 `/proc` 中提供 `VmRSS`、`VmHWM`、`VmSize` 等字段（见 `starryos_patches/`）。

**单进程在 StarryOS / QEMU 下的分阶段耗时与内存：**


| 阶段          | 耗时      | VmRSS   |
| ----------- | ------- | ------- |
| 加载参数        | 10.5 ms | 2.7 MB  |
| 创建 ORT 环境   | 784 ms  | 10.3 MB |
| 创建 Session  | 8015 ms | 75.8 MB |
| 图像解码/缩放/归一化 | 32.2 ms | 76.4 MB |
| 推理平均        | 5660 ms | 76.7 MB |


**进程峰值内存：**

```text
VmHWM ≈ 78,600 KB ≈ 76.8 MB
VmSize ≈ 90.3 MB
```

## 稳定性

- 数据集评测全程**复用同一个** ONNX Runtime Session。
- 闭环 666 帧运行**无进程错误**。
- StarryOS / QEMU 上单帧与 benchmark 运行中，通过 `/proc/self/status`、`/proc/self/statm` 与分配器跟踪，内存报告稳定。

## 相关文档

- 仓库总览：`[../../README.md](../../README.md)`

