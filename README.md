# ACT 在 StarryOS / QEMU 上的 CPU 推理

对应赛题 **Proj57 任务三**（QEMU + StarryOS + CPU + C/C++）：在模拟 RISC-V 的 StarryOS 用户态完成 ACT 模型 **CPU 推理** 与验证。

## 工作内容概览

```text
RGB 图像 → 缩放/归一化 → ONNX Runtime CPU 推理 → 动作反归一化 → 左/右/直行决策
```

采用 C++ 编写推理程序，使用 ONNX Runtime；部署模型为静态量化后的 ACT（动作头 FP16），体积约 **50 MB**（相对 FP32 约 194 MB 明显减小）。

| 产物 | 大小（约） |
| --- | --- |
| RISC-V 推理程序 | 172 KB |
| 选定量化模型 | 50 MB |

## 已验证结果摘要

### StarryOS / QEMU 单帧

```text
进程峰值内存 VmHWM：约 76.8 MB（内核补丁后可读 /proc）
推理平均耗时：      约 5660 ms（QEMU TCG，单核）
```

初态 `[0, 0]` 单帧示例：

```text
first_step: left_vel=0.000542187 right_vel=3.93152e-05
diff=0.000502872 decision=straight（直行）
```

### 闭环数据集评测（666 帧）

下一帧状态使用上一帧预测轮速；转向判定阈值 eps=0.005：

```text
样本数：     666
diff MAE：   0.019771
|误差|≤0.01： 89.79%
转向符号准确率（忽略预测为直行）： 92.31%
反向预测次数：                   1
```

```text
left_vel - right_vel >  eps  → 右转
left_vel - right_vel < -eps  → 左转
否则                         → 直行
```

更细的流水线、量化对比与分阶段耗时见交付说明章节。

## 赛题进度

| 赛题任务 | 状态 |
| --- | --- |
| **任务三**：QEMU + StarryOS + CPU | ✅ 已完成本报告范围工作 |
| **任务二**：RK3588 NPU | ❌ 未做 |
| **任务一**：SG2002 TPU | ❌ 未做 |

## 参考链接

- 赛题：[chenlongos/proj57](https://github.com/chenlongos/proj57)
- StarryOS：[Starry-OS/StarryOS](https://github.com/Starry-OS/StarryOS)
