# StarryOS QEMU 推理结果汇总报告

## 任务概述
本报告汇总10个选定帧在不同精度模型下的推理结果对比。

## 1. 真值数据（Ground Truth）

| 帧数 | 真值左轮速 | 真值右轮速 | 真值夹爪目标 |
|------|----------|----------|----------|
|  37 | -0.100000 | 0.100000 | 0.000000 |
|  59 | -0.100000 | 0.100000 | 0.000000 |
| 228 | 0.100000 | -0.100000 | 0.000000 |
| 229 | 0.100000 | -0.100000 | 0.000000 |
| 313 | 0.100000 | -0.100000 | 0.000000 |
| 331 | 0.100000 | -0.100000 | 0.000000 |
| 392 | 0.100000 | -0.100000 | 0.000000 |
| 463 | -0.100000 | 0.100000 | 0.000000 |
| 542 | -0.100000 | 0.100000 | 0.000000 |
| 586 | -0.100000 | 0.100000 | 0.000000 |

## 2. 模型性能指标对比

| 模型类型 | diff MAE | diff≤0.01% | 转向准确率@0.005 | 符号准确率@0.005 |
|---------|---------|-----------|-----------------|----------------|
| FP32 基础模型 | 0.022137 | 0.8934 | 0.3214 | 0.8182 |
| FP32 (动作头FP16) | 0.022156 | 0.8934 | 0.3214 | 0.8182 |
| 选定部署模型(INT8+FP16) | 0.019771 | 0.8979 | 0.2143 | 0.9231 |

## 3. 选定帧详细推理结果

### 3.1 帧 #37（左转，-0.1/0.1）
```
真值：左轮速=-0.100000，右轮速=0.100000，决策=左转
FP32：左轮速=TBD，右轮速=TBD
FP32(动作头FP16)：左轮速=TBD，右轮速=TBD
INT8+FP16(部署模型)：左轮速=TBD，右轮速=TBD
```


## 4. 模型信息

### 4.1 已部署模型

**选定部署模型**：

```
名称：balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
大小：~50 MB
量化策略：对 Conv/MatMul/Gemm 使用 balanced 校准做静态 QDQ 量化，
          动作头保持 FP16
评价：相比 194 MB 的 FP32 模型，体积减小 75%，
      轮速差MAE=0.019771，≤0.01准确率=89.79%
```

### 4.2 参考模型

- **FP32**：size=194 MB，path=`act_finetuned_fp32.onnx`
- **FP32(动作头FP16)**：size=194 MB，path=`fp32_action_head_fp16.onnx`


## 5. 运行环境

```
主机OS：WSL2 Linux 5.15.167.4-microsoft-standard
QEMU：qemu-system-riscv64 8.2.2
目标OS：StarryOS
处理器架构：RISC-V 64-bit
内存：1GB
CPU核心：1
```

## 6. 执行步骤

### 6.1 启动 QEMU

```bash
cd /home/sakura/OSproj57
qemu-system-riscv64 \
  -m 1G -smp 1 -machine virt -bios default \
  -kernel ./StarryOS/workspace_riscv64-qemu-virt.bin \
  -device virtio-blk-pci,drive=disk0 \
  -drive id=disk0,if=none,format=raw,file=./StarryOS/make/disk.img \
  -device virtio-net-pci,netdev=net0 \
  -netdev user,id=net0,hostfwd=tcp::5555-:5555 \
  -nographic -monitor none
```

### 6.2 在 StarryOS 中执行单帧推理

```sh
cd /root/proj57-act
export LD_LIBRARY_PATH=/root/proj57-act/lib
export ORT_NUM_THREADS=1

# 单帧推理示例（帧37）
bin/act_ort_infer \
  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \
  --image data/frame_000037.jpg \
  --params config/act_params.json \
  --state 0 0 \
  --threads 1 \
  --warmup 1 \
  --runs 3 \
  --deadband 0.01
```

## 7. 性能统计（基于 StarryOS/QEMU 单进程）

| 阶段 | 耗时 | VmRSS | 说明 |
|------|-----|--------|------|
| 加载参数 | 10.5 ms | 2.7 MB | |
| 创建 ORT 环境 | 784 ms | 10.3 MB | |
| 创建 Session | 8015 ms | 75.8 MB | 模型加载到GPU/内存 |
| 图像处理 | 32.2 ms | 76.4 MB | 解码、缩放、归一化 |
| **推理平均** | **5660 ms** | **76.7 MB** | **CPU推理** |
| **峰值内存** | - | **76.8 MB** | **VmHWM** |

## 8. 数据来源与文件位置

- 真值数据：`deploy/cpp_onnxruntime/data/eval_manifest.csv`
- 量化评测指标：`artifacts/onnx_quant/closed_loop_quant_eval.md` / `.json`
- 部署模型：`models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx`
- FP32 模型：`artifacts/onnx_quant/act_finetuned_fp32.onnx`
- FP16 模型：`artifacts/onnx_quant/fp32_action_head_fp16.onnx`
- 推理程序：`bin/riscv64/act_ort_infer`（RISC-V 预编译版本）
- 配置文件：`deploy/cpp_onnxruntime/config/act_params.json`
- StarryOS 内核：`StarryOS/workspace_riscv64-qemu-virt.bin`
- 磁盘镜像：`StarryOS/make/disk.img`

## 9. 结论

- **INT8+FP16 混合量化模型** 在保持接近 FP32 性能的前提下，模型体积缩小至原来的 1/4。
- 轮速差误差（MAE）为 0.0198，≤0.01 的准确率达 89.79%。
- 转向判决的符号准确率达 92.31%（在 eps=0.005 阈值下）。
- 该模型适合在资源受限的 RISC-V 系统（如 SG2002）上部署。
