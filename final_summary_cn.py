#!/usr/bin/env python3
"""
生成完整的推理结果汇总表格（中文）
"""

import json
import csv
from pathlib import Path
from typing import Dict

def generate_comprehensive_summary():
    repo_root = Path("/home/sakura/OSproj57/act-starryos-qemu-infer")
    
    # 加载真值数据
    gt_data = {}
    with open(repo_root / "deploy/cpp_onnxruntime/data/eval_manifest.csv", 'r') as f:
        reader = csv.DictReader(f)
        target_frames = [37, 59, 228, 229, 313, 331, 392, 463, 542, 586]
        for row in reader:
            frame_idx = int(row['index'])
            if frame_idx in target_frames:
                gt_data[frame_idx] = {
                    'left_vel': float(row['gt_left_vel']),
                    'right_vel': float(row['gt_right_vel']),
                }
    
    # 加载量化评测数据
    with open(repo_root / "artifacts/onnx_quant/closed_loop_quant_eval.json", 'r') as f:
        quant_data = json.load(f)
    
    # 提取关键模型的数据
    models_data = {}
    model_names_map = {
        'fp32': 'FP32',
        'fp32_action_head_fp16': 'FP16',
        'balanced_static_conv_matmul_head_fp16': 'INT8+FP16'
    }
    
    for item in quant_data:
        if item['name'] in model_names_map:
            models_data[item['name']] = item
    
    # 生成文本报告
    print("\n" + "=" * 100)
    print("StarryOS QEMU 推理结果汇总 - 10个选定帧的对比分析")
    print("=" * 100)
    
    print("\n【第1部分】真值数据 (Ground Truth)")
    print("-" * 100)
    print(f"{'帧数':<8} {'真值左轮速':<15} {'真值右轮速':<15} {'转向决策':<15}")
    print("-" * 100)
    
    for frame_idx in sorted(target_frames):
        if frame_idx in gt_data:
            gt = gt_data[frame_idx]
            decision = "左转" if gt['left_vel'] < 0 else "右转"
            print(f"{frame_idx:<8} {gt['left_vel']:<15.6f} {gt['right_vel']:<15.6f} {decision:<15}")
    
    print("\n【第2部分】模型性能指标对比")
    print("-" * 100)
    print(f"{'模型精度':<20} {'模型大小(MB)':<15} {'轮速差MAE':<15} {'≤0.01准确率':<15} {'转向准确率':<15} {'符号准确率':<15}")
    print("-" * 100)
    
    model_display = {
        'fp32': ('FP32基础模型', 194),
        'fp32_action_head_fp16': ('FP16参考模型', 194),
        'balanced_static_conv_matmul_head_fp16': ('INT8+FP16(部署)', 50)
    }
    
    for model_key, (display_name, size) in model_display.items():
        if model_key in models_data:
            item = models_data[model_key]
            metrics = item['thresholds']['0.005000']
            print(
                f"{display_name:<20} {size:<15} "
                f"{item.get('diff_mae', 'N/A'):<15.6f} "
                f"{item.get('diff_acc_abs_le_0.010', 'N/A'):<15.6f} "
                f"{metrics.get('turn_accuracy', 'N/A'):<15.6f} "
                f"{metrics.get('ignore_pred_straight_accuracy', 'N/A'):<15.6f}"
            )
    
    print("\n【第3部分】模型文件位置")
    print("-" * 100)
    print("1. FP32 基础模型")
    print("   路径：artifacts/onnx_quant/act_finetuned_fp32.onnx")
    print("   大小：194 MB")
    print("   特性：原始模型，全 FP32 精度\n")
    
    print("2. FP16 参考模型")
    print("   路径：artifacts/onnx_quant/fp32_action_head_fp16.onnx")
    print("   大小：194 MB")
    print("   特性：骨干网络 FP32，动作头 FP16 精度\n")
    
    print("3. INT8+FP16 混合量化模型（选定部署模型）")
    print("   路径：models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx")
    print("   大小：50 MB（相比 FP32 减小 75%）")
    print("   特性：对 Conv/MatMul/Gemm 使用 balanced 校准做静态 QDQ 量化，动作头保持 FP16")
    print("   优势：")
    print("   - 模型体积缩小 4 倍")
    print("   - 轮速差 MAE = 0.0198（优于 FP32 的 0.0221）")
    print("   - ≤0.01 准确率 = 89.8%")
    print("   - 符号准确率 = 92.31%（远超 FP32 的 81.82%）")
    
    print("\n【第4部分】推理执行环境与方式")
    print("-" * 100)
    print("运行环境：")
    print("  - 主机：WSL2 Linux 5.15.167")
    print("  - QEMU：qemu-system-riscv64 8.2.2")
    print("  - 目标系统：StarryOS")
    print("  - 处理器：RISC-V 64-bit")
    print("  - 内存：1GB")
    print("  - CPU核心：1")
    
    print("\n执行步骤（详见 inference_results_summary.md）：")
    print("  1. 启动 QEMU 虚拟机")
    print("  2. 在 StarryOS 中运行 act_ort_infer 推理程序")
    print("  3. 对 10 个选定帧分别进行 FP32、FP16、INT8+FP16 三种精度的推理")
    print("  4. 汇总并比较推理结果")
    
    print("\n【第5部分】性能统计（基于 StarryOS/QEMU）")
    print("-" * 100)
    print("单帧推理性能：")
    print("  - 加载参数：10.5 ms")
    print("  - 创建 ORT 环境：784 ms")
    print("  - 创建推理 Session：8015 ms")
    print("  - 图像处理：32.2 ms")
    print("  - 推理耗时（平均）：5660 ms")
    print("  - 峰值内存：76.8 MB")
    
    print("\n【第6部分】关键结论")
    print("-" * 100)
    print("✓ INT8+FP16 混合量化模型在保持与 FP32 相近性能的前提下，")
    print("  实现了模型体积的大幅缩小（194MB → 50MB）")
    print("\n✓ 量化后模型的轮速差误差反而更优（0.0198 vs 0.0221），")
    print("  且转向符号准确率显著提升（92.31% vs 81.82%）")
    print("\n✓ 该模型已成功部署并可在 StarryOS RISC-V 上运行，")
    print("  单帧推理耗时约 5.7 秒（在 QEMU 模拟环境下）")
    print("\n✓ 该模型适合在资源受限的 RISC-V 系统（如 SG2002）上部署")
    
    print("\n" + "=" * 100 + "\n")

if __name__ == "__main__":
    generate_comprehensive_summary()
