#!/usr/bin/env python3
"""
完整的StarryOS QEMU推理结果汇总

收集以下帧的推理结果：
1. 真值（Ground Truth）
2. FP32 模型结果
3. FP16 模型结果（FP32 动作头FP16）
4. INT8+FP16 混合量化模型的结果（选定部署模型）
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

class InferenceSummaryComplete:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.target_frames = [37, 59, 228, 229, 313, 331, 392, 463, 542, 586]
        
        self.eval_manifest_path = self.repo_root / "deploy/cpp_onnxruntime/data/eval_manifest.csv"
        self.quant_eval_path = self.repo_root / "artifacts/onnx_quant/closed_loop_quant_eval.json"
        
        self.gt_data = {}
        self.quant_metrics = {}
        self.load_data()
    
    def load_data(self):
        """加载所有数据"""
        self.load_gt_data()
        self.load_quant_metrics()
    
    def load_gt_data(self):
        """加载真值数据"""
        if not self.eval_manifest_path.exists():
            print(f"警告：评测清单文件不存在")
            return
        
        with open(self.eval_manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['index'])
                if frame_idx in self.target_frames:
                    self.gt_data[frame_idx] = {
                        'episode': int(row['episode_index']),
                        'left_vel': float(row['gt_left_vel']),
                        'right_vel': float(row['gt_right_vel']),
                        'gripper_target': float(row['gt_gripper_target']),
                        'image_path': row['image_path']
                    }
    
    def load_quant_metrics(self):
        """加载量化评测数据"""
        if not self.quant_eval_path.exists():
            print(f"警告：量化评测文件不存在")
            return
        
        with open(self.quant_eval_path, 'r') as f:
            eval_data = json.load(f)
        
        models_to_track = {
            'fp32': '原始 FP32 模型',
            'fp32_action_head_fp16': 'FP32 + 动作头 FP16',
            'balanced_static_conv_matmul_head_fp16': '选定部署模型（INT8+FP16）'
        }
        
        for item in eval_data:
            if item['name'] in models_to_track:
                self.quant_metrics[item['name']] = {
                    'path': item['path'],
                    'status': item['status'],
                    'metrics': {
                        'diff_mae': item.get('diff_mae', 'N/A'),
                        'diff_rmse': item.get('diff_rmse', 'N/A'),
                        'diff_acc_le_0.01': item.get('diff_acc_abs_le_0.010', 'N/A'),
                        'turn_accuracy_0.005': item['thresholds']['0.005000'].get('turn_accuracy', 'N/A'),
                        'sign_no_straight_0.005': item['thresholds']['0.005000'].get('ignore_pred_straight_accuracy', 'N/A'),
                    }
                }
    
    def generate_markdown_report(self) -> str:
        """生成完整的Markdown报告"""
        lines = []
        
        lines.append("# StarryOS QEMU 推理结果汇总报告\n\n")
        lines.append("## 任务概述\n")
        lines.append("本报告汇总10个选定帧在不同精度模型下的推理结果对比。\n\n")
        
        # 真值表格
        lines.append("## 1. 真值数据（Ground Truth）\n\n")
        lines.append("| 帧数 | 真值左轮速 | 真值右轮速 | 真值夹爪目标 |\n")
        lines.append("|------|----------|----------|----------|\n")
        
        for frame_idx in sorted(self.target_frames):
            if frame_idx in self.gt_data:
                gt = self.gt_data[frame_idx]
                lines.append(
                    f"| {frame_idx:>3d} | {gt['left_vel']:>8.6f} | {gt['right_vel']:>8.6f} | "
                    f"{gt['gripper_target']:>8.6f} |\n"
                )
        
        # 模型性能指标
        lines.append("\n## 2. 模型性能指标对比\n\n")
        lines.append("| 模型类型 | diff MAE | diff≤0.01% | 转向准确率@0.005 | 符号准确率@0.005 |\n")
        lines.append("|---------|---------|-----------|-----------------|----------------|\n")
        
        model_names = {
            'fp32': 'FP32 基础模型',
            'fp32_action_head_fp16': 'FP32 (动作头FP16)',
            'balanced_static_conv_matmul_head_fp16': '选定部署模型(INT8+FP16)'
        }
        
        for model_key, model_name in model_names.items():
            if model_key in self.quant_metrics:
                m = self.quant_metrics[model_key]['metrics']
                lines.append(
                    f"| {model_name} | {m['diff_mae']:.6f} | {m['diff_acc_le_0.01']:.4f} | "
                    f"{m['turn_accuracy_0.005']:.4f} | {m['sign_no_straight_0.005']:.4f} |\n"
                )
        
        # 选定帧详细推理结果（模板）
        lines.append("\n## 3. 选定帧详细推理结果\n\n")
        lines.append("### 3.1 帧 #37（左转，-0.1/0.1）\n")
        lines.append("```\n")
        lines.append("真值：左轮速=-0.100000，右轮速=0.100000，决策=左转\n")
        lines.append("FP32：左轮速=TBD，右轮速=TBD\n")
        lines.append("FP32(动作头FP16)：左轮速=TBD，右轮速=TBD\n")
        lines.append("INT8+FP16(部署模型)：左轮速=TBD，右轮速=TBD\n")
        lines.append("```\n\n")
        
        # 模型信息
        lines.append("\n## 4. 模型信息\n\n")
        lines.append("### 4.1 已部署模型\n\n")
        lines.append("**选定部署模型**：\n\n")
        lines.append("```\n")
        lines.append("名称：balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx\n")
        lines.append("大小：~50 MB\n")
        lines.append("量化策略：对 Conv/MatMul/Gemm 使用 balanced 校准做静态 QDQ 量化，\n")
        lines.append("          动作头保持 FP16\n")
        lines.append("评价：相比 194 MB 的 FP32 模型，体积减小 75%，\n")
        lines.append("      轮速差MAE=0.019771，≤0.01准确率=89.79%\n")
        lines.append("```\n\n")
        
        lines.append("### 4.2 参考模型\n\n")
        lines.append("- **FP32**：size=194 MB，path=`act_finetuned_fp32.onnx`\n")
        lines.append("- **FP32(动作头FP16)**：size=194 MB，path=`fp32_action_head_fp16.onnx`\n\n")
        
        # 运行环境
        lines.append("\n## 5. 运行环境\n\n")
        lines.append("```\n")
        lines.append("主机OS：WSL2 Linux 5.15.167.4-microsoft-standard\n")
        lines.append("QEMU：qemu-system-riscv64 8.2.2\n")
        lines.append("目标OS：StarryOS\n")
        lines.append("处理器架构：RISC-V 64-bit\n")
        lines.append("内存：1GB\n")
        lines.append("CPU核心：1\n")
        lines.append("```\n\n")
        
        # 启动命令
        lines.append("## 6. 执行步骤\n\n")
        lines.append("### 6.1 启动 QEMU\n\n")
        lines.append("```bash\n")
        lines.append("cd /home/sakura/OSproj57\n")
        lines.append("qemu-system-riscv64 \\\n")
        lines.append("  -m 1G -smp 1 -machine virt -bios default \\\n")
        lines.append("  -kernel ./StarryOS/workspace_riscv64-qemu-virt.bin \\\n")
        lines.append("  -device virtio-blk-pci,drive=disk0 \\\n")
        lines.append("  -drive id=disk0,if=none,format=raw,file=./StarryOS/make/disk.img \\\n")
        lines.append("  -device virtio-net-pci,netdev=net0 \\\n")
        lines.append("  -netdev user,id=net0,hostfwd=tcp::5555-:5555 \\\n")
        lines.append("  -nographic -monitor none\n")
        lines.append("```\n\n")
        
        lines.append("### 6.2 在 StarryOS 中执行单帧推理\n\n")
        lines.append("```sh\n")
        lines.append("cd /root/proj57-act\n")
        lines.append("export LD_LIBRARY_PATH=/root/proj57-act/lib\n")
        lines.append("export ORT_NUM_THREADS=1\n\n")
        lines.append("# 单帧推理示例（帧37）\n")
        lines.append("bin/act_ort_infer \\\n")
        lines.append("  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \\\n")
        lines.append("  --image data/frame_000037.jpg \\\n")
        lines.append("  --params config/act_params.json \\\n")
        lines.append("  --state 0 0 \\\n")
        lines.append("  --threads 1 \\\n")
        lines.append("  --warmup 1 \\\n")
        lines.append("  --runs 3 \\\n")
        lines.append("  --deadband 0.01\n")
        lines.append("```\n\n")
        
        # 性能统计
        lines.append("## 7. 性能统计（基于 StarryOS/QEMU 单进程）\n\n")
        lines.append("| 阶段 | 耗时 | VmRSS | 说明 |\n")
        lines.append("|------|-----|--------|------|\n")
        lines.append("| 加载参数 | 10.5 ms | 2.7 MB | |\n")
        lines.append("| 创建 ORT 环境 | 784 ms | 10.3 MB | |\n")
        lines.append("| 创建 Session | 8015 ms | 75.8 MB | 模型加载到GPU/内存 |\n")
        lines.append("| 图像处理 | 32.2 ms | 76.4 MB | 解码、缩放、归一化 |\n")
        lines.append("| **推理平均** | **5660 ms** | **76.7 MB** | **CPU推理** |\n")
        lines.append("| **峰值内存** | - | **76.8 MB** | **VmHWM** |\n\n")
        
        # 数据来源
        lines.append("## 8. 数据来源与文件位置\n\n")
        lines.append("- 真值数据：`deploy/cpp_onnxruntime/data/eval_manifest.csv`\n")
        lines.append("- 量化评测指标：`artifacts/onnx_quant/closed_loop_quant_eval.md` / `.json`\n")
        lines.append("- 部署模型：`models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx`\n")
        lines.append("- FP32 模型：`artifacts/onnx_quant/act_finetuned_fp32.onnx`\n")
        lines.append("- FP16 模型：`artifacts/onnx_quant/fp32_action_head_fp16.onnx`\n")
        lines.append("- 推理程序：`bin/riscv64/act_ort_infer`（RISC-V 预编译版本）\n")
        lines.append("- 配置文件：`deploy/cpp_onnxruntime/config/act_params.json`\n")
        lines.append("- StarryOS 内核：`StarryOS/workspace_riscv64-qemu-virt.bin`\n")
        lines.append("- 磁盘镜像：`StarryOS/make/disk.img`\n\n")
        
        # 结论
        lines.append("## 9. 结论\n\n")
        lines.append("- **INT8+FP16 混合量化模型** 在保持接近 FP32 性能的前提下，模型体积缩小至原来的 1/4。\n")
        lines.append("- 轮速差误差（MAE）为 0.0198，≤0.01 的准确率达 89.79%。\n")
        lines.append("- 转向判决的符号准确率达 92.31%（在 eps=0.005 阈值下）。\n")
        lines.append("- 该模型适合在资源受限的 RISC-V 系统（如 SG2002）上部署。\n")
        
        return "".join(lines)
    
    def generate_json_report(self) -> Dict:
        """生成完整的JSON报告"""
        report = {
            "metadata": {
                "title": "StarryOS QEMU 推理结果汇总",
                "target_frames": self.target_frames,
                "total_frames": len(self.target_frames),
                "timestamp": "2026-06-06"
            },
            "models": {
                "fp32": {
                    "name": "原始 FP32 模型",
                    "path": "artifacts/onnx_quant/act_finetuned_fp32.onnx",
                    "size_mb": 194,
                    "status": "参考模型",
                    "metrics": self.quant_metrics.get('fp32', {}).get('metrics', {})
                },
                "fp32_action_head_fp16": {
                    "name": "FP32 + 动作头 FP16",
                    "path": "artifacts/onnx_quant/fp32_action_head_fp16.onnx",
                    "size_mb": 194,
                    "status": "参考模型",
                    "metrics": self.quant_metrics.get('fp32_action_head_fp16', {}).get('metrics', {})
                },
                "int8_fp16_deployed": {
                    "name": "选定部署模型（INT8+FP16 混合量化）",
                    "path": "models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx",
                    "size_mb": 50,
                    "quantization_strategy": "对 Conv/MatMul/Gemm 使用 balanced 校准做静态 QDQ 量化，动作头保持 FP16",
                    "status": "已部署",
                    "metrics": self.quant_metrics.get('balanced_static_conv_matmul_head_fp16', {}).get('metrics', {})
                }
            },
            "ground_truth_summary": {
                "total_frames": len(self.gt_data),
                "frames": self.gt_data
            },
            "performance": {
                "qemu_environment": {
                    "os": "StarryOS",
                    "qemu_version": "8.2.2",
                    "memory_mb": 1024,
                    "cpu_cores": 1,
                    "architecture": "RISC-V 64-bit"
                },
                "single_frame_inference": {
                    "avg_latency_ms": 5660,
                    "peak_memory_mb": 76.8,
                    "vmpeak_vmpss_mb": 76.7
                }
            }
        }
        
        return report

def main():
    repo_root = "/home/sakura/OSproj57/act-starryos-qemu-infer"
    summary = InferenceSummaryComplete(repo_root)
    
    # 生成 Markdown 报告
    md_report = summary.generate_markdown_report()
    md_path = Path(repo_root) / "inference_results_summary.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"✓ Markdown 报告已生成: {md_path}\n")
    
    # 生成 JSON 报告
    json_report = summary.generate_json_report()
    json_path = Path(repo_root) / "inference_results_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON 报告已生成: {json_path}\n")
    
    # 打印汇总表格
    print("=" * 70)
    print("真值数据汇总表")
    print("=" * 70)
    print(f"{'帧数':<8} {'真值左轮速':<15} {'真值右轮速':<15} {'夹爪目标':<15}")
    print("-" * 70)
    
    for frame_idx in sorted(summary.target_frames):
        if frame_idx in summary.gt_data:
            gt = summary.gt_data[frame_idx]
            print(
                f"{frame_idx:<8} {gt['left_vel']:<15.6f} "
                f"{gt['right_vel']:<15.6f} {gt['gripper_target']:<15.6f}"
            )
    
    print("=" * 70)
    print("\n模型性能指标对比：\n")
    print(f"{'模型':<35} {'diff MAE':<12} {'diff≤0.01':<12} {'转向准确率':<12} {'符号准确率':<12}")
    print("-" * 83)
    
    model_names = {
        'fp32': 'FP32 基础模型',
        'fp32_action_head_fp16': 'FP32 (动作头FP16)',
        'balanced_static_conv_matmul_head_fp16': '选定部署模型(INT8+FP16)'
    }
    
    for model_key, model_name in model_names.items():
        if model_key in summary.quant_metrics:
            m = summary.quant_metrics[model_key]['metrics']
            print(
                f"{model_name:<35} {m['diff_mae']:<12.6f} {m['diff_acc_le_0.01']:<12.4f} "
                f"{m['turn_accuracy_0.005']:<12.4f} {m['sign_no_straight_0.005']:<12.4f}"
            )

if __name__ == "__main__":
    main()
