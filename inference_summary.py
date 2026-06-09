#!/usr/bin/env python3
"""
汇总StarryOS QEMU推理结果的脚本

收集以下帧的推理结果：
1. 真值
2. FP32模型结果
3. FP16模型结果  
4. INT8+FP16混合量化模型的结果
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

class InferenceSummary:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.target_frames = [37, 59, 228, 229, 313, 331, 392, 463, 542, 586]
        
        # 加载评测清单
        self.eval_manifest_path = self.repo_root / "deploy/cpp_onnxruntime/data/eval_manifest.csv"
        self.quant_eval_path = self.repo_root / "artifacts/onnx_quant/closed_loop_quant_eval.json"
        
        self.gt_data = {}  # 存储真值数据
        self.load_gt_data()
    
    def load_gt_data(self):
        """加载真值数据"""
        if not self.eval_manifest_path.exists():
            print(f"警告：评测清单文件不存在: {self.eval_manifest_path}")
            return
        
        with open(self.eval_manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_idx = int(row['index'])
                if frame_idx in self.target_frames:
                    self.gt_data[frame_idx] = {
                        'left_vel': float(row['gt_left_vel']),
                        'right_vel': float(row['gt_right_vel']),
                        'image_path': row['image_path']
                    }
        
        print(f"已加载 {len(self.gt_data)} 帧的真值数据")
    
    def generate_summary_markdown(self) -> str:
        """生成Markdown格式的汇总报告"""
        lines = []
        lines.append("# StarryOS QEMU 推理结果汇总\n")
        lines.append("## 选定帧的推理结果对比\n")
        
        # 表头
        lines.append("| 帧数 | 真值左轮速 | 真值右轮速 | FP32左轮速 | FP32右轮速 | FP16左轮速 | FP16右轮速 | INT8+FP16左轮速 | INT8+FP16右轮速 |")
        lines.append("|------|----------|----------|----------|----------|----------|----------|----------------|----------------|")
        
        for frame_idx in sorted(self.target_frames):
            if frame_idx in self.gt_data:
                gt = self.gt_data[frame_idx]
                lines.append(
                    f"| {frame_idx} | {gt['left_vel']:.6f} | {gt['right_vel']:.6f} | "
                    f"TBD | TBD | TBD | TBD | TBD | TBD |"
                )
        
        lines.append("\n## 数据来源说明\n")
        lines.append("- **真值**：从 `deploy/cpp_onnxruntime/data/eval_manifest.csv` 提取\n")
        lines.append("- **FP32 模型**：从 `artifacts/onnx_quant/closed_loop_quant_eval.md` 提取\n")
        lines.append("- **FP16 模型**：从 `artifacts/onnx_quant/closed_loop_quant_eval.md` 提取\n")
        lines.append("- **INT8+FP16 混合量化模型**：从 StarryOS QEMU 推理结果提取\n")
        
        lines.append("\n## 模型信息\n")
        lines.append("- **INT8+FP16 混合量化模型路径**：`models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx`\n")
        lines.append("- **模型大小**：约 50 MB\n")
        lines.append("- **量化策略**：对 Conv/MatMul/Gemm 使用静态量化，动作头保持 FP16\n")
        
        lines.append("\n## 执行方式\n")
        lines.append("```bash\n")
        lines.append("# 启动 QEMU/StarryOS\n")
        lines.append("qemu-system-riscv64 \\\\\n")
        lines.append("  -m 1G -smp 1 -machine virt -bios default \\\\\n")
        lines.append("  -kernel /home/sakura/OSproj57/StarryOS/workspace_riscv64-qemu-virt.bin \\\\\n")
        lines.append("  -device virtio-blk-pci,drive=disk0 \\\\\n")
        lines.append("  -drive id=disk0,if=none,format=raw,file=/home/sakura/OSproj57/StarryOS/make/disk.img \\\\\n")
        lines.append("  -device virtio-net-pci,netdev=net0 \\\\\n")
        lines.append("  -netdev user,id=net0,hostfwd=tcp::5555-:5555 \\\\\n")
        lines.append("  -nographic -monitor none\n")
        lines.append("```\n")
        
        lines.append("\n## StarryOS 中的推理命令\n")
        lines.append("```sh\n")
        lines.append("cd /root/proj57-act\n")
        lines.append("export LD_LIBRARY_PATH=/root/proj57-act/lib\n\n")
        lines.append("# 单帧推理（以帧37为例）\n")
        lines.append("bin/act_ort_infer \\\\\n")
        lines.append("  --model models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx \\\\\n")
        lines.append("  --image data/frame_000037.jpg \\\\\n")
        lines.append("  --params config/act_params.json \\\\\n")
        lines.append("  --state 0 0 \\\\\n")
        lines.append("  --threads 1 \\\\\n")
        lines.append("  --warmup 1 \\\\\n")
        lines.append("  --runs 3\n")
        lines.append("```\n")
        
        return "".join(lines)
    
    def generate_summary_json(self) -> Dict:
        """生成 JSON 格式的汇总数据"""
        summary = {
            "metadata": {
                "title": "StarryOS QEMU 推理结果汇总",
                "target_frames": self.target_frames,
                "total_frames": len(self.target_frames),
                "models": {
                    "fp32": {
                        "name": "FP32 基础模型",
                        "size_mb": 194,
                        "status": "参考模型（未在本仓库包含）"
                    },
                    "fp16": {
                        "name": "FP16 精度模型",
                        "size_mb": "TBD",
                        "status": "参考模型（未在本仓库包含）"
                    },
                    "int8_fp16": {
                        "name": "INT8+FP16 混合量化模型",
                        "size_mb": 50,
                        "path": "models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx",
                        "status": "已部署"
                    }
                }
            },
            "ground_truth": self.gt_data,
            "frames": {}
        }
        
        for frame_idx in sorted(self.target_frames):
            if frame_idx in self.gt_data:
                summary["frames"][frame_idx] = {
                    "ground_truth": self.gt_data[frame_idx],
                    "predictions": {
                        "fp32": {"left_vel": None, "right_vel": None},
                        "fp16": {"left_vel": None, "right_vel": None},
                        "int8_fp16": {"left_vel": None, "right_vel": None}
                    }
                }
        
        return summary

def main():
    summary = InferenceSummary("/home/sakura/OSproj57/act-starryos-qemu-infer")
    
    # 生成 Markdown 汇总
    markdown_output = summary.generate_summary_markdown()
    output_md = Path("/home/sakura/OSproj57/act-starryos-qemu-infer/inference_results_summary.md")
    with open(output_md, 'w') as f:
        f.write(markdown_output)
    print(f"✓ Markdown 汇总已保存到: {output_md}")
    
    # 生成 JSON 汇总
    json_output = summary.generate_summary_json()
    output_json = Path("/home/sakura/OSproj57/act-starryos-qemu-infer/inference_results_summary.json")
    with open(output_json, 'w') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON 汇总已保存到: {output_json}")
    
    # 打印摘要
    print("\n=== 真值数据摘要 ===")
    print(f"{'帧数':<6} {'真值左轮速':<12} {'真值右轮速':<12}")
    print("-" * 30)
    for frame_idx in sorted(summary.target_frames):
        if frame_idx in summary.gt_data:
            gt = summary.gt_data[frame_idx]
            print(f"{frame_idx:<6} {gt['left_vel']:<12.6f} {gt['right_vel']:<12.6f}")

if __name__ == "__main__":
    main()
