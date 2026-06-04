# StarryOS/QEMU Single-frame Inference Result

Environment:

```text
OS: StarryOS under qemu-system-riscv64
Kernel: /home/sakura/OSproj57/StarryOS/workspace_riscv64-qemu-virt.bin
CPU mode: QEMU CPU, -smp 1
Executable: bin/riscv64/act_ort_infer
```

Model:

```text
models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx
```

Frame:

```text
dataset frame: videos/observation.images.fpv/chunk-000/frame_000007.jpg
target path: /root/proj57-act/data/frame_000000.jpg
```

Input state:

```text
left_vel=-0.100000001
right_vel=0.100000001
```

Ground truth action:

```text
gt_left_vel=-0.100000001
gt_right_vel=0.100000001
gt_wheel_diff=-0.200000002
gt_direction=left
```

Predicted first action step on StarryOS:

```text
pred_left_vel=-0.10913
pred_right_vel=0.106983
pred_gripper_target=1.03931e-10
pred_wheel_diff=-0.216113
pred_direction=left
```

Comparison:

```text
wheel_diff_error=pred_wheel_diff - gt_wheel_diff = -0.016112998
direction_match=true
```

Runtime and memory:

```text
threads=1
warmup=1
runs=1
avg_latency_ms=5325.1
VmRSS_after_runs_kb=78556
VmHWM_after_runs_kb=78556
VmSize_after_runs_kb=90292
ORT_allocator_peak_after_runs_bytes=20837135
```

Raw first-step output:

```text
first_step: left_vel=-0.10913 right_vel=0.106983 gripper_target=1.03931e-10 diff=-0.216113 decision=left
```
