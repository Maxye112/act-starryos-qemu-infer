# Limitations and Notes

This file intentionally separates caveats from the main deliverable report.

## QEMU Timing

The QEMU run uses `qemu-system-riscv64` on a non-RISC-V host. It executes through
TCG emulation, so measured latency is useful for deployment-flow validation but
not for estimating native board latency.

## CPU Backend

The delivered path is ONNX Runtime CPU inference for StarryOS/QEMU. SG2002 TPU
and RK3588 NPU deployment require their vendor runtimes and model conversion
flows.

## Turn Threshold

Turn classification depends on `eps = abs(left_vel - right_vel)`. The report
uses multiple thresholds because closed-loop predictions can be directionally
correct while having low wheel-speed-difference magnitude.

## Dataset Evaluation

Closed-loop evaluation feeds the previous predicted wheel velocities into the
next frame. It resets state at episode boundaries using `episode_index` from the
manifest.

## Memory Accounting

StarryOS `/proc` now exposes process memory fields used by the benchmark.
Allocator tracking measures ONNX Runtime allocations that pass through the
registered allocator and is not a full replacement for process RSS.
