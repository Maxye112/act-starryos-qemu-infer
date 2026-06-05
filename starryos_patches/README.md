# StarryOS 内核补丁说明

为在 QEMU 上统计推理进程内存，对 StarryOS 内核做了**少量修改**：在 `/proc/self/status` 中补充 **VmRSS、VmHWM、VmSize** 等字段（与 Linux 类似），便于读取峰值占用。改完后重新编译内核，用新内核启动 QEMU 即可；**不影响** ACT 推理与评测逻辑。未打补丁时程序仍可运行，但内存数据可能读不到。
