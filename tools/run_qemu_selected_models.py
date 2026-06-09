#!/usr/bin/env python3
import csv
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO = Path("/home/sakura/OSproj57/act-starryos-qemu-infer")
STARRY = Path("/home/sakura/OSproj57/StarryOS")
BASE_DISK = STARRY / "make/disk.img"
KERNEL = STARRY / "workspace_riscv64-qemu-virt.bin"
WORK = REPO / "qemu_model_runs"
RUN_DISK = WORK / "disk_compare.img"
RAW_DIR = WORK / "raw"
SUMMARY_CSV = WORK / "selected_frame_qemu_results.csv"

TARGET_FRAMES = [37, 59, 228, 229, 313, 331, 392, 463, 542, 586]

MODELS = [
    (
        "FP32",
        REPO / "artifacts/onnx_quant/act_finetuned_fp32.onnx",
    ),
    (
        "FP16",
        REPO / "artifacts/onnx_quant/fp32_action_head_fp16.onnx",
    ),
    (
        "INT8_FP16",
        REPO / "artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx",
    ),
]


def run(cmd, *, input_text=None, timeout=None):
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def debugfs(commands):
    proc = run(["debugfs", "-w", str(RUN_DISK)], input_text="\n".join(commands) + "\n")
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(f"debugfs failed with code {proc.returncode}")
    return proc.stdout


def load_frames():
    frames = {}
    with (REPO / "deploy/cpp_onnxruntime/data/eval_manifest.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            if idx in TARGET_FRAMES:
                frames[idx] = {
                    "state_left": float(row["state_left"]),
                    "state_right": float(row["state_right"]),
                    "gt_left": float(row["gt_left_vel"]),
                    "gt_right": float(row["gt_right_vel"]),
                    "image": f"data/dataset/{row['image_path']}",
                }
    missing = [idx for idx in TARGET_FRAMES if idx not in frames]
    if missing:
        raise SystemExit(f"missing target frames in manifest: {missing}")
    return frames


def prepare_disk():
    WORK.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    if RUN_DISK.exists():
        RUN_DISK.unlink()
    print(f"copy disk image: {BASE_DISK} -> {RUN_DISK}", flush=True)
    shutil.copy2(BASE_DISK, RUN_DISK)

    # Keep libonnxruntime.so.1, remove duplicate ORT libraries and previous current model.
    debugfs(
        [
            "rm /root/proj57-act/models/current.onnx",
            "rm /root/proj57-act/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx",
            "rm /root/proj57-act/lib/libonnxruntime.so",
            "rm /root/proj57-act/lib/libonnxruntime.so.1.26.0",
            "rm /lib/libonnxruntime.so",
            "rm /lib/libonnxruntime.so.1",
            "rm /lib/libonnxruntime.so.1.26.0",
        ]
    )


def write_model(model_path):
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")
    print(f"write model into rootfs: {model_path.name}", flush=True)
    debugfs(
        [
            "rm /root/proj57-act/models/current.onnx",
            f"write {model_path} /root/proj57-act/models/current.onnx",
        ]
    )


def run_qemu_for_model(model_name, frames):
    raw_path = RAW_DIR / f"{model_name}.log"
    cmd = [
        "qemu-system-riscv64",
        "-m",
        "1G",
        "-smp",
        "1",
        "-machine",
        "virt",
        "-bios",
        "default",
        "-kernel",
        str(KERNEL),
        "-device",
        "virtio-blk-pci,drive=disk0",
        "-drive",
        f"id=disk0,if=none,format=raw,file={RUN_DISK}",
        "-device",
        "virtio-net-pci,netdev=net0",
        "-netdev",
        "user,id=net0",
        "-nographic",
        "-monitor",
        "none",
    ]
    print(f"run QEMU model={model_name}", flush=True)
    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)
    os.set_blocking(master_fd, False)
    chunks = []

    def read_available(timeout=0.2):
        data = b""
        end = time.time() + timeout
        while time.time() < end:
            readable, _, _ = select.select([master_fd], [], [], max(0.0, end - time.time()))
            if not readable:
                break
            try:
                part = os.read(master_fd, 65536)
            except BlockingIOError:
                continue
            except OSError:
                break
            if not part:
                break
            chunks.append(part)
            data += part
        return data

    def text():
        return b"".join(chunks).replace(b"\x00", b"").decode("utf-8", "replace")

    def send_line(line):
        os.write(master_fd, (line + "\r").encode())

    def wait_for(needle, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in text():
                return
            if proc.poll() is not None:
                break
            read_available(0.5)
        raw_path.write_text(text())
        raise SystemExit(f"timeout waiting for {needle!r} in model {model_name}")

    try:
        wait_for("starry:~#", 120)
        send_line("export LD_LIBRARY_PATH=/root/proj57-act/lib:/lib")
        wait_for("starry:~#", 20)
        send_line(f"echo QEMU_MODEL_BEGIN_{model_name}")
        wait_for(f"QEMU_MODEL_BEGIN_{model_name}", 20)

        for idx in TARGET_FRAMES:
            f = frames[idx]
            begin = (
                f"RESULT_BEGIN model={model_name} frame={idx} "
                f"gt_left={f['gt_left']:.9f} gt_right={f['gt_right']:.9f} "
                f"state_left={f['state_left']:.9f} state_right={f['state_right']:.9f}"
            )
            send_line("echo " + begin)
            wait_for(begin, 20)
            cmd_line = (
                "/root/proj57-act/bin/act_ort_infer "
                "--model /root/proj57-act/models/current.onnx "
                f"--image /root/proj57-act/{f['image']} "
                "--params /root/proj57-act/config/act_params.json "
                f"--state {f['state_left']:.9f} {f['state_right']:.9f} "
                "--threads 1 --warmup 0 --runs 1 --deadband 0.005"
            )
            send_line(cmd_line)
            wait_for("first_step:", 900)
            end_marker = f"RESULT_END model={model_name} frame={idx}"
            send_line("echo " + end_marker)
            wait_for(end_marker, 20)
            print(f"  {model_name} frame {idx} done", flush=True)

        done = f"QEMU_MODEL_DONE_{model_name}"
        send_line("echo " + done)
        wait_for(done, 20)
        raw_path.write_text(text())
        print(f"qemu model={model_name} raw={raw_path}", flush=True)
        return text()
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        os.close(master_fd)


def parse_outputs(log_text):
    results = []
    current = None
    begin_re = re.compile(
        r"RESULT_BEGIN model=(\S+) frame=(\d+) gt_left=([-\d.]+) gt_right=([-\d.]+) "
        r"state_left=([-\d.]+) state_right=([-\d.]+)"
    )
    pred_re = re.compile(
        r"first_step: left_vel=([-\deE+.]+) right_vel=([-\deE+.]+).*diff=([-\deE+.]+) decision=(\S+)"
    )
    for line in log_text.splitlines():
        m = begin_re.search(line)
        if m:
            current = {
                "model": m.group(1),
                "frame": int(m.group(2)),
                "gt_left": float(m.group(3)),
                "gt_right": float(m.group(4)),
                "state_left": float(m.group(5)),
                "state_right": float(m.group(6)),
                "pred_left": None,
                "pred_right": None,
                "pred_diff": None,
                "decision": None,
            }
            continue
        m = pred_re.search(line)
        if m and current is not None:
            current["pred_left"] = float(m.group(1))
            current["pred_right"] = float(m.group(2))
            current["pred_diff"] = float(m.group(3))
            current["decision"] = m.group(4)
            results.append(current)
            current = None
    return results


def write_summary(rows):
    with SUMMARY_CSV.open("w", newline="") as f:
        fieldnames = [
            "frame",
            "model",
            "gt_left",
            "gt_right",
            "state_left",
            "state_right",
            "pred_left",
            "pred_right",
            "pred_diff",
            "decision",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["frame"], r["model"])):
            writer.writerow(row)
    print(f"summary csv: {SUMMARY_CSV}", flush=True)


def main():
    for path in [BASE_DISK, KERNEL]:
        if not path.exists():
            raise SystemExit(f"required file not found: {path}")
    frames = load_frames()
    prepare_disk()
    all_rows = []
    for model_name, model_path in MODELS:
        write_model(model_path)
        log_text = run_qemu_for_model(model_name, frames)
        rows = parse_outputs(log_text)
        if len(rows) != len(TARGET_FRAMES):
            raise SystemExit(f"expected {len(TARGET_FRAMES)} rows for {model_name}, got {len(rows)}")
        all_rows.extend(rows)
    write_summary(all_rows)


if __name__ == "__main__":
    main()
