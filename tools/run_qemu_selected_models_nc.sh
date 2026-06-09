#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-4321}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-/home/sakura/OSproj57/act-starryos-qemu-infer/qemu_model_runs/nc_raw}"
mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/all_models.log"

FRAMES=(37 59 228 229 313 331 392 463 542 586)
STATES_LEFT=(-0.100000001 -0.100000001 0.100000001 0.100000001 0.100000001 0.100000001 0.100000001 -0.100000001 -0.100000001 -0.100000001)
STATES_RIGHT=(0.100000001 0.100000001 -0.100000001 -0.100000001 -0.100000001 -0.100000001 -0.100000001 0.100000001 0.100000001 0.100000001)

MODELS=(
  "FP32:/root/proj57-act/models/act_finetuned_fp32.onnx"
  "FP16:/root/proj57-act/models/fp32_action_head_fp16.onnx"
  "INT8_FP16:/root/proj57-act/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx"
)

coproc QEMU_NC { nc "${HOST}" "${PORT}"; }

{
  printf '%s\n' 'export LD_LIBRARY_PATH=/root/proj57-act/lib:/lib'
  printf '%s\n' 'BIN=/root/proj57-act/bin/act_ort_infer'
  printf '%s\n' 'PARAMS=/root/proj57-act/config/act_params.json'
  printf '%s\n' 'IMGDIR=/root/proj57-act/data/dataset/videos/observation.images.fpv/chunk-000'
  for model in "${MODELS[@]}"; do
    model_name="${model%%:*}"
    model_path="${model#*:}"
    printf 'MODEL=%s\n' "${model_path}"
    for i in "${!FRAMES[@]}"; do
      frame="${FRAMES[$i]}"
      state_left="${STATES_LEFT[$i]}"
      state_right="${STATES_RIGHT[$i]}"
      printf 'echo START model=%s frame=%s gt_left=%s gt_right=%s\n' "${model_name}" "${frame}" "${state_left}" "${state_right}"
      printf '$BIN --model $MODEL --image $IMGDIR/frame_%06d.jpg --params $PARAMS --state %s %s --threads 1 --warmup 0 --runs 1 --deadband 0.005\n' "${frame}" "${state_left}" "${state_right}"
      printf 'echo DONE_%s_%s\n' "${model_name}" "${frame}"
    done
  done
  printf '%s\n' 'echo ALL_DONE'
} >&"${QEMU_NC[1]}"

: >"${LOG}"
while IFS= read -r -t 1800 line <&"${QEMU_NC[0]}"; do
  printf '%s\n' "${line}" | tee -a "${LOG}"
  case "${line}" in
    *ALL_DONE*) break ;;
  esac
done
kill "${QEMU_NC_PID}" 2>/dev/null || true
echo "raw log: ${LOG}"
