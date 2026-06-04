#!/bin/sh
set -eu

BASE_DIR="${1:-/root/proj57-act}"
cd "${BASE_DIR}"

export LD_LIBRARY_PATH="${BASE_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

BIN="${BASE_DIR}/bin/act_ort_infer"
PARAMS="${BASE_DIR}/config/act_params.json"
IMAGE="${BASE_DIR}/data/frame_000000.jpg"
MODEL="${BASE_DIR}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx"

if [ ! -x "${BIN}" ]; then
  echo "Missing executable: ${BIN}" >&2
  exit 1
fi

echo "=== StarryOS ACT ONNXRuntime CPU Benchmark ==="
echo "date: $(date 2>/dev/null || true)"
echo "uname: $(uname -a 2>/dev/null || true)"
echo "model: ${MODEL}"
echo ""

run_case() {
  name="$1"
  extra="$2"
  echo "--- ${name} extra='${extra}' ---"
  "${BIN}" \
    --model "${MODEL}" \
    --image "${IMAGE}" \
    --params "${PARAMS}" \
    --state 0 0 \
    --threads 1 \
    --warmup 1 \
    --runs 3 \
    --deadband 0.01 \
    --track-allocator \
    ${extra}
  echo ""
}

run_case "single_thread_default" ""
run_case "single_thread_no_arena" "--no-arena"
run_case "single_thread_no_mem_pattern" "--no-mem-pattern"
