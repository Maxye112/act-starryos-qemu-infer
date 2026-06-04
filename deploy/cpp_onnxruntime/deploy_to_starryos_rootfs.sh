#!/usr/bin/env bash
set -euo pipefail

ROOTFS_IMG="${1:-/home/sakura/OSproj57/StarryOS/make/disk.img}"
DEPLOY_ROOT="${2:-/root/proj57-act}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RISCV64_BUILD_DIR="${RISCV64_BUILD_DIR:-${SCRIPT_DIR}/build-riscv64}"
ONNXRUNTIME_ROOT_RISCV64="${ONNXRUNTIME_ROOT_RISCV64:-/home/sakura/Deploy-ACT/third_party/onnxruntime-linux-riscv64-musl}"
TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-${HOME}/.cache/deploy-act/riscv64-linux-musl-cross}"
TOOLCHAIN_SYSROOT_LIB="${TOOLCHAIN_SYSROOT_LIB:-${TOOLCHAIN_ROOT}/riscv64-linux-musl/lib}"

MODEL="${MODEL:-${PROJECT_ROOT}/artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx}"
IMAGE="${IMAGE:-${PROJECT_ROOT}/output/dataset/videos/observation.images.fpv/chunk-000/frame_000000.jpg}"
PARAMS="${PARAMS:-${SCRIPT_DIR}/config/act_params.json}"
EVAL_MANIFEST="${EVAL_MANIFEST:-${SCRIPT_DIR}/data/eval_manifest.csv}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/output/dataset}"
FRAME_DIR="${FRAME_DIR:-${DATASET_ROOT}/videos/observation.images.fpv/chunk-000}"

if [[ ! -f "${ROOTFS_IMG}" ]]; then
  echo "Rootfs image not found: ${ROOTFS_IMG}" >&2
  exit 1
fi

if [[ ! -x "${RISCV64_BUILD_DIR}/act_ort_infer" ]]; then
  echo "RISC-V binary not found: ${RISCV64_BUILD_DIR}/act_ort_infer" >&2
  echo "Build it with deploy/cpp_onnxruntime/README.md instructions." >&2
  exit 1
fi

for f in "${MODEL}" "${IMAGE}" "${PARAMS}" "${SCRIPT_DIR}/run_starryos_benchmark.sh"; do
  if [[ ! -f "${f}" ]]; then
    echo "Required file is missing: ${f}" >&2
    exit 1
  fi
done

if [[ -f "${EVAL_MANIFEST}" && ! -d "${FRAME_DIR}" ]]; then
  echo "Frame directory is missing: ${FRAME_DIR}" >&2
  exit 1
fi

if ! command -v debugfs >/dev/null 2>&1; then
  echo "debugfs is required to write the ext4 StarryOS rootfs image." >&2
  exit 1
fi

tmp_cmds="$(mktemp)"
trap 'rm -f "${tmp_cmds}"' EXIT

add_write() {
  local src="$1"
  local dst="$2"
  printf 'rm %s\n' "${dst}" >>"${tmp_cmds}"
  printf 'write %s %s\n' "${src}" "${dst}" >>"${tmp_cmds}"
}

add_runtime_lib() {
  local src="$1"
  local name
  name="$(basename "${src}")"
  add_write "${src}" "${DEPLOY_ROOT}/lib/${name}"
  add_write "${src}" "/lib/${name}"
}

cat >"${tmp_cmds}" <<EOF
mkdir ${DEPLOY_ROOT}
mkdir ${DEPLOY_ROOT}/bin
mkdir ${DEPLOY_ROOT}/lib
mkdir ${DEPLOY_ROOT}/models
mkdir ${DEPLOY_ROOT}/config
mkdir ${DEPLOY_ROOT}/data
mkdir ${DEPLOY_ROOT}/data/dataset
mkdir ${DEPLOY_ROOT}/data/dataset/videos
mkdir ${DEPLOY_ROOT}/data/dataset/videos/observation.images.fpv
mkdir ${DEPLOY_ROOT}/data/dataset/videos/observation.images.fpv/chunk-000
EOF

add_write "${RISCV64_BUILD_DIR}/act_ort_infer" "${DEPLOY_ROOT}/bin/act_ort_infer"
add_write "${MODEL}" "${DEPLOY_ROOT}/models/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx"
add_write "${PARAMS}" "${DEPLOY_ROOT}/config/act_params.json"
add_write "${IMAGE}" "${DEPLOY_ROOT}/data/frame_000000.jpg"
add_write "${SCRIPT_DIR}/run_starryos_benchmark.sh" "${DEPLOY_ROOT}/run_starryos_benchmark.sh"

if [[ -f "${EVAL_MANIFEST}" ]]; then
  add_write "${EVAL_MANIFEST}" "${DEPLOY_ROOT}/data/eval_manifest.csv"
  while IFS= read -r -d '' frame; do
    add_write "${frame}" "${DEPLOY_ROOT}/data/dataset/videos/observation.images.fpv/chunk-000/$(basename "${frame}")"
  done < <(find "${FRAME_DIR}" -maxdepth 1 -type f -name 'frame_*.jpg' -print0 | sort -z)
fi

if compgen -G "${ONNXRUNTIME_ROOT_RISCV64}/lib/libonnxruntime.so*" >/dev/null; then
  for lib in "${ONNXRUNTIME_ROOT_RISCV64}"/lib/libonnxruntime.so*; do
    add_runtime_lib "${lib}"
  done
else
  echo "ONNX Runtime RISC-V libs not found: ${ONNXRUNTIME_ROOT_RISCV64}/lib" >&2
  exit 1
fi

for runtime_lib in ld-musl-riscv64.so.1 libatomic.so.1 libstdc++.so.6 libgcc_s.so.1; do
  if [[ -f "${TOOLCHAIN_SYSROOT_LIB}/${runtime_lib}" ]]; then
    add_runtime_lib "${TOOLCHAIN_SYSROOT_LIB}/${runtime_lib}"
  else
    echo "RISC-V runtime library not found, skipping: ${TOOLCHAIN_SYSROOT_LIB}/${runtime_lib}" >&2
  fi
done

debugfs -w -f "${tmp_cmds}" "${ROOTFS_IMG}"

echo "Deployment copied into ${ROOTFS_IMG}:${DEPLOY_ROOT}"
echo "Inside StarryOS run:"
echo "  sh ${DEPLOY_ROOT}/run_starryos_benchmark.sh"
if [[ -f "${EVAL_MANIFEST}" ]]; then
  echo "Dataset eval inside StarryOS:"
  echo "  cd ${DEPLOY_ROOT} && LD_LIBRARY_PATH=${DEPLOY_ROOT}/lib bin/act_ort_infer --model models/$(basename "${MODEL}") --params config/act_params.json --eval-manifest data/eval_manifest.csv --dataset-root data/dataset --threads 1 --track-allocator"
fi
