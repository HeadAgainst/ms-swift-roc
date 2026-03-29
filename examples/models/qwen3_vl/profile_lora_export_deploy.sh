#!/usr/bin/env bash
set -euo pipefail

# Export merged weights from a Qwen3-VL LoRA checkpoint, then deploy with vLLM.

ADAPTER_DIR="${ADAPTER_DIR:-output_profile_lora/checkpoint-xxx}"
MERGED_DIR="${MERGED_DIR:-output_profile_lora/checkpoint-xxx-merged}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
swift export \
    --adapters "${ADAPTER_DIR}" \
    --merge_lora true \
    --output_dir "${MERGED_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
MAX_PIXELS="${MAX_PIXELS:-1003520}" \
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}" \
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}" \
swift deploy \
    --model "${MERGED_DIR}" \
    --infer_backend vllm \
    --vllm_tensor_parallel_size "${VLLM_TP:-2}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --vllm_limit_mm_per_prompt "${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\": 5, \"video\": 2}}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-1024}" \
    --served_model_name "${SERVED_MODEL_NAME:-Qwen3-VL-Profile}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}"
