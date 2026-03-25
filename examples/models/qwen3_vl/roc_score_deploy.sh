# Deploy ROC-style Qwen3-VL rater with the local ms-swift codebase.
# IMPORTANT:
# 1. Keep `--template qwen3_vl_roc`
# 2. Keep `--infer_backend transformers`
# 3. Replace the checkpoint path before running

CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model /home3/chenb/downloaded/piaa/rater \
    --template qwen3_vl_roc \
    --infer_backend transformers \
    --attn_impl flash_attn \
    --max_batch_size 1 \
    --host 0.0.0.0 \
    --port 8000
