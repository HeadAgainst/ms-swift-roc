# Minimal smoke test for ROC-style rater training.
# Replace the dataset path before running.
# The assistant response should follow:
# <reason>...</reason> <score>[SCORE]</score>

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --dataset /path/to/rater_train.jsonl#32 \
    --load_from_cache_file false \
    --split_dataset_ratio 0 \
    --template qwen3_vl_roc \
    --roc_enable true \
    --roc_num_tokens 10 \
    --roc_min_score 1 \
    --roc_max_score 5 \
    --roc_l1_weight 1.0 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --max_steps 10 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl sdpa \
    --learning_rate 1e-4 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --save_steps 10 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 1024 \
    --output_dir output_rater_smoke_lora \
    --warmup_ratio 0 \
    --dataset_num_proc 1 \
    --dataloader_num_workers 0
