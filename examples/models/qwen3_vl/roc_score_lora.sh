# ROC-style single-task rater training with LoRA.
# The assistant response should follow:
# <reason>...</reason> <score>[SCORE]</score>

IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dataset /path/to/rater_train.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --template qwen3_vl_roc \
    --roc_enable true \
    --roc_num_tokens 10 \
    --roc_min_score 1 \
    --roc_max_score 5 \
    --roc_l1_weight 1.0 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --router_aux_loss_coef 1e-3 \
    --experts_impl grouped_mm \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_rater_lora \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
