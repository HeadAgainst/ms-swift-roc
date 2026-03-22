# User profile summarization full-parameter training.

IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dataset /path/to/profile_train.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --template qwen3_vl \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --attn_impl flash_attn \
    --padding_free true \
    --learning_rate 1e-5 \
    --router_aux_loss_coef 1e-3 \
    --experts_impl grouped_mm \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_profile_full \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
