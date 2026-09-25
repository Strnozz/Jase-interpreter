#!/usr/bin/env bash
set -euo pipefail

# Designed for a single NVIDIA GPU such as RTX 4080 Super 16 GB.
# For a 2B model, normal LoRA is preferable to QLoRA when memory allows,
# because the adapter can later be merged cleanly into the base model.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
swift sft \
  --model Qwen/Qwen3.5-2B \
  --use_hf true \
  --dataset data/train.jsonl \
  --val_dataset data/valid.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --target_modules all-linear \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_length 768 \
  --eval_steps 100 \
  --save_steps 100 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --warmup_ratio 0.05 \
  --output_dir outputs/jase-v1-nvidia
