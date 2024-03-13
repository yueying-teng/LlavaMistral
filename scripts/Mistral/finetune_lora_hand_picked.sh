#!/bin/bash

deepspeed llava/train/train-flash_attention_2.py \
  --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --version plain \
  --data_path ./playground/data/train/finetune/hand_picked/science_qa_finetune.json \
  --image_folder ./playground/data/train/finetune/hand_picked/images \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/llava-mistral-7b-pretrain/mm_projector.bin \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/llava-mistral-7b-lora-hand-picked \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --save_steps 1 \
  --save_total_limit 6 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb
