#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
MODEL_NAME=$1
MODEL_NAME_OR_PATH="./output/${MODEL_NAME}"
SAVE_PATH="./output/lora_grpo_${MODEL_NAME}"

DS_CONFIG_PATH="./config/grpo_config.json"

GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6001}
DATASET_PATH=${DATASET_PATH:-./datasets/trainset/train_all.jsonl}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

python -m torch.distributed.run $DISTRIBUTED_ARGS ./02_grpo_train_8b.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $SAVE_PATH \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16  \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --per_device_eval_batch_size 1 \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_dir ./logs_v0 \
    --logging_strategy "steps"\
    --logging_steps 1 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --report_to "tensorboard" \
    --bf16 True \
    --logging_first_step \
    --use_peft \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16 \
    --save_only_model \
    --max_prompt_length 2048 \
    --max_completion_length 6144 \
    --dataset_path ${DATASET_PATH}


