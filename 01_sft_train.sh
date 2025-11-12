sample=$1
epoch=$2
MODEL_NAME_OR_PATH="Qwen/Qwen3-8B"
SAVE_PATH="./output/sft_qwen3_8b_dir_${sample}sample_${epoch}epoch"

DATA_PATH="./datasets/OR-Instruct-Data-3K/OR-Instruct-Data-${sample}.json"

NUM_GPUS=4
BATCH_SIZE_PER_GPU=1 
PREPROCESSING_NUM_WORKERS=0
MAX_SEQ_LENGTH=8192
LEARNING_RATE=2e-5
NUM_TRAIN_EPOCHS=${epoch}

# torchrun \
python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m 01_sft_train \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps 16 \
    --save_strategy "no" \
    --save_total_limit 1 \
    --preprocessing_num_workers $PREPROCESSING_NUM_WORKERS \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed config/sft_config.json \
    --overwrite_output_dir \
    --bf16 True
