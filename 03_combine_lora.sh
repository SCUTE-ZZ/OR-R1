MODEL_NAME=$1
SFT_MODEL="./output/${MODEL_NAME}"
GRPO_MODEL="./output/lora_grpo_${MODEL_NAME}"
OUTPUT_DIR="./output/full_grpo_${MODEL_NAME}"
python 03_combine_lora.py $SFT_MODEL $GRPO_MODEL $OUTPUT_DIR
