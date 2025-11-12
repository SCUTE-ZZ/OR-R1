MODEL_PATH=$1
NUM_GPUS=$2

# TEST_DATASET_NAME="CardinalOperations/MAMO"
# TEST_DATASET_SPLIT="complex_lp"

TEST_DATASET_NAME="./datasets/testset/mamo_complex_test.jsonl"
TEST_DATASET_SPLIT="test"

Q2MC_OUTPUT_DIR="$MODEL_PATH/eval.MAMO_ComplexLP.pass8"

python -m eval.generate \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $TEST_DATASET_NAME \
    --dataset_split $TEST_DATASET_SPLIT \
    --tensor_parallel_size $NUM_GPUS \
    --save_dir $Q2MC_OUTPUT_DIR \
    --topk 8 \
    --decoding sampling \
    --verbose

python -m eval.execute \
    --input_file $Q2MC_OUTPUT_DIR/generated.jsonl \
    --output_file $Q2MC_OUTPUT_DIR/executed.jsonl \
    --question_field question \
    --answer_field answer \
    --timeout 600 \
    --max_workers 16 \
    --verbose