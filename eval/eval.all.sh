MODEL_PATH=$1
NUM_GPUS=$2

sh eval/eval.NL4OPT.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.MAMO_EasyLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.MAMO_ComplexLP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.IndustryOR.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.NLP4LP.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.ComplexOR.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.OptiBench.pass1.sh $MODEL_PATH $NUM_GPUS
sh eval/eval.ICMLTEST.pass1.sh $MODEL_PATH $NUM_GPUS

# sh eval/eval.NL4OPT.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.MAMO_EasyLP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.MAMO_ComplexLP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.IndustryOR.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.NLP4LP.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.ComplexOR.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.OptiBench.pass8.sh $MODEL_PATH $NUM_GPUS
# sh eval/eval.ICMLTEST.pass8.sh $MODEL_PATH $NUM_GPUS

python eval/read_eval_results.py $MODEL_PATH
