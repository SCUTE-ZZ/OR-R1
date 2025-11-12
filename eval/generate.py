import argparse
import json
import os
import re
import sys
import datasets

from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

OUTPUT_TEMPLATE = r"""

Fill in the following format:
## Mathematical Model:
YOUR ANSWER HERE

## Decision Variables:
YOUR ANSWER HERE

## Objective Function:
YOUR ANSWER HERE

## Constraints:
YOUR ANSWER HERE

## Python Code Solution Using `coptpy`:
```python
YOUR ANSWER HERE
```
"""

def main(args):
    assert args.dataset_name is not None
    assert args.dataset_split is not None
    assert isinstance(args.topk, int)
    assert args.decoding_method in ["greedy", "sampling"]
    assert os.path.exists(args.model_name_or_path), "We only support local model path!"
    assert args.save_dir is not None

    os.makedirs(args.save_dir, exist_ok=True)
    save_file = os.path.join(args.save_dir, "generated.jsonl")
    if os.path.exists(save_file):
        print(f"File {save_file} already exists. Exiting to avoid overwriting.")
        return 

    # Load data
    sample = []
    if( not args.dataset_name.endswith("jsonl") and not args.dataset_name.endswith("json")):
        ds = datasets.load_dataset(args.dataset_name)
        ds = ds[args.dataset_split]
    else:
        ds = datasets.load_dataset("json", data_files=args.dataset_name)
        ds = ds['train']

    # if("grpo" in args.model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    for example in ds:
        if("LLMOPT-Qwen2.5-14B" in args.model_name_or_path):
            if "en_question" in example:
                prompt = example["en_question"].strip()
            elif "question" in example:
                prompt = example["question"].strip()
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        else:
            if "en_question" in example:
                prompt = TEMPLATE_q2mc_en.replace("{Question}", example["en_question"].strip()).strip()
            elif "question" in example:
                prompt = TEMPLATE_q2mc_en.replace("{Question}", example["question"].strip()).strip()
            
            if("BASELINE" in args.model_name_or_path):
                prompt = prompt + OUTPUT_TEMPLATE
            if ("ORLM-LLaMA-3-8B" not in args.model_name_or_path) and ("llama3" not in args.model_name_or_path):
                prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
            

        example_t = {k: v for k, v in example.items() if k not in ["prompt"]}
        example_t["prompt"] = prompt
        sample.append(example_t)

    print(f"load dataset from `{args.dataset_name}` done. sample size: {len(ds)}")

    # Init model
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)
    print("init model done.")
    stop_tokens = ["</s>", "<|endoftext|>", "<|im_end|>"]
    max_tokens = 10000 # model.llm_engine.model_config.max_model_len if args.max_tokens is None else args.max_tokens
    if args.decoding_method == "greedy":
        print(f"WARNING! greedy decoding will force temperature=0, top_p=1!")
        sampling_params = SamplingParams(n=args.topk, temperature=0, top_p=1, max_tokens=max_tokens, stop=stop_tokens)
    elif args.decoding_method == "sampling":
        sampling_params = SamplingParams(n=args.topk, temperature=args.temperature, top_p=args.top_p, max_tokens=max_tokens, stop=stop_tokens)
    else:
        raise
    print(f"init sampling params done: {sampling_params}")

    # generate
    prompts = [example["prompt"] for example in sample]
    generations = model.generate(prompts, sampling_params)
    fw = open(save_file, "w", encoding='utf-8')
    num_total = 0
    num_skip_for_dup = 0
    for example, prompt, generation in zip(sample, prompts, generations):
        outputs = generation.outputs
        outputs_t = []
        touched_output = set()
        for output in outputs:
            num_total += 1
            output = output.text
            if output not in touched_output:
                outputs_t.append(output)
                touched_output.add(output)
            else:
                num_skip_for_dup += 1

        for output in outputs_t:
            example_t = {k: v for k, v in example.items()}
            example_t["q2mc_en_prompt"] = prompt
            example_t["en_math_model_coptpy_code"] = output
            if args.verbose:
                print("-" * 20 + "prompt" + "-" * 20)
                print(prompt)
                print("-" * 20 + "completion" + "-" * 20)
                print(output)
                print("-" * 80)

            dump = json.dumps(example_t, ensure_ascii=False)
            fw.write(dump + "\n")
    fw.close()
    print(f"num_total: {num_total}; num_skip_for_dup: {num_skip_for_dup}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)  # model path
    parser.add_argument("--dataset_name", type=str, default=None) 
    parser.add_argument("--dataset_split", type=str, default=None) 
    parser.add_argument("--save_dir", type=str, default=None)  
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # num_gpus
    parser.add_argument("--topk", type=int, default=1)  
    parser.add_argument("--temperature", type=float, default=0.7) 
    parser.add_argument("--top_p", type=float, default=0.95) 
    parser.add_argument("--max_tokens", type=int, default=None) 
    parser.add_argument("--decoding_method", type=str, default="greedy")  
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)