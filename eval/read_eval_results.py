import os
import sys
import json

model_dir = sys.argv[1]
eval_results = {
    "model": model_dir, 
    "eval_results": []
}

dir_ents = os.listdir(model_dir)
for de in dir_ents:
    if de.startswith("eval."):
        eval_path = os.path.join(model_dir, de)
        eval_ents = os.listdir(eval_path)
        for ee in eval_ents:
            if "metrics" in ee:
                metrics_path = os.path.join(eval_path, ee)
                with open(metrics_path) as fd:
                    metrics = json.load(fd)
                    eval_results["eval_results"].append({"eval_task": de, "metrics": metrics})

datasets = ["NL4OPT", "MAMO_EasyLP", "MAMO_ComplexLP", "IndustryOR", "NLP4LP", "ComplexOR", "OptiBench", "ICMLTEST"]
for dataset in datasets:
    # print(eval_results)
    for eval_result in eval_results["eval_results"]:
        if f"eval.{dataset}.pass1" in eval_result["eval_task"]:
            if('pass@1' in eval_result['metrics']):
                print(f"eval.{dataset}.pass1, {eval_result['metrics']['pass@1']:.3f}")
            else:
                print(f"eval.{dataset}.pass1, {eval_result['metrics']}")

for dataset in datasets:
    for eval_result in eval_results["eval_results"]:
        if f"eval.{dataset}.pass8" in eval_result["eval_task"]:
            if('pass@8' in eval_result['metrics']):
                print(f"eval.{dataset}.pass8, {eval_result['metrics']['pass@8']:.3f}")
            else:
                print(f"eval.{dataset}.pass8, {eval_result['metrics']}")
        
