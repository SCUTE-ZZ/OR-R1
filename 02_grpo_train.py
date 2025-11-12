# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import GRPOConfig, GRPOTrainer, ModelConfig, get_peft_config, setup_chat_format
from peft import PeftModel, PeftConfig
import os
import tempfile
import subprocess
import torch
from datetime import datetime


@dataclass
class DataConfig:
    dataset_path: str = ""

TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

def compile_script(script_content, timeout=10):
    # Ensure the target directory exists
    target_dir = './eval_execute'
    os.makedirs(target_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.py', dir=target_dir) as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(script_content.encode())
    try:
        # Running the Lean3 compiler on the temporary script file with a time limit
        process = subprocess.run(['python', tmp_file_name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
        # If compilation is successful, return the output and a success message
        execution_result = process.stdout
        execution_best_solution_start_pos = execution_result.find("Just print the best solution:")
        if execution_best_solution_start_pos != -1:
            execution_best_solution = execution_result[execution_best_solution_start_pos:].replace("Just print the best solution:", "").strip()
            execution_best_solution_end_pos = execution_best_solution.find("\n")
            if execution_best_solution_end_pos != -1:
                execution_best_solution = execution_best_solution[:execution_best_solution_end_pos]
            execution_state = "Execution Successful and Best Solution Found"
        else:
            if "No Best Solution" in execution_result:
                execution_best_solution = "No Best Solution"
                execution_state = "Execution Successful but No Best Solution Found"
            else:
                execution_best_solution = None
                execution_state = "Execution Suceessful but Out of Expectation"
    except subprocess.TimeoutExpired as e:
        # If compilation time exceeds the limit, kill the process and return a failure message
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = "Execution Failed: Timeout"
    except subprocess.CalledProcessError as e:
        # If compilation fails for other reasons, return the error output
        execution_result = e.stdout
        execution_best_solution = None
        execution_state = f"Execution Failed: {e.stdout}"
    finally:
        # Clean up the temporary file
        os.remove(tmp_file_name)

    execution_output = {
        "execution_result": execution_result,
        "execution_best_solution": execution_best_solution, 
        "execution_state": execution_state
    }
    return execution_output

def run_code(output):
    ADD_SCRIPT = '\nif model.status == COPT.OPTIMAL:\n    print(f"Just print the best solution: {model.objval}")\nelse:\n    print("No Best Solution")'

    # print(output)
    start = output.find("```python")
    if start == -1:
        return None
    end = output.find("```", start + 9)
    script = output[start:end].replace("```python", "") + ADD_SCRIPT
    execution_output = compile_script(script)
    return execution_output

def reward_with_reference(completions, **kwargs):

    format_rewards = []
    valid_code_rewards = []
    answer_rewards = []

    True_count = 0
    records = []
    gpu_id = torch.cuda.current_device()

    prediction_answers = []
    for i in range(len(completions)):
        format_reward = 0.0
        formats = ["## Mathematical Model:", "## Decision Variables:", "## Objective Function:", "## Constraints:", "## Python Code Solution Using `coptpy`:", "```python"]
        for format in formats:
            if completions[i].find(format) != -1:
                format_reward += 1
        format_rewards.append(format_reward/len(formats))



    for i in range(len(completions)):
        valid_code_reward = 0.0
        prediction_execution_output = run_code(completions[i])
        if( prediction_execution_output is None):
            prediction_answers.append(None)
        else:
            prediction_answers.append(prediction_execution_output['execution_best_solution'])
            if(prediction_execution_output['execution_best_solution'] is not None):
                valid_code_reward = 1
        valid_code_rewards.append(valid_code_reward)

    volting_answers = []
    for i in range(0, len(completions), 8):
        prediction_answer_dict = {}
        for j in range(8):
            if i + j >= len(prediction_answers):
                break
            if prediction_answers[i + j] == None or prediction_answers[i + j] == "No Best Solution":
                continue
            try:
                if(int(float(prediction_answers[i + j])) not in prediction_answer_dict):
                    prediction_answer_dict[int(float(prediction_answers[i + j]))] = 0
                prediction_answer_dict[int(float(prediction_answers[i + j]))] += 1
            except ValueError:
                # If conversion to int fails, skip this answer
                continue
        
        volting_answer = None
        max_count = 1
        for key, count in prediction_answer_dict.items():
            if count > max_count:
                max_count = count
                volting_answer = key

        for j in range(8):
            volting_answers.append(volting_answer)

    for i in range(len(completions)):
        answer_reward = 0.0
        gt_answer = kwargs['answer'][i]
        volting_answer = volting_answers[i]
        prediction_answer = prediction_answers[i]

        try:
            if(prediction_answer == None or prediction_answer == "No Best Solution"):
                answer_reward = 0.0
            elif(volting_answer == None):
                answer_reward = 0.0
            else:
                if int(float(prediction_answer)) == volting_answer:
                    answer_reward = 1.0
                    True_count += 1
                else:
                    answer_reward = 0.0
        except ValueError:
            answer_reward = 0
        
        answer_rewards.append(answer_reward)
        records.append([gpu_id, i, prediction_answer, volting_answer, gt_answer, format_rewards[i], valid_code_rewards[i], answer_rewards[i]])

    print(f"True Rate: {True_count/len(completions):.2f}, Mean Format Reward: {sum(format_rewards)/len(format_rewards):.2f}, Mean Valid Code Reward: {sum(valid_code_rewards)/len(valid_code_rewards):.2f}, Mean Answer Reward: {sum(answer_rewards)/len(answer_rewards):.2f}")
    # Save the records to a file
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Write the records to a CSV file
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    with open("./logs/records.csv", "a") as f:
        f.writelines("\n")
        for record in records:
            f.writelines(current_time + "," + ",".join(str(i) for i in record) + "\n")

    torch.cuda.empty_cache()
    rewards = []
    for i in range(len(completions)):
        reward = 0.0
        reward += format_rewards[i]
        reward += valid_code_rewards[i]
        reward += answer_rewards[i]
        rewards.append(reward)
    return rewards

parser = HfArgumentParser((GRPOConfig, ModelConfig, DataConfig))
grpo_args, model_args, data_args = parser.parse_args_into_dataclasses()

model_path = model_args.model_name_or_path
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_path , trust_remote_code=True
)
# Apply chat template for GRPO format
def format_dataset(example):
    prompt = TEMPLATE_q2mc_en.replace("{Question}", example["question"].strip()).strip()
    example["prompt"] = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
    return example

dataset = load_dataset('json', data_files=data_args.dataset_path)
formatted_dataset = dataset.map(format_dataset)

# Initialize the GRPO trainer
grpo_trainer = GRPOTrainer(
    model,
    args=grpo_args,
    reward_funcs=reward_with_reference,
    train_dataset=formatted_dataset["train"],
    processing_class=tokenizer,
    peft_config=get_peft_config(model_args),
)

grpo_trainer.train()
grpo_trainer.save_model(grpo_args.output_dir)
