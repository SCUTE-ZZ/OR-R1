import os
import sys
args = sys.argv
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

dir1 = args[1]
dir2 = args[2]
output_dir = args[3]
base_model = AutoModelForCausalLM.from_pretrained(dir1, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, dir2)
model = model.merge_and_unload()
model.save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(dir2, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)