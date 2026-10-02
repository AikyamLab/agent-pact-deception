import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_8b_r_16/output_20260904_132411/checkpoint-1700"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_8b_r_16/merged_best"

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()          # fold LoRA into base weights
model.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged model saved to", out)


# PYTHONNOUSERSITE=1 python merge_adapter.py