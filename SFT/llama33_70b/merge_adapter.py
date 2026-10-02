import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.3-70B-Instruct"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b/output_20260905_110317/checkpoint-1100"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b/merged_best"

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()          # fold LoRA into base weights
model.save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged model saved to", out)