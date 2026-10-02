import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.3-70B-Instruct"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b_r_16/output_20260908_225455/checkpoint-1200"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b_r_16/merged_ckpt1200"

model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("merged model saved to", out)