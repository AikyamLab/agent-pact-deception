import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/output_lora_ds70b_20260910_020915/checkpoint-2200"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/merged_ds70b_lora"

model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("merged model saved to", out)


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# base = "meta-llama/Llama-3.3-70B-Instruct"
# adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b/output_20260828_070523/checkpoint-1000"
# out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b/merged_best"

# model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
# model = PeftModel.from_pretrained(model, adapter)
# model = model.merge_and_unload()          # fold LoRA into base weights
# model.save_pretrained(out)
# AutoTokenizer.from_pretrained(base).save_pretrained(out)
# print("merged model saved to", out)