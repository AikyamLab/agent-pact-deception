import os, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_8b/output_20260908_032132/checkpoint-1782"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_8b/merged_ckpt1782"
base = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"   # must match base_model_name_or_path in adapter_config.json

model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
print("merged model saved to", out)



# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# base = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# adapter = "./output_20260908_032132/checkpoint-1200"
# out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_8b/merged_best"

# model = AutoModelForCausalLM.from_pretrained(
#     base,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )
# model = PeftModel.from_pretrained(model, adapter)
# model = model.merge_and_unload()
# model.save_pretrained(out)
# AutoTokenizer.from_pretrained(base).save_pretrained(out)
# print("merged model saved to", out)