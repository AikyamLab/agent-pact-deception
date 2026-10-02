import os
import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

base = "google/gemma-3-27b-it"
adapter = "./output_20260908_030613/checkpoint-1200"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/gemma_lora/gemma3_27b/merged_ckpt1200"

os.makedirs(out, exist_ok=True)

model = Gemma3ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")

AutoProcessor.from_pretrained(base).save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged model saved to", out)

#PYTHONNOUSERSITE=1 python merge_adapter.py