import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

base = "google/gemma-3-12b-it"
adapter = "./output_20260907_133443/checkpoint-1000"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/gemma_lora/gemma3_12b/merged_best"

# 12B is on GPU 0; merge on GPU 1 (or use device_map="cpu" if both cards are busy)
model = Gemma3ForConditionalGeneration.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  # this process should see only GPU 1 via CUDA_VISIBLE_DEVICES=1
)
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out)

tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(out)
print("merged model saved to", out)


# PYTHONNOUSERSITE=1 python merge_adapter.py