import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/merged_ds70b_lora"
ADAPTER = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama70b/opdds70b_out_20260911_233922"
MERGED  = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama70b/merged_opd_615"

model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(MERGED, safe_serialization=True)
AutoTokenizer.from_pretrained(STUDENT).save_pretrained(MERGED)
print("merged ->", MERGED)


#PYTHONNOUSERSITE=1 python merge_adapter.py