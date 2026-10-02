import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

STUDENT = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama33_70b_r_16/merged_ckpt1200"
ADAPTER = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/llama/llama33_70b/opdl33_out_20260911_233436"
MERGED  = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/llama/llama33_70b/merged_opd_615"

model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(MERGED, safe_serialization=True)
AutoTokenizer.from_pretrained(STUDENT).save_pretrained(MERGED)
print("merged ->", MERGED)


#PYTHONNOUSERSITE=1 python merge_adapter.py