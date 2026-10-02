import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # keep off the card running the 27B

STUDENT = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_8b_r_16/merged_best"
ADAPTER = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/llama/llama31_8b/opd_out_20260911_112148"
MERGED = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/llama/llama31_8b/merged_opd_615"

model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(MERGED, safe_serialization=True)
AutoTokenizer.from_pretrained(STUDENT).save_pretrained(MERGED)
print("merged ->", MERGED)