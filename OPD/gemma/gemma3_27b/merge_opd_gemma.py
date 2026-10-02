import os
import torch
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/gemma_lora/gemma3_27b/merged_ckpt1200"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/gemma/gemma3_27b/opdg27b_out_20260911_001306"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/gemma/gemma3_27b/merged_opd_615"

os.makedirs(out, exist_ok=True)
model = Gemma3ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoProcessor.from_pretrained(base).save_pretrained(out)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)


#PYTHONNOUSERSITE=1 python merge_opd_gemma.py