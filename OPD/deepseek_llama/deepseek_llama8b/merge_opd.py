#!/usr/bin/env python3
"""Merge an OPD LoRA adapter into the bf16 student it was trained on and save a standalone checkpoint.
Edit the three paths below, then: PYTHONNOUSERSITE=1 python merge_opd.py
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_8b/merged_ckpt1782"
ADAPTER = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama8b/opdds8b_out_20260910_203601"
OUT = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama8b/merged_opd_615"
DEVICE = "cuda"  # "cpu" if no card has room for the bf16 base (needs host RAM ~2 bytes/param)

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
device_map = {"": 0} if DEVICE == "cuda" else None

model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map=device_map)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
print(f"merged -> {OUT}")