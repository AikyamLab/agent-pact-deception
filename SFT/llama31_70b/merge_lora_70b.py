"""
Merge the Llama 3.1 70B LoRA adapter (checkpoint-2376) into the base model.
Run:  python merge_lora_70b.py
Needs both GPUs free (the base is 141 GB in bf16, split across them). Host RAM stays small.
"""

import json
import os

import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "meta-llama/Llama-3.1-70B-Instruct"
adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_70b/output_lora70b_20260909_182548/checkpoint-2376"
out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_70b/merged_llama70b_lora"

# ---- merge -----------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()  # fold LoRA into the base weights
os.makedirs(out, exist_ok=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
del model
print("merged model saved to", out)

# ---- verify: one layer of the written files must equal base + adapter exactly ----------------
key = "model.layers.0.self_attn.q_proj.weight"
weight_map = json.load(open(f"{out}/model.safetensors.index.json"))["weight_map"]
with safe_open(f"{out}/{weight_map[key]}", "pt") as f:
    merged = f.get_tensor(key).float()
base_map = json.load(open(hf_hub_download(base, "model.safetensors.index.json")))["weight_map"]
with safe_open(hf_hub_download(base, base_map[key]), "pt") as f:
    W = f.get_tensor(key).float()
cfg = json.load(open(f"{adapter}/adapter_config.json"))
scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
a = load_file(f"{adapter}/adapter_model.safetensors")
A = a[f"base_model.model.{key[:-7]}.lora_A.weight"].float()
B = a[f"base_model.model.{key[:-7]}.lora_B.weight"].float()
expected = (W + scale * (B @ A)).to(torch.bfloat16).float()
diff = (merged - expected).abs().max().item()
print(f"verification on {key}: max|diff| vs base+adapter = {diff:.4g}", "-> OK" if diff == 0 else "-> MISMATCH, do not use")


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel

# base = "meta-llama/Llama-3.1-70B-Instruct"
# adapter = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_70b/output_20260828_070632/checkpoint-1000"
# out = "/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_70b/merged_best"

# model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
# model = PeftModel.from_pretrained(model, adapter)
# model = model.merge_and_unload()          # fold LoRA into base weights
# model.save_pretrained(out)
# AutoTokenizer.from_pretrained(base).save_pretrained(out)
# print("merged model saved to", out)

#PYTHONNOUSERSITE=1 python merge_adapter.py