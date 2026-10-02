---
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: transformers
model_name: output_20260902_060947
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for output_20260902_060947

This model is a fine-tuned version of [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/ericnarteyonyame-university-of-virginia/Anti_deception_llama_8B_full/runs/h6c8rgwc) 



This model was trained with SFT.

### Framework versions

- TRL: 1.12.0
- Transformers: 5.16.1
- Pytorch: 2.11.0
- Datasets: 5.0.1
- Tokenizers: 0.23.1

## Citations



Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```