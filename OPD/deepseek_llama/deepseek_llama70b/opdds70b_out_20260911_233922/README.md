---
library_name: peft
model_name: opdds70b_out_20260911_233922
tags:
- base_model:adapter:/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/merged_ds70b_lora
- distillation
- lora
- transformers
- trl
licence: license
base_model: /standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/merged_ds70b_lora
pipeline_tag: text-generation
---

# Model Card for opdds70b_out_20260911_233922

This model is a fine-tuned version of [None](https://huggingface.co/None).
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

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/ericnarteyonyame-university-of-virginia/deepseek_llama70b_anti_deception_opd/runs/fil777pe) 



This model was trained with Distillation, a method introduced in [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649).

### Framework versions

- PEFT 0.20.0
- TRL: 1.12.0
- Transformers: 5.16.1
- Pytorch: 2.11.0
- Datasets: 5.0.1
- Tokenizers: 0.23.1

## Citations

Cite Distillation as:

```bibtex
@inproceedings{agarwal2024on-policy,
    title        = {{On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes}},
    author       = {Rishabh Agarwal and Nino Vieillard and Yongchao Zhou and Piotr Stanczyk and Sabela Ramos Garea and Matthieu Geist and Olivier Bachem},
    year         = 2024,
    booktitle    = {The Twelfth International Conference on Learning Representations, {ICLR} 2024, Vienna, Austria, May 7-11, 2024},
    publisher    = {OpenReview.net},
    url          = {https://openreview.net/forum?id=3zKtaqxLhW},
}
```

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