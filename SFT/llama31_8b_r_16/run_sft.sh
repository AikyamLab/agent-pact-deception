#!/bin/bash -l
#SBATCH --job-name=anti_deception_sft_r16_llama8b
#SBATCH --output=logs/anti_deception_sft_r16_8b_%j.out
#SBATCH --error=logs/anti_deception_sft_r16_8b_%j.err


set -euo pipefail

module purge
module load gcc/11.4.0

# Reset Conda stack
unset CONDA_SHLVL CONDA_PREFIX CONDA_PROMPT_MODIFIER
for var in $(env | grep -E '^CONDA_PREFIX_[0-9]+=' | cut -d= -f1); do unset $var; done
unset CONDA_DEFAULT_ENV
source /home/reh6ed/miniconda3/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate monitoring_bw


export HF_HOME=/scratch/${USER}/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# wandb key 
export WANDB_API_KEY="your key here"

mkdir -p logs

echo "=============================================="
echo "Anti-Deception SFT (bf16 LoRA, r=32, +lm_head) — Llama-3.1-8B"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================="
nvidia-smi

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
PY

# ---- training parameters ----
uid="$(date +%Y%m%d_%H%M%S)"
base_model="meta-llama/Llama-3.1-8B-Instruct"
lr=1e-4
weight_decay=0.0
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_8b_r_16/output_${uid}"
mkdir -p "${output_dir}"

# bf16 LoRA on 1x A100 80GB. r=32/alpha=64, targets attention+MLP+lm_head.
# lm_head in the adapter requires --loss_type=nll (chunked_nll is unsupported there).

python anti_deceptive_sft.py \
    --model_name=${base_model} \
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA \
    --block_size=4096 \
    --val_fraction=0.05 \
    --lora_r=16 --lora_alpha=16 --lora_dropout=0.05 \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --num_train_epochs=5 \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --warmup_steps=0.03 \
    --lr_scheduler_type=cosine \
    --bf16=True \
    --loss_type=nll \
    --logging_steps=10 \
    --eval_strategy=steps \
    --eval_steps=100 \
    --save_strategy=steps \
    --save_steps=100 \
    --save_total_limit=3 \
    --load_best_model_at_end=True \
    --metric_for_best_model=eval_loss \
    --greater_is_better=False \
    --output_dir=${output_dir} \
    --report_to=wandb \
    --run_name=sft_8b_${uid} \
    --push_to_hub=false

echo "=============================================="
echo "Training completed. End: $(date)"
echo "Adapter + tokenizer saved to: ${output_dir}"
echo "=============================================="