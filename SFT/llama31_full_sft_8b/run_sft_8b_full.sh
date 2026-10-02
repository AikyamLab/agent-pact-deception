#!/bin/bash -l
#SBATCH --job-name=anti_deception_sft_8b_full
#SBATCH --output=logs/sft_8b_full_%j.out
#SBATCH --error=logs/sft_8b_full_%j.err


set -euo pipefail

module purge
module load gcc/11.4.0

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

# wandb key (hardcoded)
export WANDB_API_KEY="your key here"
export WANDB_START_METHOD=thread


mkdir -p logs

echo "=============================================="
echo "Anti-Deception FULL FINE-TUNING (ZeRO-3) — Llama-3.1-8B"
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

uid="$(date +%Y%m%d_%H%M%S)"
base_model="meta-llama/Llama-3.1-8B-Instruct"
lr=1e-5                 # FULL FT learning rate (NOT the 1e-4 LoRA rate)
weight_decay=0.0
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_8b_full/output_${uid}"
mkdir -p "${output_dir}"

# FULL fine-tuning across 2x RTX PRO 6000 with DeepSpeed ZeRO-3.
# ZeRO-3 shards weights + grads + optimizer states across the 2 cards.
# effective batch = per_device(1) * grad_accum(8) * num_gpus(2) = 16

python -m accelerate.commands.launch \
    --config_file deepspeed_zero3.yaml \
    --num_processes 2 \
    --num_machines 1 \
    anti_deceptive_sft_full.py \
    --model_name=${base_model} \
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA \
    --block_size=4096 \
    --val_fraction=0.05 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --num_train_epochs=3 \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --warmup_steps=0.03 \
    --lr_scheduler_type=cosine \
    --bf16=True \
    --logging_steps=10 \
    --eval_strategy=steps \
    --eval_steps=100 \
    --save_strategy=steps \
    --save_steps=200 \
    --save_total_limit=2 \
    --load_best_model_at_end=True \
    --metric_for_best_model=eval_loss \
    --greater_is_better=False \
    --output_dir=${output_dir} \
    --report_to=wandb \
    --run_name=sft_8b_full_${uid} \
    --push_to_hub=false

echo "=============================================="
echo "Training completed. End: $(date)"
echo "Full model saved to: ${output_dir}"
echo "=============================================="