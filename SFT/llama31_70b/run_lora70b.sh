#!/bin/bash -l
#SBATCH --job-name=anti_deception_sft_llama3170b
#SBATCH --output=logs/anti_deception_sft_70b_%j.out
#SBATCH --error=logs/anti_deception_sft_70b_%j.err


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

# wandb key (hardcoded as requested)
export WANDB_API_KEY="I will put the key here"

mkdir -p logs

echo "=============================================="
echo "Anti-Deception SFT (bf16 LoRA, r=32, +lm_head) — Llama-3.1-70B"
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

# ---- MODE: pass "smoke" as first arg for a 10-step memory test, else full run ----
MODE="${1:-full}"

uid="$(date +%Y%m%d_%H%M%S)"
base_model="meta-llama/Llama-3.1-70B-Instruct"
lr=1e-4
weight_decay=0.0
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/llama31_70b/output_${uid}"
mkdir -p "${output_dir}"

# bf16 LoRA across 4x A100 80GB via device_map="auto" (set in the script).
# NO DeepSpeed/ZeRO-3: device_map shards the 70B; LoRA trains only small adapters.
# per_device batch = 1 (70B bf16 is memory-heavy); grad-accum 16 -> effective batch 16.
# lm_head in the adapter requires --loss_type=nll.

COMMON_ARGS="
    --model_name=${base_model}
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA
    --block_size=4096
    --val_fraction=0.05
    --lora_r=16 --lora_alpha=16 --lora_dropout=0.05
    --per_device_train_batch_size=1
    --per_device_eval_batch_size=1
    --gradient_accumulation_steps=16
    --gradient_checkpointing=True
    --learning_rate=${lr}
    --weight_decay=${weight_decay}
    --warmup_steps=0.03
    --lr_scheduler_type=cosine
    --bf16=True
    --loss_type=nll
    --logging_steps=10
    --output_dir=${output_dir}
    --report_to=wandb
    --run_name=sft_70b_${uid}
    --push_to_hub=false
"

if [ "$MODE" = "smoke" ]; then
    echo ">>> SMOKE TEST: 10 steps, no eval/save (memory check)"
    python anti_deceptive_sft.py ${COMMON_ARGS} \
        --num_train_epochs=1 \
        --max_steps=10 \
        --eval_strategy=no \
        --save_strategy=no \
        --report_to=none
else
    echo ">>> FULL RUN: 5 epochs, best checkpoint on eval_loss"
    python anti_deceptive_sft.py ${COMMON_ARGS} \
        --num_train_epochs=5 \
        --eval_strategy=steps \
        --eval_steps=100 \
        --save_strategy=steps \
        --save_steps=100 \
        --save_total_limit=3 \
        --load_best_model_at_end=True \
        --metric_for_best_model=eval_loss \
        --greater_is_better=False
fi

echo "=============================================="
echo "Training completed. End: $(date)"
echo "Adapter + tokenizer saved to: ${output_dir}"
echo "=============================================="