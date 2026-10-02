#!/bin/bash -l
#SBATCH --job-name=lora_gemma12b
#SBATCH --output=logs/lora_gemma12b_%j.out
#SBATCH --error=logs/lora_gemma12b_%j.err


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
export WANDB_START_METHOD=thread
export WANDB_API_KEY="your key here"


mkdir -p logs

echo "=== LoRA Gemma-3-12B-IT | Job ${SLURM_JOB_ID:-local} | $(hostname) | $(date) ==="
nvidia-smi
python -c "import transformers, trl, torch, peft; print('torch', torch.__version__); print('transformers', transformers.__version__); print('trl', trl.__version__); print('peft', peft.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

python - <<'PY'
import transformers
major, minor, *rest = transformers.__version__.split(".")
if (int(major), int(minor.split("+")[0])) < (4, 50):
    raise SystemExit(
        f"transformers {transformers.__version__} is too old for Gemma 3. "
        "pip install -U 'transformers>=4.50'"
    )
print("transformers version OK")
PY

MODE="${1:-full}"
uid="$(date +%Y%m%d_%H%M%S)"
base_model="google/gemma-3-27b-it"
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/gemma_lora/output_${uid}"
mkdir -p "${output_dir}"

COMMON="
    --model_name=${base_model}
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA
    --block_size=4096
    --val_fraction=0.05
    --attn_implementation=sdpa
    --freeze_vision=True
    --lora_r=32
    --lora_alpha=64
    --lora_dropout=0.05
    --per_device_train_batch_size=1
    --per_device_eval_batch_size=1
    --gradient_accumulation_steps=16
    --gradient_checkpointing=True
    --learning_rate=1e-4
    --weight_decay=0.0
    --warmup_steps=0.03
    --lr_scheduler_type=cosine
    --bf16=True
    --logging_steps=10
    --output_dir=${output_dir}
    --report_to=wandb
    --run_name=lora_gemma12b_${uid}
    --push_to_hub=false
    --optim=adamw_torch_fused
    --max_grad_norm=1.0
    --dataloader_num_workers=4
    --dataloader_pin_memory=True
"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/anti_deceptive_sft_lora.py"

if [ "$MODE" = "smoke" ]; then
    python "${TRAIN_PY}" ${COMMON} \
        --num_train_epochs=1 --max_steps=10 --eval_strategy=no --save_strategy=no --report_to=none
else
    python "${TRAIN_PY}" ${COMMON} \
        --num_train_epochs=3 \
        --eval_strategy=steps --eval_steps=100 \
        --save_strategy=steps --save_steps=200 --save_total_limit=2 \
        --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False
fi

echo "=== Done $(date) | Adapter: ${output_dir} ==="