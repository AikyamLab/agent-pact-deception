#!/bin/bash -l
#SBATCH --job-name=full_ft_gemma12b
#SBATCH --output=logs/full_gemma12b_%j.out
#SBATCH --error=logs/full_gemma12b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --account=.........
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=...............

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


export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p logs

echo "=== FULL FT Gemma-3-12B-IT (ZeRO-3) | Job ${SLURM_JOB_ID:-local} | $(hostname) | $(date) ==="
nvidia-smi
python -c "import transformers, trl, torch; print('torch', torch.__version__); print('transformers', transformers.__version__); print('trl', trl.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# Gemma 3 needs transformers >= 4.50
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
base_model="google/gemma-3-12b-it"
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/GEMMA/output_${uid}"
mkdir -p "${output_dir}"

# Matches TrainingConfig + SFTConfig in anti_deceptive_sft_full.py
# Use warmup_ratio, not warmup_steps=0.03
COMMON="
    --model_name=${base_model}
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA
    --block_size=4096
    --val_fraction=0.05
    --attn_implementation=sdpa
    --freeze_vision=True
    --per_device_train_batch_size=1
    --per_device_eval_batch_size=1
    --gradient_accumulation_steps=4
    --gradient_checkpointing=True
    --learning_rate=1e-5
    --weight_decay=0.0
    --warmup_ratio=0.03
    --lr_scheduler_type=cosine
    --bf16=True
    --logging_steps=10
    --output_dir=${output_dir}
    --report_to=wandb
    --run_name=full_gemma12b_${uid}
    --push_to_hub=false
    --optim=adamw_torch_fused
    --max_grad_norm=1.0
    --dataloader_num_workers=4
    --dataloader_pin_memory=True
"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/anti_deceptive_sft_full.py"
if [ ! -f "${TRAIN_PY}" ]; then
    TRAIN_PY="anti_deceptive_sft_full.py"
fi
DS_CFG="${SCRIPT_DIR}/deepspeed_zero3.yaml"
if [ ! -f "${DS_CFG}" ]; then
    DS_CFG="deepspeed_zero3.yaml"
fi

if [ "$MODE" = "smoke" ]; then
    python -m accelerate.commands.launch --config_file "${DS_CFG}" \
        --num_processes 4 --num_machines 1 "${TRAIN_PY}" ${COMMON} \
        --num_train_epochs=1 --max_steps=10 --eval_strategy=no --save_strategy=no --report_to=none
else
    python -m accelerate.commands.launch --config_file "${DS_CFG}" \
        --num_processes 4 --num_machines 1 "${TRAIN_PY}" ${COMMON} \
        --num_train_epochs=3 \
        --eval_strategy=steps --eval_steps=100 \
        --save_strategy=steps --save_steps=200 --save_total_limit=2 \
        --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False
fi

echo "=== Done $(date) | Output: ${output_dir} ==="