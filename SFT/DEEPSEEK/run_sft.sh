#!/bin/bash -l
#SBATCH --job-name=full_ft_deepseek8b
#SBATCH --output=logs/full_deepseek8b_%j.out
#SBATCH --error=logs/full_deepseek8b_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:
#SBATCH --constraint=
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=48:00:00
#SBATCH --account=..........
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=reh6ed@virginia.edu

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
export WANDB_API_KEY="your key here"
export WANDB_START_METHOD=thread

mkdir -p logs
echo "=== FULL FT DeepSeek-Llama-8B (ZeRO-3) | Job ${SLURM_JOB_ID} | $(hostname) | $(date) ==="
nvidia-smi

MODE="${1:-full}"
uid="$(date +%Y%m%d_%H%M%S)"
base_model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
output_dir="/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DEEPSEEK/output_${uid}"
mkdir -p "${output_dir}"

COMMON="
    --model_name=${base_model}
    --data_dir=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DATA
    --block_size=4096
    --val_fraction=0.05
    --per_device_train_batch_size=1
    --per_device_eval_batch_size=1
    --gradient_accumulation_steps=8
    --gradient_checkpointing=True
    --learning_rate=1e-5
    --weight_decay=0.0
    --warmup_steps=0.03
    --lr_scheduler_type=cosine
    --bf16=True
    --logging_steps=10
    --output_dir=${output_dir}
    --report_to=wandb
    --run_name=full_deepseek8b_${uid}
    --push_to_hub=false
"

if [ "$MODE" = "smoke" ]; then
    python -m accelerate.commands.launch --config_file deepspeed_zero3.yaml \
        --num_processes 2 --num_machines 1 anti_deceptive_sft_full.py ${COMMON} \
        --num_train_epochs=1 --max_steps=10 --eval_strategy=no --save_strategy=no --report_to=none
else
    python -m accelerate.commands.launch --config_file deepspeed_zero3.yaml \
        --num_processes 2 --num_machines 1 anti_deceptive_sft_full.py ${COMMON} \
        --num_train_epochs=3 \
        --eval_strategy=steps --eval_steps=100 \
        --save_strategy=steps --save_steps=200 --save_total_limit=2 \
        --load_best_model_at_end=True --metric_for_best_model=eval_loss --greater_is_better=False
fi
echo "=== Done $(date) | Output: ${output_dir} ==="