#!/bin/bash -l
#SBATCH --job-name=opd_ds70b
#SBATCH --output=/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama70b/logs/opd_ds70b_%j.out
#SBATCH --error=/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama70b/logs/opd_ds70b_%j.err



# Usage:  bash run_opd_ds70b.sh
# Outside the batch job this creates the log directory and submits itself; inside the job it runs training.
WORKDIR=/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/deepseek_llama/deepseek_llama70b
if [[ "${SLURM_JOB_NAME:-}" != "opd_ds70b" ]]; then
    mkdir -p "${WORKDIR}/logs"
    cd "${WORKDIR}"
    exec sbatch "$0"
fi

set -euo pipefail

module purge
module load gcc/11.4.0

unset CONDA_SHLVL CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_DEFAULT_ENV
for var in $(env | grep -E '^CONDA_PREFIX_[0-9]+=' | cut -d= -f1 || true); do unset "$var"; done
source /home/reh6ed/miniconda3/etc/profile.d/conda.sh
export CONDA_NO_PLUGINS=true
conda activate monitoring_bw

cd "${WORKDIR}"
mkdir -p logs

export HF_HOME=/scratch/${USER}/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=deepseek_llama70b_anti_deception_opd
export WANDB_API_KEY="the key here"

STUDENT=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/deepseek_lora/deepseek_llama_70b/merged_ds70b_lora
TEACHER=/standard/AikyamLab/eric/Multi_Agent_Monitoring/SFT/DEEPSEEK/output_20260907_211949/checkpoint-1000
DATA=/standard/AikyamLab/eric/Multi_Agent_Monitoring/OPD/honesty_opd_dataset.jsonl

# One epoch: (9932 prompts - 96 eval) / (batch 1 * grad_accum 16) = 615 steps
MAX_STEPS="${MAX_STEPS:-615}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_OVERRIDE:-${WORKDIR}/opdds70b_out_${STAMP}}"
RESUME="${RESUME:-}"   # to resume: RESUME=<out>/checkpoint-N OUT_OVERRIDE=<out> bash run_opd_ds70b.sh
mkdir -p "${OUT}"

echo "job=${SLURM_JOB_ID} host=$(hostname) start=$(date) out=${OUT} max_steps=${MAX_STEPS}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

python opd_distill.py \
    --student "${STUDENT}" \
    --teacher "${TEACHER}" \
    --data "${DATA}" \
    --out "${OUT}" \
    --max-steps "${MAX_STEPS}" \
    --lr 2e-5 \
    --beta 1.0 \
    --temperature 1.0 \
    --max-completion-length 1024 \
    --batch 1 --grad-accum 16 --student-4bit --teacher-8bit \
    --lora-r 16 --lora-alpha 32 \
    --eval-size 96 --eval-steps 50 --save-steps 50 \
    --report-to wandb \
    --run-name "opd_ds70b_${STAMP}" \
    ${RESUME:+--resume-from "${RESUME}"}

echo "done=$(date) out=${OUT}"