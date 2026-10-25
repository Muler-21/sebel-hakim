#!/bin/bash
# Full training run on a SLURM cluster: teacher → distilled student → evaluation → export.
# Edit the #SBATCH lines for your cluster (partition, account, GPU type), then:
#   sbatch hpc/run_all.sh
# Download and prepare data first, on a node WITH internet (see README):
#   python pipeline/download.py && python pipeline/prepare.py --workers 16
#
#SBATCH --job-name=agrodoc
#SBATCH --partition=gpu            # ← your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs runs reports

# module load cuda/12.1            # ← if your cluster uses modules
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agrodoc

W=${SLURM_CPUS_PER_TASK:-8}
TEACHER=${TEACHER:-convnextv2_tiny.fcmae_ft_in22k_in1k}
STUDENT=${STUDENT:-efficientnet_b0.ra_in1k}

echo "== 1/4 teacher: $TEACHER"
python pipeline/train.py --model "$TEACHER" --img 288 --epochs 30 --bs 64 --lr 4e-4 \
    --workers "$W" --out runs/teacher

echo "== 2/4 student: $STUDENT (distilled)"
python pipeline/train.py --model "$STUDENT" --img 224 --epochs 40 --bs 128 --lr 1e-3 \
    --teacher runs/teacher/best.pt --workers "$W" --out runs/student

echo "== 3/4 evaluate on the held-out test set"
python pipeline/evaluate.py runs/teacher/best.pt --workers "$W"
python pipeline/evaluate.py runs/student/best.pt --workers "$W"

echo "== 4/4 export for the app"
python pipeline/export.py runs/student/best.pt --quantize
python pipeline/evaluate.py app/model/model.onnx --workers "$W" --out reports/eval_onnx_fp32.json
python pipeline/evaluate.py app/model/model.int8.onnx --workers "$W" --out reports/eval_onnx_int8.json
echo "done → app/model/ (send me model.onnx, model.int8.onnx, labels.json and the reports/ folder)"
