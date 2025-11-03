#!/bin/bash
#SBATCH -A mp94                     # Allocation name
#SBATCH -C gpu                      # Request GPU node(s)
#SBATCH -q debug                    # Queue type
#SBATCH -t 00:30:00                 # Walltime (4 hours)
#SBATCH -N 1                        # One node
#SBATCH -G 1                        # One GPU
#SBATCH -c 16                       # 16 CPU cores
#SBATCH -J cgyro_nn_train           # Job name
#SBATCH -o logs/cgyro_train_%j.out  # Stdout
#SBATCH -e logs/cgyro_train_%j.err  # Stderr
#SBATCH --mail-type=END,FAIL        # Optional notifications
#SBATCH --mail-user=arash.ashourvan@gmail.com  # Change this!

# ---- Load modules ----
module load pytorch/2.1.2
module load cudnn
module load gcc/11.2.0

# ---- Activate your conda env ----
source ~/.bashrc
conda activate cgyro-nn

# ---- Set environment variables ----
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# ---- Run training ----
srun python train_with_earlystop_rmse_deep.py \
    --data ./mnt/data/myrun_phi_flux.npz \
    --Tc 64 --horizons 1 5 10 \
    --epochs 40 \
    --batch 8 --lr 1e-3 \
    --weight_decay 1e-4 --dropout 0.2 \
    --device cuda \
    --log_dir ./mnt/data/myrun_logs_deep_gpu \
    --early_stop 10 \
    --phys_units 1

