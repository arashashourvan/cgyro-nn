#!/bin/bash
#SBATCH -A mp94
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00

#SBATCH -N 1                  # one node
#SBATCH -G 1                  # one GPU
#SBATCH -n 1                  # one MPI task
#SBATCH --cpus-per-task=16    # 16 CPU threads for this task

#SBATCH -J cgyro_nn_debug
#SBATCH -o logs/cgyro_debug_%j.out
#SBATCH -e logs/cgyro_debug_%j.err

module load pytorch/2.8.0
module list

cd /pscratch/sd/a/arash/cgyro-nn/code/3DCNN

export OMP_NUM_THREADS=64
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
# (optional, can help with fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun -n 1 python train_with_earlystop_rmse_deep.py \
    --data ./mnt/data/myrun_phi_flux.npz \
    --Tc 16 --horizons 1 5 10 \
    --epochs 30 \
    --batch 16 --lr 1e-4 \
    --weight_decay 1e-3 --dropout 0.2 \
    --device cuda \
    --log_dir ./mnt/data/myrun_logs_ot3 \
    --early_stop 10 \
    --phys_units 1\
    --time_split 1
