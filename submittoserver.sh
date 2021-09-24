#!/bin/bash
#SBATCH -J 092401   # ======
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu1
#SBATCH -c 2
#SBATCH -N 1
#SBATCH -w node6
#SBATCH -o train092401.out    # ===== trainMMDDii.out, testMMDDi.out

echo "Submitted from:"$SLURM_SUBMIT_DIR" on node:"$SLURM_SUBMIT_HOST
echo "Running on node "$SLURM_JOB_NODELIST
echo "Allocate Gpu Units:"$CUDA_VISIBLE_DEVICES

#nvidia-smi
# python -m script.[script_file_name] (no .py)
python -m train_semi
