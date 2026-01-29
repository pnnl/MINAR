#!/bin/bash
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -p a100_shared
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=7
#SBATCH -o ./train_logs/%x-%j.out
#SBATCH -e ./train_logs/%x-%j.err
#SBATCH -J train_salsa-clrs_models

module load python
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate minar

# python train_salsa_clrs_distributed.py --epochs 100 --model GINE --lr 0.001 --eta 0.8 --batch_size 128 --devices 0 1 2 3 4 5 6 --seed 0
python train_salsa_clrs_distributed_l1.py --epochs 100 --model GINE --lr 0.001 --eta 0.0005 --weight_decay 0.1 --batch_size 32 --devices 0 1 2 3 4 5 6 --seed 0
# python train_salsa_clrs_distributed_l1_schedule.py --epochs 100 --model GINE --lr 0.001 --eta 0.005 --weight_decay 0.1 --batch_size 32 --devices 0 1 2 3 4 5 6 --seed 0