#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH -p slurm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -J generate_salsa-clrs_data

module load python
source /share/apps/python/miniconda25.5.1/etc/profile.d/conda.sh
conda activate minar

python generate_new_salsa_clrs_data.py