#!/bin/bash
#SBATCH -J riborthrus_ft
#SBATCH -o logs/riborthrus_ft_%j.out
#SBATCH -e logs/riborthrus_ft_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -D /work/11244/j7p16/vista/RibOrthrus

#SBATCH --mail-user=johncapocyan@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate orthrus

python scripts/train.py
