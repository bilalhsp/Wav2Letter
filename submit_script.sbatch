#!/bin/sh

#SBATCH --output=/home/ahmedb/projects/Wav2Letter/output.out

#SBATCH	-A partner

#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --time=23:55:00

#conda activate wav2letter

python run.py
