#!/bin/sh

#SBATCH --output=/home/ahmedb/projects/Wav2Letter/putput.out

#SBATCH	-A partner

#SBATCH --nodes=1 --gpus-per-node=1
#SBATCH --time=23:55:00

module purge
module load use.own
module load conda-env/wav2letter_lightning-py3.8.5
hostname

python run.py
