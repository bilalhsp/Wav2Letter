#!/bin/sh

#SBATCH --output=./result_evaluation.out

#SBATCH	-A standby  

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=4:00:00
 
hostname
echo $CUDA_VISIBLE_DEVICES
# module purge
module load anaconda/2020.11-py38
module load cuda/11.2.0
module load use.own
module load conda-env/w2l_cortex-py3.8.5

python ../scripts/evaluate.py
