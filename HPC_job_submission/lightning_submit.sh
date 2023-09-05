#!/bin/sh

#SBATCH --output=./result_lightning.out

#SBATCH	-A standby    
# --constraint=E|F

#|G|I|J|K


#SBATCH --nodes=2 
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH --time=4:00:00
 
hostname
echo $CUDA_VISIBLE_DEVICES
# module purge
module load anaconda/2020.11-py38
module load cuda/11.2.0
module load use.own
module load conda-env/w2l_cortex-py3.8.5
# module load conda-env/wav2letter-py3.8.5
# module load learning/conda-2020.11-py38-gpu
# module load conda-env/wav2letter_lightning-py3.8.5

# module load conda-env/wav2letter_pretrained-py3.8.5
#module load NCCL/2.4.7-1-cuda.10.0

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=^docker,eth,lo
# export NCCL_SOCKET_IFNAME=^lo,eth,em,en,docker0,ib
# export NCCL_SOCKET_IFNAME=^lo,eth,en,docker0
# export CUDA_VISIBLE_DEVICES=2

# os.environ[""] = "4"


srun python ../scripts/run_lightning.py
# python run_lightning.py
