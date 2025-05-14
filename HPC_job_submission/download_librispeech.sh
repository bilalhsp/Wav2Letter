#!/bin/sh

# --output=./result_download_dataset.out

#SBATCH	-A training

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
# --mem=0
#SBATCH --time=1-04:00:00
 
hostname
echo $CUDA_VISIBLE_DEVICES
module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
# module load conda-env/w2l_cortex-py3.8.5
# module load conda-env/wav2letter_lightning-py3.8.5


# module load conda-env/wav2letter_pretrained-py3.8.5
#module load NCCL/2.4.7-1-cuda.10.0

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=^docker,eth,lo
# export NCCL_SOCKET_IFNAME=^lo,eth,em,en,docker0,ib
export NCCL_SOCKET_IFNAME=^lo,eth,en,docker0


python ../scripts/download_LibriSpeech.py $@
#srun python run_lightning.py
# python run_lightning.py
