#!/bin/bash -l

#SBATCH --job-name=cnn_parallel_gpu
#SBATCH --output={logs_folder}/output_%j.txt
#SBATCH --error={logs_folder}/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

cd {current_dir}
eval "$(micromamba shell hook --shell bash)"

export PATH="/home/users/lgreco/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/home/users/lgreco/cuda/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME=/home/users/lgreco/cuda

micromamba activate cnn_parallel_gpu_env

python implementations/main_gpu.py {data_folder} {num_epochs}
