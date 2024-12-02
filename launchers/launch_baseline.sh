#!/bin/bash -l

#SBATCH --job-name=cnn_parallel
#SBATCH --output={logs_folder}/output_%j.txt
#SBATCH --error={logs_folder}/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

cd {current_dir}
eval "$(micromamba shell hook --shell bash)"
micromamba activate cnn_parallel_env

python implementations/baseline.py {data_folder} {num_epochs}