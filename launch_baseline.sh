#!/bin/bash -l

#SBATCH --job-name=cnn_parallel
#SBATCH --output={logs_folder}/output_%j.txt
#SBATCH --error={logs_folder}/error_%j.txt
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --ntasks-per-socket=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

cd {current_dir}
eval "$(micromamba shell hook --shell bash)"
micromamba activate cnn_parallel_env

python baseline.py {data_folder} {num_epochs}