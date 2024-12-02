# Parallel CNN Project

This project uses parallel computing techniques to train Convolutional Neural Networks (CNNs) efficiently.

## Setup Instructions

### Step 0: Install Micromamba

To install Micromamba, follow the instructions below:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```


### 1. Create a Micromamba Environment

First, create a new environment using Micromamba:

```bash
micromamba create -n cnn_parallel_env python=3.12
micromamba activate cnn_parallel_env
```

### 2. Allocate a Node

Allocate a node for your computations. This step may vary depending on your cluster management system. For example, using SLURM:

```bash
 srun --partition=gpu --gres=gpu:1 --ntasks-per-node=4 --time=01:00:00  --pty bash
 ```

### 3. Load the OpenMPI Module

Load the OpenMPI module to enable MPI support:

```bash
module load mpi/OpenMPI/4.0.5-GCC-10.2.0
```

### 4. Install `mpi4py` and Other Requirements

Finally, install the `mpi4py` package and other dependencies listed in `requirements.txt`:

```bash
pip install mpi4py
pip install -r requirements.txt
```

### 5. Optionally install cuda

If you are using a GPU, you may need to install the CUDA toolkit. You can download the CUDA toolkit from the NVIDIA website.

```bash
pip install -U "jax[cuda12]"
```

You are now ready to run the parallel CNN training scripts.