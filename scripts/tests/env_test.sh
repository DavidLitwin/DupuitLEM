#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:0:0
#SBATCH --partition=debug
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#### load and unload modules you may need
conda info --envs > env.out
which python > python_version.out
