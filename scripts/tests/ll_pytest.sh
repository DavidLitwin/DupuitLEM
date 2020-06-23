#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:0:0
#SBATCH --partition=debug
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#### load and unload modules you may need
cd ~/data/dlitwin3/landlab
pytest landlab  > test_new.out
