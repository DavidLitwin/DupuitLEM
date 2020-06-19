#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=1:0:0
#SBATCH --partition=debug
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#### load and unload modules you may need
module load python
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev
cd ~/data/dlitwin3/landlab
pytest landlab  > test_new.out
