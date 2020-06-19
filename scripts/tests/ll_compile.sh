#!/bin/bash
#SBATCH --job-name=build
#SBATCH --time=1:0:0
#SBATCH --partition=debug
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#### load and unload modules you may need
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
cd ~/data/dlitwin3/landlab
conda env create --file=environment-dev.yml
conda activate
conda activate landlab_dev
python setup.py develop
pytest landlab  > test.out
