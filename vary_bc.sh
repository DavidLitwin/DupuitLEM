#!/bin/bash
#SBATCH --job-name=DupuitLEMTest
#SBATCH --time=24:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#### load and unload modules you may need
module load git
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev
mkdir ~/data/dlitwin3/DupuitLEMResults/vary_bc
mkdir ~/data/dlitwin3/DupuitLEMResults/vary_bc/data
cd ~/data/dlitwin3/DupuitLEMScripts
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/vary_bc/script_id.txt
cd ~/data/dlitwin3/landlab
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/vary_bc/gdp_id.txt
cp ~/data/dlitwin3/DupuitLEMScripts/vary_bc.py ~/data/dlitwin3/DupuitLEMResults/vary_bc
cd ~/data/dlitwin3/DupuitLEMResults/vary_bc
python vary_bc.py
