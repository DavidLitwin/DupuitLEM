#!/bin/bash
#SBATCH --job-name=stoch_ss
#SBATCH --time=48:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=stoch_vary_k.py
output_folder=stoch_vary_k
module load git
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev
export PYTHONPATH="/home-1/dlitwin3@jhu.edu/data/dlitwin3/DupuitLEMScripts"
mkdir ~/data/dlitwin3/DupuitLEMResults/$output_folder
mkdir ~/data/dlitwin3/DupuitLEMResults/$output_folder/data
cd ~/data/dlitwin3/DupuitLEMScripts
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/$output_folder/script_id.txt
cd ~/data/dlitwin3/landlab
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/$output_folder/gdp_id.txt
cp ~/data/dlitwin3/DupuitLEMScripts/scripts/$script ~/data/dlitwin3/DupuitLEMResults/$output_folder
cd ~/data/dlitwin3/DupuitLEMResults/$output_folder
python $script
