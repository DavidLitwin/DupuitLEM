#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=24:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
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
echo $SLURM_JOBID > slurm.txt
python -u $script > pythonlog.out