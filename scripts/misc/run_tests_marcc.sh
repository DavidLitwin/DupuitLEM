#!/bin/bash
#SBATCH --job-name=hydrotest
#SBATCH --time=1:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../../DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
mkdir ~/work/dlitwin3/DupuitLEMResults/$output_folder
cd ~/work/dlitwin3/DupuitLEM
git rev-parse HEAD > ~/work/dlitwin3/DupuitLEMResults/$output_folder/script_id.txt
cd ~/work/dlitwin3/landlab
git rev-parse HEAD > ~/work/dlitwin3/DupuitLEMResults/$output_folder/gdp_id.txt
cp ~/work/dlitwin3/DupuitLEM/scripts/misc/$script ~/work/dlitwin3/DupuitLEMResults/$output_folder
cd ~/work/dlitwin3/DupuitLEMResults/$output_folder
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > slurm.txt
# python -u $script > pythonlog.out
python $script
