#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=8-0:0:0
#SBATCH --partition=unlimited
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../../DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
mkdir ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
mkdir ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/data
cd ~/work/dlitwin3/DupuitLEM
git rev-parse HEAD > ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/script_id.txt
cd ~/work/dlitwin3/landlab
git rev-parse HEAD > ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/gdp_id.txt
cp ~/work/dlitwin3/DupuitLEM/scripts/HPC/$script ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
cd ~/work/dlitwin3/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > slurm.txt
# python -u $script > pythonlog.out
python $script
