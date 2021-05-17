#!/bin/bash
#SBATCH --job-name=cross
#SBATCH --time=6:0:0
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
export BASE_OUTPUT_FOLDER=$2
cd ~/work/dlitwin3/DupuitLEMResults/post_proc/$output_folder
python -u ~/work/dlitwin3/DupuitLEM/scripts/analysis/$script > analysis_pythonlog_$SLURM_ARRAY_TASK_ID.out
