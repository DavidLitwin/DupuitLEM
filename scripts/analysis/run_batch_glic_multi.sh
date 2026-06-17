#!/bin/bash
#SBATCH --job-name=post
#SBATCH --time=2:0:0
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin@gfz-potsdam.de
#SBATCH -o ../../../../data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
t_index=$3
export BASE_OUTPUT_FOLDER=$2
git pull
mkdir ~/data/DupuitLEMResults/post_proc/$output_folder
cd ~/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > analysis_slurm.txt
python -u ~/code/DupuitLEM/scripts/analysis/$script $t_index > analysis_pythonlog.out
