#!/bin/bash
#SBATCH --job-name=post
#SBATCH --time=1:0:0
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o /projects/dali4360/data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
export BASE_OUTPUT_FOLDER=$2
mkdir /projects/dali4360/data/DupuitLEMResults/post_proc/$output_folder
cd /projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > analysis_slurm.txt
python -u ~/DupuitLEM/scripts/analysis/$script > analysis_pythonlog.out
