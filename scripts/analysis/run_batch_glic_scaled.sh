#!/bin/bash
#SBATCH --job-name=post
#SBATCH --time=1:0:0
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin@gfz-potsdam.de
#SBATCH -o ../../../../data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
# Usage: sbatch --array=0-(N-1) run_batch_glic_scaled.sh <script> <output_folder> <source_base> <run_ID>
#   output_folder: folder in post_proc where results are saved
#   source_base, run_ID: the original coevolved run is ~/data/DupuitLEMResults/<source_base>-<run_ID>
#   N = len(VARIANTS) in the python script; array task i is variant i.
script=$1
output_folder=$2
source_base=$3
run_id=$4
export BASE_OUTPUT_FOLDER=$output_folder
export SOURCE_RUN_ID=$run_id
git pull
mkdir -p ~/data/DupuitLEMResults/post_proc/$output_folder
cd ~/data/DupuitLEMResults/$source_base-$run_id
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > analysis_slurm_$SLURM_ARRAY_TASK_ID.txt
python -u ~/code/DupuitLEM/scripts/analysis/$script > analysis_pythonlog_$SLURM_ARRAY_TASK_ID.out
