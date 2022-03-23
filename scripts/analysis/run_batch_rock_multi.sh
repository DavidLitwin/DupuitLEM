#!/bin/bash
#SBATCH --job-name=post
#SBATCH --time=3:0:0
#SBATCH --partition=defq
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=3
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../../../data_charman1/DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
model_run=$3
export BASE_OUTPUT_FOLDER=$2
export MODEL_RUN=$3
cd ~/data_charman1/DupuitLEMResults/$output_folder-$model_run
python -u ~/dlitwin3/DupuitLEM/scripts/analysis/$script > analysis_pythonlog.out
