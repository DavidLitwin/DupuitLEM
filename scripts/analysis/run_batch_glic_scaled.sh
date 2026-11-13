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
# Flat array over (run, variant): task t -> run = t / N, variant = t % N.
# Usage: sbatch --array=0-(R*N-1)%50 run_batch_glic_scaled.sh <script> <output_folder> <source_base> <N>
#   output_folder: folder in post_proc where all results are saved
#   source_base: original coevolved runs are ~/data/DupuitLEMResults/<source_base>-<run_ID>, run_ID = 0..R-1
#   N: number of variants per run = len(VARIANTS) in the python script (7 for scales 0.75, 1.0, 1.25)
# e.g. runs 0-26: sbatch --array=0-188%50 run_batch_glic_scaled.sh stoch_sp_svm_analysis_scaled.py scaled_X X 7
script=$1
output_folder=$2
source_base=$3
N=$4
run_id=$((SLURM_ARRAY_TASK_ID / N))
export BASE_OUTPUT_FOLDER=$output_folder
export SOURCE_RUN_ID=$run_id
export VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID % N))
export N_VARIANTS=$N
git pull
mkdir -p ~/data/DupuitLEMResults/post_proc/$output_folder
cd ~/data/DupuitLEMResults/$source_base-$run_id
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > analysis_slurm_v$VARIANT_INDEX.txt
python -u ~/code/DupuitLEM/scripts/analysis/$script > analysis_pythonlog_v$VARIANT_INDEX.out
