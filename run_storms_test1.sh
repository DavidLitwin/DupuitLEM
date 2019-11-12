#!/bin/bash
#SBATCH --job-name=DupuitLEMStorms1
#SBATCH --time=48:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=6
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#### load and unload modules you may need
module load git
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev
mkdir ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID/data
cd ~/data/dlitwin3/DupuitLEMScripts
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID/script_id.txt
cd ~/data/dlitwin3/landlab
git rev-parse HEAD > ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID/gdp_id.txt
cp ~/data/dlitwin3/DupuitLEMScripts/run_storms_test1.py ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
cd ~/data/dlitwin3/DupuitLEMResults/$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
python run_storms_test1.py
