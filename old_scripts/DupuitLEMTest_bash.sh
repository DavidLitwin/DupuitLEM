#!/bin/bash
#SBATCH --job-name=DupuitLEMTest
#SBATCH --time=04:0:0
#SBATCH --partition=shared
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=end
#SBATCH --mail-user=dlitwin3@jhu.edu
#### load and unload modules you may need
module load git
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev
mkdir ~/data/dlitwin3/$SLURM_JOBID
mkdir ~/data/dlitwin3/$SLURM_JOBID/data
cd ~/data/dlitwin3/DupuitLEMScripts
git rev-parse HEAD > ~/data/dlitwin3/$SLURM_JOBID/script_id.txt
cd ~/data/dlitwin3/landlab
git rev-parse HEAD > ~/data/dlitwin3/$SLURM_JOBID/gdp_id.txt
cp ~/data/dlitwin3/DupuitLEMScripts/DupuitLEMTestMSFConverge.py ~/data/dlitwin3/$SLURM_JOBID
cd ~/data/dlitwin3/$SLURM_JOBID
python DupuitLEMTestMSFConvergence.py
