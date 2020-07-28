#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=72:0:0
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o /data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
mkdir projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
mkdir projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/data
cd ~/DupuitLEM
git rev-parse HEAD > projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/script_id.txt
cd ~/landlab
git rev-parse HEAD > projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID/gdp_id.txt
cp ~/DupuitLEM/scripts/HPC/$script projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
cd projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > slurm.txt
python -u $script > pythonlog.out
