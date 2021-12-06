#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=48:0:0
#SBATCH --partition=defq
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o ../../../../data_charman1/DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
savedir=~/data_charman1/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
if [ ! -d $savedir ]; then
  mkdir $savedir
fi
cd ~/dlitwin3/DupuitLEM
git rev-parse HEAD > $savedir/params_script_id.txt
cd ~/dlitwin3/landlab
git rev-parse HEAD > $savedir/params_gdp_id.txt

scriptloc=~/dlitwin3/DupuitLEM/scripts/generate_parameters/$script
cp $scriptloc $savedir
cd $savedir
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > params_slurm.txt
python $script
