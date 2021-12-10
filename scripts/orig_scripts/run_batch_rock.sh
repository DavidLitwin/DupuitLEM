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
  mkdir $savedir/data
  cd ~/dlitwin3/DupuitLEM
  git rev-parse HEAD > $savedir/script_id.txt
  cd ~/dlitwin3/landlab
  git rev-parse HEAD > $savedir/gdp_id.txt
fi
scriptloc=~/dlitwin3/DupuitLEM/scripts/orig_scripts/$script
if [ ! -f $savedir/$script ]; then
  cp $scriptloc $savedir
fi
cd $savedir
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > slurm.txt
python $script
