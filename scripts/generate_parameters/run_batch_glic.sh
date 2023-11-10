#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=0:10:0
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin@gfz-potsdam.de
#SBATCH -o ../../../../data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
savedir=~/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
if [ ! -d $savedir ]; then
  mkdir $savedir
fi
cd ~/code/DupuitLEM
git rev-parse HEAD > $savedir/params_script_id.txt
cd ~/dlitwin3/landlab
git rev-parse HEAD > $savedir/params_gdp_id.txt

scriptloc=~/code/DupuitLEM/scripts/generate_parameters/$script
cp $scriptloc $savedir
cd $savedir
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > params_slurm.txt
python $script
