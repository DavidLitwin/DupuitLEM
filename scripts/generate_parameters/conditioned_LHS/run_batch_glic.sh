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
clhs_file=$3
savedir=~/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
if [ ! -d $savedir ]; then
  mkdir $savedir
fi

cp $clhs_file $savedir
cp $script $savedir

cd ~/code/DupuitLEM
git rev-parse HEAD > $savedir/params_script_id.txt
cd ~/code/landlab
git rev-parse HEAD > $savedir/params_gdp_id.txt

cd $savedir
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > params_slurm.txt
python $script
