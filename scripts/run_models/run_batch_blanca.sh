#!/bin/bash
#SBATCH --job-name=dlem
#SBATCH --time=6-0:0:0
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=dlitwin3@jhu.edu
#SBATCH -o /projects/dali4360/data/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=$1
output_folder=$2
savedir=/projects/dali4360/data/DupuitLEMResults/$output_folder-$SLURM_ARRAY_TASK_ID
mkdir $savedir/data
cd ~/DupuitLEM
git rev-parse HEAD > $savedir/lem_script_id.txt
cd ~/landlab
git rev-parse HEAD > $savedir/lem_gdp_id.txt

scriptloc=~/DupuitLEM/scripts/run_models/$script
cp $scriptloc $savedir
cd $savedir
echo $SLURM_JOBID-$SLURM_ARRAY_TASK_ID > lem_slurm.txt
python $script
