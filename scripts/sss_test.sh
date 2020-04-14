#!/bin/bash
#SBATCH --job-name=steady_ss
#SBATCH --time=10:00
#SBATCH --partition=debug
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
##SBATCH --mail-type=begin,end
##SBATCH --mail-user=dlitwin3@jhu.edu
##SBATCH -o ../../DupuitLEMResults/slurm/slurm-%A_%a.out
#### load and unload modules you may need
script=steady_shear_stress_marcc_test.py
output_folder=steady_ss
module load git
module load python/3.7-anaconda
. /software/apps/anaconda/5.2/python/3.7/etc/profile.d/conda.sh
conda activate
conda activate landlab_dev

export PYTHONPATH="/home-1/dlitwin3@jhu.edu/data/dlitwin3/DupuitLEMScripts"

python -c "from landlab import RasterModelGrid"
python -c "from DupuitLEM import SteadyRechargeShearStress"
python -c "from DupuitLEM.grid_functions.grid_funcs import bind_avg_hydraulic_conductivity"

output_folder=test_1
mkdir ~/data/dlitwin3/DupuitLEMResults/$output_folder/$SLURM_JOBID
