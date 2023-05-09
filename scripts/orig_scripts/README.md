## Generate parameters and run DupuitLEM models

Scripts here are used to 1) generate parameters and 2) run DupuitLEM models and
3) save model output. These models evolve landscapes with hydrology driven by
the selected hydrological model. This folder contains legacy script to do this,
as now models are run in two steps: generate parameters using a script from the 
generate_parameters folder, and then run the model with a script from the run_model
folder. Scripts assume that SLURM is used on an HPC to generate an array of model 
runs. Example of a SLURM call that would run these models is:

`sbatch --array=0-10 run_batch_rock.sh steady_sp_gam_hi.py steady_sp_gam_hi_1`

Where `run_batch_rock.sh` is the shell script with instructions for making
the directory for each model run, copying the model script,
and running that script. `steady_gam_sigma_1` would be the base name of the
output folder created. In this case, folders would be created with names
`steady_gam_sigma_1-0` through `steady_gam_sigma_1-10`. Within those folders,
a folder called `data` is created to store the saved model output.
