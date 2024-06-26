## Run DupuitLEM models from generated parameters

Scripts here are used to run DupuitLEM models from a csv file of parameters.
Files in the `generate_parameters` folder can be used to generate these parameters.
These models evolve landscapes with hydrology driven by the selected hydrological model. 
Make sure the model used here corresponds to that intended by the generate_parameters
script. Scripts assume that SLURM is used on an HPC to generate
an array of model runs. Example of a SLURM call that would run these models is:

`sbatch --array=0-24 run_batch_rock.sh model_stoch_sp_nld_svm.py stoch_sp_gam_sigma_1`

Where `run_batch_rock.sh` is the shell script for copying the model run script,
and running that script. Calling the shell script when generating parameters
has already created folders `stoch_sp_gam_sigma_1-0` through
`stoch_sp_gam_sigma_1-10`. Within those folders, a folder called `data` is made
to store the saved model output.
