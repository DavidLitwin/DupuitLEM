## Generate parameters

Scripts here generate parameters (parameters.csv) used to run DupuitLEM
models using scripts in the `run_models` folder. Scripts assume that SLURM is
used on an HPC to generate parameters for an array of model runs. Example of a
SLURM call that would generate parameters:

`sbatch --array=0-24 run_batch_rock.sh params_stoch_ndl_svm_gam_sigma.py stoch_sp_gam_sigma_1`

Where `run_batch_rock.sh` is the shell script with instructions for making
the directory for each model run, copying the parameter-generating script,
and running that script. `stoch_sp_gam_sigma_1` would be the base name of the
output folder created. In this case, folders would be created with names
`stoch_sp_gam_sigma_1-0` through `stoch_sp_gam_sigma_1-24`.
