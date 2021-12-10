## Run 1D hillslope hydrological models

Scripts here are used to 1) generate parameters, 2) run hydrological models,
3) analyze model results on 1D hillslopes, and 3) save analysis. Note there is
no landscape evolution in these models--they are used instead to analyze
hydrological behavior. Scripts assume that SLURM is used on an HPC to generate
an array of model runs. Example of a SLURM call that would run these models is:

`sbatch --array=0-10 run_batch_rock.sh hyd1d_gam_sigma_nld_svm.py hyd1d_gam_sigma_nld_svm_1`

Where `run_batch_rock.sh` is the shell script with instructions for making
the output directory, copying the hyd1d script, and running that script.
The folder `steady_gam_sigma_1` will be created, and all model output placed
in this folder.
