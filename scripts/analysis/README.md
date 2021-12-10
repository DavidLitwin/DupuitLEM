## Post process DupuitLEM model results

Scripts here are used to post-process DupuitLEM model output and save results
in a folder given by the base output name.  Scripts assume that SLURM is
used on an HPC to post-process an array of model runs. Example of a
SLURM call that would generate parameters:

`sbatch --array=0-24 run_batch_rock.sh stoch_sp_svm_analysis.py stoch_sp_gam_sigma_1`

Where `run_batch_rock.sh` is the shell script with instructions for making
the directory where the results will be saved, copying the post processing script,
and running that script. Results will be saved in a folder `stoch_sp_gam_sigma_1`
within the `post_proc` folder. See individual shell scripts for the location
of that folder.
