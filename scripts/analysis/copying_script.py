
import os
import glob
import shutil

input_name = input('Source run name:')
output_name = input('Output run name:')
N = int(input('Number of runs:'))

home = '~/data_charman1/DupuitLEMResults'

for ID in range(N):
    grid_files = glob.glob(os.join(home, "%s-%d"%(input_name, ID), "data", "*.nc"))
    files = sorted(grid_files, key=lambda x:int(x.split('_')[-1][:-3]))
    grid_path = files[-1]

    os.makedir(os.join(home, "%s-%d"%(output_name, ID)))
    shutil.copy2(grid_path, os.join(home, "%s-%d"%(output_name, ID), 'grid_%d.nc'%ID) )
