import sys 
from pathlib import Path

### Paths ### (absolute and relative to the script location)
abs_path = Path(__file__).parent # Absolute path to the current file
module_path = (abs_path / '../').resolve(strict=True)
path2config = (abs_path / Path('../config/config.yaml')).resolve(strict=True) # Get the path of the config file (resolve for symlinks)
path = (abs_path / Path('../data')).resolve(strict=True) # Path to save the results
sys.path.append(str(module_path)+'/') # Add the parent directory to the path

import logging
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

from mockfactory import setup_logging
from CorrHOD.utils import create_logger

from CorrHOD.pipeline import CorrHOD

### Logging ###
setup_logging() # Initialize the logging for all the loggers that will not have a handler 
# Create the loggers for this script, and add a handler to it
logger = create_logger('CorrHOD', level='debug', stream=sys.stdout) 
#Temporary logger for densitysplit
logger2 = create_logger('DS', level='debug', stream=sys.stdout) 

### MPI ###
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.size # Number of processes
rank = comm.Get_rank() # Rank of the current process (root process has rank 0)
root = 0

# comm = None
# root = None
# rank = 0
# size = 1

# Get the number of threads available
import multiprocessing
nthread = multiprocessing.cpu_count()

logger.info(f'Rank: {rank} / {size}, {nthread} threads available')
logger.newline()


### Parameters ###
HOD_params = {
    'logM_cut': 12.0,
    'logM1': 14.525, 
    'sigma': 0.029, 
    'alpha': 1.141, 
    'kappa': 1.089, 
    'alpha_c': 0.0, 
    'alpha_s': 0.0, 
    'Bcent': 0.0, 
    'Bsat': 0.0
}

nquantiles=10
los = 'z'
smooth_radius = 10
cellsize = 5

Object = CorrHOD(HOD_params, path2config)

Object.run_all(los_to_compute=los,
               display_times=True,
               smooth_radius=smooth_radius,
               cellsize=cellsize,
               mpicomm=comm,
               mpiroot=root,
               nquantiles=nquantiles,
               nthread=nthread,
               path = path,
               los='all', # Save all the lines of sight, including the average
               save_all=True)




#%% Plot the CFs
if rank == 0:
    fig, ax = plt.subplots(2, 3, sharex=True, figsize=(16, 7))

    colors = [f'C{i}' for i in range(nquantiles)]

    s = Object.CF['average']['s'] # Get the separation s (identical for all CFs)

    for i in range(nquantiles):
        
        color = colors[i]
        alpha = 0.6
        
        # Plot the multipoles of the auto correlation
        poles1 = Object.CF['average']['Auto'][f'DS{i}']
        ax[0,0].plot(s, poles1[0]*s**2, alpha=alpha, color=color, label = f'DS{i}')
        ax[0,1].plot(s, poles1[1]*s**2, alpha=alpha, color=color)
        ax[0,2].plot(s, poles1[2]*s**2, alpha=alpha, color=color)

        # Plot the multipoles of the cross correlation
        poles2 = Object.CF['average']['Cross'][f'DS{i}']
        ax[1,0].plot(s, poles2[0]*s**2, alpha=alpha, color=color)
        ax[1,1].plot(s, poles2[1]*s**2, alpha=alpha, color=color)
        ax[1,2].plot(s, poles2[2]*s**2, alpha=alpha, color=color)
        
    #Add the 2PCF on both plots

    poles = Object.CF['average']['2PCF']
    for i in [0, 1]:
        ax[i,0].plot(s, poles[0]*s**2, alpha=alpha, color='k', ls='--', label='2PCF')
        ax[i,1].plot(s, poles[1]*s**2, alpha=alpha, color='k', ls='--')
        ax[i,2].plot(s, poles[2]*s**2, alpha=alpha, color='k', ls='--')


    ax[0,0].legend()
    ax[1,0].set_xlabel(r'$s$')
    ax[1,0].set_ylabel(r'$s^2\xi_l(s)$')
    ax[0,0].set_ylabel(r'$s^2\xi_l(s)$')
    ax[0,0].set_title('Monopole', fontsize=11)
    ax[0,1].set_title('Quadrupole', fontsize=11)
    ax[0,2].set_title('Hexadecapole', fontsize=11)
    fig.suptitle('Auto (top) and Cross (bottom) Correlation for c000', fontsize=15, y=1)

    fig.savefig(path+'/CFs.png', dpi=300, bbox_inches='tight')
    
#%% Access the times
if rank == 0 : 
    # The times are stored in a dictionary
    times_dict = Object.times_dict

    # It contains the following keys:
    pprint(times_dict)

    # Let's compute some interesting times to look at

    total_time = times_dict['run_all']

    mean_los_time = np.mean([times_dict[los]['run_los'] for los in ['x', 'y', 'z']], axis=0)

    mean_2PCF_time = np.mean([times_dict[los]['compute_2pcf'] for los in ['x', 'y', 'z']], axis=0)

    mean_auto_time = np.mean([times_dict[los]['compute_auto_corr'][f'DS{i}'] for los in ['x', 'y', 'z'] for i in range(nquantiles)], axis=0)

    mean_cross_time = np.mean([times_dict[los]['compute_cross_corr'][f'DS{i}'] for los in ['x', 'y', 'z'] for i in range(nquantiles)], axis=0)

    total_auto_time = np.mean([np.sum([times_dict[los]['compute_auto_corr'][f'DS{i}'] for i in range(nquantiles)], axis=0) for los in ['x', 'y', 'z']], axis=0)

    total_cross_time = np.mean([np.sum([times_dict[los]['compute_cross_corr'][f'DS{i}'] for i in range(nquantiles)], axis=0) for los in ['x', 'y', 'z']], axis=0)

    #Display the times
    print(f'Total time: {total_time:.2f} s')

    print(f'Mean time for computing a LOS: {mean_los_time:.2f} s')

    print(f'Mean time for computing the 2PCF: {mean_2PCF_time:.2f} s')

    print(f'Mean time for computing an auto correlation: {mean_auto_time:.2f} s')

    print(f'Mean time for computing a cross correlation: {mean_cross_time:.2f} s')

    print(f'Mean time for computing all auto correlations: {total_auto_time:.2f} s')

    print(f'Mean time for computing all cross correlations: {total_cross_time:.2f} s')
