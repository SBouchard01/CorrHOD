#!/bin/bash -l

#SBATCH --nodes 5
#SBATCH --constraint cpu
#SBATCH --qos regular
#SBATCH --account desi

#SBATCH --time 10:00:00

#SBATCH --job-name CorrHOD_cubic
#SBATCH --output /global/homes/s/sbouchar/Abacus_HOD/Out_jobs/%J.%x.out
#SBATCH --error /global/homes/s/sbouchar/Abacus_HOD/Out_jobs/%J.%x.err

# Load the modules of the DESI environment (cosmodesi)
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

cd /scripts/ # Change this to the path of the script (I recommend an absolute path here)
srun -n 5 python 'run_CorrHOD.py'
