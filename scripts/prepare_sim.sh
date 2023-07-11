#!/bin/bash 

# Path to the config file, which must contain the path to the dark matter simulation in sim_dir
# and the name of the simulation we want to prepare in sim_name
# The prepared simulation will be saved in subsample_dir
cd /global/homes/s/sbouchar/Abacus_HOD/CorrHOD/scripts
PATH2CONFIG='../config/config.yaml' 

# Run the simulation preparation (see https://abacusutils.readthedocs.io/en/latest/hod.html#short-example)
# python -m abacusnbody.hod.prepare_sim --path2config $PATH2CONFIG

# For BGS, we use a modified prepare_sim file : 
python prepare_sim_bgs.py --path2config $PATH2CONFIG