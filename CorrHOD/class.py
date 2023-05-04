import yaml
import numpy as np
from pathlib import Path

from cosmoprimo.fiducial import AbacusSummit
from abacusnbody.hod.abacus_hod import AbacusHOD

from densitysplit.pipeline import DensitySplit
from densitysplit.utilities import sky_to_cartesian
from densitysplit.cosmology import Cosmology

from mockfactory import setup_logging
from pycorr import TwoPointCorrelationFunction

from utils import apply_rsd

class CorrHOD():
    
    def __init__(self, 
                 HOD_params, 
                 path2config:str,  
                 los:str = 'z', 
                 boxsize:float = 2000, 
                 cosmo:int = 0, 
                 phase:int = 0):
                
        # Set the cosmology objects
        self.cosmo = AbacusSummit(cosmo)
        h=self.cosmo.get('h')
        omega_m=self.cosmo.get('Omega_m')
        self.ds_cosmo = Cosmology(h=h, omega_m=omega_m) # Cosmology object from DensitySplit initialized with the cosmology from cosmoprimo
        
        # Read the config file
        config = yaml.safe_load(open(path2config))
        self.sim_params = config['sim_params']
        self.data_params = config['data_params']
        self.config_HOD_params = config['HOD_params'] # Temporary HOD parameters from the config file of AbacusHOD
        
        # Set the fixed parameters
        self.boxsize = boxsize
        self.los = los
        self.redshift = self.sim_params['z_mock']
        
        # Set the HOD parameters
        self.HOD_params = HOD_params
        # TODO : Add self parameters everywhere ??
    
        # Check if the simulation is already precomputed
        sim_name = Path(self.sim_params['sim_name'])
        subsample_dir = Path(self.sim_params['subsample_dir'])
        sim_path = subsample_dir / sim_name / (f'z{self.redshift:4.3f}') # Path to where the simulation is supposed to be
        
        if not sim_path.exists(): 
            err = f'The simulation {sim_path} does not exist. Run prepare_sim first. ',\
                    '(see https://abacusutils.readthedocs.io/en/latest/hod.html#short-example for more details)'
            raise ValueError(err)
    
    

    def initialize_halo(self, nthread:int = 16):
        # Create the AbacusHOD object and load the simulation
        self.Ball = AbacusHOD(self.sim_params, self.config_HOD_params)
        
        # Update the parameters of the AbacusHOD object
        self.Ball.params['Lbox'] = self.boxsize
        # TODO : Update HOD parameters for the tracers
        
        
        # Compute the incompleteness for the LRGs 
        # TODO : generalize to other tracers or make it optional with a tracer selection !!!
        self.Ball.tracers['LRG']['ic'] = 1 # Set the incompleteness to 1 for LRGs (i.e. the fraction of LRGs that are observed)
        ngal_dict = self.Ball.compute_ngal(Nthread = nthread)[0] # Compute the number of galaxies in the box
        N_lrg = ngal_dict['LRG'] # Get the number of LRGs in the box
        self.Ball.tracers['LRG']['ic'] = min(1, self.data_params['tracer_density_mean']['LRG']*self.Ball.params['Lbox']**3/N_lrg) # Compute the actual incompleteness
        
        return self.Ball
    
    
    
    def populate_halos(self, nthread:int = 16):
        
        # Run the HOD and get the dictionary containing the HOD catalogue
        self.mock_dict = self.Ball.run_hod(self.Ball.tracers, self.Ball.want_rsd, Nthread = nthread)
        
        return self.mock_dict
    
    
    def set_tracer_positions(self, tracer:str = 'LRG'):
        pass
    
    
    def get_tracer_positions(self, tracer:str = 'LRG'):
        
        # Get the positions of the galaxies
        self.data_positions = np.c_([apply_rsd(self.mock_dict, self.boxsize, self.redshift, self.cosmo, tracer=tracer, los=self.los)])
        
        return self.data_positions
    
    # TODO : Compute the cutsky, randoms, weights
    
    def compute_DensitySplit(self,
                             smooth_radius:float = 10,
                             cellsize:float = 10,
                             nquantiles:int = 10,
                             sampling:str = 'randoms',
                             filter_shape:str = 'Gaussian',
                             return_density:bool = True):
        
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        ds = DensitySplit(data_positions=self.data_positions, boxsize=self.boxsize)
        
        self.density = ds.get_density(smooth_radius=smooth_radius, cellsize=cellsize, sampling=sampling, filter_shape=filter_shape)
        
        self.quantiles = ds.get_quantiles(nquantiles=nquantiles)

        if return_density:
            return self.density, self.quantiles
        
        return self.quantiles
    
    
    # TODO : Autocorrelation and cross-correlation of the quantiles
    
    def save(self, path:str):
        pass
    
    # TODO : Option for the HOD parameters to be a dictionary or a list of dictionaries (for the MCMC)
    
    # TODO : Option for printing the times of each step
    
    
    def compute_2pcf(self, 
                     tracer:str = 'LRG', 
                     mode:str = 'smu',
                     edges:list = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                     mpicomm=None,
                     mpiroot=None,
                     nthread:int = 16):
        
        # Get the positions of the galaxies
        data_positions = np.c_([apply_rsd(self.mock_dict, self.boxsize, self.redshift, self.cosmo, tracer=tracer, los=self.los)])
    
        # Compute the 2pcf
        setup_logging()
        self.xi_HOD = TwoPointCorrelationFunction(mode, 
                                                  edges, 
                                                  data_positions1=data_positions, 
                                                  boxsize=self.boxsize, 
                                                  los=self.los, 
                                                  position_type='pos', 
                                                  mpicomm=mpicomm,
                                                  mpiroot=mpiroot,
                                                  num_threads=256)
        
        return self.xi_HOD