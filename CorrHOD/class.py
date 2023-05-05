import yaml
import numpy as np
import pandas as pd
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
                 HOD_params:dict, 
                 path2config:str,  
                 los:str = 'z', 
                 boxsize:float = 2000, 
                 cosmo:int = 0, 
                 phase:int = 0):
        
        # TODO : Option for printing the times of each step
            
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
        self.tracer = 'LRG' # Tracer to use for the HOD. Since this code is only for BGS, this is fixed but can be changed later by reassigning the parameter
        self.boxsize = boxsize
        self.los = los
        self.redshift = self.sim_params['z_mock']
        
        # Set the HOD parameters
        self.HOD_params = HOD_params
    
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
        for key in self.HOD_params.keys():
            self.Ball.tracers[self.tracer][key] = self.HOD_params[key]
        
        
        # Compute the incompleteness for the tracer (i.e. the fraction of galaxies that are observed)
        self.Ball.tracers[self.tracer]['ic'] = 1 # Set the incompleteness to 1 by default
        ngal_dict = self.Ball.compute_ngal(Nthread = nthread)[0] # Compute the number of galaxies in the box
        N_trc = ngal_dict[self.tracer] # Get the number of galaxies of the tracer type in the box
        self.Ball.tracers[self.tracer]['ic'] = min(1, self.data_params['tracer_density_mean'][self.tracer]*self.Ball.params['Lbox']**3/N_trc) # Compute the actual incompleteness
        
        return self.Ball
    
    
    
    def populate_halos(self, nthread:int = 16):
        
        # Run the HOD and get the dictionary containing the HOD catalogue
        self.cubic_dict = self.Ball.run_hod(self.Ball.tracers, self.Ball.want_rsd, Nthread = nthread)
        
        return self.cubic_dict
    
    
    def set_tracer_data(self, 
                        positions:np.ndarray,
                        velocities:np.ndarray,
                        masses:np.ndarray,
                        halo_id:np.ndarray,
                        Ncent:int,
                        tracer:str = 'LRG'):
        
        # Check if positions has keys x, y, z
        if not all([key in positions.keys() for key in ['x', 'y', 'z']]):
            # If the array has a shape (3,N), we transpose it to (N,3)
            if positions.shape[0] == 3:
                positions = positions.T
            # Then, we check if the array has the right shape (i.e. (N,3))
            if positions.shape[1] != 3:
                raise ValueError('The positions array must have a shape (N,3) or (3,N)')
            
            x = positions[:,0]
            y = positions[:,1]
            z = positions[:,2]
        else:
            x = positions['x']
            y = positions['y']
            z = positions['z']
        
        # Check if velocities has keys vx, vy, vz
        if not all([key in velocities.keys() for key in ['vx', 'vy', 'vz']]):
            # If the array has a shape (3,N), we transpose it to (N,3)
            if velocities.shape[0] == 3:
                velocities = velocities.T
            # Then, we check if the array has the right shape (i.e. (N,3))
            if velocities.shape[1] != 3:
                raise ValueError('The velocities array must have a shape (N,3) or (3,N)')
            
            vx = velocities[:,0]
            vy = velocities[:,1]
            vz = velocities[:,2]
        else:
            vx = velocities['vx']
            vy = velocities['vy']
            vz = velocities['vz']
        
        # Set the tracer dictionary
        tracer_dict = {
            'x': x,
            'y': y,
            'z': z,
            
            'vx': vx,
            'vy': vy,
            'vz': vz,
            
            'mass': masses,
            'id': halo_id,
            
            'Ncent': Ncent  
        }
        
        # Set the tracer dictionary in the mock dictionary
        if not hasattr(self, 'cubic_dict'):
            self.cubic_dict = {}
        self.cubic_dict[tracer] = tracer_dict # Overwrite the tracer dictionary if it already exists
        
        # Set the tracer type to the one provided
        if hasattr(self, 'tracer'):
            # TODO : warn that the tracer type has been changed
            pass
        self.tracer = tracer # Set the tracer type to the one given 
        
    
    
    def get_tracer_positions(self):
        
        # Get the positions of the galaxies
        self.data_positions = np.c_([apply_rsd(self.cubic_dict, self.boxsize, self.redshift, self.cosmo, tracer=self.tracer, los=self.los)])
        
        return self.data_positions
    
    
    
    # TODO : Compute the cutsky, randoms, weights
    # TODO : Add the option to use the cutsky in the functions
    
    
    
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
    
    
    
    
    # TODO : Option for the HOD parameters to be a dictionary or a list of dictionaries (for the MCMC)
    # Not a good idea ?
    
    
    
    # TODO : 2PCF, autocorrelation and cross-correlation of the quantiles
    
    def compute_auto_corr(self,
                          quantile:int,
                          mode:str = 'smu',
                          edges:list = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                          mpicomm = None,
                          mpiroot = None,
                          nthread:int = 16):
        
        # Get the positions of the points in the quantile
        quantile_data = self.quantiles[quantile]
        quantile_positions = np.c_[quantile_data['x'], quantile_data['y'], quantile_data['z']]
        
        if not hasattr(self, 'CF'):
            # Initialize the dictionary for the correlations
            self.CF = {} 
        if not ('Auto' in self.CF.keys()):
            # Initialize the dictionary for the auto-correlations
            self.CF['Auto'] = {}
        
        # Compute the 2pcf
        setup_logging()
        xi_quantile = TwoPointCorrelationFunction(mode, edges,
                                                  data_positions1=quantile_positions,
                                                  boxsize=self.boxsize,
                                                  los=self.los,
                                                  position_type='pos',
                                                  mpicomm=mpicomm, mpiroot=mpiroot, num_threads=nthread) 
        
        # Add the 2pcf to the dictionary
        self.CF['Auto'][f'Q{quantile}'] = xi_quantile # TODO : check if the key is correct (might be DS{quantile} ?)
        
        return xi_quantile
    
    
    
    def compute_cross_corr(self,
                           quantile:int,
                           mode:str = 'smu',
                           edges:list = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                           mpicomm = None,
                           mpiroot = None,
                           nthread:int = 16):
    
        # Get the positions of the points in the quantile
        quantile_data = self.quantiles[quantile]
        quantile_positions = np.c_[quantile_data['x'], quantile_data['y'], quantile_data['z']]
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        if not hasattr(self, 'CF'):
            # Initialize the dictionary for the correlations
            self.CF = {} 
        if not ('Cross' in self.CF.keys()):
            # Initialize the dictionary for the auto-correlations
            self.CF['Cross'] = {}
        
        # Compute the 2pcf
        setup_logging()
        xi_quantile = TwoPointCorrelationFunction(mode, edges,
                                                  data_positions1=quantile_positions,
                                                  data_positions2=self.data_positions,
                                                  boxsize=self.boxsize,
                                                  los=self.los,
                                                  position_type='pos',
                                                  mpicomm=mpicomm, mpiroot=mpiroot, num_threads=nthread)
        
        # Add the 2pcf to the dictionary
        self.CF['Cross'][f'Q{quantile}'] = xi_quantile # TODO : check if the key is correct (might be DS{quantile} ?)
        
        return xi_quantile
    
    
    def compute_2pcf(self,  
                     mode:str = 'smu',
                     edges:list = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                     mpicomm = None,
                     mpiroot = None,
                     nthread:int = 16):
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        if not hasattr(self, 'CF'):
            # Initialize the dictionary for the correlations
            self.CF = {} 
        if not ('2PCF' in self.CF.keys()):
            # Initialize the dictionary for the auto-correlations
            self.CF['2PCF'] = {}
    
        # Compute the 2pcf
        setup_logging()
        xi = TwoPointCorrelationFunction(mode, edges, 
                                         data_positions1 = self.data_positions, 
                                         boxsize = self.boxsize, 
                                         los = self.los, 
                                         position_type = 'pos', 
                                         mpicomm = mpicomm, mpiroot = mpiroot, num_threads = nthread)
        
        # Add the 2pcf to the dictionary
        self.CF['2PCF'] = xi
        
        return xi
    
    def save(self,
             path:str = None,
             save_HOD:bool = True,
             save_cubic:bool = True,
             save_cutsky:bool = True,
             save_density:bool = True,
             save_quantiles:bool = True,
             save_CF:bool = True,
             save_all:bool = True):
        
        if path is None:
            output_dir = Path(self.sim_params['output_dir'])
        sim_name = Path(self.sim_params['sim_name'])
        
        # TODO : Check the naming conventions for the files and save them accordingly
        # TODO : Check the format of the files we want to save
        
        if save_all:
            pass
        
        if save_HOD:
            pass
        
        if save_cubic:
            pass
                    
        if save_density:
            pass
        
        if save_quantiles:
            pass
        
        if save_CF:
            pass
        
        
    def run_all(self,
                # Parameters for the DensitySplit
                smooth_radius:float = 10,
                cellsize:float = 10,
                nquantiles:int = 10,
                sampling:str = 'randoms',
                filter_shape:str = 'Gaussian',
                # Parameters for the 2PCF, autocorrelation and cross-correlations
                mpicomm = None,
                mpiroot = None,
                nthread:int = 16
                # Parameters for saving the results
                ):
        
        self.initialize_halo()
        
        self.populate_halos()
    
        self.get_tracer_positions()
        
        self.compute_DensitySplit(smooth_radius=smooth_radius, 
                                  cellsize=cellsize, 
                                  nquantiles=nquantiles,
                                  sampling=sampling, 
                                  filter_shape=filter_shape,
                                  return_density=False)
        
        self.compute_2pcf(mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
        
        for quantile in range(nquantiles):
            self.compute_auto_corr(quantile, mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
            self.compute_cross_corr(quantile, mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
        
        self.save(save_all=True) # TODO : Add the options in the parameters
    
    
    # TODO : Document the functions
    
    # TODO : Add checks and warnings if needed
    
    # TODO : Functions to turn arrays to dictionaries and vice versa (With option for log_sigma)
    
    # TODO : Script to prepare the simulation if needed