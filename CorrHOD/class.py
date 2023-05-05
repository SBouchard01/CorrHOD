import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

from cosmoprimo.fiducial import AbacusSummit
from abacusnbody.hod.abacus_hod import AbacusHOD

from densitysplit.pipeline import DensitySplit
from densitysplit.utilities import sky_to_cartesian
from densitysplit.cosmology import Cosmology

from mockfactory import setup_logging
from pycorr import TwoPointCorrelationFunction

from utils import apply_rsd

class CorrHOD():
    """
    This class is used to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles of the DensitySplit.
    It takes HOD parameters and a cosmology as input and uses AbacusHOD to generate a mock.
    It then uses DensitySplit to compute the density field and the quantiles.
    Finally, it uses pycorr to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles.
    """
    
    def __init__(self, 
                 HOD_params:dict, 
                 path2config:str,  
                 los:str = 'z', 
                 boxsize:float = 2000, 
                 cosmo:int = 0, 
                 phase:int = 0,
                 display_times:bool = False):
        """
        Initialises the CorrHOD object.
        Note that the 'tracer' parameter is fixed to 'LRG' by default. It can be changed later by manually reassigning the parameter.
        This tracer is chosen to study LRG and BGS. It can be changed but the code is not guaranteed to work.

        Parameters
        ----------
        HOD_params : dict
            Dictionary containing the HOD parameters. The keys must be the same as the ones used in AbacusHOD.
            
        path2config : str
            Path to the config file of AbacusHOD. It is used to load the simulation. See the documentation for more details.
            
        los : str, optional
            Line-of-sight direction in which to apply the RSD. Defaults to 'z'.
            
        boxsize : float, optional
            Size of the simulation box in Mpc/h. Defaults to 2000.
            
        cosmo : int, optional
            Index of the cosmology to use. This index is used to load the cosmology from the AbacusSummit object. Defaults to 0.
            See the documentation of AbacusSummit for more details.
            
        phase : int, optional
            Index of the phase to use. This index is used to load the cosmology from the AbacusSummit object. Defaults to 0.
            The phase describes the initial conditions of the simulation. See the documentation of AbacusSummit for more details.
            
        display_times : bool, optional
            If True, the time taken by each step is printed. Defaults to False.

        Raises
        ------
        ValueError
            If the simulation is not precomputed. See the documentation of AbacusHOD for more details.
        """
        
        
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
        """
        Initializes the AbacusHOD object and loads the simulation.

        Parameters
        ----------
        nthread : int, optional
            Number of threads to use. Defaults to 16.

        Returns
        -------
        Ball : AbacusHOD object
            AbacusHOD object containing the simulation.
        """
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
        """
        Populates the halos with galaxies using the HOD parameters.
        
        Parameters
        ----------
        nthread : int, optional
            Number of threads to use. Defaults to 16.

        Returns
        -------
        cubic_dict : dict
            Dictionary containing the HOD catalogue. It contains for each tracer :
            * 'x', 'y', 'z' : the positions of the galaxies, 
            * 'vx', 'vy', 'vz' : their velocities, 
            * 'id' and 'mass' : the ID and mass of the halo they belong to,
            * 'Ncent' : the number of central galaxies in the simulation. The first Ncent galaxies are the centrals. 
        """
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
        """
        A method to set a tracer data in the cubic dictionary.
        This will overwrite the tracer data if it already exists. Otherwise, it will create it.

        Parameters
        ----------
        positions : np.ndarray
            The positions of the galaxies. It can be a (N,3) array or a dictionary with keys 'x', 'y', 'z'.
            
        velocities : np.ndarray
            The velocities of the galaxies. It can be a (N,3) array or a dictionary with keys 'vx', 'vy', 'vz'.
            
        masses : np.ndarray
            The masses of the galaxies. It must be a (N,) array.
            
        halo_id : np.ndarray
            The ID of the halo each galaxy belongs to. It must be a (N,) array.
            
        Ncent : int
            The number of central galaxies in the simulation. The first Ncent galaxies are the centrals.
            
        tracer : str, optional
            The tracer type. Defaults to 'LRG'.

        Raises
        ------
        ValueError
            If an array is not of the right shape or if a dictionary does not have the right keys.
        """
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
            
        # Check if masses and halo_id have the right shape
        if masses.shape != x.shape:
            raise ValueError('The masses array must have the same shape as the positions array')
        if halo_id.shape != x.shape:
            raise ValueError('The halo_id array must have the same shape as the positions array')
        
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
        """
        Get the positions of the galaxies and apply RSD.

        Returns
        -------
        data_positions : np.ndarray
            The positions of the galaxies after RSD. It is a (N,3) array.
        """
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
        """
        Compute the density field and the quantiles using DensitySplit.
        (see https://github.com/epaillas/densitysplit for more details)

        Parameters
        ----------
        smooth_radius : float, optional
            The radius of the Gaussian smoothing in Mpc/h. Defaults to 10.
            
        cellsize : float, optional
            The size of the cells in the mesh. Defaults to 10.
            
        nquantiles : int, optional
            The number of quantiles to compute. Defaults to 10.
            
        sampling : str, optional
            The sampling to use. Defaults to 'randoms'.
            
        filter_shape : str, optional
            The shape of the filter to use. Can be 'Gaussian' or 'TopHat'. Defaults to 'Gaussian'.
            
        return_density : bool, optional
            If True, the density field is returned. Defaults to True.

        Returns
        -------
        density : np.ndarray
            The density field. It is a (N,3) array.
        
        quantiles : np.ndarray
            The quantiles. It is a (nquantiles,N,3) array. 
        """
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        # Initialize the DensitySplit object
        ds = DensitySplit(data_positions=self.data_positions, boxsize=self.boxsize)
        
        # Compute the density field and the quantiles
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
        """
        Compute the auto-correlation of a quantile.
        The result will also be saved in the class, as `self.CF['Auto'][f'Q{quantile}']`

        Parameters
        ----------
        quantile : int
            Index of the quantile to compute. It must be between 0 and nquantiles-1.
            
        mode : str, optional
            The mode to use for the 2PCF. Defaults to 'smu'.
            See pycorr for more details .
            
        edges : list, optional
            The edges to use for the 2PCF. Defaults to [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)].
            See pycorr for more details.
             
        mpicomm : _type_, optional
            The MPI communicator. Defaults to None.
            
        mpiroot : _type_, optional
            The MPI root. Defaults to None.
            
        nthread : int, optional
            The number of threads to use. Defaults to 16.

        Returns
        -------
        xi_quantile : pycorr.TwoPointCorrelationFunction
            The auto-correlation of the quantile.
        """
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
        """
        Compute the cross-correlation of a quantile with the galaxies.
        The result will also be saved in the class, as `self.CF['Cross'][f'Q{quantile}']`
        
        Parameters
        ----------
        quantile : int
            Index of the quantile to compute. It must be between 0 and nquantiles-1.
            
        mode : str, optional
            The mode to use for the 2PCF. Defaults to 'smu'.
            See pycorr for more details .
            
        edges : list, optional
            The edges to use for the 2PCF. Defaults to [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)].
            See pycorr for more details.
            
        mpicomm : _type_, optional
            The MPI communicator. Defaults to None.
            
        mpiroot : _type_, optional
            The MPI root. Defaults to None.
            
        nthread : int, optional
            The number of threads to use. Defaults to 16.

        Returns
        -------
        xi_quantile : pycorr.TwoPointCorrelationFunction
            The cross-correlation of the quantile.
        """
        
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
        """
        Compute the 2PCF of the galaxies. The result will also be saved in the class, as `self.CF['2PCF']`

        Parameters
        ----------
        mode : str, optional
            The mode to use for the 2PCF. Defaults to 'smu'.
            See pycorr for more details .
            
        edges : list, optional
            The edges to use for the 2PCF. Defaults to [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)].
            See pycorr for more details.
             
        mpicomm : _type_, optional
            The MPI communicator. Defaults to None.
            
        mpiroot : _type_, optional
            The MPI root. Defaults to None.
            
        nthread : int, optional
            The number of threads to use. Defaults to 16.

        Returns
        -------
        xi : pycorr.TwoPointCorrelationFunction
            The 2PCF of the galaxies.
        """
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        if not hasattr(self, 'CF'):
            # Initialize the dictionary for the correlations
            self.CF = {} 
        if not ('2PCF' in self.CF.keys()):
            # Initialize the dictionary for the 2PCF
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
             save_all:bool = False):
        
        if path is None:
            output_dir = Path(self.sim_params['output_dir'])
        sim_name = Path(self.sim_params['sim_name'])
        
        # TODO : Check the naming conventions for the files and save them accordingly
        # TODO : Check the format of the files we want to save
        
        if save_HOD or save_all:
            pass
        
        if save_cubic or save_all:
            pass
        
        if save_cutsky or save_all:
            pass
                    
        if save_density or save_all:
            pass
        
        if save_quantiles or save_all:
            pass
        
        if save_CF or save_all:
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
        
        # On rank 0, initialize the halo, populate it and get the positions of the galaxies
        
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
    
    # TODO : Handle MPI
    
    
    
# Utils and scripts outside the class
    # TODO : Functions to turn arrays to dictionaries and vice versa (With option for log_sigma)
    
    # TODO : Script to prepare the simulation if needed