import yaml, logging, sys
import numpy as np
import pandas as pd
from pathlib import Path
from warnings import warn, filterwarnings, catch_warnings
from time import time

from cosmoprimo.fiducial import AbacusSummit
from abacusnbody.hod.abacus_hod import AbacusHOD

from densitysplit.pipeline import DensitySplit
from densitysplit.cosmology import Cosmology

from pycorr import TwoPointCorrelationFunction, project_to_multipoles

from CorrHOD.utils import apply_rsd


class CorrHOD_cubic():
    """
    This class is used to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles of the DensitySplit.
    It takes HOD parameters and a cosmology as input and uses AbacusHOD to generate a cubic mock.
    It then uses DensitySplit to compute the density field and the quantiles.
    Finally, it uses pycorr to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles.
    """
    
    def __init__(self, 
                 path2config:str, 
                 HOD_params:dict = None,  
                 los:str = 'z', 
                 boxsize:float = 2000, 
                 cosmo:int = 0):
        """
        Initialises the CorrHOD object.
        Note that the 'tracer' parameter is fixed to 'LRG' by default. It can be changed later by manually reassigning the parameter.
        This tracer is chosen to study LRG and BGS. It can be changed but the code is not guaranteed to work.

        Parameters
        ----------
        path2config : str
            Path to the config file of AbacusHOD. It is used to load the simulation. See the documentation for more details.
            
        HOD_params : dict, optional
            Dictionary containing the HOD parameters. The keys must be the same as the ones used in AbacusHOD.
            If not provided, the default parameters of AbacusHOD are used. Defaults to None.
            
        los : str, optional
            Line-of-sight direction in which to apply the RSD. Defaults to 'z'.
            
        boxsize : float, optional
            Size of the simulation box in Mpc/h. Defaults to 2000.
            
        cosmo : int, optional
            Index of the cosmology to use. This index is used to load the cosmology from the AbacusSummit object. Defaults to 0.
            See the documentation of AbacusSummit for more details.

        Raises
        ------
        ValueError
            If the simulation is not precomputed. See the documentation of AbacusHOD for more details.
        """
        
            
        # Set the cosmology objects
        self.cosmo = AbacusSummit(cosmo)
        h=self.cosmo.get('h')
        omega_m=self.cosmo.get('Omega_m')
        self.ds_cosmo = Cosmology(h=h, omega_m=omega_m) # Cosmology object from DensitySplit initialized with the cosmology from cosmoprimo
        
        # Read the config file
        config = yaml.safe_load(open(path2config))
        self.sim_params = config['sim_params']
        self.config_HOD_params = config['HOD_params'] # Temporary HOD parameters from the config file of AbacusHOD
        # Make data_params an optional parameter
        if 'data_params' in config.keys():
            self.data_params = config['data_params']
        else:
            self.data_params = None
        
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
        if not hasattr(self, 'Ball'): # To avoid loading the slabs if they are already loaded
            self.Ball = AbacusHOD(self.sim_params, self.config_HOD_params)
        
        # Update the parameters of the AbacusHOD object
        self.Ball.params['Lbox'] = self.boxsize
        if self.HOD_params is not None:
            # Update the HOD parameters if different ones are provided
            for key in self.HOD_params.keys():
                self.Ball.tracers[self.tracer][key] = self.HOD_params[key]
        
        
        # Compute the incompleteness for the tracer (i.e. the fraction of galaxies that are observed)
        self.Ball.tracers[self.tracer]['ic'] = 1 # Set the incompleteness to 1 by default
        ngal_dict = self.Ball.compute_ngal(Nthread = nthread)[0] # Compute the number of galaxies in the box
        N_trc = ngal_dict[self.tracer] # Get the number of galaxies of the tracer type in the box
        if self.data_params is not None:
            self.Ball.tracers[self.tracer]['ic'] = min(1, self.data_params['tracer_density_mean'][self.tracer]*self.Ball.params['Lbox']**3/N_trc) # Compute the actual incompleteness
        
        return self.Ball
    
    
    
    def populate_halos(self, nthread:int = 16, remove_Ball:bool = False):
        """
        Populates the halos with galaxies using the HOD parameters.
        
        Parameters
        ----------
        nthread : int, optional
            Number of threads to use. Defaults to 16.
            
        remove_Ball : bool, optional
            If True, the Ball object (Dark matter simulation loaded from AbacusHOD) is removed after populating the halos. 
            This is useful to save memory. Defaults to False.

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
        
        # Log the number density of the populated halos
        logger = logging.getLogger('CorrHOD') # Get the logger for the CorrHOD class
        actual_n = len(self.cubic_dict['LRG']['x']) / self.boxsize**3
        logger.debug(f'Number density of the populated halos : {actual_n:.2e} h^3/Mpc^3')
        
        # Check that the number density of the populated halos is close to the target number density
        if self.data_params is not None:
            expected_n = self.data_params['tracer_density_mean'][self.tracer]
            std_n = self.data_params['tracer_density_std'][self.tracer]
        
            if abs(expected_n - actual_n) > std_n :
                warn(f'The number density of the populated halos ({actual_n:.2e}) is not close to the target number density ({expected_n}).', UserWarning) 

        # Remove the Ball object to save memory
        if remove_Ball:
            del self.Ball
            logger.debug('Removed the Ball object to save memory')

        self.nbar = actual_n
        return self.cubic_dict
    
    
    #TODO : Change this : leave it to the user and just run checks to make sure the data is in the right format
    def set_cubic(self, 
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
        
        # Set the tracer type to the one provided (Warn if it is different from the one already set)
        if hasattr(self, 'tracer'):
            if self.tracer != tracer: 
                warn(f"The tracer provided ('{tracer}') will be used instead of the already existing one ('{self.tracer}').", UserWarning)
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
        self.data_positions = np.c_[apply_rsd(self.cubic_dict, self.boxsize, self.redshift, self.cosmo, tracer=self.tracer, los=self.los)]
        
        return self.data_positions    
    
    
    
    def compute_DensitySplit(self,
                             smooth_radius:float = 10,
                             cellsize:float = 5,
                             nquantiles:int = 10, 
                             sampling:str = 'randoms',
                             randoms=None,
                             filter_shape:str = 'Gaussian',
                             return_density:bool = True,
                             nthread=16):
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
            The sampling to use. Can be 'randoms' or 'data'. Defaults to 'randoms'.
            
        randoms : np.ndarray, optional
            The positions of the randoms. It is a (N,3) array of points in the range [0, boxsize] for each coordinate.
            If set to none, a random array will be created of size nquantiles * len(data_positions). 
            Defaults to None.
            
        filter_shape : str, optional
            The shape of the filter to use. Can be 'Gaussian' or 'TopHat'. Defaults to 'Gaussian'.
            
        return_density : bool, optional
            If True, the density field is returned. Defaults to True.

        nthread : int, optional
            The number of threads to use. Defaults to 16.
        
        Returns
        -------
        quantiles : np.ndarray
            The quantiles. It is a (nquantiles,N,3) array. 
        
        density : np.ndarray, optional
            The density field. It is a (N,3) array.
        """
        logger = logging.getLogger('DensitySplit') #tmp
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        # Initialize the DensitySplit object and compute the density field
        try : #Main branch
            logger.debug('Launched densitysplit on main branch')
            
            ds = DensitySplit(data_positions=self.data_positions, boxsize=self.boxsize)

            self.density = ds.get_density_mesh(smooth_radius=smooth_radius, cellsize=cellsize, sampling=sampling, filter_shape=filter_shape)
        except : #OpenMP branch (an error will be raised because the OpenMP branch used differently)
            
            logger.debug('Launched densitysplit on openmp branch') #tmp
            
            ds = DensitySplit(data_positions=self.data_positions, 
                              boxsize=self.boxsize, 
                              boxcenter=self.boxsize/2,
                              cellsize=cellsize,
                              nthreads=nthread)
            
            if sampling == 'randoms' and randoms is None:
                # Sample the positions on random points that we have to create in that branch
                logger.debug('No randoms provided, creating randoms')
                sampling_positions = np.random.uniform(0, self.boxsize, (nquantiles * len(self.data_positions), 3))
            elif sampling == 'randoms':
                # Sample the positions on the provided randoms
                sampling_positions = randoms
            elif sampling == 'data':
                # Sample the positions on the data positions
                sampling_positions = self.data_positions
            else:
                raise ValueError('The sampling parameter must be either "randoms" or "data"')
            
            self.density = ds.get_density_mesh(sampling_positions=sampling_positions, smoothing_radius=smooth_radius)
        
        # Temporary fix waiting for an update of the code : Remove the unphysical values (density under -1)
        self.density[self.density < -1] = -1 # Remove the outliers
        
        self.quantiles = ds.get_quantiles(nquantiles=nquantiles)

        if return_density:
            return self.quantiles, self.density
        
        return self.quantiles    



    def downsample_data(self,
                        frac: float=None,
                        new_n: float=None,
                        npoints: int=None):
        """
        Downsample the data to a new number of galaxies.
        The quantiles are also downsampled to the same number of galaxies.
        (They are generated to have the same number of points as the data in each quantile)

        Parameters
        ----------
        new_n : float
            The new number density in the box.

        Returns
        -------
        data_positions : np.ndarray
            The positions of the galaxies after downsampling. It is a (N,3) array.
            
        quantiles : np.ndarray, optional
            The quantiles after downsampling. It is a (nquantiles,N,3) array.
            Returns None if the quantiles have not been computed yet.
        """
        logger = logging.getLogger('CorrHOD') # Log some info just in case
        
        # First, check that only one of the three parameters is set
        if np.sum([frac is not None, new_n is not None, npoints is not None]) != 1:
            raise ValueError('Only one of the parameters frac, new_n and npoints must be set.')
        
        if not hasattr(self, 'nbar'):
            self.nbar = len(self.data_positions) / self.boxsize**3
        
        N = len(self.data_positions)
        
        # Then, get the other parameters from the one that is set
        if frac is not None:
            npoints = int(frac * N)
            new_n = self.nbar * frac
        if npoints is not None:
            frac = npoints / N
            new_n = self.nbar * frac
        if new_n is not None:
            frac = new_n / self.nbar
            npoints = int(frac * N)
        
        if new_n is None or new_n > self.nbar : # Do nothing
            logger.warning(f'Data not downsampled due to number density {new_n} too small or None')
            return self.data_positions, self.quantiles
        
        wanted_number = int(new_n*self.boxsize**3) # Get the wanted number of galaxies in the box
        sample_indices = np.random.choice(len(self.data_positions), size=wanted_number, replace=False)
        self.data_positions = self.data_positions[sample_indices]
        
        logger.info(f'Downsampling the data to a number density of {new_n:.2e} h^3/Mpc^3: {len(self.data_positions)} galaxies remaining from {N} galaxies')
        
        if not hasattr(self, 'quantiles'):
            return self.data_positions, None
        
        # Downsample the quantiles to the new number of galaxies (with the same proportions)
        for i in range(len(self.quantiles)):
            wanted_number = len(self.data_positions)
            sample_indices = np.random.choice(len(self.quantiles[i]), size=wanted_number, replace=False)
            self.quantiles[i] = self.quantiles[i][sample_indices]
        
        return self.data_positions, self.quantiles


    
    def compute_auto_corr(self,
                          quantile:int,
                          mode:str = 'smu',
                          edges:list = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                          mpicomm = None,
                          mpiroot = None,
                          nthread:int = 16):
        """
        Compute the auto-correlation of a quantile.
        The result will also be saved in the class, as `self.CF['Auto'][f'DS{quantile}']`

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
        
        if not hasattr(self, 'quantiles'):
            raise ValueError('The quantiles have not been computed yet. Run compute_DensitySplit first.')
        
        # Get the positions of the points in the quantile
        quantile_positions = self.quantiles[quantile] # An array of 3 columns (x,y,z)
        
        # Initialize the dictionary for the correlations
        if not hasattr(self, 'CF'):
            self.CF = {} 
        if not (self.los in self.CF.keys()):
            self.CF[self.los] = {} 
        if not ('Auto' in self.CF[self.los].keys()):
            self.CF[self.los]['Auto'] = {}
            
        # Initialize the dictionary for the pycorr objects
        if not hasattr(self, 'xi'):
            self.xi = {}
        if not (self.los in self.xi.keys()):
            self.xi[self.los] = {}
        if not ('Auto' in self.xi[self.los].keys()):
            self.xi[self.los]['Auto'] = {}
        
        # Compute the 2pcf
        xi_quantile = TwoPointCorrelationFunction(mode, edges,
                                                  data_positions1=quantile_positions,
                                                  boxsize=self.boxsize,
                                                  los=self.los,
                                                  position_type='pos',
                                                  mpicomm=mpicomm, mpiroot=mpiroot, num_threads=nthread) 
        
        # Add the 2pcf to the dictionary
        if not ('s' in self.CF[self.los]):
            # Note that the s is the same for all the lines of sight as long as we give the same edges to the 2PCF function
            s, poles = project_to_multipoles(xi_quantile)
            self.CF[self.los]['s'] = s
        else:
            poles = project_to_multipoles(xi_quantile, return_sep=False)
            
        self.CF[self.los]['Auto'][f'DS{quantile}'] = poles 
        self.xi[self.los]['Auto'][f'DS{quantile}'] = xi_quantile 
        
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
        The result will also be saved in the class, as `self.CF['Cross'][f'DS{quantile}']`
        
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
        
        if not hasattr(self, 'quantiles'):
            raise ValueError('The quantiles have not been computed yet. Run compute_DensitySplit first.')
        
        # Get the positions of the points in the quantile
        quantile_positions = self.quantiles[quantile] # An array of 3 columns (x,y,z)
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        # Initialize the dictionary for the correlations
        if not hasattr(self, 'CF'):
            self.CF = {} 
        if not (self.los in self.CF.keys()):
            self.CF[self.los] = {} 
        if not ('Cross' in self.CF[self.los].keys()):
            self.CF[self.los]['Cross'] = {}
            
        # Initialize the dictionary for the pycorr objects
        if not hasattr(self, 'xi'):
            self.xi = {}
        if not (self.los in self.xi.keys()):
            self.xi[self.los] = {}
        if not ('Cross' in self.xi[self.los].keys()):
            self.xi[self.los]['Cross'] = {}
        
        # Compute the 2pcf
        xi_quantile = TwoPointCorrelationFunction(mode, edges,
                                                  data_positions1=quantile_positions,
                                                  data_positions2=self.data_positions,
                                                  boxsize=self.boxsize,
                                                  los=self.los,
                                                  position_type='pos',
                                                  mpicomm=mpicomm, mpiroot=mpiroot, num_threads=nthread)
        
        # Add the 2pcf to the dictionary
        if not ('s' in self.CF[self.los]):
            # Note that the s is the same for all the lines of sight as long as we give the same edges to the 2PCF function
            s, poles = project_to_multipoles(xi_quantile)
            self.CF[self.los]['s'] = s
        else:
            poles = project_to_multipoles(xi_quantile, return_sep=False)
            
        self.CF[self.los]['Cross'][f'DS{quantile}'] = poles 
        self.xi[self.los]['Cross'][f'DS{quantile}'] = xi_quantile 
        
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
        
        # Initialize the dictionary for the correlations
        if not hasattr(self, 'CF'):
            self.CF = {} 
        if not (self.los in self.CF.keys()):
            self.CF[self.los] = {} 
        if not ('2PCF' in self.CF[self.los].keys()):
            self.CF[self.los]['2PCF'] = {}
        
        # Initialize the dictionary for the pycorr objects
        if not hasattr(self, 'xi'):
            self.xi = {}
        if not (self.los in self.xi.keys()):
            self.xi[self.los] = {}
        if not ('Auto' in self.xi[self.los].keys()):
            self.xi[self.los]['2PCF'] = {}
    
        # Compute the 2pcf
        xi = TwoPointCorrelationFunction(mode, edges, 
                                         data_positions1 = self.data_positions, 
                                         boxsize = self.boxsize, 
                                         los = self.los, 
                                         position_type = 'pos', 
                                         mpicomm = mpicomm, mpiroot = mpiroot, num_threads = nthread)
        
        # Add the 2pcf to the dictionary
        if not ('s' in self.CF[self.los]):
            # Note that the s is the same for all the lines of sight as long as we give the same edges to the 2PCF function
            s, poles = project_to_multipoles(xi)
            self.CF[self.los]['s'] = s
        else:
            poles = project_to_multipoles(xi, return_sep=False)
            
        self.CF[self.los]['2PCF'] = poles
        self.xi[self.los]['2PCF'] = xi
        
        return xi
    
    
    
    def average_CF(self,
                   average_on:list = ['x', 'y', 'z']
                   ):
        """
        Averages the 2PCF, autocorrelation and cross-correlation of the quantiles on the three lines of sight. available (x, y and z)
        
        Parameters
        ----------
        average_on : list, optional
            The lines of sight to average on. Defaults to ['x', 'y', 'z']. The CFs must have been computed on these lines of sight before calling this function.
            
        Returns
        -------
        CF_average : dict
            Dictionary containing the averaged 2PCF, autocorrelation and cross-correlation of the quantiles on the lines of sight. 
            It contains :
            * `s` : the separation bins (identical for all the lines of sight, as long as the same edges are used for the 2PCF),
            * `2PCF` : the averaged poles of the 2PCF,
            * `Auto` : a dictionary containing the averaged autocorrelation of the quantiles. It contains for each quantile `DS{quantile}` as the poles of the autocorrelation,
            * `Cross` : a dictionary containing the averaged cross-correlation of the quantiles. It contains for each quantile `DS{quantile}` as the poles of the cross-correlation.
            
        """
        
        los_list = average_on
        
        # Initialize the dictionary for the averaged correlations
        self.CF['average'] = {}
        
        # Check if the 2PCF, autocorrelation and cross-correlation of the quantiles are already computed
        if not hasattr(self, 'CF'):
            raise ValueError('No correlation has been computed yet. Run compute_2pcf, compute_auto_corr and/or compute_cross_corr first.')
        for los in los_list:
            if not (los in self.CF.keys()):
                raise ValueError(f'The {los} line of sight has not been computed yet. Run compute_2pcf, compute_auto_corr and/or compute_cross_corr first.')
            if not ('2PCF' in self.CF[los].keys()):
                raise ValueError(f'The 2PCF of the {los} line of sight has not been computed yet. Run compute_2pcf first.')
            if not ('Auto' in self.CF[los].keys()):
                raise ValueError(f'The autocorrelation of the {los} line of sight has not been computed yet. Run compute_auto_corr first.')
            if not ('Cross' in self.CF[los].keys()):
                raise ValueError(f'The cross-correlation of the {los} line of sight has not been computed yet. Run compute_cross_corr first.')
        
        # Note that the s is the same for all the lines of sight as long as we give the same edges to the 2PCF function
        s = self.CF[los_list[0]]['s'] # Get the separation bins (Same for all the CFs, we take the first one available)
        self.CF['average']['s'] = s
        
        poles=[]
        # Average the 2PCF
        for los in los_list:
            poles.append(self.CF[los]['2PCF'])
        self.CF['average']['2PCF'] = np.mean(poles, axis=0)
        
        # Average the autocorrelation of the quantiles
        self.CF['average']['Auto'] = {}
        for quantile in range(len(self.quantiles)):
            poles = []
            for los in los_list:
                poles.append(self.CF[los]['Auto'][f'DS{quantile}'])
            self.CF['average']['Auto'][f'DS{quantile}'] = np.mean(poles, axis=0)
        
        # Average the cross-correlation of the quantiles
        self.CF['average']['Cross'] = {}
        for quantile in range(len(self.quantiles)):
            poles = []
            for los in los_list:
                poles.append(self.CF[los]['Cross'][f'DS{quantile}'])
            self.CF['average']['Cross'][f'DS{quantile}'] = np.mean(poles, axis=0)
    
        return self.CF['average']
    
    
    
    def save(self,
             hod_indice:int = 0,
             path:str = None,
             save_HOD:bool = True,
             save_pos:bool = True,
             save_density:bool = True,
             save_quantiles:bool = True,
             save_CF:bool = True,
             save_xi:bool = True,
             los:str = 'average', 
             save_all:bool = False):
        """
        Save the results of the CorrHOD object. 
        Some of the results are saved as dictionaries in numpy files. They can be accessed using the `np.load` function,
        and the .item() method of the loaded object. 

        Parameters
        ----------
        hod_indice : int, optional
            The indice of the set of HOD parameters provided to the CorrHOD Object. Defaults to 0.
            Be careful to set the right indice if you want to save the HOD parameters !
            
        path : str, optional
            The path where to save the results. If None, the path is set to the output_dir of config file. Defaults to None.
            
        save_HOD : bool, optional
            If True, the HOD parameters are saved. 
            File saved as `hod{hod_indice}_c{cosmo}_p{phase}.npy`. Defaults to True.
            
        save_pos : bool, optional
            If True, the cubic mock dictionary is saved. 
            File saved as `pos_hod{hod_indice}_c{cosmo}_p{phase}.npy`. Defaults to True.
            
        save_density : bool, optional
            If True, the density PDF is saved. 
            File saved as `density_hod{hod_indice}_c{cosmo}_p{phase}.npy`. Defaults to True.
            
        save_quantiles : bool, optional
            If True, the quantiles of the densitysplit are saved. 
            File saved as `quantiles_hod{hod_indice}_c{cosmo}_p{phase}.npy`. Defaults to True.
            
        save_CF : bool, optional
            If True, the 2PCF, the autocorrelation and cross-correlation of the quantiles are saved. 
            The 2PCF is saved as `tpcf_hod{hod_indice}_c{cosmo}_p{phase}.npy`.
            It contains the separation bins in the `s` key and the poles in the `2PCF` key.
            The Auto and Corr dictionaries are saved as `ds_auto_hod{hod_indice}_c{cosmo}_p{phase}.npy` and `ds_cross_hod{hod_indice}_c{cosmo}_p{phase}.npy`.
            Each dictionnary contains `DS{quantile}` keys with the CF of the quantile. The `s` key contains the separation bins.
            Defaults to True.
            
        save_xi : bool, optional
            If True, the pycorr objects are saved in an unique dictionary.
            The 2PCF can be accessed using the `xi[los]['2PCF']` key.
            The auto and cross-correlations of the quantiles can be accessed using the `xi[los]['Auto']['DS{i}']` and `xi[los]['Cross']['DS{i}']` keys.
            Defaults to True.
            
        los : str, optional
            The line of sight along which to save the 2PCF, the autocorrelation and cross-correlation of the quantiles. 
            Can be 'x', 'y', 'z' or 'average'. Defaults to 'average'.
            
        save_all : bool, optional
            If True, all the results are saved. This overrides the other options. Defaults to False.
        """
        
        
        if path is None:
            output_dir = Path(self.sim_params['output_dir'])
        else :
            output_dir = Path(path)
            
        sim_name = self.sim_params['sim_name']
        
        # Get the cosmo and phase from sim_name (Naming convention has to end by '_c{cosmo}_p{phase}' !)
        cosmo = sim_name.split('_')[-2].split('c')[-1] # Get the cosmology number by splitting the name of the simulation
        phase = sim_name.split('_')[-1].split('ph')[-1] # Get the phase number by splitting the name of the simulation
        
        # Get the HOD indice in the right format (same as the cosmology and phase)
        hod_indice = f'{hod_indice:03d}'
        
        # Define the base names of the files
        base_name = f'hod{hod_indice}_{los}_c{cosmo}_ph{phase}.npy'
        hod_name = f'hod{hod_indice}_c{cosmo}_ph{phase}.npy' # HOD parameters don't depend on the line of sight
        cubic_name = f'pos_hod{hod_indice}_c{cosmo}_ph{phase}.npy' # The cubic dictionary does not depend on the line of sight either
        
        # Note : If the user explicitly wants to save somethig, it is assumed that it has been computed before.
        # No error is explicitly raised if the user tries to save something that has not been computed yet.
        # If the user wants to save everything, the results are saved only if they exist.
        
        if save_HOD or (save_all and hasattr(self, 'HOD_params')):
            # Add the number density to the HOD parameters
            self.HOD_params['nbar'] = self.nbar
            path = output_dir / 'hod' 
            path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
            np.save(path / hod_name, self.HOD_params)
        
        if save_pos or (save_all and hasattr(self, 'cubic_dict')):
            # Pass if the cubic dictionary has not been computed yet
            if not hasattr(self, 'cubic_dict'):
                warn('The cubic dictionary has not been computed yet. Run populate_halos first.', UserWarning)
                pass
            path = output_dir 
            path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
            np.save(path / cubic_name, self.cubic_dict)
                    
        if save_density or (save_all and hasattr(self, 'density')):
            path = output_dir / 'ds' / 'density'
            path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
            np.save(path / (f'density_' + base_name), self.density)
        
        if save_quantiles or (save_all and hasattr(self, 'quantiles')):
            path = output_dir / 'ds' / 'quantiles'
            path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
            np.save(path / (f'quantiles_' + base_name), self.quantiles)
            
        if save_xi or (save_all and hasattr(self, 'xi')):
            path = output_dir / 'xi'
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / (f'xi_' + base_name), self.xi)
        
        if save_CF or (save_all and hasattr(self, 'CF')):
            
            # First, we check that the provided los is expected
            if los not in ['x', 'y', 'z', 'average']:
                raise ValueError(f'The line of sight must be "x", "y", "z" or "average". Got {los}.')
            
            # Check that the CF has been computed on the provided los
            if los not in self.CF.keys():
                raise ValueError(f'The {los} line of sight has not been computed yet. Run compute_2pcf, compute_auto_corr and/or compute_cross_corr first.', UserWarning)
            
            # From now on, we only save the CFs that have been computed on the provided los
            if '2PCF' in self.CF[los].keys():
                tpcf_dict = {'s': self.CF[los]['s'], '2PCF': self.CF[los]['2PCF']} 
            
                path = output_dir / 'tpcf'
                path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
                np.save(path / (f'tpcf_' + base_name), tpcf_dict)
            
            if 'Auto' in self.CF[los].keys():
                auto_dict = {'s': self.CF[los]['s'], **self.CF[los]['Auto']}
                
                path = output_dir / 'ds' / 'gaussian'
                path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
                np.save(path / (f'ds_auto_' + base_name), auto_dict)
            
            if 'Cross' in self.CF[los].keys():
                cross_dict = {'s': self.CF[los]['s'], **self.CF[los]['Cross']}
                path = output_dir / 'ds' / 'gaussian'
                path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
                np.save(path / (f'ds_cross_' + base_name), cross_dict)
        
        
        
    def run_all(self,
                is_populated:bool = False,
                exit_on_underpopulated:bool = False,
                los_to_compute='average',
                downsample_to:float = None,
                remove_Ball:bool = False,
                # Parameters for the DensitySplit
                smooth_radius:float = 10,
                cellsize:float = 5,
                nquantiles:int = 10,
                sampling:str = 'randoms',
                filter_shape:str = 'Gaussian',
                # Parameters for the 2PCF, autocorrelation and cross-correlations
                edges = [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)],
                mpicomm = None,
                mpiroot = None,
                nthread:int = 16,
                # Parameters for saving the results
                hod_indice:int = 0,
                path:str = None,
                **kwargs
                ):
        """
        Run all the steps of the CorrHOD object. See tests/test_corrHOD.py for an example of how to use it and its contents.
        
        Note that the **kwargs are used to set the parameters of the save function. If nothing is provided, nothing will be saved.
        See the documentation of the save function for more details.
        
        Times will be saved in the `times_dict` attribute of the CorrHOD object and displayed in the logs.

        Parameters
        ----------
        los_to_compute : str, optional
            The line of sight along which to compute the 2PCF, the autocorrelation and cross-correlation of the quantiles.
            If set to 'average', the 2PCF, the autocorrelation and cross-correlation of the quantiles will be averaged on the three lines of sight. Defaults to 'average'.
           
        is_populated : bool, optional
            If True, the halos have already been populated before running this method. Defaults to False.
        
        exit_on_underpopulated : bool, optional
            If True, the function will exit if the halos are underpopulated compared to the requested number density. Defaults to False.
        
        downsample_to : float, optional
            If provided, the galaxies will be randomly downsampled to the provided number density in h^3/Mpc^3 before computing the CFs. 
            This can be useful to reduce the computation time.
            Defaults to None.
            
        remove_Ball : bool, optional
            If True, the Ball object (Dark matter simulation loaded from AbacusHOD) is removed after populating the halos. 
            This is useful to save memory. Defaults to False.
            
        smooth_radius : float, optional
            The radius of the Gaussian smoothing in Mpc/h used in the densitysplit. 
            See https://github.com/epaillas/densitysplit for more details. Defaults to 10.
            
        cellsize : float, optional
            The size of the cells in the mesh used in the densitysplit. 
            See https://github.com/epaillas/densitysplit for more details. Defaults to 10.
            
        nquantiles : int, optional
            The number of quantiles to define in the densitysplit. 
            see https://github.com/epaillas/densitysplit for more details. Defaults to 10.
            
        sampling : str, optional
            The type of sampling to use in the densitysplit. 
            See https://github.com/epaillas/densitysplit for more details. Defaults to 'randoms'.
            
        filter_shape : str, optional
            The shape of the smoothing filter to use in the densitysplit.
            see https://github.com/epaillas/densitysplit for more details. Defaults to 'Gaussian'.
            
        edges : list, optional
            The edges of the s, mu bins to use for the 2PCF, autocorrelation and cross-correlations. Defaults to [np.linspace(0.1, 200, 50), np.linspace(-1, 1, 60)].
            
        mpicomm : _type_, optional
            The MPI communicator used in the 2PCF, autocorrelation and cross-correlations. Defaults to None.
            
        mpiroot : _type_, optional
            The MPI root used in the 2PCF, autocorrelation and cross-correlations. Defaults to None.
            
        nthread : int, optional
            The number of threads to use in the 2PCF, autocorrelation and cross-correlations. Defaults to 16.
            
        hod_indice : int, optional
            The indice of the set of HOD parameters provided to the CorrHOD Object. Defaults to 0.
            Be careful to set the right indice if you want to save the HOD parameters !
            
        path : str, optional
            The path where to save the results. If None, the path is set to the output_dir of config file. Defaults to None.
            
        **kwargs : dict, optional
            The parameters to pass to the save function. See the documentation of the save function for more details.
            If nothing is provided, nothing will be saved.
        """
        
        logger = logging.getLogger('CorrHOD') # Get the logger for the CorrHOD class
        
        # Get the MPI communicator and rank
        if mpicomm is not None and mpiroot is not None:
            size = mpicomm.size # Number of processes
            rank = mpicomm.Get_rank() # Rank of the current process (root process has rank 0)
            root = (rank == mpiroot) # True if the current process is the root process
        else:
            root = True
            rank = 0
        
        if los_to_compute=='average':
            los_list = ['x', 'y', 'z']
        else:
            los_list = [los_to_compute]
        
        # The Barrier and bcast method can only be called if the communicator is not None
        if mpicomm is not None:
            mpicomm.Barrier() # Wait for all the processes to reach this point before starting
        start_time = time()
        self.times_dict = {} # Saving the times taken by each step in a dict to call them later if needed
        
        if root:
            logger.info(f'Running CorrHOD with the following parameters ({hod_indice}) :')
            logger.info(f"\t Simulation : {self.sim_params['sim_name']}")
            for key in self.HOD_params.keys():
                logger.info(f'\t {key} : {self.HOD_params[key]}')
            if self.data_params is not None:
                logger.info(f"\t Number density : {self.data_params['tracer_density_mean'][self.tracer]:.2e} h^3/Mpc^3")
            logger.newline()
        
        if root and not is_populated:
            logger.info('Initializing and populating the halos ...')
            self.initialize_halo() # Initialize the halo
        
            self.times_dict['initialize_halo'] = time()-start_time
            logger.info(f"Initialized the halos in {self.times_dict['initialize_halo']:.2f} s\n")

            # Catch the UserWarning raised by the densitysplit if the halos are underpopulated
            with catch_warnings():
                filterwarnings("error") 
                try : 
                    self.populate_halos(nthread=nthread, remove_Ball=remove_Ball) # Populate the halos
                except UserWarning as w: # Catch the warning if needed
                    logger.warning('Warning : ' + str(w)) # Display the warning without triggering the error
                    if exit_on_underpopulated: 
                        logger.info('The halos are underpopulated. Exiting the function as requested.\n')
                        return None # Exit the function to avoid crashing the code
            
            logger.newline() # Just to add a space because populate_halos has a built-in print I can't remove
        elif not is_populated:
            self.cubic_dict = None
        elif root:
            logger.info('The halos have already been populated. Skipping the initialization and population steps.\n')

        # Here, we run all the CF computations for each los given in los_to_compute.
        # We will then average the results on the lines of sight if needed
        for los in los_list:
            
            los_time = time()
            self.times_dict[los] = {} # Initialize the dict for the times taken by each step for this los
            
            self.los = los # Reassign the los we will work on
            
            if root:
                logger.info(f'Computing along the {los} line of sight ...')
                logger.newline()
                
                self.get_tracer_positions() # Get the positions of the galaxies with RSD on the line of sight
            
                # Compute the DensitySplit
                logger.info('Computing the DensitySplit ...')
                tmp_time = time()
                self.compute_DensitySplit(smooth_radius=smooth_radius, 
                                          cellsize=cellsize, 
                                          nquantiles=nquantiles,
                                          sampling=sampling, 
                                          filter_shape=filter_shape,
                                          return_density=False,
                                          nthread=nthread)
                
                self.times_dict[los]['compute_DensitySplit'] = time()-tmp_time
                logger.info(f"Computed the DensitySplit in {self.times_dict[los]['compute_DensitySplit']:.2f} s\n")
                
                # Downsample the galaxies if needed
                if downsample_to is not None:
                    self.downsample_data(new_n=downsample_to)
                    logger.info(f'Downsampled the galaxies to {downsample_to:.2e} h^3/Mpc^3\n')
                    
            else:
                self.data_positions = None
                self.density = None
                self.quantiles = None
            
            if mpicomm is not None:
                self.data_positions = mpicomm.bcast(self.data_positions, root=mpiroot) # Broadcast the positions of the galaxies to all the processes
                self.quantiles = mpicomm.bcast(self.quantiles, root=mpiroot) # Broadcast the quantiles to all the processes
            
            if rank == 1:
                logger.debug(f'(Broadcast test) Number of galaxies : {len(self.data_positions)} on rank {rank}')
                logger.newline()
            
            # Compute the 2PCF
            if root : 
                logger.info('Computing the 2PCF ...')
            tmp_time = time()
            self.compute_2pcf(edges=edges, mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
            
            self.times_dict[los]['compute_2pcf'] = time()-tmp_time
            if root:
                logger.info(f"Computed the 2PCF in {self.times_dict[los]['compute_2pcf']:.2f} s\n")
            
            # For each quantile, compute the autocorrelation and cross-correlation
            self.times_dict[los]['compute_auto_corr'] = {}
            self.times_dict[los]['compute_cross_corr'] = {}
            
            for quantile in range(nquantiles):
                if root:
                    logger.info(f'Computing the auto-correlation and cross-correlation of quantile {quantile} ...')
                tmp_time = time()
                self.compute_auto_corr(quantile, edges=edges,mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
                
                self.times_dict[los]['compute_auto_corr'][f'DS{quantile}'] = time()-tmp_time
                if root:
                    logger.info(f"Computed the auto-correlation of quantile {quantile} in {self.times_dict[los]['compute_auto_corr'][f'DS{quantile}']:.2f} s")
                
                tmp_time = time()
                self.compute_cross_corr(quantile, edges=edges,mpicomm=mpicomm, mpiroot=mpiroot, nthread=nthread)
                
                self.times_dict[los]['compute_cross_corr'][f'DS{quantile}'] = time()-tmp_time
                if root:
                    logger.info(f"Computed the cross-correlation of quantile {quantile} in {self.times_dict[los]['compute_cross_corr'][f'DS{quantile}']:.2f} s")
            
            self.times_dict[los]['run_los'] = time()-los_time
            if root:
                logger.newline()
                logger.info(f"Ran los '{los}' in {self.times_dict[los]['run_los']:.2f} s\n")
        
        if root: 
            # Average the 2PCF, autocorrelation and cross-correlation of the quantiles on the lines of sight  
            self.average_CF(average_on=los_list) 
        
        if mpicomm is not None:
            mpicomm.Barrier() # Wait for all the processes to reach this point before ending the timer
        self.times_dict['run_all'] = time()-start_time
        if root:
            logger.info(f"Run_all in {self.times_dict['run_all']:.2f} s")
        
        # Here, we define the arguments to pass to the save function by default (nothing will be saved)
        save_args = {
            'hod_indice': hod_indice,
            'path': path,
            'save_HOD': False,
            'save_pos': False,
            'save_density': False,
            'save_quantiles': False,
            'save_CF': False,
            'save_xi': False,
            'los': 'average', # We save the averaged CFs by default
            'save_all': False
        }
        # We update the arguments with the kwargs provided by the user
        for key, value in kwargs.items():
            if key in save_args.keys():
                save_args[key] = value
            else:
                # If the argument is not an argument of the save function, we warn the user
                warn(f'Unknown argument {key}={value} in run_all. It will be ignored.', UserWarning)
        
        # Save the results
        if root:
            if save_args['los']=='all':
                del save_args['los'] # We will manually save the results for each los
                self.save(los='average', **save_args) 
                self.save(los='x', **save_args)
                self.save(los='y', **save_args)
                self.save(los='z', **save_args)
            else:
                self.save(**save_args)