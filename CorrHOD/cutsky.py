import numpy as np
import logging
from pathlib import Path

from warnings import warn

from densitysplit.pipeline import DensitySplit

from pycorr import TwoPointCorrelationFunction, project_to_multipoles

from CorrHOD.cubic import CorrHOD_cubic
from CorrHOD.weights import n_z, w_fkp, get_quantiles_weight, sky_fraction

# Imports specific to packages versions
try:
    from densitysplit.utilities import sky_to_cartesian, cartesian_to_sky
except ImportError: # For main branch of densitysplit
    from densitysplit.utils import sky_to_cartesian, cartesian_to_sky

class CorrHOD_cutsky(CorrHOD_cubic):
    """
    This class is used to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles of the DensitySplit.
    It takes HOD parameters and a cosmology as input and uses AbacusHOD to generate a DESI cutsky mock.
    It then uses DensitySplit to compute the density field and the quantiles.
    Finally, it uses pycorr to compute the 2PCF and the autocorrelation and cross-correlation of the quantiles.
    
    If a cutsky file is to be provided, the cutsky must be in the form of a dictionary in sky coordinates.
    Every computation in this class is done in sky coordinates, except for the densitysplit. The conversion is done using the densitysplit package.
    Unless the name of the variable indicates it, the positions are in (ra, dec, comoving distance) coordinates. If the name of the variable contains 'sky', the positions are in (ra, dec, z) coordinates.
    """
    
    # __init__ method is the same as in the CorrHOD_cubic class
        
    # The initialize_halos method is the same as in the CorrHOD_cubic class
    
    # The populate_halos method is the same as in the CorrHOD_cubic class
    
    
    
    # TODO (last) : Create the cutsky from the box --> Mockfactory package ! 
    
    # TODO : Add a way to read the cutsky from a file 
    
    # TODO : create_randoms method
    
    
    
    
    def get_tracer_positions(self,
                             return_cartesian:bool=False):
        """
        Get the positions of the tracers (data and randoms) in cartesian coordinates.
        Note : For now, we assume that the RSD are already applied to the data
        
        Returns
        -------
        data_positions : array_like
            The positions of the galaxies in cartesian coordinates.
            
        randoms_positions : array_like
            The positions of the randoms in cartesian coordinates.
        """       
        
        # Create these parameters for readability and make sure that we have the right dict passed
        try:
            data = self.cutsky_dict[self.tracer] 
            randoms = self.randoms_dict[self.tracer]
        except:
            # Handle the case where we have set the dicts without using the set_cutsky method
            warn('The object is not a hod_dict. Trying to load it as a positional dataset.', UserWarning)

            data = self.cutsky_dict
            randoms = self.randoms_dict
            
            # Check that the data contains the right keys
            if not ( all([key in data.keys() for key in ['RA', 'DEC', 'Z']]) or all([key in randoms.keys() for key in ['RA', 'DEC', 'Z']]) ):
                raise ValueError('The object is not a hod_dict and does not contain the keys "RA", "DEC", "Z"')
        
        data_cd = self.cosmo.comoving_radial_distance(data['Z']) # Convert the z to comoving distance
        random_cd = self.cosmo.comoving_radial_distance(randoms['Z']) # Convert the z to comoving distance
        
        # Concatenate the positions with the comoving distance
        self.data_positions = np.c_[data['RA'], data['DEC'], data_cd]
        self.randoms_positions = np.c_[randoms['RA'], randoms['DEC'], random_cd]
        
        # Concatenate the positions in sky coordinates
        self.data_sky = np.c_[data['RA'], data['DEC'], data['Z']]
        self.randoms_sky = np.c_[randoms['RA'], randoms['DEC'], randoms['Z']]
        
        # Use densitysplit to convert to cartesian coordinates
        self.data_cartesian = sky_to_cartesian(self.data_sky, self.ds_cosmo)
        self.randoms_cartesian = sky_to_cartesian(self.randoms_sky, self.ds_cosmo)

        if return_cartesian:
            return self.data_cartesian, self.randoms_cartesian
    
        return self.data_positions, self.randoms_positions
    
    
    
    def get_tracer_weights(self,
                    edges:list=None,
                    area:float=None,
                    fsky:float=None,
                    P0:float=7000):
        """
        Compute the weights for the tracer and the randoms.
        Note : If neither `area` nor `fsky` are set, the sky fraction of the data will be computed, using the current data sky positions.
        
        Parameters
        ----------
        edges: list, optional
            The edges of the bins used to compute the number density. 
            If set to `None`, the edges are computed using Scott's rule. Defaults to `None`.
        
        area: float, optional
            The area of the survey in square degrees. Defaults to `None`.
            
        fsky: float, optional
            The fraction of the sky covered by the survey. Defaults to `None`.
        
        P0: float, optional
            The power spectrum normalization (TODO : Check this definition). Defaults to `7000` for the BGS.        
        
        Returns
        -------
        data_weights : array_like
            The weights of the galaxies.
            
        randoms_weights : array_like
            The weights of the randoms.
        """
        # Get the sky fraction if it is not set
        if fsky is None and area is None:
            ra = self.randoms_sky[:,0]
            dec = self.randoms_sky[:,1]
            fsky = sky_fraction(ra, dec)
        
        z_data = self.data_positions[:,2]
        z_random = self.randoms_positions[:,2]        
        
        # Compute the weights
        self.data_weights, self.randoms_weights = w_fkp(z_data, z_random, self.cosmo, edges=edges, area=area, fsky=fsky, P0=P0)
 
        if hasattr(self, 'quantiles'):
            # Compute the weights for the quantiles
            self.quantiles_weights = get_quantiles_weight(self.density, self.randoms_weights, nquantiles=len(self.quantiles))
            return self.data_weights, self.randoms_weights, self.quantiles_weights
            
        return self.data_weights, self.randoms_weights
    
    
    def compute_DensitySplit(self, 
                             smooth_radius: float = 10, 
                             cellsize: float = 5, 
                             nquantiles: int = 10, 
                             sampling: str = 'randoms', 
                             randoms=None, 
                             filter_shape: str = 'Gaussian', 
                             return_density: bool = True, 
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
            If set to none, a the randoms positions of the data will be used. 
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
        
        # Set the weights of the galaxies to 1 if they are not set
        if not hasattr(self, 'data_weights'):
            self.data_weights = np.ones(len(self.data_positions))
        if not hasattr(self, 'randoms_weights'):
            self.randoms_weights = np.ones(len(self.randoms_positions))

        try: # Main branch
            logger.debug('Launched densitysplit on main branch')
            
            ds = DensitySplit(data_positions=self.data_cartesian, 
                              data_weights=self.data_weights,
                              randoms_positions=self.randoms_cartesian,
                              randoms_weights=self.randoms_weights)

            self.density = ds.get_density_mesh(smooth_radius=smooth_radius, cellsize=cellsize, sampling=sampling, filter_shape=filter_shape)
        except : #OpenMP branch (an error will be raised because the OpenMP branch used differently)
            logger.debug('Launched densitysplit on openmp branch') #tmp
            
            ds = DensitySplit(data_positions=self.data_cartesian, 
                              data_weights=self.data_weights,
                              randoms_positions=self.randoms_cartesian,
                              randoms_weights=self.randoms_weights,
                              boxpad=1.1,
                              cellsize=cellsize,
                              nthreads=nthread)

            if sampling == 'randoms' and randoms is None:
                # Sample the positions on random points that we have to create in that branch
                logger.debug('No randoms provided, creating randoms')
                sampling_positions = self.randoms_cartesian
            elif sampling == 'randoms':
                # Sample the positions on the provided randoms
                sampling_positions = randoms
            elif sampling == 'data':
                # Sample the positions on the data positions
                sampling_positions = self.data_cartesian
            else:
                raise ValueError('The sampling parameter must be either "randoms" or "data"')
            
            self.density = ds.get_density_mesh(sampling_positions=sampling_positions, smoothing_radius=smooth_radius)
        
        # Temporary fix waiting for an update of the code : Remove the unphysical values (density under -1)
        self.density[self.density < -1] = -1 # Remove the outliers
        
        # Compute the quantiles in cartesian coordinates
        self.quantiles_cartesian = ds.get_quantiles(nquantiles=nquantiles)
        
        # Convert the quantiles to sky coordinates
        self.quantiles=[]
        self.quantiles_sky = []
        for i in range(nquantiles):
            self.quantiles_sky.append(cartesian_to_sky(self.quantiles_cartesian[i], self.ds_cosmo)) # Save the quantiles in sky coordinates (for the n(z) computation)
            sky_quantile = cartesian_to_sky(self.quantiles_cartesian[i], self.ds_cosmo) # Convert the quantiles to sky coordinates
            sky_quantile[:,2] = self.cosmo.comoving_radial_distance(sky_quantile[:,2]) # Convert the z to comoving distance
            self.quantiles.append(sky_quantile) # Save the quantiles with the comoving distance

        if return_density:
            return self.quantiles, self.density
        
        return self.quantiles
    
    
    def get_nz(self,
               edges:list=None,
               area:float=None,
               fsky:float=None):
        """
        Computes the n(z) function of the data and the quantiles (if they exist).
        Note : If neither `area` nor `fsky` are set, the sky fraction of the data will be computed, using the current data sky positions.

        Parameters
        ----------
        edges: list, optional
            The edges of the bins used to compute the number density. 
            If set to `None`, the edges are computed using Scott's rule. Defaults to `None`.
        
        area: float, optional
            The area of the survey in square degrees. Defaults to `None`.
        
        fsky: float, optional
            The fraction of the sky covered by the survey. Defaults to `None`.

        Returns
        -------
        nz_data : InterpolatedUnivariateSpline
            The n(z) function of the data. 
            
        nz_randoms : InterpolatedUnivariateSpline
            The n(z) function of the randoms.
            
        nz_functions : list, optional
            The n(z) function for each quantile. It is a list of InterpolatedUnivariateSpline objects.
        """
        # Get the sky fraction if it is not set
        if fsky is None and area is None:
            ra = self.randoms_sky[:,0]
            dec = self.randoms_sky[:,1]
            fsky = sky_fraction(ra, dec)
        
        # Check that the mean n(z) is approx. the same for each quantile
        if hasattr(self, 'quantiles'):
            nquantiles = len(self.quantiles)
            self.nz_functions = [n_z(self.quantiles_sky[i][:,2], self.cosmo, edges=edges, area=area, fsky=fsky) for i in range(nquantiles)] # Compute the n(z) functions
        
        # Get the positions of the galaxies
        if not hasattr(self, 'data_positions'):
            self.get_tracer_positions()
        
        self.nz_data = n_z(self.data_sky[:,2], self.cosmo, edges=edges, area=area, fsky=fsky)
        
        self.nz_randoms = n_z(self.randoms_sky[:,2], self.cosmo, edges=edges, area=area, fsky=fsky)
        
        if hasattr(self, 'nz_functions') and hasattr(self, 'nz_data'):
            return self.nz_data, self.nz_randoms, self.nz_functions
        
        return self.nz_data, self.nz_randoms
    
    
    
    def downsample_data(self, 
                        frac: float=None,
                        new_n: float=None,
                        npoints: int=None):
        """
        Downsamples the data to a given number density.
        The number of points in the randoms and the quantiles will be adjusted accordingly.
        (i.e. if the number density of the data is 10 times smaller, the number of points 
        in the randoms and the quantiles will be 10 times smaller, while keeping the proportionality to the data)

        Since the number density varies with the redshift, the downsampling to the new number density is
        done by taking the mean n(z) of the data as reference.
        This should have n(z) remaining approx. the same.
        (TODO : Test this ?)

        Parameters
        ----------
        new_n : float
            The wanted number density of the data after downsampling.
            The number density of the randoms and the quantiles will be adjusted accordingly.

        Returns
        -------
        data_positions : np.ndarray
            The positions of the data after downsampling.
        
        randoms_positions : np.ndarray
            The positions of the randoms after downsampling.
        
        quantiles : np.ndarray, optional
            The quantiles after downsampling.
        """
        
        logger = logging.getLogger('CorrHOD') # Log some info just in case
        
        # First, get the n(z) of the data and quantiles if they exist
        self.get_nz()
        
        mean_n = np.mean(self.nz_data(self.data_sky[:,2])) # Get the mean n(z) of the data
        N = len(self.data_positions) # Get the number of galaxies in the data
        
        # First, check that only one of the three parameters is set
        if np.sum([frac is not None, new_n is not None, npoints is not None]) != 1:
            raise ValueError('Only one of the parameters frac, new_n and npoints must be set.')
        
        # Then, get the other parameters from the one that is set
        if frac is not None:
            npoints = int(frac * N)
            new_n = mean_n * frac
        if npoints is not None:
            frac = npoints / N
            new_n = mean_n * frac
        if new_n is not None:
            frac = new_n / mean_n
            npoints = int(frac * N)
        
        # Check that the new number density is not too small
        if mean_n < new_n or new_n is None:
            logger.warning(f'Data not downsampled due to number density {new_n:.2e} ({npoints} points) too small or None')
            if not hasattr(self, 'quantiles'):
                return self.data_positions, self.randoms_positions
            else:
                return self.data_positions, self.randoms_positions, self.quantiles
        
        # First, downsample the data
        wanted_number = int(N * new_n / mean_n) # Get the wanted number of galaxies after downsampling
        sample_indices = np.random.choice(N, size=wanted_number, replace=False) # Get the indices of the galaxies to keep
        self.data_positions = self.data_positions[sample_indices] # Keep only the selected galaxies
        self.data_weights = self.data_weights[sample_indices] # Keep only the selected weights
        self.data_sky = self.data_sky[sample_indices] # Keep only the selected galaxies in sky coordinates
        self.data_cartesian = self.data_cartesian[sample_indices] # Keep only the selected galaxies in cartesian coordinates
        
        logger.info(f'Downsampling the data to a number density of {new_n:.2e} h^3/Mpc^3: {len(self.data_positions)} galaxies remaining from {N} galaxies')
        
        # Then, downsample the randoms
        N = len(self.randoms_positions) # Get the number of galaxies in the randoms
        wanted_number = int(N * new_n / mean_n) # Get the wanted number of randoms after downsampling
        sample_indices = np.random.choice(N, size=wanted_number, replace=False) # Get the indices of the randoms to keep
        self.randoms_positions = self.randoms_positions[sample_indices] # Keep only the selected randoms
        self.randoms_weights = self.randoms_weights[sample_indices] # Keep only the selected weights
        self.randoms_sky = self.randoms_sky[sample_indices] # Keep only the selected randoms in sky coordinates
        self.randoms_cartesian = self.randoms_cartesian[sample_indices] # Keep only the selected randoms in cartesian coordinates
        
        logger.info(f'Downsampling the randoms : {len(self.randoms_positions)} randoms remaining from {N} randoms')
        
        # If the quantiles have been computed, downsample them too
        if not hasattr(self, 'quantiles'):
            return self.data_positions, self.randoms_positions
        
        # Finally, downsample the quantiles
        nquantiles = len(self.quantiles)
        wanted_number = int(len(self.randoms_positions)/nquantiles) # Get the wanted number of quantiles after downsampling
        N = np.min([len(self.quantiles[i]) for i in range(nquantiles)]) # The number of points CAN VARY (by 1 or 2) between the quantiles, so we take the smallest one
        for i in range(nquantiles):
            sample_indices = np.random.choice(N, size=wanted_number, replace=False) # Get the indices of the quantiles to keep
            self.quantiles[i] = self.quantiles[i][sample_indices] # Keep only the selected quantiles
            self.quantiles_weights[i] = self.quantiles_weights[i][sample_indices] # Keep only the selected weights
            self.quantiles_sky[i] = self.quantiles_sky[i][sample_indices] # Keep only the selected quantiles in sky coordinates
            self.quantiles_cartesian[i] = self.quantiles_cartesian[i][sample_indices] # Keep only the selected quantiles in cartesian coordinates
        
        logger.info(f'Downsampling the quantiles: {len(self.quantiles[0])} points remaining from {N} points')
        
        return self.data_positions, self.randoms_positions, self.quantiles
    
    
    
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
        
        # Get the positions of the galaxies
        if not hasattr(self, 'randoms_positions'):
            self.get_tracer_positions()
        
        # Get the positions of the points in the quantile
        quantile_positions = self.quantiles[quantile] # An array of 3 columns (ra, dec, z)
        
        # Set the weights of the galaxies to 1 if they are not set
        if not hasattr(self, 'randoms_weights'):
            self.randoms_weights = np.ones(len(self.randoms_positions))
        if not hasattr(self, 'quantiles_weights'):
            quantile_weights = np.ones(len(quantile_positions))
        else:
            quantile_weights = self.quantiles_weights[quantile]
        
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
                                                  data_positions1=quantile_positions.T, # Note the transpose to get the right shape
                                                  randoms_positions1=self.randoms_positions.T,
                                                  data_weights1=quantile_weights,
                                                  randoms_weights1=self.randoms_weights,
                                                  position_type='rdd',
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
        
        # Set the weights of the galaxies to 1 if they are not set
        if not hasattr(self, 'data_weights'):
            self.data_weights = np.ones(len(self.data_positions))
        if not hasattr(self, 'randoms_weights'):
            self.randoms_weights = np.ones(len(self.randoms_positions))
        if not hasattr(self, 'quantiles_weights'):
            quantile_weights = np.ones(len(quantile_positions))
        else:
            quantile_weights = self.quantiles_weights[quantile]
            
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
                                                  data_positions1=quantile_positions.T, # Note the transpose to get the right shape
                                                  data_positions2=self.data_positions.T,
                                                  randoms_positions1=self.randoms_positions.T,
                                                  randoms_positions2=self.randoms_positions.T,
                                                  data_weights1=quantile_weights,
                                                  data_weights2=self.data_weights,
                                                  randoms_weights1=self.randoms_weights,
                                                  randoms_weights2=self.randoms_weights,
                                                  position_type='rdd',
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
        
        # Set the weights of the galaxies to 1 if they are not set
        if not hasattr(self, 'data_weights'):
            self.data_weights = np.ones(len(self.data_positions))
        if not hasattr(self, 'randoms_weights'):
            self.randoms_weights = np.ones(len(self.randoms_positions))
        
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
                                         data_positions1 = self.data_positions.T, # Note the transpose to get the right shape 
                                         randoms_positions1 = self.randoms_positions.T,
                                         data_weights1=self.data_weights,
                                         randoms_weights1=self.randoms_weights,
                                         position_type = 'rdd', 
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
    
    
    
    # Average_cf method should be the same as in the CorrHOD_cubic class
    
    
    
    # Save method should be the same as in the CorrHOD_cubic class
    def save(self, 
             hod_indice: int = 0, 
             path: str = None, 
             save_HOD: bool = True, 
             save_pos: bool = True, 
             save_density: bool = True, 
             save_quantiles: bool = True, 
             save_CF: bool = True, 
             save_xi: bool = True, 
             los: str = 'average', 
             save_all: bool = False):
        
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
            
        cutsky_name = f'pos_hod{hod_indice}_c{cosmo}_ph{phase}.npy' # The cubic dictionary does not depend on the line of sight either
        
        if save_pos or (save_all and hasattr(self, 'cubic_dict')):
            # Pass if the cubic dictionary has not been computed yet
            if not hasattr(self, 'cubic_dict'):
                warn('The cubic dictionary has not been computed yet. Run populate_halos first.', UserWarning)
                pass
            path = output_dir 
            path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
            np.save(path / cutsky_name, self.cubic_dict)
        
        if save_all:
            save_HOD = save_density = save_quantiles = save_CF = save_xi = True
        
        # Everything else is the same as in the CorrHOD_cubic class, so we can use the super method
        super().save(hod_indice, 
                     path=path, 
                     save_HOD=save_HOD, 
                     save_pos=False, 
                     save_density=save_density, 
                     save_quantiles=save_quantiles, 
                     save_CF=save_CF, 
                     save_xi=save_xi, 
                     los=los, 
                     save_all=False)

    # TODO : Run_all 

# To convert the cutsky to cartesian coordinates --> Use the densitysplit package, it's already there ! 