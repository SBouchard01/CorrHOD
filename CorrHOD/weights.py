import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import InterpolatedUnivariateSpline

# Ignore the warning that appears when we are creating a new column in a dataframe
import warnings
try : 
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning) # Ignore the warning that appears when we are creating a new column in a dataframe
except AttributeError:
    warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning) # Old version pf pandas


def sky_fraction(ra, dec, nside=256):
    """
    Computes the fraction of the sky covered by the survey.
    
    Parameters
    ----------
    ra : array_like
        Right ascension of the galaxies in degrees.
    
    dec : array_like
        Declination of the galaxies in degrees.
        
    nside : int, optional
        The nside parameter of the HEALPix map. Defaults to `256`.
        
    Returns
    -------
    fsky : float
        The fraction of the sky covered by the survey.
    """
    
    npix = hp.nside2npix(nside)
    phi = np.radians(ra)
    theta = np.radians(90.0 - dec)
    
    pixel_indices = hp.ang2pix(nside, theta, phi)
    pixel_unique, counts = np.unique(pixel_indices, return_counts=True)
    fsky = len(pixel_unique)/npix   # fsky
    
    return fsky

def comoving_volume(cosmo, 
                    z_min, 
                    z_max, 
                    area: float = 14000, 
                    fsky: float = None):
    """ 
    Computes the comoving volume associated to the redshift bin [`z_min`, `z_max`].

    Parameters
    ----------
        cosmo
            Cosmology object that can be used to compute the comoving distance.
            It must have a method `comoving_radial_distance(z)` that returns the comoving distance at redshift `z`.
        
        z_min: float or array_like
            The minimum redshift of the redshift bin. (Can be an array, in which case the output is an array)
            Must be of the same format as `z_max` and same length if it is an array.
        
        z_max: float or array_like
            The maximum redshift of the redshift bin. (Can be an array, in which case the output is an array)
            Must be of the same format as `z_min` and same length if it is an array.
        
        area: float, optional
            The area of the survey in square degrees. Defaults to `14000` (the area of the DESI footprint).
        
        fsky: float, optional
            The fraction of the sky covered by the survey. Can be provided instead of area. Defaults to `None`.
        
    Returns
    -------
        comov_vol: float or array_like
            Comoving volume associated to the redshift bin [`z_min`, `z_max`].
            The output has the same format as `z_min` and `z_max`.
    """
    
    if fsky is None and area is None:
        raise ValueError("Either `fsky` or `area` must be specified.")
    elif fsky is not None and area is not None:
        warnings.warn("Both `fsky` and `area` are specified. `fsky` will be used.")
        area = fsky * 4*np.pi * (180.0/np.pi)**2
    elif area is None:
        area = fsky * 4*np.pi * (180.0/np.pi)**2

    # Compute the comoving distance associated to the redshift bin
    comov_dist_min = cosmo.comoving_radial_distance(z_min)
    comov_dist_max = cosmo.comoving_radial_distance(z_max)
    
    # Convert the area from square degrees to square radians
    area = area * (np.pi/180)**2

    # Compute the comoving volume associated to the redshift bin
    comov_vol = (area/3) * (comov_dist_max**3 - comov_dist_min**3)

    return comov_vol


def ScottsBinEdges(data) -> np.ndarray :
    """
    Computes the bin edges for a histogram using Scott's rule.
    Scott's rule is a rule of thumb for choosing the bin width of a histogram.
    It is based on the standard deviation of the data and is a function of the
    sample size. It is a good compromise when no other information is known
    about the data.
    
    Parameters
    ----------
    data : array_like
        The data we want to bin.
        
    Returns
    -------
    edges : ndarray
        The edges of the bins. Length `nbins + 1`.
    
    Notes
    -----
    Scott's rule defines the bin width as `dx = 3.5 * sigma / n**(1/3)`, where
    sigma is the standard deviation of the data and n is the sample size.
    (The factor of `3.5` comes from a `24*np.sqrt(np.pi)` factor at the power of 1/3).
    """
    
    # Convert the data to a numpy array
    data = np.asarray(data)
    
    # Extract the information needed from the data
    n = len(data) # Number of data points
    sigma = np.std(data) # Standard deviation of the data
    xmin = np.min(data) # Minimum of the data
    xmax = np.max(data) # Maximum of the data
    
    # Compute the bin width 
    factor = (24 * np.sqrt(np.pi))**(1/3) # A factor that appears in Scott's rule, approximated by 3.5
    dx = factor * sigma / n**(1/3) # The bin width according to Scott's rule
    
    # Compute the bin edges
    nbins = int(np.ceil((xmax - xmin) / dx)) # The number of bins
    nbins = max(nbins, 1) # Make sure that there is at least one bin
    edges = xmin + dx * np.arange(nbins + 1) # The edges of the bins
    
    return edges


def n_z(z, 
        cosmo, 
        edges: list = None, 
        area: float = 14000, 
        fsky:float=None) -> InterpolatedUnivariateSpline:
    """ 
    Computes the number density of galaxies in the data in the given redshift bin.

    Parameters
    ----------
        z: array_like
            The redshift of the galaxies in the data. 
            
        cosmo
            Cosmology object that can be used to compute the comoving distance.
            It must have a method `comoving_radial_distance(z)` that returns the comoving distance at redshift `z`.
            
        edges: list, optional
            The edges of the bins used to compute the number density. 
            If set to `None`, the edges are computed using Scott's rule. Defaults to `None`.
        
        area: float, optional
            The area of the survey in square degrees. Defaults to `14000` (the area of the DESI footprint).
        
        fsky: float, optional
            The fraction of the sky covered by the survey. Can be provided instead of area. Defaults to `None`.
        
    Returns
    -------
        n_func : InterpolatedUnivariateSpline
            The number density as a function of redshift. 
            It can be called as `n_func(z)` to get the number density at redshift `z`.
            
    Notes
    -----
    The number density is computed as `n(z) = N(z) / V(z)`, 
    where `N(z)` is the number of galaxies in the redshift bin [`z_min`, `z_max`] 
    and `V(z)` is the comoving volume associated to the redshift bin [`z_min`, `z_max`].
    """
    
    # Convert the z to a numpy array
    z = np.asarray(z)
    
    # Compute the number of galaxies in the redshift bin
    if edges is None:
        edges = ScottsBinEdges(z)
    bin_centers = 0.5*(edges[:-1] + edges[1:])
    
    # Compute the comoving volume associated to the redshift bin
    z_min = edges[:-1]
    z_max = edges[1:]
    V = comoving_volume(cosmo, z_min, z_max, area, fsky) # Comoving volume associated to the redshift bin
    
    # Compute the number of galaxies in the bins 
    N, _ = np.histogram(z, bins=edges) # Number of galaxies in the redshift bin
    nbar = N / V # Number density in the redshift bin
    
    # Interpolate the number density as a function of redshift
    n_func = InterpolatedUnivariateSpline(bin_centers, nbar, ext='zeros') # (ext='zeros' means that the number density is set to zero outside of the redshift range of the data)
    return n_func
    

def w_fkp(z_data, 
          z_random, 
          cosmo, 
          edges: list = None, 
          area: float = 14000, 
          fsky: float = None, 
          P0: float = 7000):
    """ 
    Computes the FKP weights for the data, and returns a column containing the FKP weights. (If cuts need to be applied, they should be applied before calling this function.)

    Parameters
    ----------
        z_data: array_like
            The redshift of the galaxies in the data.
        
        z_random: array_like
            The redshift of the galaxies in the randoms.
            
        cosmo
            Cosmology object that can be used to compute the comoving distance.
            It must have a method `comoving_radial_distance(z)` that returns the comoving distance at redshift `z`.
            
        edges: list, optional
            The edges of the bins used to compute the number density. 
            If set to `None`, the edges are computed using Scott's rule. Defaults to `None`.
        
        area: float, optional
            The area of the survey in square degrees. Defaults to `14000` (the area of the DESI footprint).
        
        fsky: float, optional
            The fraction of the sky covered by the survey. Can be provided instead of area. Defaults to `None`.
        
        P0: float, optional
            The power spectrum normalization (TODO : Check this definition). Defaults to `7000` for the BGS.
        
    Returns
    -------
        weight_data: array_like
            The FKP weights for the data for the respective galaxies in z_data.
            
        weight_random: array_like
            The FKP weights for the data for the respective galaxies in z_random.
    """

    # Convert the z to a numpy array
    z_data = np.asarray(z_data)
    z_random = np.asarray(z_random)
    
    # Compute the number density
    n_func = n_z(z_random, cosmo, edges=edges, area=area, fsky=fsky) # Number density as a function of redshift, computed from the randoms (the data should follow the same distribution as the randoms)
    
    # Re-normalize n(z) to the total size of the data catalog
    alpha = 1.0 * len(z_data) / len(z_random)
    
    n_data = n_func(z_data) * alpha # Number density at the redshift of the data
    n_random = n_func(z_random) * alpha # Number density at the redshift of the randoms

    # Compute the FKP weights
    weight_data = 1.0 / (1 + n_data*P0)
    weight_random = 1.0 / (1 + n_random*P0)
    
    return weight_data, weight_random



from pandas import qcut
def get_quantiles_weight(density, 
                         randoms_weights,
                         nquantiles=10):
    """
    Gets the weights of the quantiles of the density.

    Parameters
    ----------
    density : array_like
        The density of the data.
        
    randoms_weights : array_like
        The weights of the randoms.
        
    nquantiles : int, optional
        The number of quantiles to use. Defaults to `10`.
    """
    quantiles_idx = qcut(density, nquantiles, labels=False)
    quantiles_weights = []
    for i in range(nquantiles):
        quantiles_weights.append(randoms_weights[quantiles_idx == i])
    
    return quantiles_weights