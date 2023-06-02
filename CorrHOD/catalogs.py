from astropy.io import fits
import pandas as pd
import numpy as np
from pathlib import Path


def read_fits(file:str,  
              apply_E_correction:bool=True,
              z_cubic:float=0.2,
              check_cutsky_format:bool=False, 
              check_cubic_format:bool=False,
              print_column_names:bool=False) -> pd.DataFrame:
    """
    Reads a fits file from the Abacus simulation and returns the data as a Pandas DataFrame.
    
    A DESI cutsky file must be in the format of the DESI simulations. (i.e. contain the following columns :
        * `RA` : Right ascension (in degrees)
        * `DEC` : Declination (in degrees)
        * `Z` : Redshift of the observed galaxy (in redshift space)
        * `Z_COSMO` : Redshift of the galaxy in the comoving frame (in real space)
        * `R_MAG_ABS` : Absolute magnitude
        * `R_MAG_APP` : Apparent magnitude
        * `G_R_REST` : Color (g-r) of the galaxy in the rest frame
        * `HALO_MASS` : Halo mass of the galaxy
        * `STATUS` : Status of the galaxy (if the galaxy is in the DESI footprint or not)
    )
    
    A DESI cubic box file must be in the format of the DESI simulations. (i.e. contain the following columns :
        * `x` : x coordinate of the galaxy in the comoving frame (in real space)
        * `y` : y coordinate of the galaxy in the comoving frame (in real space)
        * `z` : z coordinate of the galaxy in the comoving frame (in real space)
        * `vx` : x-component of velocity, in km/s (not comoving)
        * `vy` : y-component of velocity, in km/s (not comoving)
        * `vz` : z-component of velocity, in km/s (not comoving)
        * `R_MAG_ABS` : Absolute magnitude
        * `HALO_MASS` : Halo mass of the galaxy
    )
    
    Parameters
    ----------
        file: str
            Path to the file.
        
        apply_E_correction: bool, optional
            If True, corrects the evolution in the spectrum over the redshift interval on the absolute magnitude. 
            Defaults to `True`.
            
        z_cubic: float, optional
            Redshift of the cubic snapshot for E-correction. Defaults to `0.2`.
            
        check_cutsky_format: bool, optional
            If True, checks if the file is in the format of a DESI cutsky (see above). Defaults to `False`.
            
        check_cubic_format: bool, optional
            If True, checks if the file is in the format of a DESI cubic box (see above). Defaults to `False`.
        
        print_column_names: bool, optional
            If True, prints the names of the columns in the file. Defaults to `False`.
        
    Returns
    -------
        data: Pandas DataFrame
            Data contained in the file.
    """

    # Read the fits file
    with fits.open(Path(file)) as hdul:
        data = hdul[1].data
    
    # Checks if the columns we need are present in the file
    cols = ['RA', 'DEC', 'Z', 'R_MAG_ABS', 'R_MAG_APP', 'STATUS', 'HALO_MASS']
    is_cutsky = True 
    for col in cols:
        if col not in data.columns.names:
            is_cutsky = False 
            if check_cutsky_format:
                raise KeyError(f'The column "{col}" is not present in the file.')
        
    # Checks if the columns we need are present in the file
    cols = ['x', 'y', 'z', 'R_MAG_ABS', 'HALO_MASS']
    is_cubic = True
    for col in cols:
        if col not in data.columns.names:
            is_cubic = False
            if check_cubic_format:
                raise KeyError(f'The column "{col}" is not present in the file.')

    # Convert the data to a Pandas DataFrame
    data = pd.DataFrame(data)
    
    if apply_E_correction and is_cubic: # We apply a constant E-correction if the file is in the format of a DESI cubic box
        z = np.full(len(data), z_cubic)
        data['R_MAG_ABS'] = data['R_MAG_ABS'] - E_correction(z)
    
    if apply_E_correction and is_cutsky: # We apply the E-correction only if the file is in the format of a DESI cutsky
        # Check if the column 'Z' is present in the file
        if 'Z' not in data.columns.values:
            raise KeyError(f'The column "Z" is not present in the file. Impossible to apply the E-correction.')
        z = data['Z']
        data['R_MAG_ABS'] = data['R_MAG_ABS'] - E_correction(z)
    
    if print_column_names:
        print("The columns of the dataframe are :", data.columns.values)

    return data


def E_correction(z, Q0=-0.97, z0=0.1):
    """
    Corrects the evolution in the spectrum over the redshift interval on the absolute magnitude.

    Parameters
    ----------
        z: array_like
            Redshift of the observed galaxy (in redshift space).
        
        Q0: float, optional
            Corrective factor.
            
        z0: float, optional
            Reference redshift
    Returns
    -------
        E_correction: array_like
            Correction to apply on the absolute magnitude.
    """
    return Q0 * (z - z0)



def status_mask(main=0, nz=0, Y5=0, sv3=0):
    """
    Returns the status value of the points to select in the catalog
    """
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)



def catalog_cuts(data, 
                 zmin:float=0.1, 
                 zmax:float=2.0, 
                 abs_mag:float=-21.5, 
                 app_mag:float=19.5, 
                 status_mask = status_mask(nz=1, Y5=1),
                 cap:str='NGC'):
    """
    Applies a mask to the provided data to reduce it according to the parameters

    Parameters
    ----------
    data : pandas.DataFrame
        Data on which the mask should be applied
        
    zmin : float, optional
        Minimum value of redshift to keep. Defaults to 0.1
        
    zmax : float, optional
        Maximum value of redshift to keep. Defaults to 0.1
        
    abs_mag : float, optional
        The cut in absolute magnitude. Only the brighter galaxies (smaller magnitudes) will be kept.
        Only applied if the data has a comumn named 'R_MAG_ABS'.
        Set to None to disable. Defaults to -21.5
        
    app_mag : float, optional
        The cut in apparent magnitude. Only the brighter galaxies (smaller magnitudes) will be kept.
        Only applied if the data has a comumn named 'R_MAG_APP'.
        Set to None to disable. Defaults to 19.5
        
    status_mask : _type_, optional
       The value of the STATUS column to keep.
       Requires a bit value, as in https://desi.lbl.gov/trac/wiki/CosmoSimsWG/FirstGenerationMocks
       The status_mask() function can also be used.
       Defaults to status_mask(nz=1, Y5=1) (i.e. Y5 survey, with n(z) applied)
       
    cap : str, optional
        The cap to keep. Can be 'NGC' or 'SGC'. Only applied if the cap column exists.
        Set to None to disable. Defaults to 'NGC'

    Returns
    -------
    _type_
        _description_
    """
    # Get the redshift mask
    z = data['Z']
    zcut = (z > zmin) & (z < zmax)
    
    # Get the status mask
    status = data['STATUS']
    status_cut = status & status_mask == status_mask
    
    # Get the magnitude mask
    mag_cut = np.full(len(data), True, dtype=bool) # Mask that keeps all the galaxies
    if 'R_MAG_APP' in data.columns and app_mag is not None:
        m_r = data['R_MAG_APP']
        mag_cut = mag_cut & (m_r < app_mag)
    
    if 'R_MAG_ABS' in data.columns and abs_mag is not None:
        M_r = data['R_MAG_ABS']
        mag_cut = mag_cut & (M_r < abs_mag)
    
    # Get the cap mask
    cap_cut = np.full(len(data), True, dtype=bool) # Mask that keeps all the galaxies
    if cap in data.columns and cap is not None:
        cap_cut = data[cap] == 1
    
    # 
    mask = mag_cut & status_cut & zcut & cap_cut
    
    # Apply the mask
    data = pd.DataFrame(data.values[mask], columns=data.columns) # Get the values of the mask
    
    return data



def create_random(path:str, multiplier:int=5) -> pd.DataFrame :
    """ 
    Creates a random catalog with the x1 catalogs in the provided path. The number of randoms is equal to `multiplier` times the number of galaxies in the data catalog.
    
    Parameters
    ----------
        path: str
            The path to the file containing the x1 random catalogs.
            
        multiplier: int
            The number of randoms in the random catalog is equal to `multiplier` times the number of galaxies in the data catalog.
            
    Returns
    -------
        random: array_like
            The random catalog.
    """
    
    path = Path(path)
    
    # Create a list of the name of the random catalogs
    random_catalogs = [f"random_S{i}00_1X.fits" for i in range(1, multiplier+1)] # List of the random files
    
    # Read the first random catalog
    random = read_fits(path / random_catalogs[0]) # Returns a pandas dataframe
    
    # Loop over the other random catalogs to concatenate them to the first one
    for i in range(1, len(random_catalogs)):
        random = pd.concat([random, read_fits(path / random_catalogs[i])], ignore_index=True)
        
    return random
