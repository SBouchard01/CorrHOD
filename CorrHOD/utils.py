import numpy as np
from os.path import exists
from pathlib import Path
from warnings import warn

def apply_rsd(data, boxsize, redshift, cosmo, tracer='LRG', los = 'z'):
    """Read positions and velocities from HOD dict and applies Redshift-Space Distortions (RSD).
    
    Parameters
    ----------
    data : dict
        HOD dict containing the positions and velocities of the tracer (data[tracer])
        It can also be a positional dataset, in which case it must contain the keys 'x', 'y', 'z', 'vx', 'vy', 'vz'
        
    boxsize : float
        Size of the simulation box in Mpc/h.
        
    redshift : float
        Redshift of the simulation snapshot.
        
    cosmo : cosmology object (cosmoprimo)
        Cosmology used for the simulation. It must contain an `efunc(z)` method.
        
    tracer : str, optional
        Tracer to use. Default is 'LRG'.
        
    los : str, optional
        Line-of-sight direction in which to apply the RSD. Default is 'z'.
        
    Returns
    -------
    x, y, z : arrays
        Redshift-space positions of the tracer. The los axis is replaced by the redshift-space position.
    """

    try:
        data = data[tracer] # Load the LRG data
    except:
        warn('The object is not a hod_dict. Trying to load it as a positional dataset.', TypeError)
        
        # Check that the data contains the right keys
        if not all([key in data.keys() for key in ['x', 'y', 'z', 'vx', 'vy', 'vz']]):
            raise ValueError('The object is not a hod_dict and does not contain the keys "x", "y", "z", "vx", "vy", "vz"')
        
    # Get the scale factor and Hubble parameter at the redshift of the snapshot
    az = 1 / (1 + redshift)
    hubble = 100 * cosmo.efunc(redshift)
    
    # Get the velocities
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    
    # Get the positions (Add boxsize/2 to center the box at boxsize/2)
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    
    # Get the redshift-space positions for each axis. That way, we can replace the los axis with the redshift-space position
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    # Periodic boundary conditions
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    
    # Replace the los axis with the redshift-space position
    if los == 'x':
        x = x_rsd
    elif los == 'y':
        y = y_rsd
    elif los == 'z':
        z = z_rsd
    else:
        raise ValueError('los must be x, y or z')
    
    return x, y, z



def array_to_dict(array, is_log_sigma=False):
    """
    Converts an array of values into a dictionary of HOD parameters
    
    Parameters
    ----------
    array: array_like
        Array of values for the HOD parameters. The order of the parameters must be the following:
        [logM_cut, logM1, sigma, alpha, kappa, alpha_c, alpha_s, Bcent, Bsat]
    
    Returns
    -------
    hod_dict: dict
        Dictionary of HOD parameters
    """
        
    hod_dict = {
        'logM_cut': array[0],
        'logM1':    array[1], 
        'sigma':    array[2], 
        'alpha':    array[3], 
        'kappa':    array[4], 
        'alpha_c':  array[5], 
        'alpha_s':  array[6], 
        'Bcent':    array[7], 
        'Bsat':     array[8]
    }
    
    # Handle the case where sigma is in log10
    if is_log_sigma:
        hod_dict['sigma'] = 10 ** hod_dict['sigma']
    
    return hod_dict



def dict_to_array(dic:dict):
    """
    Converts a dictionary of HOD parameters into an array of values
    
    Parameters
    ----------
    dic: dict
        Dictionary of HOD parameters. The keys must be the following:
        ['logM_cut', 'logM1', 'sigma', 'alpha', 'kappa', 'alpha_c', 'alpha_s', 'Bcent', 'Bsat']
    
    Returns
    -------
    array: array_like
        Array of values for the HOD parameters
    """
    
    array = np.array([
        dic['logM_cut'],
        dic['logM1'],
        dic['sigma'],
        dic['alpha'],
        dic['kappa'],
        dic['alpha_c'],
        dic['alpha_s'],
        dic['Bcent'],
        dic['Bsat']
    ])
    
    return array



def format_HOD_CFs(path:str,
                   output_dir:str,
                   cosmo:int=0,
                   phase:int=0,
                   HOD_start:int=0,
                   HOD_number:int=1,
                   nquantiles=10,
                   smoothing_filter:str='gaussian',
                   smoothing_radius:float=10,
                   merge_2PCF:bool=True,
                   merge_DS_auto:bool=True,
                   merge_DS_cross:bool=True):
    """
    Creates a dictionary of the CFs from the saved files.
    This should bring all the files in the same format as the one used by Sunbird.
    
    It will return for each CF, one file per cosmology and phase, with the following format:
    * `s`: array of separation bins
    * `multipoles`: array of shape (HOD_number, los_number(3), nquantiles, npoles(3), sep_bin_number)	
    
    The name of the file is : 
    * `tpcf_c{cosmo}_p{phase}.npy` for the 2PCF
    * `ds_auto_zsplit_Rs{smoothing_radius}_c{cosmo}_p{phase}.npy` for the DS auto (zsplit because the density is split in redshift space)

    Note that it is assumed that the files that we want to merge are all exist, and are respectively in a directory named `tpcf` and `ds/{smoothing_filter}` from the given path.
    An error will be raised if any of the files is missing.
    
    It is also assumed that the separation bins are the same for all the files (tpcf, cross and auto). A check is done to verify that.

    Parameters
    ----------
    path : str
        Path to the directory containing the saved files
        
    output_dir : str
        Path to the directory where the formatted files will be saved
        
    cosmo : int, optional
        Cosmology number, same as the one used in AbacusSummit. Defaults to 0
        
    phase : int, optional
        Phase number, same as the one used in AbacusSummit. Defaults to 0
        
    HOD_start : int, optional
        HOD number to start from. Defaults to 0
        
    HOD_number : int, optional
        Number of HODs to merge. Defaults to 1
        
    nquantiles : int, optional
        Number of quantiles used to split the sample. Defaults to 10
        
    smoothing_filter : str, optional
        Smoothing filter used to compute the DensitySplit (see https://github.com/epaillas/densitysplit for more details).
        Defaults to 'gaussian'
        
    smoothing_radius : float, optional
        Smoothing radius used to compute the DensitySplit (see https://github.com/epaillas/densitysplit for more details).
        Defaults to 10
        
    merge_2PCF : bool, optional
        Whether to merge the 2PCF files. Defaults to True
        
    merge_DS_auto : bool, optional
        Whether to merge the quantiles auto-correlation files. Defaults to True
        
    merge_DS_cross : bool, optional
        Whether to merge the quantiles cross-correlation files. Defaults to True
    """
    
    # Note : We consider that the files that we want to merge must all exist in the same directory
    # We also consider than the separation bins are the same for all the files (tpcf, cross and auto) ! A check is done to verify that.
    
    path = Path(path)
    output_dir = Path(output_dir)
    cosmo = f'{cosmo:03d}'
    phase = f'{phase:03d}'
    
    # Check that HOD_number is bigger than 0
    if HOD_number < 1:
        raise ValueError('HOD_number must be bigger than 0')
    
    # Path to the directories containing the files
    tpcf_path = path / 'tpcf'
    ds_path = path / 'ds' / smoothing_filter 
    
    # Load the fisrt file to get the separation bins length
    dic = np.load(tpcf_path / f'tpcf_hod{HOD_start:03d}_x_c{cosmo}_p{phase}.npy').item()
    s = dic['s']
    # The number of poles should be 3, but just to be sure : 
    npoles = dic['2PCF'].shape[0]
    
    # Initialize the dictionaries with arrays of size (HOD_number, los_number, nquantiles, npoles, sep_bin_number) for the poles
    tpcf_dict = {
        's': s, 
        'multipoles':np.empty((HOD_number, 3, npoles, len(s))) # No nquantiles dimension here since it's the 2PCF
        }
    ds_auto_dict = {
        's': s, 
        'multipoles':np.empty((HOD_number, 3, nquantiles, npoles, len(s)))
        }
    ds_cross_dict = {
        's': s, 
        'multipoles':np.empty((HOD_number, 3, nquantiles, npoles, len(s)))
        }
    
    # Loop over the HODs
    for i in range(HOD_start, HOD_start + HOD_number):
        hod_indice = f'{i:03d}'
        
        # Loop over the LOS
        for los_indice, los in enumerate(['x', 'y', 'z']):
        
            base_name = f'hod{hod_indice}_{los}_c{cosmo}_p{phase}.npy'
            
            # Get the 2PCF path
            filename = f'tpcf_' + base_name
            if merge_2PCF:
                dic = np.load(tpcf_path / filename).item() # Load the CF
            
                # Check that the separation bins are the same
                if not np.array_equal(tpcf_dict['s'], dic['s']):
                    raise ValueError(f'The separation bins are not equal. Impossible to merge the {i}th 2PCF.')
            
                # Add the 2PCF to the dictionary at the right place
                tpcf_dict['multipoles'][hod_indice, los_indice, :, :] = dic['2PCF']
            
            # Get the DS auto path
            filename = f'ds_auto_' + base_name
            if merge_DS_auto:
                dic = np.load(ds_path / filename).item()
            
                # Check that the separation bins are the same
                if not np.array_equal(ds_auto_dict['s'], dic['s']):
                    raise ValueError(f'The separation bins are not equal. Impossible to merge the {i}th DS auto.')

                # Loop over the quantiles
                for quantile_indice in range(nquantiles):
                    # Add the DS auto to the dictionary at the right place
                    ds_auto_dict['multipoles'][hod_indice, los_indice, quantile_indice, :, :] = dic[f'DS{quantile_indice}']

            # Get the DS cross path
            filename = f'ds_cross_' + base_name
            if merge_DS_cross:
                dic = np.load(ds_path / filename).item()
            
                # Check that the separation bins are the same
                if not np.array_equal(ds_cross_dict['s'], dic['s']):
                    raise ValueError(f'The separation bins are not equal. Impossible to merge the {i}th DS cross.')

                # Loop over the quantiles
                for quantile_indice in range(nquantiles):
                    # Add the DS cross to the dictionary at the right place
                    ds_cross_dict['multipoles'][hod_indice, los_indice, quantile_indice, :, :] = dic[f'DS{quantile_indice}']
                

    # Save the dictionaries
    path = output_dir / 'tpcf'
    path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
    if merge_2PCF:
        np.save(path / f'tpcf_c{cosmo}_p{phase}.npy', tpcf_dict)
    
    base_name = f'zsplit_Rs{smoothing_radius}_c{cosmo}_p{phase}.npy'
    path = output_dir / 'ds' / smoothing_filter
    path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
    if merge_DS_auto:
        np.save(path / f'ds_auto_' + base_name, ds_auto_dict)
    if merge_DS_cross:
        np.save(path / f'ds_cross_' + base_name, ds_cross_dict)

       
#%% Logging utils
import sys
import logging
import types

def log_newline(self, how_many_lines=1):
    """
    Add a blank to the logger. 
    Accessed as a method of the logger object with `types.MethodType(log_newline, logger)`
    """
    # From https://stackoverflow.com/a/45032701
    
    # Switch formatter, output a blank line
    self.handler.setFormatter(self.blank_formatter)

    for i in range(how_many_lines):
        self.info('')

    # Switch back
    self.handler.setFormatter(self.formatter)

def create_logger(name:str,
                  level=logging.INFO, 
                  stream=sys.stdout, 
                  filename:str=None,
                  filemode:str='w',
                  propagate=False): 
    """
    Will get or create the loger with the given name, and add a method to the logger object to output a blank line.
    A handler is created, with the given level and stream (or file output).

    Parameters
    ----------
    name : str
        Name of the logger
        
    level : optional
        Level of the logger and handler. Can be a string ('info', 'debug', 'warning') or a logging level (`logging.INFO`, `logging.DEBUG`, `logging.WARNING`).
        Defaults to `logging.INFO`
        
    stream : optional
        Stream to which the logger will output. Defaults to sys.stdout
        
    filename : str, optional
        Path to the file to which the logger will output. Defaults to None
        
    filemode : str, optional
        Mode to open the file. Defaults to 'w'
        
    propagate : bool, optional
        Whether to propagate the logs to the root logger. Defaults to False
        Warning : If sets to False, the logs will not be propagated to the root logger, and will not be output by the root logger. 
        If the root logger outputs to a file, the logs will not be saved in the file, unless the logger has the same output file.
        However, if propagate is True, the logs will be output twice if the root has a different handler (once by the logger, once by the root logger)

    Returns
    -------
    logger : logging.Logger 
        Logger object 
    """
    # From https://stackoverflow.com/a/45032701 
    
    if isinstance(level, str):
        level = {'info': logging.INFO, 'debug': logging.DEBUG, 'warning': logging.WARNING}[level.lower()]
    
    # Create a handler
    if filename is not None:
        path = Path(filename).parent[0] # Get the path to the directory containing the file
        path.mkdir(parents=True, exist_ok=True) # Create the directory if it does not exist
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setLevel(level)
    
    # Create formatter and add it to the handler
    formatter = logging.Formatter(fmt="%(message)s")   
    blank_formatter = logging.Formatter(fmt="")
    handler.setFormatter(formatter)

    # Create a logger, with the previously-defined handler
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Remove all handlers already associated with the logger object
    for hd in logger.handlers:
        logger.removeHandler(hd)
    logger.propagate = propagate # Prevent the logs from being propagated to the root logger that will have different handlers
    logger.addHandler(handler) # Add the handler to the logger

    # Save some data and add a method to logger object
    logger.handler = handler
    logger.formatter = formatter
    logger.blank_formatter = blank_formatter
    logger.newline = types.MethodType(log_newline, logger)
    
    return logger
