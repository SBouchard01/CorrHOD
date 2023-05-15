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


# TODO : Function to compile the saved CFs as a dictionary (with the right format) for sunbird
def format_HOD_CFs(path:str,
                   cosmo:int=0,
                   phase:int=0,
                   HOD_start:int=0,
                   HOD_number:int=1,
                   merge_2PCF:bool=True,
                   merge_DS_auto:bool=True,
                   merge_DS_cross:bool=True):
    
    path = Path(path)
    cosmo = f'{cosmo:03d}'
    phase = f'{phase:03d}'
    
    # Check that HOD_number is bigger than 0
    if HOD_number < 1:
        raise ValueError('HOD_number must be bigger than 0')
    
    # Initialize the dictionaries
    tpcf_dict = {}
    tpcf_poles = []
    ds_auto_dict = {}
    ds_cross_dict = {}
    
    # Loop over the HODs
    for i in range(HOD_start, HOD_start + HOD_number):
        hod_indice = f'{i:03d}'
        
        # Loop over the LOS
        for los in ['x', 'y', 'z']:
        
            base_name = f'hod{hod_indice}_{los}_c{cosmo}_p{phase}.npy'
            
            # Get the 2PCF path
            path = path / 'tpcf'
            filename = f'tpcf_' + base_name
            if merge_2PCF and exists(path / filename):
                # Load the 2PCF
                dic = np.load(path / f'2PCF_{hod_indice}.npy').item()
            
                if 's' not in tpcf_dict.keys():
                    # Initialize the dictionary
                    tpcf_dict['s'] = dic['s']
                    tpcf_poles.append(dic['2PCF'])
                elif not np.array_equal(tpcf_dict['s'], dic['s']):
                    raise ValueError(f'The separation bins are not equal. Impossible to merge the {i}th 2PCF.')
                
                







       
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
        Level of the logger and handler, by default logging.INFO
        
    stream : optional
        Stream to which the logger will output, by default sys.stdout
        
    filename : str, optional
        Path to the file to which the logger will output, by default None
        
    filemode : str, optional
        Mode to open the file, by default 'w'
        
    propagate : bool, optional
        Whether to propagate the logs to the root logger, by default False
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
