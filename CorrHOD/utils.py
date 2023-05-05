
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
    
    # Get the positions (Add boxsize/2 to center the box at 0)
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
