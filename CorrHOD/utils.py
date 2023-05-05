

def apply_rsd(data, boxsize, redshift, cosmo, tracer='LRG', los = 'z'):
    """Read positions and velocities from HOD dict
    return real and redshift-space galaxy positions."""

    try:
        data = data[tracer] # Load the LRG data
    except:
        print('The object is not a hod_dict. Trying to load it as a positional dataset.')
        # TODO : Check that the data contains the right keys
        # TODO : Transform the print into a warning
        
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
