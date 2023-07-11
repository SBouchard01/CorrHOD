"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
import os

install_requires=[
        'numpy',
        'pandas',
        'abacusutils',
        'densitysplit @ git+https://github.com/epaillas/densitysplit@openmp',        
    ]

if not os.getenv('READTHEDOCS'):
    install_requires.append('cosmoprimo[class,camb,astropy,extras] @ git+https://github.com/cosmodesi/cosmoprimo')
    install_requires.append('mockfactory @ git+https://github.com/cosmodesi/mockfactory')
    install_requires.append('pycorr[mpi,jackknife,corrfunc] @ git+https://github.com/cosmodesi/pycorr')

setup(
    name = 'CorrHOD',  
    version = '0.1',  
    description = 'Correlation function estimation using HOD parameters',  
    url = 'https://github.com/SBouchard01/CorrHOD',
    author = 'Simon Bouchard',
    author_email = 'simonbouchard47@gmail.com',
    packages = find_packages(),
    python_requires='>=3.6, <4',
    license = 'MIT',
    include_package_data = True,
    install_requires = install_requires
)