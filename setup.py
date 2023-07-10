"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

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
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'cosmoprimo @ git+https://github.com/cosmodesi/cosmoprimo#egg=cosmoprimo[class,camb,astropy,extras]',
        'abacusutils',
        'densitysplit @ git+https://github.com/epaillas/densitysplit@openmp',
        'mockfactory @ git+https://github.com/cosmodesi/mockfactory',
        'pycorr @ git+https://github.com/cosmodesi/pycorr#egg=pycorr[mpi,jackknife,corrfunc]'
    ]
)