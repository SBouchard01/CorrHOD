"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages

setup(
    name = 'CorrHOD',  # Required
    version = '0.1',  # Required
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
        'cosmoprimo',
        'abacusutils',
        'densitysplit',
        'mockfactory',
        'pycorr'
    ]
)