# CorrHOD
[![Documentation Status](https://readthedocs.org/projects/corrhod/badge/?version=latest)](https://corrhod.readthedocs.io/en/latest/?badge=latest)

CorrHOD is a python3 package for modeling the clustering of galaxies in the 
[AbacusSummit](https://abacussummit.readthedocs.io>) Dark Matter simulations. 

## Requirements
Strict requirements are : 
* `numpy`
* `pandas`
* [`cosmoprimo`](https://github.com/cosmodesi/cosmoprimo)
* `AbacusHOD` (from [`abacusutils`](https://abacusutils.readthedocs.io/en/latest/index.html))
* [`densitysplit`](https://github.com/epaillas/densitysplit/tree/master)
* [`mockfactory`](https://github.com/cosmodesi/mockfactory)
* [`pycorr`](https://github.com/cosmodesi/pycorr/tree/main)

## Installation

### Pip install
Simply run
```bash
$ pip install git+https://github.com/SBouchard01/CorrHOD
```

### Manual install
Download the package from github, and unzip it. Then, under the main directory, install the package with:
```bash
$ python setup.py install --user
```
You can also install it in developer mode, so that any changes you make to the code take place immediately:
```bash
$ python setup.py develop --user
```

For more information, see https://corrhod.readthedocs.io/en/latest/installation.html


## Documentation
The documentation can be found at https://corrhod.readthedocs.io


## Examples
Examples can be found under the `examples` directory.


## TODO-list of the project
- [x] Implement cutsky class
  - [x] Implement downsampling function
  - [ ] Create cutsky from boxes (see mockfactory)
  - [ ] Create randoms from cutsky (see mockfactory)
  - [ ] Implement `read_cutsky` function (name to be changed)
  - [ ] Implement `run_all` function
- [x] Make the package pip-installable
- [x] Create documentation
  - [ ] Change style 
  - [x] Add link to example notebooks
  - [ ] Find an icon ?