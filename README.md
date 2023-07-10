# CorrHOD
A code that populates dark matter halos and computes correlation functions


## Documentation
(Coming later)

## Requirements
Strict requirements are : 
* `numpy`
* `pandas`
* [`cosmoprimo`](https://github.com/cosmodesi/cosmoprimo)
* `AbacusHOD` (from [`abacusutils`](https://abacusutils.readthedocs.io/en/latest/index.html))
* [`densitysplit`](https://github.com/epaillas/densitysplit/tree/master)
* [`mockfactory`](https://github.com/cosmodesi/mockfactory)
* [`pycorr`](https://github.com/cosmodesi/pycorr/tree/main)

*Note : The densitysplit package is not yet available on pip, so you will have to install it manually first. I recommend using the `openmp` branch, which is MPI-compatible and faster than the master branch.*

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

## Examples
Examples can be found under the `examples` directory.


## TODO-list of the project
- [X] Implement cutsky class
  - [X] Implement downsampling function
  - [ ] Create cutsky from boxes (see mockfactory)
  - [ ] Create randoms from cutsky (see mockfactory)
  - [ ] Implement `read_cutsky` function (name to be changed)
  - [ ] Implement `run_all` function
- [ ] Make the package pip-installable
- [ ] Create documentation
- [ ] Create examples
  - [ ] Modify cubic example
  - [ ] Create cutsky example notebook
- [ ] Finish the README
- [ ] Add a note about coordinates used in cutsky class (and the densitysplit functions)