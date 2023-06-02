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

## Installation
(Nothing implemented yet, come back later)

## Examples
Examples can be found under the `examples` directory.


## TODO-list of the project
- [ ] Finish the README
- [X] Implement cutsky class
  - [ ] Implement downsampling function
  - [ ] Create cutsky from boxes (see mockfactory)
  - [ ] Create randoms from cutsky (see mockfactory)
  - [ ] Implement `read_cutsky` function (name to be changed)
  - [ ] Implement `run_all` function
- [ ] Create cutsky example notebook
- [ ] Add a note about coordinates used in cutsky class (and the densitysplit functions)