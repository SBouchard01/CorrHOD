.. CorrHOD documentation master file, created by
   sphinx-quickstart on Mon Jul 10 11:36:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CorrHOD
===================================

CorrHOD is a python3 package for modeling the clustering of galaxies in the 
`AbacusSummit <https://abacussummit.readthedocs.io>`_ Dark Matter simulations. 
It is based on the HOD model of `Zheng et al. 2007 <https://arxiv.org/abs/astro-ph/0703457>`_ 
and the `AbacusHOD <https://abacusutils.readthedocs.io/en/latest/hod.html>`_ package from 
`abacusutils <https://github.com/abacusorg/abacusutils>`_.

CorrHOD is hosted in the `CorrHOD <https://github.com/SBouchard01/CorrHOD>`_ GitHub repository. 
Please report bugs and ask questions by opening an issue in that repository.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   usage

.. toctree::
   :maxdepth: 2
   :caption: Main Modules

   CorrHOD_cubic
   CorrHOD_cutsky

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   logging
   catalogs 
   weights
   utils

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   _collections/examples

.. warning::
   These notebooks were computed on a LRG-like halo simulation (see :doc:`CorrHOD_cubic` for details).
   This means that the results are not physically meaningful, but they can be used to understand how to use the code.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   source/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
