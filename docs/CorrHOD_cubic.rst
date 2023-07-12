CorrHOD_cubic
=============

For the utility of the class, see its desctiption in the `API`_ section below.

Before using the class
----------------------
As we will be using the AbacusSummit Dark Matter simulations, 
AbacusHOD requires to prepare these simulations.

.. warning::
   Without this step, the simulation creation will fail.

To do so, you need to follow the instructions in the 
`AbacusHOD documentation <https://abacusutils.readthedocs.io/en/latest/hod.html#short-example>`_.


The BGS filter
~~~~~~~~~~~~~~

The ``prepare_sim`` script is scaled for LRG galaxies. For computational reasons, a filter [#]_
is applied to the halo catalog to remove the light halos that will not be populated. 
However, the Bright Galaxy Survey (BGS) galaxies has lower mass conditions than LRGs. 
Therefore, the filter needs to be changed.

.. figure:: images/Prepare_sim.png
   :scale: 50 %
   :align: center
   :alt: prepare_sim halos distribution
   :figclass: align-center

   The Halo mass distribution for different filters (blue for the halo catalog, yellow for BGS
   and green for LRGs)

The ``prepare_sim.sh`` file in the ``scripts`` folder is the script used to prepare the BGS simulations.

.. warning::
   The ``prepare_sim.sh`` script and ``prepare_sim_bgs.py`` script are not included in the package, as
   they are supposed to be included in the AbacusHOD package. However, they are available in the
   ``scripts`` folder of the `GitHub repository <https://github.com/SBouchard01/CorrHOD>`_.

The script needs as an imput a config file (the same as AbacusHOD), with the path to 
the dark matter simulation in ``sim_dir``, the name of the simulation we want to prepare in 
``sim_name`` and the path in which the prepared simulation will be saved in ``subsample_dir``.
(see ``config\config.yaml`` for an example, and 
`AbacusHOD <https://abacusutils.readthedocs.io/en/latest/hod.html#short-example>`_ 
for more details).

.. note::
   The only difference between the LRG and BGS scripts are the ``subsample_halos`` and ``submask_particles``
   functions. 


How to use the class
--------------------

Initialization of the class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, you need to create a config file for AbacusHOD. 
See the `AbacusHOD documentation <https://abacusutils.readthedocs.io/en/latest/hod.html>`_ for more details.

Then, you need to create a HOD dictionnary that contains the following parameters 
(provided here with a random set for example):

::

   {'Acent': 0,
   'Asat': 0,
   'Bcent': 0.30609746972148444,
   'Bsat': -0.0737193257210993,
   'alpha': 0.8689115800548024,
   'alpha_c': 0.7700801491165564,
   'alpha_s': 1.036122317356142,
   'ic': 1,
   'kappa': 0.3005439816787442,
   'logM1': 13.481589622029889,
   'logM_cut': 13.274157859189234,
   's': 0,
   's_p': 0,
   's_r': 0,
   's_v': 0,
   'sigma': 0.00011413582336912553}

The CorrHOD_cubic class can then be initialized ::

   from CorrHOD import CorrHOD_cubic

   Object = CorrHOD_cubic(config_file, HOD_dict)

   Object.initialize_halos() # Load the halos catalog

   Object.populate_halos() # Populate the halos with galaxies

From that point, you can use the different methods of the class to perform different analysis.

.. tip::
   The analysis can be performed on a pre-existing catalog. To do so, you need to either use the 
   ``set_cubic`` method, or pass the catalog to the ``cubic_dict`` variable after initialization.

   .. note::
      In that case, no ``HOD_dict`` is needed in the initialization, as no simulation is computed. 
      However, due to the construction of the class, the config file for AbacusHOD 
      still needs to be passed and valid.

.. note::
   The class initialization can also take a line of sight, a boxsize and a cosmology 
   as arguments. See the API for more details.


Getting the positions
~~~~~~~~~~~~~~~~~~~~~

The CorrHOD code is designed to work for BGS only, but should *in theory* work for all AbacusHOD tracers.
Only galaxies of this tracer will be populated in the simulation, and only one tracer can be used at a time.
The tracer used can be changed in the ``tracer`` variable ::

   Object.tracer = 'LRG'

.. tip::
   The ``tracer`` parameter is set to 'LRG' because the HOD model has the same functions for BGS and LRG, 
   so only 'LRG' is coded in AbacusHOD.

The ``get_tracer_positions()`` method can then be used to get the positions of the galaxies in the catalog.
This will also apply the Redshift Space Distortion (RSD) to the positions.

::
   
      Object.get_tracer_positions()


.. warning::
   This step is important, as the positionnal array returned in the class by this function 
   will be used for all the analysis.


DensitySplit [#]_
~~~~~~~~~~~~

The ``compute_densitysplit()`` method applies the ``densitysplit`` package to the catalog,
and separates the galaxies in the catalog in quantiles of local density::

   quantiles, density = Object.compute_densitysplit(return_density=True) # If return_density is False, only the quantiles are returned

.. note::
   The quantiles also can be accessed with ``Object.quantiles``.


Downsampling the catalog
~~~~~~~~~~~~~~~~~~~~~~~~

For computational reasons, the number of galaxies in the catalog can be too big, causing the 
Correlations functions to take too much time to compute.

While CorrHOD supports multiprocessing and MPI (see the `Parallelism`_ section), downsampling the catalog
can be a good way to reduce the computation time, without loosing too much information.

The ``downsample_data()`` method can be used to uniformly downsample the catalog ::

   new_nbar = 1e-3 # New number density of galaxies
   Object.downsample_data(new_n=new_n) 

.. warning::
   Any downsampling will cause a loss of information, and should be used with caution.
   For example, the `DensitySplit`_ theory is based on the local density of galaxies,
   that will be affected by the downsampling.


Computing the correlation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three methods are available to compute the correlation functions:
``compute_2pcf()``, ``compute_auto_corr()`` and ``compute_cross_corr()``.
Their use and parameters are detailled in the `API`_ section.

Their results are stored in two dictionnaires inside the class :

* ``Object.xi`` stores the ``pycorr.TwoPointCorrelationFunction`` objects
* ``Object.CF`` stores the multipoles of the correlation functions, and assumes that all 
   the correlation functions are computed with the same separation bins.

.. warning::
   This last point is important, as the separation bins are stored only once in this dictionnary.

The two dictionnaires have the same structure :

   los = 'z' # Line of sight on which the RSD have been applied
   Object.CF[los]['2PCF'] # gets the poles of the 2PCF
   Object.CF['Cross']['DS0'] # For quantile 0 ...

.. tip::
   The *line of sight* key is here so that several lines of sight can be computed in the same CorrHOD object.
   This allows for the `Averaging the correlation functions`_ part.


Averaging the correlation functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If several lines of sight have been computed in the same CorrHOD object, the ``average_CF()`` method can be used
to average the poles of the correlation functions.

.. tip::
   The method has an ``average_on`` parameter that can be used to average only on a subset of the lines of sight.

This allows for a reduction of the noise in the correlation functions, and a better estimation of the covariance matrix.

Once this method called, a new key is added to the ``Object.CF`` dictionnary, with the key ``'average'``.

.. warning::
   For the method to work, all the correlation functions must have been computed (and with the same separation bins).
   This means the 2PCF, and the auto and cross correlation functions for **each quantile !**


Parallelism
~~~~~~~~~~~

The ``CorrHOD_cubic`` class supports multiprocessing and MPI, and will transfer the ``nthreads``, 
communicator and rank to the functions that support it, if they are provided.
See the `API`_ section for more details.

Saving the results
~~~~~~~~~~~~~~~~~~

The ``save()`` method can be used to save the selected results in numpy files.
The arguments of the method can be found in the `API`_ section.

The saved results are saved in the following structure::

   Folder/
   ├── ds/
   │   ├── density/        (Density)
   │   ├── quantiles/      (Quantiles)
   │   ├── gaussian/       (Auto and cross correlation functions)
   ├── hod/                (HOD parameters)
   ├── tpcf/               (2PCF)
   ├── xi/                 (Correlation functions)

The ``run_all`` method
~~~~~~~~~~~~~~~~~~~~~~

This method is a shortcut to run all the analysis in one go. 
It is not recommended to use it, as it will not allow for a good control of the analysis, and 
requires a good understanding of the class to be used properly.

However, if you know what you are doing, it makes for a very powerful tool to run the analysis,
as the entire code only takes a few lines::

   from CorrHOD import CorrHOD_cubic

   Object = CorrHOD_cubic(config_file, HOD_dict)

   Object.run_all() # Several arguments can be passed to the method, see the API for more details

.. tip::
   The ``run_all()`` method times every step of the execution. These times are displayed in the logs, 
   but are also stored in a times_dict object, that can be accessed.

API
---

.. automodule:: CorrHOD.cubic
   :members:
   :undoc-members:
   :show-inheritance:



.. rubric:: Footnotes

.. [#] See `Yuan et Al. (2021) <https://arxiv.org/pdf/2110.11412.pdf>`_ eq.13 for more details.
.. [#] See `Paillas et al. (2022) <https://arxiv.org/pdf/2209.04310.pdf>`_ for more details.