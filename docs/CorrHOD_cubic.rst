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

The ``prepare_sim`` script is scaled for LRG galaxies. For computational reasons, a filter 
(see `Yuan et Al. (2021) <https://arxiv.org/pdf/2110.11412.pdf>`_ eq.13) is applied to the
halo catalog to remove the light halos that will not be populated. However, the Bright Galaxy Survey 
(BGS) galaxies has lower mass conditions than LRGs. Therefore, the filter needs to be changed.

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

First, you need to create a config file for AbacusHOD. 
See the `AbacusHOD documentation <https://abacusutils.readthedocs.io/en/latest/hod.html>`_ for more details.

Then, you need to create a config file for CorrHOD.



.. tip::
   The analysis can be performed on a pre-existing catalog. To do so, you need to either use the 
   ``set_cubic`` method, or pass the catalog to the ``cubic_dict`` variable after initialization.

   .. note::
      In that case, no ``HOD_dict`` is needed, as no simulation is computed. However, due to the construction
      of the class, the config file for AbacusHOD still needs to be passed and valid.



Paralellism
~~~~~~~~~~~



API
---

.. automodule:: CorrHOD.cubic
   :members:
   :undoc-members:
   :show-inheritance: