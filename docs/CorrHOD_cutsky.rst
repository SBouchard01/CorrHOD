CorrHOD_cutsky
=============

.. warning::
   This class is still in development and is not yet ready for use.

   While the analysis code is stable, the creation of cutsky simulation is still in development.

What is a cutsky ?
------------------


In theory
~~~~~~~~~

While the cubic box is a snapshot of the Universe at a given time (i.e. a redshift), a cutsky aims
to be a simulation of an acual survey. It is a simulation of the sky, with a given geometry.

Several effects are included in a cutsky, such as Redshift Space Distortions (RSD), 
a survey geometry (footprint), some observational contraints 
(like fiber assignment, incompleteness, ...), etc.

As the further we look in space the further we look in time, a cutsky is not a snapshot of the
Universe at a given time, as the redshift evolves with the distance. this causes the number density 
of galaxies to change with the redshift. 

In theory, a cutsky and cubic mock of the same cosmology should have the same clustering properties.


In practice (computationaly)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cutsky is usually created from a cubic box, and is therefore a subset of the cubic box.

As it is not a periodic box, the computation method changes to become a bit more complicated.
We will need random points matching the footprint of the survey, to be able to compute the 
analyis. The more points, the better the analysis will be, but the longer it will take to compute.

Some methods to handle the data and randoms catalogs can be found in the ``CorrHOD.catalogs`` module.
(see :doc:`catalogs`)


How to use the class
--------------------

::
   
   from CorrHOD import CorrHOD_cutsky

.. warning::
   The whole "Creating the simulation and turn it in a cutsky" patrt of the class is not yet
   implemented. This part of the documentation will be updated when the class will be ready.

   However, the analysis part of the class is ready and can be used.

.. tip::
   You can import the cubic_dict catalog in the class like this ::

      Object = CorrHOD_cutsky(path2config) # Useless here, but we need it to create the object
      Object.cutsky_dict = data_cutsky
      Object.randoms_dict = randoms_cutsky

As the first steps of this class are the same as the ``CorrHOD_cubic`` class, 
please refer to the documentation of that class for the first steps of the analysis.

::

   Object.get_tracer_positions()
   Object.get_tracer_weights()


Weights and number density
~~~~~~~~~~~~~~~~~~~~~~~~~~

To normalize the power spectrum, some weights (called 'FKP weights' [#]_ ) need to be applied 
in the computation of the correlation functions.

They can be computed by the ``get_tracer_weights()`` method of the class, and are all set to 1 if
not computed before calling the correlation functions.

.. note::
   The densitysplit has not method to compute the weights yet, as it si not clear yet what those 
   mean in this context.

   However, the correlation functions will return wierd results if the weights of the catalogs 
   are not evolving in the same way with the redshift. Therefore, setting all the weights to 1 
   for the quantiles and using the FKP weights for the data and randoms is not a good idea.

   In the ``get_tracer_weights()`` method, the weights are computed for the data and randoms and
   splitted in the quantiles to their respectives galaxies (the same way as the densitysplit).

   *This is a temporary fix and doesn't impact a lot the results, compared with setting all the
   weights to 1.*

The number density functions can also be computed within the class, which can allow for some 
verifications. For example, for the densitysplit to agree with the cubic box, the number density
of the quantiles should have roughly the same shape as the data and randoms catalogs.

The number density can be computed with the ``get_nz()`` method of the class.


Computing the analysis
~~~~~~~~~~~~~~~~~~~~~~

Computing the densitysplit and correlation functions is the same as for the ``CorrHOD_cubic`` class. 
(Under the hood, the process is a bit different but the use of the methods are the same).

.. tip::
   After computing the densitysplit, if you want to downsample the data to reduce the computing
   time and memory usage, it is recommended to use the ``npoints`` or ``frac`` parameters of the
   ``downsample_data()`` method, rather than the ``new_n`` parameter, as the number density 
   parameter is computed as the mean number density in the cutsky, and is less accurate than the
   one computed in the cubic box.



API
---

.. automodule:: CorrHOD.cutsky
   :members:
   :undoc-members:
   :show-inheritance:



.. rubric:: Footnotes

.. [#] See `Feldman et al. (1993) <https://arxiv.org/pdf/astro-ph/9304022.pdf>`_ for more details.