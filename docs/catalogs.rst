Catalogs handling
=================

This module contains some functions to handle catalogs.

Reading ``.fits`` files
-----------------------

The ``read_fits()`` function reads a FITS file and returns a pandas DataFrame, with the same
columns names. It can check the cubic or cutsky formatting that we expect for a DESI file, but this
is optional.
It also applies the *E-correction* [#]_ to the magnitudes by default.

Applying some cuts
------------------

The ``catalog_cuts()`` function applies some cuts on the catalog, and returns a new DataFrame.
See the `API`_ section for more details.	

Creating a random file
----------------------

The ``create_random()`` function creates a random catalog from several randoms catalogs provided.

.. note::
   This function is mainly used to avoid concatenating all the randoms catalogs in a single file, 
   and does not actually generates randoms. (This is done by the ``CorrHOD_cutsky`` class.)


API
---

.. automodule:: CorrHOD.catalogs
   :members:
   :undoc-members:
   :show-inheritance:



.. rubric:: Footnotes

.. [#] An empiric correction on the evolution in the spectrum over the redshift interval on the absolute magnitude.
