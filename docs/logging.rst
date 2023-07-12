Logging
=======

Logging is done all trough the package execution. Two loggers are defined in the classes : 

* ``DensitySplit`` in the ``compute_Densitysplit`` method (on ``DEBUG`` level)
* ``CorrHOD`` trough the classes.

The logging is done using the standard python logging module, usually on the ``INFO`` level 
(except for some specific messages not really useful)

In the ``run_all()`` method, the logging is used to print the progress of the computation and
some useful informations. 

.. tip::
   This is done to avoid the use of the ``print`` function, which has a buffering system that
   will not print the messages in the correct order or in real time.

To follow the progress of the correlation functions, the ``setup_logging()`` from the ``mockfactory``
package can be used. However, this method will format every log message, and this might not be
the desired behavior.

Therefore, the ``create_logger()`` method from the ``logging`` module can be used to initialize a
logger that will not be formatted by the ``setup_logging()`` method. 

.. tip::
   The ``create_logger()`` method is a wrapper around the ``logging.getLogger()`` method, and
   will return the same logger if it has already been initialized.

   It is also possible to get the same formatting as the ``setup_logging()`` method by setting the
   ``propagate`` attribute of the logger to ``True``.

Recommended logger initialization
---------------------------------

::

   from CorrHOD import create_logger
   from mockfactory import setup_logging

   # Get a logger object
   setup_logging() # Initialize the logging for all the loggers that will not have a handler 
   logger = create_logger('CorrHOD', level='debug', stream=sys.stdout)


API
---

.. automodule:: CorrHOD.logging
   :members:
   :undoc-members:
   :show-inheritance: