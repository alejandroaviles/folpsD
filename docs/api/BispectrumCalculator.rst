BispectrumCalculator
====================

Description
-----------

Class for bispectrum computations derived from the nonlinear tables. Includes a Sugiyama basis implementation.

Usage
-----

.. code-block:: python

   from folps import BispectrumCalculator

   bispec = BispectrumCalculator(model='FOLPSD')
   B000, B202 = bispec.Bisp_Sugiyama(...)
