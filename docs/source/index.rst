.. title:: Home

.. define new line for html
.. |br| raw:: html

   <br />

.. image:: https://tsbinns.com/assets/pyparrm/logo.gif
   :alt: PyPARRM

|br|
A Python signal processing package for identifying and removing stimulation
artefacts from electrophysiological data using the Period-based Artefact
Reconstruction and Removal Method (PARRM) of Dastin-van Rijn *et al.* (2021)
(https://doi.org/10.1016/j.crmeth.2021.100010).

This package is based on the original MATLAB implementation of the method
(https://github.com/neuromotion/PARRM). All credit for the method goes to its
original authors.

Parallel processing and `Numba <https://numba.pydata.org/>`_ optimisation are
implemented to reduce computation times.

.. toctree::
   :maxdepth: 3
   :titlesonly:
   :caption: Contents:

   installation
   examples
   api
