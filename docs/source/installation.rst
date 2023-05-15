Installation
============

Pyparrm requires at least Python 3.10. To setup a new conda environment, use:

.. code-block:: console

    $ conda create -n pyparrm_env python=3.10 anaconda

Install then the package into the desired environment using pip:

.. code-block:: console
    
    $ pip install pyparrm

`See here for the list of requirements <_static/requirements.txt>`_.

Development
-----------

To install the package in editable mode for development, clone the `GitHub
repository <https://github.com/neuromodulation/pyparrm/tree/main>`_ and
navigate to the desired installation location, then install the package and its
`development requirements <https://github.com/neuromodulation/pyparrm/tree/main/requirements_dev.txt>`_
using pip:

.. code-block:: console
    
    $ pip install -e .
    $ pip install -r requirements_dev.txt
