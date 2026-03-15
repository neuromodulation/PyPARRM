Installation
============

PyPARRM is available on `PyPI <https://pypi.org/project/pyparrm/>`_, and
`conda-forge <https://anaconda.org/channels/conda-forge/packages/pyparrm/overview>`_
for version ≥ 1.1.1.

PyPARRM requires Python ≥ 3.10.

To install PyPARRM, activate the desired environment or project in which you want the
package, then install it using `pip <https://pip.pypa.io/en/stable/>`_:

.. code-block::
    
    pip install pyparrm

`uv <https://docs.astral.sh/uv/>`_:

.. code-block::
    
    uv pip install pyparrm

`conda <https://docs.conda.io/en/latest/>`_:

.. code-block::
    
    conda install -c conda-forge pyparrm

or `pixi <https://pixi.prefix.dev/latest/>`_:

.. code-block::
    
    pixi add pyparrm

|

If you need to create an environment or project in which to install PyPARRM, you can do
so using `venv <https://docs.python.org/3/library/venv.html>`_,
`uv <https://docs.astral.sh/uv/>`_, `pixi <https://pixi.prefix.dev/latest/>`_, or
`conda <https://docs.conda.io/en/latest/>`_.

With ``venv``
-------------

In a shell with Python available, navigate to your project location and create the
environment:

.. code-block::

    python -m venv pyparrm_env

Activate the environment using the
`appropriate venv command for your operating system and shell <https://docs.python.org/3/library/venv.html#how-venvs-work>`_,
then install the package:

.. code-block::

    pip install pyparrm

With ``uv``
-----------

In a shell with ``uv`` available, navigate to your project location and create the
environment:

.. code-block::

    uv venv pyparrm_env

Activate the environment using the
`appropriate uv command for your operating system and shell <https://docs.astral.sh/uv/pip/environments/#using-a-virtual-environment>`_,
then install the package:

.. code-block::

    uv pip install pyparrm

With ``pixi``
-------------

In a shell with ``pixi`` available, run the following commands:

.. code-block::

    pixi init
    pixi shell-hook
    pixi add pyparrm

With ``conda``
--------------

In a shell with ``conda`` available, run the following commands:

.. code-block::

    conda create -n pyparrm_env
    conda activate pyparrm_env
    conda install -c conda-forge pyparrm
