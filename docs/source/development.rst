Development
===========

If you want to make changes to the package, you may wish to install it in
editable mode. To do so, first clone the `GitHub repository
<https://github.com/neuromodulation/pyparrm/tree/main>`_ to your desired
location. Once cloned, navigate to this location and install the package
alongside its `development requirements
<https://github.com/neuromodulation/pyparrm/tree/main/requirements_dev.txt>`_
using pip:

.. code-block:: console
    
    $ pip install -e .
    $ pip install -r requirements_dev.txt

If you encounter any issues with the package or wish to suggest improvements,
please submit a report on the `GitHub issues page
<https://github.com/neuromodulation/pyparrm/issues>`_.

If you have made any changes which you would like to see officially added to
the package, consider submitting a `pull request
<https://github.com/neuromodulation/pyparrm/pulls>`_. When submitting a pull
request, please check that the existing test suite passes, and if you add new
features, please make sure that these are covered in the unit tests. The tests
can be run by navigating to the ``/tests`` directory and calling `pytest
<https://docs.pytest.org/en/7.3.x/>`_:

.. code-block:: console
    
    $ pytest test_parrm.py

Please also check that the documentation can be built following any changes,
which can be done using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in
the ``/docs`` directory:

.. code-block:: console
    
    $ make html

Finally, features of the code such as compliance with established styles and
spelling errors in the documentation are also checked. Please ensure that the
code is formatted using `Black <https://black.readthedocs.io/en/stable/>`_, and
check that there are no egregious errors from the following commands:

.. code-block:: console
    
    $ pycodestyle
    $ pydocstyle
    $ codespell
