Installation
============

You can install ``PaTSEmb`` in various ways, described below, but the preferred way to install
``PaTSEmb`` is via PyPI.

.. note::
   To mine frequent, sequential patterns, ``PaTSEmb`` relies on the SPMF-library. Therefore,
   you also need to have Java 1.7 or higher installed on your machine.

From PyPI
---------

The easiest way to install ``PaTSEmb`` is via the `PyPI <https://pypi.org/project/patsemb/>`_, by
simply running the following command:

.. code-block:: bash

    pip install patsemb

From GitHub
-----------

You can also install ``PaTSEmb`` directly from `GitHub <https://github.com/ML-KULeuven/PaTSEmb>`_.
To install version ``X.Y.Z``, you can use the following command:

.. code-block:: bash

    pip install git+https://github.com/ML-KULeuven/PaTSEmb.git@X.Y.Z

The `release page <https://github.com/ML-KULeuven/PaTSEmb/releases>`_ contains more
information regarding the different versions. It is also possible to install the
latest, *unreleased* version using the following command:

.. code-block:: bash

    pip install git+https://github.com/ML-KULeuven/PaTSEmb.git

From source
-----------

It is also possible to install ``PaTSEmb`` directly from the source code. First, download
the source from `GitHub <https://github.com/ML-KULeuven/PaTSEmb.git>`_. It is also
possible to download the source code for a specific release on `the release page <https://github.com/ML-KULeuven/PaTSEmb/releases>`_.
Unzip the files, and navigate to the root directory of the repository in the terminal.
Finally, ``PaTSEmb`` can be installed through the following command:

.. code-block:: bash

    pip install .
