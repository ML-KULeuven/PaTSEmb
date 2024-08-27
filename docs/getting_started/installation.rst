Installation
============

You can install ``PaTSEmb`` in various ways, described below, but the preferred way to install
``PaTSEmb`` is via `PyPi <_install_from_pypi>`_.

.. note::
   To mine frequent, sequential patterns, ``PaTSEmb`` relies on the SPMF-library. Therefore,
   you also need to have Java 1.7 or higher installed on your machine.

.. _install_from_pypi:

From PyPi
---------

The easiest way to install ``PaTSEmb`` is via the `PyPi <https://pypi.org/project/patsemb/>`_, by
simply running the following command:

.. code-block:: bash

    pip install patsemb

From GitLab
-----------

You can also install ``PaTSEmb`` directly from `GitLab <https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb>`_.
To install version ``X.Y.Z``, you can use the following command:

.. code-block:: bash

    pip install git+https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb.git@X.Y.Z

The `release page <https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb/-/releases>`_ contains more
information regarding the different versions. It is also possible to install the
latest, *unreleased* version using the following command:

.. code-block:: bash

    pip install git+https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb.git

From source
-----------

It is also possible to install ``PaTSEmb`` directly from the source code. First, download
the source from `GitLab <https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb>`_. It is also
possible to download the source code for a specific release on `the release page <https://gitlab.kuleuven.be/m-group-campus-brugge/dtai_public/patsemb/-/releases>`_.
Unzip the files, and navigate to the root directory of the repository in the terminal.
Finally, ``PaTSEmb`` can be installed through the following command:

.. code-block:: bash

    pip install .
