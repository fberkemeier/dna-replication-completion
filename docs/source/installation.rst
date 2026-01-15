Installation
============

DNAscape is currently distributed via its GitHub repository:

https://github.com/fberkemeier/DNAscape

The repository contains the core source code together with a set of Jupyter
notebooks that reproduce the main workflows and illustrate typical use cases.

Clone the repository and create an environment
--------------------

Clone DNAscape from GitHub and move into the project directory:

.. code-block:: bash

   git clone https://github.com/fberkemeier/DNAscape.git
   cd DNAscape

It is recommended to work inside a dedicated Python environment to avoid
dependency conflicts. A standard ``venv`` workflow is sufficient.

Create and activate a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

On Windows (PowerShell):

.. code-block:: powershell

   py -m venv .venv
   .\.venv\Scripts\Activate.ps1

Install DNAscape
----------------

With the environment activated, install DNAscape in editable mode from the
repository root:

.. code-block:: bash

   pip install -U pip
   pip install -e .

This installs DNAscape into the active environment while allowing changes to
the source code to be picked up immediately during development.

Install Jupyter and run notebooks (optional)
--------------------------

To run the example notebooks, install Jupyter into the same environment:

.. code-block:: bash

   pip install notebook ipykernel

If using notebooks regularly, it is recommended to register a kernel for this
environment:

.. code-block:: bash

   python -m ipykernel install --user --name dnascape --display-name "DNAscape"

Launch Jupyter from the repository root:

.. code-block:: bash

   python -m notebook

or JupyterLab

.. code-block:: bash

   python -m jupyterlab


Open one of the example notebooks (for instance,
``examples/DNAscape_tests.ipynb``) and run the cells sequentially
(see :doc:`quickstart` for a first look). DNAscape is provided as a standard Python package. Once installed, it can be
imported directly in notebooks and scripts.

For convenience, the example notebooks use:

.. code-block:: python

   from dnascape import *

This makes all public functions defined in the DNAscape API available in the
notebook namespace and simplifies interactive exploration.

Users who prefer more explicit imports may instead write:

.. code-block:: python

   from dnascape import simulate_replication

depending on their workflow.

Notes on usage
--------------

DNAscape is under active development. At this stage, most workflows are
demonstrated through Jupyter notebooks, which provide a transparent record of
data loading, mapping, simulation, and visualisation steps.

As the API stabilises, DNAscape is designed to support use as a standard Python
library, enabling its integration into scripts, pipelines, and automated
analyses beyond interactive notebook use.
