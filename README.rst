****************
EAGERx Tutorials
****************

.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: license

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: codestyle

.. image:: https://github.com/eager-dev/eagerx_tutorials/actions/workflows/ci.yml/badge.svg?branch=master
  :target: https://github.com/eager-dev/eagerx_tutorials/actions/workflows/ci.yml
  :alt: Continuous Integration

.. contents:: Table of Contents
    :depth: 2

What is the *eagerx_tutorials* package?
=======================================
This repository/package contains Jupyter Notebooks with examples on how to use EAGERx.
EAGERx (Engine Agnostic Graph Environments for Robotics) enables users to easily define new tasks, switch from one sensor to another, and switch from simulation to reality with a single line of code by being invariant to the physics engine.

`The core repository is available here <https://github.com/eager-dev/eagerx>`_.

`Full documentation and tutorials (including package creation and contributing) are available here <https://eagerx.readthedocs.io/en/master/>`_.

Tutorials
=========

The following tutorials are currently available.

**Introduction to EAGERx**

- `Tutorial 1: Getting started <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/icra/getting_started.ipynb>`_
- `Tutorial 2: Advanced usage <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/icra/advanced_usage.ipynb>`_

`The solutions are available in here <https://github.com/eager-dev/eagerx_tutorials/tree/master/tutorials/icra/solutions/>`_.

**Developer tutorials**

- `Tutorial 1: Environment Creation and Training with EAGERx <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/1_environment_creation.ipynb>`_
- `Tutorial 2: Reset and Step <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/2_reset_and_step.ipynb>`_ 
- `Tutorial 3: Space and Processors <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/3_space_and_processors.ipynb>`_
- `Tutorial 4: Nodes and Graph Validity <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/4_nodes.ipynb>`_
- `Tutorial 5: Adding Engine Support for an Object <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/5_engine_implementation.ipynb>`_
- `Tutorial 6: Defining a new Object <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/6_objects.ipynb>`_
- `Tutorial 7: More Informative Rendering <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/7_rendering.ipynb>`_
- `Tutorial 8: Reset Routines <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/8_reset_routine.ipynb>`_
`The solutions are available in here <https://github.com/eager-dev/eagerx_tutorials/tree/master/tutorials/pendulum/solutions/>`_.

How to run the tutorials locally?
=================================

As an alternative to running the tutorials in Google Colab, they can also be run locally in order to speed up computations.

.. 
   *Prequisites*:  Install `Poetry <https://python-poetry.org/docs/master/#installation>`_.

Clone this repository and go to its root:

.. code-block:: bash

    git clone git@github.com:eager-dev/eagerx_tutorials.git
    cd eagerx_tutorials

*Optional* Create and source a virtual environment, (if venv is not installed run `python3 -m pip install --user virtualenv`):

.. code-block:: bash

    python3 -m venv tutorial_env
    source tutorial_env/bin/activate

Install the *eagerx_tutorials* package:

.. code-block:: bash

    pip3 install -e .

Start Jupyter Lab:

.. code-block:: bash

    jupyter lab

You will find the tutorials in the *tutorials* directory.

Cite EAGERx
===========
If you are using EAGERx for your scientific publications, please cite:

.. code:: bibtex

      @article{vanderheijden2024eagerx,
        title={EAGERx: Graph-Based Framework for Sim2real Robot Learning},
        author={van der Heijden, Bas and Luijkx, Jelle and Ferranti, Laura and Kober, Jens and Babuska, Robert},
        journal={arXiv preprint arXiv:2407.04328},
        year={2024}
      }

Acknowledgements
================
EAGERx is funded by the `OpenDR <https://opendr.eu/>`_ Horizon 2020 project.
