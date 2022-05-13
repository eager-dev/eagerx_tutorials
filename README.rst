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
EAGERx (Engine Agnostic Gym Environments for Robotics) enables users to easily define new tasks, switch from one sensor to another, and switch from simulation to reality with a single line of code by being invariant to the physics engine.

`The core repository is available here <https://github.com/eager-dev/eagerx>`_.

`Full documentation and tutorials (including package creation and contributing) are available here <https://eagerx.readthedocs.io/en/master/>`_.

Tutorials
=========

The following tutorials are currently available:

- `Tutorial 1: Environment Creation and Training with EAGERx <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/1_environment_creation.ipynb>`_
- `Tutorial 2: Reset and Step Function <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/2_reset_and_step.ipynb>`_
- `Tutorial 3: Converters <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/3_converters.ipynb>`_
- `Tutorial 4: Nodes and Graph Validity <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/4_nodes.ipynb>`_
- `Tutorial 5: Adding Engine Support for an Object <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/5_engine_implementation.ipynb>`_
- `Tutorial 6: More Informative Rendering <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/6_rendering.ipynb>`_
- `Tutorial 7: Reset Routines <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/7_reset_routine.ipynb>`_

A page with the exercises only is available `here <https://araffin.github.io/tools-for-robotic-rl-icra2022/notebooks/eagerx_exercises.html>`_.
The solutions are available in [here](https://github.com/eager-dev/eagerx_tutorials/tree/master/tutorials/pendulum/solutions/).

How to run the tutorials locally?
=================================

As an alternative to running the tutorials in Google Colab, they can also be run locally in order to speed up computations.

*Prequisites*:  Install `ROS1 <http://wiki.ros.org/ROS/Installation>`_ and `Poetry <https://python-poetry.org/docs/master/#installation>`_.

Clone this repository and go to its root:

.. code-block:: bash

    git clone git@github.com:eager-dev/eagerx_tutorials.git
    cd eagerx_tutorials

Install the *eagerx_tutorials* package:

.. code-block:: bash

    poetry install

*Optional*: To support eagerx visualization tools, install the *eagerx-gui* package:

.. code-block:: bash

    poetry run pip3 install eagerx-gui

Start Jupyter Lab (make sure ROS1 is sourced):

.. code-block:: bash

    poetry run jupyter lab

You will find the tutorials in the *tutorials* directory.

Cite EAGERx
===========
If you are using EAGERx for your scientific publications, please cite:

.. code:: bibtex

    @article{eagerx,
        author  = {van der Heijden, Bas and Luijkx, Jelle, and Ferranti, Laura and Kober, Jens and Babuska, Robert},
        title = {EAGERx: Engine Agnostic Gym Environment for Robotics},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/eager-dev/eagerx}}
    }

Acknowledgements
================
EAGERx is funded by the `OpenDR <https://opendr.eu/>`_ Horizon 2020 project.
