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

Installation
============

You can install the package using pip:

.. code:: shell

    pip3 install eagerx-tutorials

.. note::
    EAGERx depends on a minimal ROS installation. Fortunately, you **can** use eagerx anywhere as you would any python package,
    so it does **not** impose a ROS package structure on your project.

Tutorials
=========

The following tutorials are currently available:

Pendulum:

- `Tutorial 1: EAGERx Environment Creation and Training <https://colab.research.google.com/github/eager-dev/eagerx_tutorials/blob/master/tutorials/pendulum/pendulum_1.ipynb>`_

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
=================
EAGERx is funded by the `OpenDR <https://opendr.eu/>`_ Horizon 2020 project.
