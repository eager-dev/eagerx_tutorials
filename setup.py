# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['eagerx_tutorials',
 'eagerx_tutorials.pendulum',
 'eagerx_tutorials.quadruped',
 'eagerx_tutorials.quadruped.go1']

package_data = \
{'': ['*'],
 'eagerx_tutorials.quadruped.go1': ['go1_description/meshes/*',
                                    'go1_description/urdf/*']}

install_requires = \
['PyVirtualDisplay>=3.0,<4.0',
 'eagerx-ode>=0.1.25,<0.2.0',
 'eagerx-pybullet>=0.1.9,<0.2.0',
 'eagerx>=0.1.30,<0.2.0',
 'ipywidgets>=7.7.0,<8.0.0',
 'jupyterlab>=3.3.4,<4.0.0',
 'nbconvert>=6.5.0,<7.0.0',
 'sb3-contrib>=1.5.0,<2.0.0',
 'stable-baselines3>=1.2,<3.0',
 'tqdm>=4.64.0,<5.0.0']

setup_kwargs = {
    'name': 'eagerx-tutorials',
    'version': '0.1.19',
    'description': 'Tutorials on how to use EAGERx.',
    'long_description': None,
    'author': 'Jelle Luijkx',
    'author_email': 'j.d.luijkx@tudelft.nl',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://github.com/eager-dev/eagerx_tutorials',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
