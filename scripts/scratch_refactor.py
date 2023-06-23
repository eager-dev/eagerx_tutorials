#@title Notebook Setup

#@markdown In order to be able to run the code, we need to install the *eagerx_tutorials* package.
import eagerx_tutorials

# Setup interactive notebook
# Required in interactive notebooks only.
from eagerx_tutorials import helper
# helper.setup_notebook()

# Import eagerx
import eagerx
eagerx.set_log_level(eagerx.WARN)

from eagerx_tutorials.pendulum.objects import Pendulum

Pendulum.info()

#@markdown We can make the *Pendulum* object with the `eagerx.Object.make` method with (a unique) *name*. Furthermore, we will specify which actuators, sensors and states we will use.

# Make pendulum
pendulum = Pendulum.make("pendulum", actuators=["u"], sensors=["theta", "theta_dot", "image"], states=["model_state"])

#@markdown Next, we create a [Graph](https://eagerx.readthedocs.io/en/master/guide/api_reference/graph/graph.html) and add the pendulum to it.

# Define rate (depends on rate of ode)
rate = 30.0

# Initialize empty graph
graph = eagerx.Graph.create()

# Add pendulum to the graph
graph.add(pendulum)

# Connect the pendulum to an action and observation
graph.connect(action="voltage", target=pendulum.actuators.u)
graph.connect(source=pendulum.sensors.theta, observation="angle")
graph.connect(source=pendulum.sensors.theta_dot, observation="angular_velocity")

# Render image
graph.render(source=pendulum.sensors.image, rate=rate)

from typing import Dict
import numpy as np


class PendulumEnv(eagerx.BaseEnv):
	def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec,
	             backend: eagerx.specs.BackendSpec,
	             render_mode: str = "rgb_array"):
		"""Initializes an environment with EAGERx dynamics.

		:param name: The name of the environment. Everything related to this environment
					 (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
		:param rate: The rate (Hz) at which the environment will run.
		:param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
		:param engine: The physics engine that will govern the environment's dynamics.
		:param backend: The backend that manages all communication and the parameter server.
		:param render_mode: Defines the render mode (e.g. "human", "rgb_array").
		"""
		self.eval = eval

		# Maximum episode length
		self.max_steps = 100

		# Step counter
		self.steps = None
		super().__init__(name, rate, graph, engine, backend, force_start=True, render_mode=render_mode)

	def step(self, action: Dict):
		"""A method that runs one timestep of the environment's dynamics.

		:params action: A dictionary of actions provided by the agent.
		:returns: A tuple (observation, reward, done, info).

			- observation: Dictionary of observations of the current timestep.

			- reward: amount of reward returned after previous action

			- truncated: wether the episode ended due to a timeout

			- done: whether the episode has ended, in which case further step() calls will return undefined results

			- info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
		"""
		# Take step
		observation = self._step(action)
		self.steps += 1

		# Get angle and angular velocity
		# Take first element because of window size (covered in other tutorial)
		th = observation["angle"][0]
		thdot = observation["angular_velocity"][0]

		# Convert from numpy array to float
		u = float(action["voltage"])

		# Normalize angle so it lies in [-pi, pi]
		th -= 2 * np.pi * np.floor((th + np.pi) / (2 * np.pi))

		# Calculate cost
		# Penalize angle error, angular velocity and input voltage
		cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2

		# Determine when is the episode over
		# currently just a timeout after 100 steps
		done = self.steps > self.max_steps
		truncated = self.steps > self.max_steps

		# Set info, tell the algorithm the termination was due to a timeout
		# (the episode was truncated)
		info = {"TimeLimit.truncated": self.steps > self.max_steps}

		# Render
		if self.render_mode == "human":
			self.render()
		return observation, -cost, truncated, done, info

	def reset(self, seed=None, options=None):
		"""Resets the environment to an initial state and returns an initial observation.

		:returns: The initial observation.
		"""
		# Determine reset states
		states = self.state_space.sample()

		# Perform reset
		observation = self._reset(states)
		info = {}

		# Reset step counter
		self.steps = 0

		# Render
		if self.render_mode == "human":
			self.render()
		return observation, info

#@markdown Next, we will make a specification with which we can initialize an [Engine](https://eagerx.readthedocs.io/en/master/guide/api_reference/engine/index.html).

from eagerx_ode.engine import OdeEngine

# Make the engine specification
engine = OdeEngine.make(rate=rate)

#@markdown Then, we will make a specification with which we can initialize an [Backend](https://eagerx.readthedocs.io/en/master/guide/api_reference/backend/index.html).

from eagerx.backends.single_process import SingleProcess

# Make the backend specification
backend = SingleProcess.make()

#@title Training

#@markdown Having created a graph, an engine and a step function, we can now construct the EAGERx environment. We can use it like any other Gym environment. Here we will now train a policy to swing up the pendulum using the Soft Actor Critic (SAC) reinforcement learning algorithm implementation from [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from eagerx.wrappers import Flatten
from gymnasium.wrappers.rescale_action import RescaleAction
from gym.wrappers.rescale_action import RescaleAction

# Initialize Environment
env = PendulumEnv(name="PendulumEnv", rate=rate, graph=graph, engine=engine, backend=backend, render_mode="human")

# Print action & observation space
print("action_space: ", env.action_space)
print("observation_space: ", env.observation_space)

# Stable Baselines3 expects flattened actions & observations
# Convert observation and action space from Dict() to Box(), normalize actions
env = Flatten(env)
env = helper.RescaleAction(env, min_action=-1.0, max_action=1.0)

# Check that env follows Gym API and returns expected shapes
check_env(env)

# Initialize learner
model = sb3.SAC("MlpPolicy", env, verbose=1)

# Train for 1 minute (sim time)
model.learn(total_timesteps=int(150 * rate))

env.shutdown()