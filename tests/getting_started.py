import eagerx_tutorials
import huggingface_sb3

# Setup interactive notebook
# Required in interactive notebooks only.
from eagerx_tutorials import helper

# Import eagerx
import eagerx

eagerx.set_log_level(eagerx.WARN)

# Next, we will download a pretrained policy in order to see what a successful policy looks like.

import sys
import stable_baselines3 as sb3
from huggingface_sb3 import load_from_hub

# Download pretrained policy from hugging face
newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
custom_objects = {}
if newer_python_version:
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
checkpoint = load_from_hub(
    repo_id="sb3/ppo-Pendulum-v1",
    filename="ppo-Pendulum-v1.zip",
)

# Initialize model
pretrained_model = sb3.PPO.load(checkpoint, custom_objects=custom_objects, device="cpu")

#@markdown We will create a standard *Pendulum-v1* Gym environment in order to evaluate the policy.

import gym

# Initalize pendulum environment
env = gym.make("Pendulum-v1")

# Evaluate policy and record video
# helper.record_video(env=env, model=pretrained_model, prefix="pretrained")

# Show video
# helper.show_video("pretrained-step-0-to-step-500")

from eagerx_tutorials.pendulum.objects import Pendulum
import eagerx_tutorials.pendulum.gym_implementation  # NOOP to register Gym implementation of the pendulum.

Pendulum.info()

#@markdown We can make the parameter specification of the `Pendulum` object with the `Pendulum.make` method and can then add it to a [Graph](https://eagerx.readthedocs.io/en/master/guide/api_reference/graph/graph.html).

# Define rate (Hz)
rate = 20.0

# Initialize empty graph
graph = eagerx.Graph.create()

# Select sensors, actuators and states of Pendulum
sensors = ["theta", "theta_dot", "image"]
actuators = ["u"]
states = ["model_state", "mass", "length", "max_speed"]

# Make pendulum
pendulum = Pendulum.make("pendulum", rate=rate, actuators=actuators, sensors=sensors, states=states, render_fn="disc_pendulum_render_fn")

# Decompose angle [cos(theta), sin(theta)]
from eagerx_tutorials.pendulum.processor import DecomposedAngle
pendulum.sensors.theta.processor = DecomposedAngle.make()
pendulum.sensors.theta.space.low = -1
pendulum.sensors.theta.space.high = 1
pendulum.sensors.theta.space.shape = [2]

# Add pendulum to the graph
graph.add(pendulum)

# Connect the pendulum to an action and observations
graph.connect(action="voltage", target=pendulum.actuators.u)
graph.connect(source=pendulum.sensors.theta, observation="angle")
graph.connect(source=pendulum.sensors.theta_dot, observation="angular_velocity")

# Render image
graph.render(source=pendulum.sensors.image, rate=rate)

from typing import Dict
import numpy as np


class PendulumEnv(eagerx.BaseEnv):
    def __init__(self, name: str, rate: float, graph: eagerx.Graph, engine: eagerx.specs.EngineSpec, eval: bool):
        """Initializes an environment with EAGERx dynamics.

        :param name: The name of the environment. Everything related to this environment
                     (parameters, topics, nodes, etc...) will be registered under namespace: "/[name]".
        :param rate: The rate (Hz) at which the environment will run.
        :param graph: The graph consisting of nodes and objects that describe the environment's dynamics.
        :param engine: The physics engine that will govern the environment's dynamics.
        :param eval: If True we will create an evaluation environment, i.e. not performing domain randomization.
        """
        # Make the backend specification
        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()

        self.eval = eval

        # Maximum episode length
        self.max_steps = 270 if eval else 100

        # Step counter
        self.steps = None
        super().__init__(name, rate, graph, engine, backend, force_start=True)

    def step(self, action: Dict):
        """A method that runs one timestep of the environment's dynamics.

        :params action: A dictionary of actions provided by the agent.
        :returns: A tuple (observation, reward, done, info).

            - observation: Dictionary of observations of the current timestep.

            - reward: amount of reward returned after previous action

            - done: whether the episode has ended, in which case further step() calls will return undefined results

            - info: contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        # Take step
        observation = self._step(action)
        self.steps += 1

        # Extract observations
        cos_th, sin_th = observation["angle"][0]
        thdot = observation["angular_velocity"][0]
        u = action["voltage"][0]

        # Calculate reward
        # We want to penalize the angle error, angular velocity and applied voltage
        th = np.arctan2(sin_th, cos_th)
        cost = th ** 2 + 0.1 * (thdot / (1 + 10 * abs(th))) ** 2 + 0.01 * u ** 2

        # Determine done flag
        done = self.steps > self.max_steps

        # Set info:
        info = {"TimeLimit.truncated": self.steps > self.max_steps}

        return observation, -cost, done, info

    def reset(self) -> Dict:
        """Resets the environment to an initial state and returns an initial observation.

        :returns: The initial observation.
        """
        # Determine reset states
        states = self.state_space.sample()

        if self.eval:
            theta = 3.14 * np.random.uniform(low=0.75, high=1.0) * [-1, 1][np.random.randint(2)]
            states["pendulum/model_state"][:] = [theta, 0.0]
        else:
            # YOUR CODE HERE
            # TODO:
            # During training we want to vary the length and mass of the pendulum.
            # This will improve the robustness against model inaccuracies.
            # Randomly sample values for the mass and length of the pendulum.
            # Try to estimate the mass and length of the real pendulum system in Figure 1.
            # You can adjust the low and the high in the lines below to define the distributions for sampling.
            # Hint: the Gym pendulum is a rod, while the real pendulum is not.
            # They have different moments of inertia, therefore overestimating the length will help.

            # key = "[object_name]/[state_name]"
            # value should be of type np.ndarray

            # Sample mass (kg)
            states["pendulum/mass"] = np.random.uniform(low=0.04, high=0.04, size=()).astype("float32")  # Sample mass (kg)
            # Sample length (m)
            states["pendulum/length"] = np.random.uniform(low=0.04, high=0.04, size=()).astype("float32")  # Sample length (m)

            # END OF YOUR CODE

        # Perform reset
        observation = self._reset(states)

        # Reset step counter
        self.steps = 0
        return observation

#@markdown Next, we will create the Engines corresponding to the simulators we will use.

# Import the two supported engines
from eagerx_ode.engine import OdeEngine
from eagerx.engines.openai_gym.engine import GymEngine

# Initialize engines
gym_engine = GymEngine.make(rate=rate)
ode_engine = OdeEngine.make(rate=rate)

#@title Now we are ready to make the environments!

#@markdown We will create one with the `gym_engine` for training and one with the `ode_engine` for evaluation.

from eagerx.wrappers import Flatten

# Initialize environments
train_env = PendulumEnv(name="train", rate=rate, graph=graph, engine=gym_engine, eval=False)
eval_env = PendulumEnv(name="eval", rate=rate, graph=graph, engine=ode_engine, eval=True)

# Stable Baselines3 expects flattened actions & observations
# Convert observation and action space from Dict() to Box()
train_env = Flatten(train_env)
eval_env = Flatten(eval_env)

#@markdown Let's first check if the pretrained policy we downloaded at the beginning transfers to the simulated disc pendulum...

# helper.evaluate(pretrained_model, eval_env, episode_length=270, video_rate=rate, video_prefix="pretrained_disc")

#@title If you have added your code, you can train a policy as follows (this will take a couple of minutes in Colab).

#@markdown **NOTE: If you want to rerun code, we advice you to restart and run all code (in Colab there is the option Restart and run all under Runtime).**

# Initialize learner
model = sb3.SAC("MlpPolicy", train_env, verbose=1, learning_rate=7e-4)

# Train for 40 episodes
train_env.render("human")
model.learn(total_timesteps=int(500))
train_env.close()

# Save model
model.save("pendulum")

#@markdown Next, you can evaluate your policy again on the simulated disc pendulum.

# Create evaluation environment
eval_env = PendulumEnv(name="disc", rate=rate, graph=graph, engine=ode_engine, eval=True)
eval_env = Flatten(eval_env)

helper.evaluate(model, eval_env, episode_length=270, video_rate=rate, video_prefix="trained_disc")