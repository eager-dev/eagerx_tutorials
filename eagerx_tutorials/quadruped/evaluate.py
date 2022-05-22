import numpy as np
from typing import Dict, Tuple
import eagerx
import pybullet
import gym


class EvaluateEnv(eagerx.BaseEnv):
    def __init__(self, env, engine, episode_timeout, render="pybullet"):
        name = f"{env.name}_eval"
        self.rate = env.rate
        graph = env.graph
        self._wrapped = env
        if render == "pybullet":
            robot = graph.get_spec("quadruped")
            graph.set(5, robot.sensors.image, parameter="rate")
            graph.add_component(robot.sensors.image)
            graph.render(robot.sensors.image, rate=5)
            graph.remove("xy_plane")

        super(EvaluateEnv, self).__init__(name, self.rate, graph, engine, force_start=True)
        self.timeout_steps = int(episode_timeout * self.rate)
        self.steps = None

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return self._wrapped.observation_space

    @property
    def action_space(self) -> gym.spaces.Dict:
        return self._wrapped.action_space

    def reset(self):
        # Reset number of steps
        self.steps = 0

        # Sample desired states
        states = self.state_space.sample()

        # set camera location
        if "quadruped/image/pos" in states:
            states["quadruped/image/pos"] = np.array([-1, -1, 0.5])
        if "quadruped/image/orientation" in states:
            tmp = list(pybullet.getQuaternionFromEuler(np.deg2rad([-90, 0, 0])))
            states["quadruped/image/orientation"] = np.array(tmp)

        # Perform reset
        obs = self._reset(states)
        return obs

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        obs = self._step(action)
        self.steps += 1

        done = self.steps >= self.timeout_steps

        # Determine done flag
        # Set info about episode truncation
        info = {"TimeLimit.truncated": done}
        return obs, 0.0, done, info
