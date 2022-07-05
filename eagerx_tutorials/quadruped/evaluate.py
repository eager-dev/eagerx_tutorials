import numpy as np
from typing import Dict, Tuple
import eagerx
import pybullet
import gym
import copy


class EvaluateEnv(eagerx.BaseEnv):
    def __init__(self, env, engine, backend, episode_timeout, render="pybullet"):
        self.rate = env.rate
        graph = copy.deepcopy(env.graph)
        self._wrapped = env
        if render == "pybullet":
            name = f"{env.name}_pybullet_eval"
            robot = graph.get_spec("quadruped")
            graph.set(5, robot.sensors.image, parameter="rate")
            graph.add_component(robot.sensors.image)
            graph.render(robot.sensors.image, rate=5)
            graph.remove("xy_plane")
        else:
            name = f"{env.name}_xyplane_eval"
            xy_plane = graph.get_spec("xy_plane")
            graph.render(xy_plane.outputs.image, rate=5)

        super(EvaluateEnv, self).__init__(name, self.rate, graph, engine, backend, force_start=True)
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
        if "xs_zs" in obs and self.steps == 0:
            obs["xs_zs"][0][:] = [-0.01354526, -0.26941818, 0.0552178, -0.25434446]
        return obs

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        obs = self._step(action)
        self.steps += 1

        done = self.steps >= self.timeout_steps

        # Determine done flag
        # Set info about episode truncation
        info = {"TimeLimit.truncated": done}
        return obs, 0.0, done, info
