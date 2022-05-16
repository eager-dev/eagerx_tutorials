
import eagerx
import eagerx_tutorials.pendulum  # Registers Pendulum
import eagerx_ode  # Registers OdeEngine
import eagerx.engines.openai_gym  # Registers GymEngine


def test_gymengine():
    # Initialize eagerx (starts roscore if not already started.)
    eagerx.initialize("eagerx_core", log_level=eagerx.log.INFO)

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Define rate in Hz
    rate = 30.0

    # Object
    pendulum = eagerx.Object.make("Pendulum", "pendulum", actuators=["u"], sensors=["theta", "dtheta", "image"], states=["model_state"])
    graph.add(pendulum)

    # Create reset node
    import eagerx_tutorials.pendulum.reset  # noqa: Registers reset node
    u_min = pendulum.actuators.u.space.low
    u_max = pendulum.actuators.u.space.high
    reset = eagerx.ResetNode.make("ResetAngle", "angle_reset", rate=rate, gains=[2.0, 0.2, 1.0], u_range=[u_min, u_max])
    graph.add(reset)

    # Connect reset node
    graph.connect(action="voltage", target=reset.feedthroughs.u, window=1)
    graph.connect(source=pendulum.sensors.theta, observation="angle", window=1)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u, window=1)
    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(source=pendulum.sensors.theta, target=reset.inputs.theta)
    graph.connect(source=pendulum.sensors.dtheta, target=reset.inputs.dtheta)
    graph.connect(source=pendulum.sensors.dtheta, observation="angular_velocity", window=1)

    # Create overlay node
    import eagerx_tutorials.pendulum.overlay  # noqa:
    overlay = eagerx.Node.make("Overlay", "overlay", rate)
    overlay.inputs.u.space = pendulum.actuators.u.space
    graph.add(overlay)

    # Render image
    graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
    graph.connect(source=reset.outputs.u, target=overlay.inputs.u)
    graph.connect(source=pendulum.sensors.theta, target=overlay.inputs.theta)
    graph.render(source=overlay.outputs.image, rate=rate)

    # Make OdeEngine
    engine = eagerx.Engine.make("OdeEngine", rate=rate)
    # engine = eagerx.Engine.make("GymEngine", rate=rate, process=eagerx.process.ENVIRONMENT)

    # Open GUI
    # graph.gui()

    import numpy as np
    from typing import Dict, Tuple
    import stable_baselines3 as sb
    from eagerx.wrappers import Flatten

    from eagerx.core.env import BaseEnv
    import gym

    class PendulumEnv(eagerx.core.env.BaseEnv):
        def __init__(self, name, rate, graph, engine, force_start=True):
            super(PendulumEnv, self).__init__(name, rate, graph, engine, force_start=force_start)
            self.steps = None

        @property
        def observation_space(self) -> gym.spaces.Dict:
            return self._observation_space

        @property
        def action_space(self) -> gym.spaces.Dict:
            return self._action_space

        def reset(self):
            # Reset number of steps
            self.steps = 0

            # Sample desired states
            states = self.state_space.sample()
            offset = np.random.rand() - 0.5
            theta = np.pi - offset if offset > 0 else -np.pi - offset
            states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")

            # Perform reset
            obs = self._reset(states)
            return obs

        # todo: UPDATE DOCS
        def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
            obs = self._step(action)
            self.steps += 1

            # Get angle and angular velocity
            # Take first element because of window size (covered in other tutorial)

            # START ASSIGNMENT 1.3
            th = obs["angle"][0]
            # END ASSIGNMENT 1.3

            thdot = obs["angular_velocity"][0]

            # Convert from numpy array to float
            u = float(action["voltage"])

            # Calculate cost
            # Penalize angle error, angular velocity and input voltage
            cost = th ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2

            # Determine when is the episode over
            # currently just a timeout after 100 steps
            done = self.steps > 10000

            # Set info, tell the algorithm the termination was due to a timeout
            # (the episode was truncated)
            info = {"TimeLimit.truncated": self.steps > 100}

            return obs, -cost, done, info

    # Initialize Environment
    import eagerx_tutorials.pendulum.gym_implementation
    env = PendulumEnv(name="PendulumEnv", rate=rate, graph=graph, engine=engine)

    # Toggle render
    env.render("human")

    # Stable Baselines3 expects flattened actions & observations
    # Convert observation and action space from Dict() to Box()
    env = Flatten(env)

    # Initialize learner
    model = sb.SAC("MlpPolicy", env, verbose=1, device="cpu")

    action = env.action_space.sample()*0
    while True:
        _, done = env.reset(), False
        while not done:
            obs, reward, done, info = env.step(action)

    # Train for 1 minute (sim time)
    model.learn(total_timesteps=int(10000 * rate))

    env.shutdown()


if __name__ == "__main__":
    test_gymengine()
