import eagerx
import pytest

GYM = 0
ODE = 1
ROS1 = 0
SP = 1


@pytest.mark.timeout(60)
@pytest.mark.parametrize("eng", [ODE])
@pytest.mark.parametrize("backend", [SP])
def test_gymengine(eng, backend):
    # Start virtual display
    from pyvirtualdisplay import Display
    display = Display(visible=False, backend="xvfb")
    display.start()

    eagerx.set_log_level(eagerx.WARN)

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Define rate in Hz
    rate = 30.0

    # Object
    from eagerx_tutorials.pendulum.objects import Pendulum
    pendulum = Pendulum.make("pendulum", actuators=["u"], sensors=["theta", "theta_dot", "image"], states=["model_state"])
    graph.add(pendulum)

    # Create reset node
    from eagerx_tutorials.pendulum.reset import ResetAngle

    u_min = pendulum.actuators.u.space.low
    u_max = pendulum.actuators.u.space.high
    reset = ResetAngle.make("angle_reset", rate=rate, gains=[2.0, 0.2, 1.0], u_range=[u_min, u_max])
    graph.add(reset)

    # Connect reset node
    graph.connect(action="voltage", target=reset.feedthroughs.u, window=1)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u, window=1)
    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(source=pendulum.sensors.theta, target=reset.inputs.theta)
    graph.connect(source=pendulum.sensors.theta_dot, target=reset.inputs.theta_dot)
    graph.connect(source=pendulum.sensors.theta, observation="angle", window=1)
    graph.connect(source=pendulum.sensors.theta_dot, observation="angular_velocity", window=1)

    # Create overlay node
    from eagerx_tutorials.pendulum.overlay import Overlay

    overlay = Overlay.make("overlay", rate)
    overlay.inputs.u.space = pendulum.actuators.u.space
    overlay.inputs.base_image.space = pendulum.sensors.image.space
    overlay.outputs.image.space = pendulum.sensors.image.space
    graph.add(overlay)

    # Render image
    graph.connect(source=pendulum.sensors.image, target=overlay.inputs.base_image)
    graph.connect(source=reset.outputs.u, target=overlay.inputs.u)
    graph.connect(source=pendulum.sensors.theta, target=overlay.inputs.theta)
    graph.render(source=overlay.outputs.image, rate=rate)

    # Make OdeEngine
    if eng == ODE:
        from eagerx_ode.engine import OdeEngine
        engine = OdeEngine.make(rate=rate)
    elif eng == GYM:
        from eagerx.engines.openai_gym.engine import GymEngine
        engine = GymEngine.make(rate=rate, process=eagerx.ENVIRONMENT)
    else:
        raise NotImplementedError("Select valid engine.")

    # Make backend
    if backend == ROS1:
        from eagerx.backends.ros1 import Ros1
        backend = Ros1.make()
    elif backend == SP:
        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()
    else:
        raise NotImplementedError("Select valid backend.")

    # Open GUI
    # graph.gui()

    # Define environment
    class PendulumEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, backend, force_start, render_mode: str = None):
            self.steps = 0
            super().__init__(name, rate, graph, engine, backend=backend, force_start=force_start, render_mode=render_mode)

        def step(self, action):
            obs = self._step(action)
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
            self.steps += 1
            terminated = False
            truncated = self.steps > 100
            info = {}

            # Render
            if self.render_mode == "human":
                self.render()
            return obs, -cost, terminated, truncated, info

        def reset(self, seed=None, options=None):
            # Reset steps counter
            self.steps = 0

            # Sample states
            states = self.state_space.sample()
            offset = np.random.rand() - 0.5
            theta = np.pi - offset if offset > 0 else -np.pi - offset
            states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")

            # Perform reset
            obs = self._reset(states)
            if self.render_mode == "human":
                self.render()
            return obs, {}

    # Initialize Environment
    import eagerx_tutorials.pendulum.gym_implementation  # noqa: registers gym implementation
    env = PendulumEnv("PendulumEnv", rate, graph, engine, backend, force_start=True)

    import numpy as np
    from eagerx.wrappers import Flatten

    # Toggle render
    env.render()

    # Stable Baselines3 expects flattened actions & observations
    # Convert observation and action space from Dict() to Box()
    env = Flatten(env)

    # Evaluate in simulation
    (_obs, _info), action = env.reset(), env.action_space.sample()
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            (_obs, _info), action = env.reset(), env.action_space.sample()
            print(f"Episode {i}")
    print("\n[Finished]")

    # # Initialize learner
    # import stable_baselines3 as sb
    # model = sb.SAC("MlpPolicy", env, verbose=1, device="auto")
    #
    # # Train for 1 minute (sim time)
    # model.learn(total_timesteps=int(3 * rate))

    env.shutdown()


if __name__ == "__main__":
    for e in [GYM, ODE]:
        for b in [ROS1, SP]:
            test_gymengine(eng=e, backend=b)
