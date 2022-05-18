import eagerx
import eagerx_tutorials.pendulum  # Registers Pendulum
import eagerx_ode  # Registers OdeEngine
import eagerx.engines.openai_gym  # Registers GymEngine


def test_gymengine():
    # Initialize eagerx (starts roscore if not already started.)
    eagerx.initialize("eagerx_core")

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Define rate in Hz
    rate = 30.0

    # Object
    pendulum = eagerx.Object.make(
        "Pendulum", "pendulum", actuators=["u"], sensors=["theta", "dtheta", "image"], states=["model_state"]
    )
    graph.add(pendulum)

    # Create reset node
    import eagerx_tutorials.pendulum.reset  # noqa: Registers reset node

    u_min = pendulum.actuators.u.space_converter.low[0]
    u_max = pendulum.actuators.u.space_converter.high[0]
    reset = eagerx.ResetNode.make("ResetAngle", "angle_reset", rate=rate, gains=[2.0, 0.2, 1.0], u_range=[u_min, u_max])
    graph.add(reset)

    # Connect reset node
    graph.connect(action="voltage", target=reset.feedthroughs.u, window=1)
    graph.connect(source=reset.outputs.u, target=pendulum.actuators.u, window=1)
    graph.connect(source=pendulum.states.model_state, target=reset.targets.goal)
    graph.connect(source=pendulum.sensors.theta, target=reset.inputs.theta)
    graph.connect(source=pendulum.sensors.dtheta, target=reset.inputs.dtheta)
    graph.connect(source=pendulum.sensors.theta, observation="angle", window=1)
    graph.connect(source=pendulum.sensors.dtheta, observation="angular_velocity", window=1)

    # Create overlay node
    import eagerx_tutorials.pendulum.overlay  # noqa:

    overlay = eagerx.Node.make("Overlay", "overlay", rate)
    overlay.inputs.u.space_converter = pendulum.actuators.u.space_converter
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
    graph.gui()

    import numpy as np
    from typing import Dict
    import stable_baselines3 as sb
    from eagerx.wrappers import Flatten

    # Define step function
    def step_fn(prev_obs: Dict[str, np.ndarray], obs: Dict[str, np.ndarray], action: Dict[str, np.ndarray], steps: int):
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
        cost = th**2 + 0.1 * thdot**2 + 0.001 * u**2

        # Determine when is the episode over
        # currently just a timeout after 100 steps
        done = steps > 100

        # Set info, tell the algorithm the termination was due to a timeout
        # (the episode was truncated)
        info = {"TimeLimit.truncated": steps > 100}

        return obs, -cost, done, info

    def reset_fn(environment):
        states = environment.state_space.sample()
        offset = np.random.rand() - 0.5
        theta = np.pi - offset if offset > 0 else -np.pi - offset
        states["pendulum/model_state"] = np.array([theta, 0], dtype="float32")
        return states

    # Initialize Environment
    import eagerx_tutorials.pendulum.gym_implementation

    env = eagerx.EagerxEnv(name="PendulumEnv", rate=rate, graph=graph, engine=engine, step_fn=step_fn, reset_fn=reset_fn)

    # Toggle render
    env.render("human")

    # Stable Baselines3 expects flattened actions & observations
    # Convert observation and action space from Dict() to Box()
    env = Flatten(env)

    # Initialize learner
    model = sb.SAC("MlpPolicy", env, verbose=1, device="cpu")

    # Train for 1 minute (sim time)
    model.learn(total_timesteps=int(10 * rate))

    env.shutdown()


if __name__ == "__main__":
    test_gymengine()
