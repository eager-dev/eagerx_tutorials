
import eagerx
import eagerx_tutorials.pendulum  # Registers Pendulum
import eagerx_ode  # Registers OdeBridge
import eagerx.bridges.openai_gym  # Registers GymBridge


def test_gymbridge():
    # Initialize eagerx (starts roscore if not already started.)
    eagerx.initialize("eagerx_core")

    # Object
    pendulum = eagerx.Object.make("Pendulum", "pendulum", actuators=["u"], sensors=["theta", "dtheta", "image"], states=["model_state"])

    # Define rate in Hz
    rate = 30.0

    # Make OdeBridge
    # bridge = eagerx.Bridge.make("OdeBridge", rate=rate)
    bridge = eagerx.Bridge.make("GymBridge", rate=rate, process=eagerx.process.ENVIRONMENT)
    pendulum.config.states = []

    # Initialize empty graph
    graph = eagerx.Graph.create()

    # Add pendulum to the graph
    graph.add(pendulum)

    # Connect the pendulum to an action and observation
    # We will now explicitly set the window size
    graph.connect(action="voltage", target=pendulum.actuators.u, window=1)
    graph.connect(source=pendulum.sensors.theta, observation="angle", window=1)
    graph.connect(source=pendulum.sensors.dtheta, observation="angular_velocity", window=1)

    # Create layover node
    import eagerx_tutorials.pendulum.layover  # noqa:
    layover = eagerx.Node.make("Layover", "layover", rate)
    layover.inputs.u.space_converter = pendulum.actuators.u.space_converter
    graph.add(layover)

    # Render image
    graph.connect(source=pendulum.sensors.image, target=layover.inputs.base_image)
    graph.connect(action="voltage", target=layover.inputs.u)
    graph.render(source=layover.outputs.image, rate=rate)

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
        cost = th ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2

        # Determine when is the episode over
        # currently just a timeout after 100 steps
        done = steps > 100

        # Set info, tell the algorithm the termination was due to a timeout
        # (the episode was truncated)
        info = {"TimeLimit.truncated": steps > 100}

        return obs, -cost, done, info

    # Initialize Environment
    import eagerx_tutorials.pendulum.gym_implementation
    env = eagerx.EagerxEnv(name="PendulumEnv", rate=rate, graph=graph, bridge=bridge, step_fn=step_fn)

    # Toggle render
    env.render("human")

    # Stable Baselines3 expects flattened actions & observations
    # Convert observation and action space from Dict() to Box()
    env = Flatten(env)

    # Initialize learner
    model = sb.SAC("MlpPolicy", env, verbose=1, device="cpu")

    # Train for 1 minute (sim time)
    model.learn(total_timesteps=int(60 * rate))

    env.shutdown()


if __name__ == "__main__":
    test_gymbridge()
