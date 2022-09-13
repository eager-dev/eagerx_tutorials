
GYM = 0
ODE = 1
ROS1 = 0
SP = 1


def advanced_usage(backend):
    import eagerx_tutorials
    from eagerx_tutorials import helper

    # Import eagerx
    import eagerx
    eagerx.set_log_level(eagerx.WARN)

    # import eagerx_gui

    # Available sensors
    sensors = ["joint_position", "joint_velocity", "force_torque", "orientation", "position", "velocity"]
    actuators = ["joint_control"]

    # Create the GO 1 quadruped
    from eagerx_tutorials.quadruped.object import Quadruped
    robot = Quadruped.make("quadruped", actuators=actuators, sensors=sensors, rate=20)

    # Set the quadruped's control rate to 200 Hz.
    robot.actuators.joint_control.rate = 200

    # Create central pattern generator (uses Hopf Oscillator)
    from eagerx_tutorials.quadruped.cpg_gait import CpgGait
    cpg = CpgGait.make("cpg", rate=200, gait="TROT", omega_swing=16 * 3.14, omega_stance=4 * 3.14)

    # Create Cartesian control (uses the quadruped's forward kinematics)
    from eagerx_tutorials.quadruped.cartesian_control import CartesiandPDController
    cartesian_control = CartesiandPDController.make("cartesian_control", rate=200)

    # Initialize empty graph
    graph = eagerx.Graph.create([robot, cartesian_control, cpg])

    # Interconnect the nodes that results in an initial trot (that moves straight ahead).
    graph.connect(source=cpg.outputs.cartesian_pos, target=cartesian_control.inputs.cartesian_pos)
    graph.connect(source=cartesian_control.outputs.joint_pos, target=robot.actuators.joint_control)
    #@markdown Therefore, we will define an environment action called `offset`.
    graph.connect(action="offset", target=cpg.inputs.offset)
    # Select the sensors that are to be used as observations
    graph.connect(observation="joint_position", source=robot.sensors.joint_position)
    graph.connect(observation="joint_velocity", source=robot.sensors.joint_velocity)
    graph.connect(observation="force_torque", source=robot.sensors.force_torque)
    graph.connect(observation="velocity", source=robot.sensors.velocity)
    graph.connect(observation="orientation", source=robot.sensors.orientation, window=2)
    # The open-loop pattern is probably also informative to determine relevant offsets.
    graph.connect(observation="xs_zs", source=cpg.outputs.xs_zs, skip=True)

    # Create xy-plane render node
    from eagerx_tutorials.quadruped.overlay import XyPlane
    xy_plane = XyPlane.make("xy_plane", rate=5)
    # Add node to graph
    graph.add(xy_plane)

    # The node renders images, based on the x,y position sensor measurements.
    graph.connect(source=robot.sensors.position, target=xy_plane.inputs.position)

    # Select the output of the node for rendering.
    # Can be commented out for faster training
    graph.render(xy_plane.outputs.image, rate=5)

    # Define Gym Environment
    import pybullet
    import numpy as np
    import gym
    from typing import Dict, Tuple

    # Make backend
    if backend == ROS1:
        from eagerx.backends.ros1 import Ros1
        backend = Ros1.make()
    elif backend == SP:
        from eagerx.backends.single_process import SingleProcess
        backend = SingleProcess.make()
    else:
        raise NotImplementedError("Select valid backend.")

    class QuadrupedEnv(eagerx.BaseEnv):
        def __init__(self, name, rate, graph, engine, episode_timeout):
            super().__init__(name, rate, graph, engine, backend, force_start=True)

            self.steps = None
            self.timeout_steps = int(episode_timeout * rate)
            self.rate = rate  # [Hz] Sensor rate

        @property
        def observation_space(self) -> gym.spaces.Dict:
            """The Space object corresponding to valid observations.

            Per default, the observation space of all registered observations in the graph is used.
            """
            return self._observation_space

        @property
        def action_space(self) -> gym.spaces.Dict:
            """The Space object corresponding to valid actions

            Per default, the action space of all registered actions in the graph is used.
            """
            return self._action_space

        def reset(self):
            """A method that resets the environment to an initial state and returns an initial observation."""
            # Reset number of steps
            self.steps = 0

            # Sample desired states
            states = self.state_space.sample()

            # Perform reset
            obs = self._reset(states)

            # Set initial observation for skipped connection 'xs_zs`
            if "xs_zs" in obs:
                obs["xs_zs"][0][:] = [-0.01354526, -0.26941818, 0.0552178, -0.25434446]
            return obs

        def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
            """A method that runs one timestep of the environment's dynamics."""

            # Here, we apply a step (i.e. we step the graph dynamics).
            # It returns a dict containing measurements of all registered observations.
            obs = self._step(action)
            self.steps += 1

            # We have access to the last two orientation sensor measurements,
            # because we used window=2 when connecting `orientation` as an observation in the graph.
            _, _, prev_yaw = pybullet.getEulerFromQuaternion(obs["orientation"][-2])
            roll, pitch, yaw = pybullet.getEulerFromQuaternion(obs["orientation"][-1])

            # YOUR CODE HERE
            # 1. Calculate the yaw rate using prev_yaw and yaw (don't forget to scale with self.rate).
            # 2. Calculate the desired yaw_rate (20 degrees) in radians.
            # 3. Calculate the negative squared error between the desired and actual yaw rate.
            # 4. Add a little alive bonus to promote not falling down.
            reward = 0.
            # END OF YOUR CODE

            # Determine termination condition
            has_fallen = abs(np.rad2deg(roll)) > 40 or abs(np.rad2deg(pitch)) > 40
            timeout = self.steps >= self.timeout_steps

            # Determine done flag
            done = timeout or has_fallen

            # Set info about episode truncation
            info = {"TimeLimit.truncated": timeout and not has_fallen}
            return obs, reward, done, info


    # Define the pybullet engine
    from eagerx_pybullet.engine import PybulletEngine

    engine = PybulletEngine.make(rate=200, gui=False, egl=False, process=eagerx.ENVIRONMENT)

    # Initialize Environment
    episode_timeout = 10  # [s] number of seconds before timing-out an episode.
    env = QuadrupedEnv(name="QuadEnv", rate=20, graph=graph, engine=engine, episode_timeout=episode_timeout)

    # Stable-baselines
    from sb3_contrib import TQC
    from eagerx.wrappers import Flatten

    # Define hyper parameters for the TQC policy.
    hyperparams = dict(
        learning_rate=1e-3,
        tau=0.02,
        gamma=0.98,
        buffer_size=300000,
        learning_starts=100,
        use_sde=True,
        use_sde_at_warmup=True,
        train_freq=8,
        gradient_steps=10,
        verbose=1,
        top_quantiles_to_drop_per_net=0,
        policy_kwargs=dict(n_critics=1, net_arch=dict(pi=[64, 64], qf=[64, 64])),
    )

    # Initialize the model
    model = TQC("MlpPolicy", Flatten(env), **hyperparams)

    # Train for 30 episodes
    train_episodes = 30
    try:
        train_steps = int(train_episodes * episode_timeout * 20)
        # Render top-view of the quadruped's movement
        env.render("human")
        # Start training!
        model.learn(10)
        # Save the final policy
        model.save("last_policy")
    except KeyboardInterrupt:
        model.save("last_policy")

    # # Evaluate
    # from eagerx_tutorials.quadruped.evaluate import EvaluateEnv
    #
    # # Load last policy
    # model = TQC.load("last_policy")
    #
    # # Create an evaluation environment (renders 3D images).
    # eval_env = EvaluateEnv(env, graph, engine, episode_timeout=40, render="pybullet")
    # eval_env.render("human")
    #
    # # Evaluate policy
    # # helper.evaluate(model, Flatten(eval_env), episode_length=int(40*20), video_rate=20, video_prefix="3d_eval", n_eval_episodes=2)
    #
    # # Create an evaluation environment (renders xy-plane).
    # eval_env = EvaluateEnv(env, graph, engine, episode_timeout=40, render="xy-plane")
    # eval_env.render("human")
    #
    # # Evaluate policy
    # helper.evaluate(model, Flatten(eval_env), episode_length=int(40*20), video_rate=20, video_prefix="xy_eval", n_eval_episodes=2)


if __name__ == "__main__":
    for b in [SP, ROS1]:
        advanced_usage(backend=b)
